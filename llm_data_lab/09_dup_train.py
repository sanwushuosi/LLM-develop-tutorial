#!/usr/bin/env python3
"""第 9 步：重复数据到底伤在哪里？——同样的训练量，数据重复的程度不同。

固定训练预算（默认 800 步 × 32 × 128 字节 ≈ 3.3 MB），只用英文网页，比较五种数据：
  unique     从 4 MB 数据里取，不到 1 轮，几乎不重复
  rep4       只用 1/4 的数据，重复约 4 轮
  rep16      只用 1/16，重复约 16 轮
  rep64      只用 1/64，重复约 64 轮
  boiler10   数据不重复，但 10% 的文档开头插入同一段模板（cookie 告知）——模拟“没去掉的跨站模板”
  boiler30   同上，30% 的文档
每个模型报告：
  · 留出验证集 BPB（泛化：越低越好）
  · 训练片段 BPB 与验证 BPB 之差（记忆：差得越多越“背书”）
  · 逐字复现率：给训练片段的前 64 字节，贪心续写 64 字节，与原文逐字相同的比例
  · 模板 BPB 与续写：模型对那段 cookie 模板“背”得有多熟
对照手册 4.3.1 节与 Muennighoff 等（2023）：重复约 4 轮影响很小，再多收益迅速递减；少量高度重复的内容会被逐字记住。
用法：python 09_dup_train.py [--workers 4] [--steps 800]
输出：work/09_dup/results.jsonl 与 report.html
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F

from labkit import report as R
from labkit.mixture import eval_bpb, generate, train_one

TEMPLATE = ("This website uses cookies to improve your experience. We'll assume you're ok with this, "
            "but you can opt-out if you wish. Accept Read More Privacy & Cookies Policy\n")


def build(train, cond, budget, seed=0):
    if cond == "unique":
        return train
    if cond.startswith("rep"):
        k = int(cond[3:])
        return train[:max(budget // k, 4096)]
    if cond.startswith("boiler"):                 # boiler10 / boiler30：10% / 30% 的文档开头插入同一段模板
        frac = int(cond[6:]) / 100
        docs = bytes(train).split(b"\n\n")
        rng = np.random.default_rng(seed)
        tb = TEMPLATE.encode()
        out = [tb + d if rng.random() < frac else d for d in docs]
        return np.frombuffer(b"\n\n".join(out), dtype=np.uint8)
    raise ValueError(cond)


@torch.no_grad()
def aligned_bpb(model, data, T, n=256):
    """在训练时的切分位置（每 T+1 字节一段）上算 BPB：模型“背”得多熟。"""
    starts = list(range(0, len(data) - T - 1, T + 1))
    idx = np.linspace(0, len(starts) - 1, min(n, len(starts))).astype(int)
    x = torch.from_numpy(np.stack([data[starts[i]:starts[i] + T + 1] for i in idx]).astype(np.int64))
    model.eval()
    tot = 0.0
    for k in range(0, len(x), 32):
        xb = x[k:k + 32]
        tot += F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1), reduction="sum").item()
    model.train()
    return tot / (len(x) * T) / math.log(2)


@torch.no_grad()
def exact_prefix(model, data, T, aligned, n=40, prefix=64, cont=64, seed=0):
    """给 prefix 字节，贪心续写；返回“第一个错字节之前连续正确的字节数”的平均值（上限 cont）。
    aligned=True 时从训练切分位置开头取样（检验逐字背诵），否则随机位置。"""
    rng = np.random.default_rng(seed)
    starts = list(range(0, len(data) - T - 1, T + 1)) if aligned else None
    model.eval()
    lens = []
    for _ in range(n):
        s = int(starts[rng.integers(0, len(starts))]) if aligned else int(rng.integers(0, len(data) - prefix - cont - 1))
        ids = [int(v) for v in data[s:s + prefix]]
        target = [int(v) for v in data[s + prefix:s + prefix + cont]]
        k = 0
        for j in range(cont):
            nxt = int(model(torch.tensor([ids[-T:]]))[0, -1].argmax())
            if nxt != target[j]:
                break
            ids.append(nxt)
            k += 1
        lens.append(k)
    model.train()
    return float(np.mean(lens))


@torch.no_grad()
def template_probe(model, T, head=40):
    """给模板开头 head 个字节，贪心续写，看能逐字续出多少。"""
    tb = list(TEMPLATE.encode())
    ids, rest = tb[:head], tb[head:]
    model.eval()
    k = 0
    out = []
    for j in range(len(rest)):
        nxt = int(model(torch.tensor([ids[-T:]]))[0, -1].argmax())
        out.append(nxt)
        ids.append(nxt)
        if k == j and nxt == rest[j]:
            k += 1
    model.train()
    return k, len(rest), bytes(tb[:head] + out).decode("utf-8", "replace")


def run(args):
    cond, pool_dir, steps, threads = args
    tr = np.fromfile(Path(pool_dir) / "web_en" / "train.bin", dtype=np.uint8)
    va = np.fromfile(Path(pool_dir) / "web_en" / "val.bin", dtype=np.uint8)
    T, B = 128, 32
    budget = steps * B * T
    data = build(tr, cond, budget)
    pool = {"web_en": {"train": data, "val": va}}
    res, model = train_one(pool, {"web_en": 1.0}, steps=steps, T=T, batch=B, threads=threads, gen=False)
    tb = np.frombuffer(TEMPLATE.encode() * 3, dtype=np.uint8)
    torch.set_num_threads(threads)
    out = {"cond": cond, "unique_train_bytes": int(len(data)), "epochs": res["epochs"]["web_en"],
           "val_bpb": res["val_bpb"]["web_en"],
           "train_aligned_bpb": round(aligned_bpb(model, data, T), 4),
           "train_shifted_bpb": round(eval_bpb(model, data, T, n_batches=12), 4),
           "exact_train": round(exact_prefix(model, data, T, aligned=True), 1),
           "exact_val": round(exact_prefix(model, va, T, aligned=False), 1),
           "template_bpb": round(eval_bpb(model, tb, T, n_batches=1, batch=8), 4),
           "gen_free": generate(model, "The city council announced on Monday that", n=160, T=T),
           "seconds": res["seconds"]}
    k, n, txt = template_probe(model, T)
    out.update(template_exact=k, template_len=n, gen_template=txt)
    out["gap"] = round(out["val_bpb"] - out["train_aligned_bpb"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/mix_pool")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--conds", default="unique,rep4,rep16,rep64,boiler10,boiler30")
    ap.add_argument("--out", default="work/09_dup")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = a.conds.split(",")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, [(c, a.pool, a.steps, a.threads) for c in conds]))
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'条件':10}{'独立数据':>9}{'轮数':>6}{'验证':>7}{'训练原位':>8}{'训练错位':>8}{'逐字(训)':>8}{'逐字(验)':>8}{'模板BPB':>8}{'模板续写':>9}")
    for r in res:
        print(f"{r['cond']:10}{r['unique_train_bytes'] / 1e6:8.2f}M{r['epochs']:6.1f}{r['val_bpb']:7.3f}{r['train_aligned_bpb']:8.3f}"
              f"{r['train_shifted_bpb']:8.3f}{r['exact_train']:8.1f}{r['exact_val']:8.1f}{r['template_bpb']:8.3f}"
              f"{r['template_exact']:>5}/{r['template_len']}")
    for r in res:
        print(f"--- {r['cond']} 模板续写（贪心）→ {r['gen_template'][:160]!r}")
    import matplotlib.pyplot as plt
    reps = [r for r in res if r["cond"] == "unique" or r["cond"].startswith("rep")]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    xs = [max(r["epochs"], 0.5) for r in reps]
    ax.plot(xs, [r["val_bpb"] for r in reps], "o-", label="验证集（没见过的文本）", color=R.COLORS[0])
    ax.plot(xs, [r["train_shifted_bpb"] for r in reps], "^-", label="训练文本·错开切分位置", color=R.COLORS[2])
    ax.plot(xs, [r["train_aligned_bpb"] for r in reps], "s--", label="训练片段·原切分位置", color=R.COLORS[1])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("同一份数据被重复的轮数（训练总量不变）")
    ax.set_ylabel("BPB（越低越好）")
    ax.legend(fontsize=8)
    ax.set_title("重复越多：背得越熟，泛化越差")
    S = ["<h2>1. 重复轮数与泛化</h2>", R.fig_to_img(fig),
         R.table([(r["cond"], f"{r['unique_train_bytes'] / 1e6:.2f} MB", f"{r['epochs']:.1f}", f"{r['val_bpb']:.3f}",
                   f"{r['train_aligned_bpb']:.3f}", f"{r['train_shifted_bpb']:.3f}", f"{r['exact_train']:.1f}", f"{r['exact_val']:.1f}",
                   f"{r['template_bpb']:.3f}", f"{r['template_exact']}/{r['template_len']}") for r in res],
                 ["条件", "独立数据量", "轮数", "验证 BPB", "训练片段·原位置", "训练文本·错位", "逐字续写（训练，字节）",
                  "逐字续写（验证，字节）", "模板 BPB", "模板逐字续写"]),
         "<div class='note'>“逐字续写”= 给 64 字节开头，贪心续写，第一个错字节之前连续写对了多少字节（最多 64）。"
         "“训练片段·原位置”是在训练时的切分边界上测；同样的文字错开几个字节就是“错位”。"
         "重复很多轮后，模型能把训练片段逐字背出来，但只认得那几个固定的切片——这是记忆，不是理解。</div>",
         "<h2>2. 用模板开头贪心续写</h2>",
         R.table([(r["cond"], r["gen_template"][:200]) for r in res], ["条件", "续写（贪心）"]),
         "<div class='note'>boiler10 / boiler30 条件下，同一段 cookie 告知出现在约 10% / 30% 的文档开头。对照其他条件，看模型对这段模板“熟”到什么程度。"
         "放大到万亿 token：没去掉的模板、被转载上千次的文章，会挤占模型容量、被逐字记住，带来隐私与版权风险。</div>"]
    R.page("第 9 步：重复数据的代价", S, out / "report.html")
    print(f"报告：{out / 'report.html'}；用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
