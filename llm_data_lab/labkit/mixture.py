"""配比实验：在 CPU 上用几十万参数的字节级小模型，观察“数据配比 → 各领域能力”的变化。

  数据池：每个领域一份 train.bin / val.bin（UTF-8 字节流，文档之间用空行分隔），见 08_mix_data.py；
  采样：手册 4.6.3 节的 MixtureSampler 原样使用——每一步先按配比选领域，再在该领域内按“每轮一个打乱顺序”取片段，
        因此可以直接读出每个领域被重复了多少轮（4.4.4 节）；
  模型：手册 2.6.1 节的 MiniGPT（V=256，按字节建模，中文一个字 3 个字节）；
  评测：每个领域留出的验证集上的 BPB（bits per byte，越低越好）。BPB 与分词无关，可以跨领域、跨语言比较。
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .book import MixtureSampler
from .book_model import MiniGPT

DOMAINS = ["web_en", "web_zh", "pdf", "code", "math"]
DOMAIN_CN = {"web_en": "英文网页", "web_zh": "中文网页", "pdf": "文档(PDF)", "code": "代码", "math": "数学"}
V = 256


def load_pool(pool_dir, domains=None):
    pool = {}
    for d in domains or DOMAINS:
        p = Path(pool_dir) / d
        if (p / "train.bin").exists():
            pool[d] = {"train": np.fromfile(p / "train.bin", dtype=np.uint8),
                       "val": np.fromfile(p / "val.bin", dtype=np.uint8)}
    return pool


def parse_mix(s, domains):
    """'web_en=0.4,code=0.2,...' → 归一化后的 {领域: 权重}；未写的领域权重为 0。"""
    w = {d: 0.0 for d in domains}
    for part in s.split(","):
        if part.strip():
            k, v = part.split("=")
            if k.strip() not in w:
                raise ValueError(f"未知领域 {k}，可选：{domains}")
            w[k.strip()] = float(v)
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def mix_str(w):
    return ",".join(f"{k}={v:.3f}" for k, v in w.items() if v > 0)


@torch.no_grad()
def eval_bpb(model, data, T, n_batches=24, batch=32):
    """在固定位置的验证片段上计算 BPB（确定性，便于不同运行之间比较）。"""
    model.eval()
    n = len(data) - T - 1
    starts = np.linspace(0, n - 1, n_batches * batch).astype(np.int64)
    tot, cnt = 0.0, 0
    for i in range(n_batches):
        idx = starts[i * batch:(i + 1) * batch]
        x = torch.from_numpy(np.stack([data[j:j + T + 1] for j in idx]).astype(np.int64))
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")
        tot += loss.item()
        cnt += x[:, 1:].numel()
    model.train()
    return tot / cnt / math.log(2)


@torch.no_grad()
def generate(model, prompt, n=160, T=128, temperature=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = list(prompt.encode("utf-8"))
    model.eval()
    for _ in range(n):
        x = torch.tensor([ids[-T:]])
        logits = model(x)[0, -1] / temperature
        ids.append(int(torch.multinomial(logits.softmax(-1), 1, generator=g)))
    model.train()
    return bytes(ids).decode("utf-8", "replace")


PROMPTS = {"web_en": "The city council announced on Monday that",
           "web_zh": "近日，记者从市教育局了解到，",
           "code": "def parse_args(argv):\n    ",
           "math": "To solve the equation 3x + 5 = 20, we",
           "pdf": "Table 2 shows the results of"}


def train_one(pool, mix, steps=500, d=96, L=4, T=128, batch=32, lr=3e-3, warmup=50, seed=0, threads=2,
              eval_domains=None, log_every=0, gen=True):
    """按配比训练一个小模型，返回各领域验证 BPB、各领域被读了几轮、耗时与生成样例。"""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    doms = [k for k, v in mix.items() if v > 0]
    # MixtureSampler 的“文档”在这里是固定长度片段的起点：每个领域切成不重叠的 (T+1) 字节片段
    sources = {k: list(range(0, len(pool[k]["train"]) - T - 1, T + 1)) for k in doms}
    sampler = MixtureSampler(sources, {k: mix[k] for k in doms}, seed=seed)
    model = MiniGPT(V=V, d=d, L=L, h=4, d_ff=int(d * 8 / 3) // 8 * 8, max_T=T)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for gp in opt.param_groups:
            gp["lr"] = cur
        rows = []
        for _ in range(batch):
            dom, off = sampler.next()
            rows.append(pool[dom]["train"][off:off + T + 1])
        x = torch.from_numpy(np.stack(rows).astype(np.int64))
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item() / math.log(2):.3f} bpb", flush=True)
    train_s = time.time() - t0
    ev = eval_domains or list(pool)
    bpb = {k: round(eval_bpb(model, pool[k]["val"], T), 4) for k in ev}
    n_chunks = {k: len(v) for k, v in sources.items()}
    epochs = {k: round((sampler.epoch[k] * n_chunks[k] + sampler.cursor[k]) / n_chunks[k], 3) for k in doms}
    out = {"mix": {k: round(v, 4) for k, v in mix.items()}, "steps": steps, "d": d, "L": L, "T": T, "batch": batch,
           "seed": seed, "train_bytes": steps * batch * T, "val_bpb": bpb,
           "val_bpb_mean": round(sum(bpb.values()) / len(bpb), 4), "epochs": epochs,
           "non_emb_params": sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("emb", "head"))),
           "seconds": round(time.time() - t0, 1), "train_seconds": round(train_s, 1)}
    if gen:
        out["samples"] = {k: generate(model, p, T=T) for k, p in PROMPTS.items() if k in ev}
    return out, model


def save_result(res, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
