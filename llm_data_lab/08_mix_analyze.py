#!/usr/bin/env python3
"""第 8 步（4/4）：分析配比实验——配比怎样改变各领域能力？能不能用小实验预测最优配比？

读取 work/08_mix/runs.jsonl（教师课前跑的“蜂群” + 同学们自己跑的配比），做四件事：
  1. 每个领域：该领域在配比中的权重 vs 该领域的验证 BPB（自己的数据越多越好，但收益递减）；
  2. 领域之间的相互影响：拟合回归，读出“多加 10% 的 A 数据，B 领域的 BPB 变化多少”（正迁移 / 负迁移）；
  3. 用回归预测最优配比（RegMix / Data Mixing Laws 的思路），并用留一法检验回归到底准不准；
  4. 排行榜：谁的配比在目标上最好？
目标可以自定义，例如更看重中文与数学：--target web_zh=0.5,math=0.5（默认：五个领域的平均 BPB）。
用法：python 08_mix_analyze.py [--target web_zh=0.5,math=0.5]
输出：work/08_mix/report.html；并打印“验证预测最优配比”的训练命令。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from labkit import report as R
from labkit.mixture import DOMAIN_CN, mix_str, parse_mix


def load_runs(path, steps=None):
    runs = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if steps is None or r["steps"] == steps:
                runs.append(r)
    return runs


def feats(W, kind):
    if kind == "linear":
        return W
    return np.hstack([W, np.sqrt(W)])             # sqrt 特征刻画“收益递减”


def fit_ridge(X, y, lam=1e-3):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    I = np.eye(X1.shape[1])
    I[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + lam * I, X1.T @ y)


def predict(coef, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ coef


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="work/08_mix/runs.jsonl")
    ap.add_argument("--steps", type=int, default=800, help="只分析同一训练步数的运行，保证可比")
    ap.add_argument("--target", default="", help="目标权重，如 web_zh=0.5,math=0.5；默认各领域等权")
    ap.add_argument("--model", default="sqrt", choices=["linear", "sqrt"])
    ap.add_argument("--sources", default="", help="只用这些来源的运行拟合，如 swarm 或 swarm,student；默认全部")
    ap.add_argument("--out", default="work/08_mix/report.html")
    a = ap.parse_args()
    runs = load_runs(a.runs, a.steps)
    if a.sources:
        keep = set(a.sources.split(","))
        runs = [r for r in runs if r.get("source") in keep]
    if len(runs) < 8:
        raise SystemExit(f"只有 {len(runs)} 个 {a.steps} 步的运行，至少需要 8 个才能拟合；先跑 08_mix_swarm.py")
    doms = list(runs[0]["val_bpb"])
    tgt = parse_mix(a.target, doms) if a.target else {d: 1 / len(doms) for d in doms}
    W = np.array([[r["mix"].get(d, 0.0) for d in doms] for r in runs])
    Y = np.array([[r["val_bpb"][d] for d in doms] for r in runs])
    t = np.array([tgt[d] for d in doms])
    obj = Y @ t

    S = [R.cards([("运行数", len(runs)), ("每个运行的训练步数", a.steps),
                  ("目标", "，".join(f"{DOMAIN_CN[d]} {w:.0%}" for d, w in tgt.items() if w > 0)),
                  ("目标 BPB 最好 / 最差", f"{obj.min():.3f} / {obj.max():.3f}")])]

    # 1. 自身权重 vs 自身 BPB
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(doms), figsize=(3.0 * len(doms), 2.9), sharey=False)
    for i, d in enumerate(doms):
        ax = axes[i]
        ax.scatter(W[:, i], Y[:, i], s=14, color=R.COLORS[i])
        ax.set_title(DOMAIN_CN[d], fontsize=10)
        ax.set_xlabel("该领域在配比中的权重", fontsize=8)
        if i == 0:
            ax.set_ylabel("该领域验证 BPB", fontsize=8)
    fig.tight_layout()
    S.append("<h2>1. 自己领域的数据越多，自己领域越好——但收益递减</h2>" + R.fig_to_img(fig))
    span = Y.max(0) - Y.min(0)
    S.append(R.table([(DOMAIN_CN[d], f"{Y[:, i].min():.3f}", f"{Y[:, i].max():.3f}", f"{span[i]:.3f}")
                      for i, d in enumerate(doms)], ["领域", "最好 BPB", "最差 BPB", "极差"], num_cols=(1, 2, 3)))
    S.append("<div class='note'>同样的模型、同样的训练量，只改变配比，各领域 BPB 的极差可以达到 1 bit/字节以上——"
             "相当于把这个领域的“能力”从会写变成不会写。配比是一个实实在在的“能力分配器”。</div>")

    # 2. 相互影响矩阵（线性模型：把 10% 权重从“其余领域平均”挪到领域 i，领域 j 的 BPB 变化）
    lin = [fit_ridge(W, Y[:, j]) for j in range(len(doms))]
    M = np.zeros((len(doms), len(doms)))
    for i in range(len(doms)):
        for j in range(len(doms)):
            b = lin[j][1:]
            M[i, j] = 0.1 * (b[i] - np.mean(np.delete(b, i)))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(doms)), [DOMAIN_CN[d] for d in doms], rotation=30)
    ax.set_yticks(range(len(doms)), [f"+10% {DOMAIN_CN[d]}" for d in doms])
    for i in range(len(doms)):
        for j in range(len(doms)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("多加 10% 某类数据 → 各领域 BPB 的变化（蓝=变好）")
    fig.colorbar(im, fraction=0.04)
    S.append("<h2>2. 领域之间互相影响（线性回归估计）</h2>" + R.fig_to_img(fig))
    S.append("<div class='note'>对角线是“自己帮自己”；非对角线是迁移：哪些数据彼此有帮助（蓝），哪些互相争夺容量（红）。"
             "RegMix 在 1B 模型上发现，网页数据与下游能力的正相关最强、领域间的相互作用常常违反直觉——"
             "这正是配比需要靠实验而不能靠拍脑袋的原因。</div>")

    # 3. 回归预测与留一法检验
    X = feats(W, a.model)
    loo = np.zeros(len(runs))
    for k in range(len(runs)):
        m = np.ones(len(runs), bool)
        m[k] = False
        coefs = [fit_ridge(X[m], Y[m, j]) for j in range(len(doms))]
        loo[k] = sum(t[j] * predict(coefs[j], X[k:k + 1])[0] for j in range(len(doms)))
    rho = spearman(loo, obj)
    mae = float(np.mean(np.abs(loo - obj)))
    coefs = [fit_ridge(X, Y[:, j]) for j in range(len(doms))]
    rng = np.random.default_rng(0)
    cand = rng.dirichlet(np.ones(len(doms)) * 0.7, 200_000)
    cand = np.vstack([cand, np.eye(len(doms))])
    pred = sum(t[j] * predict(coefs[j], feats(cand, a.model)) for j in range(len(doms)))
    best = cand[np.argmin(pred)]
    best_mix = {d: float(v) for d, v in zip(doms, best)}
    i_obs = int(np.argmin(obj))
    uni = np.ones(len(doms)) / len(doms)
    pred_uni = float(sum(t[j] * predict(coefs[j], feats(uni[None], a.model))[0] for j in range(len(doms))))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(obj, loo, s=16)
    lo_, hi_ = min(obj.min(), loo.min()), max(obj.max(), loo.max())
    ax.plot([lo_, hi_], [lo_, hi_], "--", color="#999")
    ax.set_xlabel("实际目标 BPB")
    ax.set_ylabel("留一法预测的目标 BPB")
    ax.set_title(f"回归（{a.model}）留一法：Spearman ρ={rho:.2f}")
    S.append("<h2>3. 用小实验预测最优配比</h2>" + R.fig_to_img(fig))
    S.append(R.table([("回归预测的最优配比", mix_str(best_mix), f"{pred.min():.3f}（预测）"),
                      ("已跑过的最好配比", mix_str(runs[i_obs]["mix"]), f"{obj[i_obs]:.3f}（实测）"),
                      ("均匀配比", mix_str(dict(zip(doms, uni))), f"{pred_uni:.3f}（预测）")],
                     ["", "配比", "目标 BPB"]))
    cmd = f"python 08_mix_train.py --mix {mix_str(best_mix)} --name 回归预测最优"
    S.append(f"<p>验证预测：<code>{R.esc(cmd)}</code>——预测和实测差多少？如果差得多，可能是什么原因？</p>")
    S.append("<div class='note'>这就是 RegMix（512 个 1M 参数模型 → 预测 1B 模型的最优配比）与 Data Mixing Laws 的缩微版。"
             "真正的难点在两次“迁移”：小模型上的最优配比能否迁移到大模型？验证集 BPB 的最优能否迁移到下游能力？"
             "Olmix、CLIMB、DataDecide 等 2025—2026 年的工作都在回答这两个问题。</div>")

    # 4. 排行榜
    order = np.argsort(obj)
    rows = [(k + 1, runs[i].get("name", ""), runs[i].get("source", ""), mix_str(runs[i]["mix"]), f"{obj[i]:.3f}",
             " ".join(f"{runs[i]['val_bpb'][d]:.2f}" for d in doms)) for k, i in enumerate(order[:25])]
    S.append("<h2>4. 排行榜（目标 BPB 越低越好）</h2>")
    S.append(R.table(rows, ["名次", "名字", "来源", "配比", "目标 BPB", " / ".join(DOMAIN_CN[d] for d in doms)], num_cols=(0, 4)))
    R.page("第 8 步：配比实验", S, a.out)
    print(f"{len(runs)} 个运行；目标 {mix_str(tgt)}")
    print(f"各领域 BPB 极差：" + "，".join(f"{DOMAIN_CN[d]} {span[i]:.2f}" for i, d in enumerate(doms)))
    print(f"回归（{a.model}）留一法 Spearman ρ = {rho:.3f}，平均绝对误差 {mae:.3f}")
    print(f"已跑过的最好：{runs[i_obs].get('name')} {mix_str(runs[i_obs]['mix'])} → {obj[i_obs]:.3f}")
    print(f"回归预测的最优：{mix_str(best_mix)} → 预测 {pred.min():.3f}")
    print(f"验证：{cmd}")
    print(f"报告：{a.out}")


if __name__ == "__main__":
    main()
