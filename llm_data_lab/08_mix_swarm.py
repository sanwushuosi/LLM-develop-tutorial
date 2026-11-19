#!/usr/bin/env python3
"""第 8 步（3/4）：配比“蜂群”——批量训练随机配比的小模型（RegMix 的思路，规模缩小一万倍）。

配比从 Dirichlet 分布采样（以“自然分布”即各领域数据量为中心，用 --alpha 控制分散程度），
再加上 5 个“单领域”配比和 1 个均匀配比作为锚点。多个进程并行，每个进程用 --threads 个线程。
  python 08_mix_swarm.py --n 48 --workers 4 --threads 2          # 教师课前跑（约 30 分钟）
  python 08_mix_swarm.py --n 8 --workers 4 --steps 500           # 学生课上加几个点
结果追加到 work/08_mix/runs.jsonl（source=swarm）。
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from labkit.mixture import load_pool, save_result, train_one

_POOL = None


def _init(pool_dir):
    global _POOL
    _POOL = load_pool(pool_dir)


def _run(args):
    mix, steps, d, L, seed, threads, tag = args
    res, _ = train_one(_POOL, mix, steps=steps, d=d, L=L, seed=seed, threads=threads, gen=False)
    res["name"] = tag
    res["source"] = "swarm"
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet 浓度，越小配比越极端")
    ap.add_argument("--anchors", action="store_true", help="同时加入 5 个单领域与 1 个均匀配比")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pool", default="data/mix_pool")
    ap.add_argument("--out", default="work/08_mix/runs.jsonl")
    a = ap.parse_args()
    meta = json.loads(Path(a.pool, "meta.json").read_text("utf-8"))
    doms = list(meta["domains"])
    nat = np.array([meta["domains"][d]["train_bytes"] for d in doms], dtype=float)
    nat /= nat.sum()
    rng = np.random.default_rng(a.seed)
    jobs = []
    if a.anchors:
        for i, d in enumerate(doms):
            jobs.append(({k: float(k == d) for k in doms}, f"only_{d}"))
        jobs.append(({k: 1 / len(doms) for k in doms}, "uniform"))
    for i in range(a.n):
        w = rng.dirichlet(a.alpha * len(doms) * nat)
        w = np.where(w < 0.02, 0, w)
        w /= w.sum()
        jobs.append(({k: float(v) for k, v in zip(doms, w)}, f"swarm_{a.seed}_{i:03d}"))
    print(f"共 {len(jobs)} 个配比，{a.workers} 个进程 × {a.threads} 线程，每个 {a.steps} 步")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.pool,)) as ex:
        args = [(m, a.steps, a.d, a.L, 0, a.threads, tag) for m, tag in jobs]
        for i, res in enumerate(ex.map(_run, args), 1):
            save_result(res, a.out)
            mix = " ".join(f"{k}={v:.2f}" for k, v in res["mix"].items() if v > 0)
            print(f"  [{i}/{len(jobs)}] {res['name']:16} 平均 BPB {res['val_bpb_mean']:.3f}  ({mix})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
