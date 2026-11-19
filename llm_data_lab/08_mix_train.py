#!/usr/bin/env python3
"""第 8 步（2/4）：按一个配比训练一个小模型，报告各领域的验证 BPB，并让它续写几段文字。

  python 08_mix_train.py --mix web_en=0.4,web_zh=0.2,pdf=0.1,code=0.2,math=0.1
  python 08_mix_train.py --mix code=1 --steps 800                       # 只用代码训练
  python 08_mix_train.py --mix web_en=1,math=1 --name 我的第一个配比
结果追加到 work/08_mix/runs.jsonl（所有人的结果都可以汇总到一起，用 08_mix_analyze.py 分析）。
默认模型约 44 万非嵌入参数，800 步 × 32 × 128 字节 ≈ 3.3 MB 训练数据，2 个线程约 2～3 分钟。
"""
import argparse
import time

from labkit.mixture import DOMAIN_CN, DOMAINS, load_pool, parse_mix, save_result, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pool", default="data/mix_pool")
    ap.add_argument("--out", default="work/08_mix/runs.jsonl")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    pool = load_pool(a.pool)
    mix = parse_mix(a.mix, list(pool))
    print("配比：", "，".join(f"{DOMAIN_CN[k]} {v:.0%}" for k, v in mix.items() if v > 0))
    res, _ = train_one(pool, mix, steps=a.steps, d=a.d, L=a.L, seed=a.seed, threads=a.threads,
                       log_every=max(1, a.steps // 5))
    res["name"] = a.name or a.mix
    res["source"] = "student"
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_result(res, a.out)
    print("\n各领域验证 BPB（越低越好）：")
    for k in DOMAINS:
        if k in res["val_bpb"]:
            print(f"  {DOMAIN_CN[k]:8} {res['val_bpb'][k]:.3f}    该领域数据被读了 {res['epochs'].get(k, 0):.2f} 轮")
    print(f"  平均      {res['val_bpb_mean']:.3f}    用时 {res['seconds']}s")
    print("\n续写样例：")
    for k, s in res["samples"].items():
        print(f"--- [{DOMAIN_CN[k]}] {s[:240]!r}")


if __name__ == "__main__":
    main()
