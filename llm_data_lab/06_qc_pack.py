#!/usr/bin/env python3
"""第 6 步（1/3）：抽样打包，准备质检。

  python 06_qc_pack.py --run work/03_clean/clean_text_trafilatura --status kept --n 40     # 查“漏网之鱼”
  python 06_qc_pack.py --run work/03_clean/clean_text_trafilatura --status removed --n 20  # 查“误杀”
  python 06_qc_pack.py --rubric rubrics/web_qc.prompt ...                                  # 换成你改过的规则
  python 06_qc_pack.py --run work/03_clean/v2 --paired-with work/06_qc/<上一轮>             # 配对复检：同一批文档
输出目录 work/06_qc/<名字>/：
  samples/NNN_xxxx.txt   每条样本一个文件（给人看、给 Agent 读）
  samples.jsonl          同样的样本，给 06_qc_run.py 用
  rubric.prompt          本次使用的质检规则（拷贝）
  AGENT_TASK.md          交给 Agent 的任务说明
"""
import argparse
from pathlib import Path

from labkit.qc import make_pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/03_clean/clean_text_trafilatura")
    ap.add_argument("--rubric", default="rubrics/web_qc.prompt")
    ap.add_argument("--status", default="kept", choices=["kept", "removed", "all"])
    ap.add_argument("--lang", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--paired-with", default="", help="上一轮的打包目录：复检同一批文档（配对比较，推荐用于迭代前后对比）")
    a = ap.parse_args()
    if a.paired_with:
        name = a.name or f"{Path(a.run).name}_paired_{Path(a.paired_with).name}"
    else:
        name = a.name or f"{Path(a.run).name}_{a.status}{('_' + a.lang) if a.lang else ''}_n{a.n}_s{a.seed}"
    out = Path("work/06_qc") / name
    n = make_pack(a.run, out, a.rubric, n=a.n, status=a.status, seed=a.seed, lang=a.lang, paired_with=a.paired_with)
    print(f"已打包 {n} 条 → {out}")
    if a.paired_with:
        import json
        rm = json.loads((out / "pack.json").read_text("utf-8"))["removed_now"]
        print(f"  配对复检：上一轮的样本中有 {len(rm)} 篇已被新版本清洗删掉（记在 pack.json），其余 {n} 篇重新送检")
    print(f"  API 模式：python 06_qc_run.py --pack {out}")
    print(f"  Agent 模式：把 {out}/AGENT_TASK.md 交给你的 Agent，结果写到 {out}/results.jsonl")
    print(f"  汇总：python 06_qc_report.py --pack {out}")


if __name__ == "__main__":
    main()
