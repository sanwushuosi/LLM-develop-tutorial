#!/usr/bin/env python3
"""【教师课前运行】为配比实验准备“代码”与“数学”两个领域的小语料。

  代码：本机 Python 标准库的 .py 源文件（PSF 许可，可再分发；去掉 test 目录），约 6 MB；
  数学：FineMath-4plus 的一个分片（HuggingFaceTB/finemath，ODC-By），随机取约 3,000 篇。
输出：data/extra/code.jsonl.gz、data/extra/math.jsonl.gz（字段 id, text, source）
用法：python scripts/prepare_extra_domains.py --finemath <finemath-4plus 分片.parquet>
"""
import argparse
import hashlib
import random
import sysconfig
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labkit.io import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finemath", required=True)
    ap.add_argument("--n-math", type=int, default=3000)
    ap.add_argument("--out", default="data/extra")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    std = Path(sysconfig.get_paths()["stdlib"])
    rows = []
    for p in sorted(std.rglob("*.py")):
        rel = p.relative_to(std)
        if any(part in ("test", "tests", "idle_test", "site-packages", "__pycache__") for part in rel.parts):
            continue
        try:
            t = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if 500 <= len(t) <= 200_000:
            rows.append({"id": hashlib.md5(str(rel).encode()).hexdigest()[:16], "text": t,
                         "source": f"cpython-stdlib/{rel}"})
    n = write_jsonl(out / "code.jsonl.gz", rows)
    print(f"代码：{n} 个文件，{sum(len(r['text']) for r in rows) / 1e6:.1f} M 字符")

    df = pd.read_parquet(a.finemath, columns=["text", "url"])
    idx = random.Random(0).sample(range(len(df)), min(a.n_math, len(df)))
    rows = [{"id": hashlib.md5(df.url.iloc[i].encode()).hexdigest()[:16], "text": df.text.iloc[i],
             "source": df.url.iloc[i]} for i in idx]
    n = write_jsonl(out / "math.jsonl.gz", rows)
    print(f"数学：{n} 篇，{sum(len(r['text']) for r in rows) / 1e6:.1f} M 字符")


if __name__ == "__main__":
    main()
