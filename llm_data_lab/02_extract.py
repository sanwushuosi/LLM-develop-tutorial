#!/usr/bin/env python3
"""第 2 步：正文提取。把原始 HTML 变成纯文本，并与 CC 官方 WET 文本对照。

每个分片交给一个进程处理（各自读、各自写），输出不再包含 HTML，后续步骤读这里的结果即可。
输出：work/02_extract/part-*.jsonl.gz，字段：
  id, url, host, cc_lang, html_bytes, wet_text, text_trafilatura[, text_resiliparse], ms_*
用法：
  python 02_extract.py                       # 全部样本，trafilatura
  python 02_extract.py --max-per-part 300    # 课上先跑一个小子集（24×300≈7 千篇）
  python 02_extract.py --methods trafilatura,resiliparse
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

from labkit.extract import EXTRACTORS
from labkit.io import files_of, read_jsonl, write_jsonl


def work(args):
    src, dst, methods, max_n = args
    rows = []
    for rec in read_jsonl(src, limit=max_n or None):
        out = {k: rec.get(k) for k in ("id", "url", "host", "date", "cc_lang", "html_bytes", "truncated")}
        out["wet_text"] = rec.get("wet_text") or ""
        for m in methods:
            t0 = time.time()
            try:
                out[f"text_{m}"] = EXTRACTORS[m](rec["html"])
            except Exception as e:                                   # noqa: BLE001
                out[f"text_{m}"] = ""
                out[f"error_{m}"] = type(e).__name__
            out[f"ms_{m}"] = round((time.time() - t0) * 1000, 1)
        rows.append(out)
    write_jsonl(dst, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cc_sample")
    ap.add_argument("--out", default="work/02_extract")
    ap.add_argument("--methods", default="trafilatura")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-part", type=int, default=0)
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    parts = files_of(a.input)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(str(p), str(Path(a.out) / p.name), methods, a.max_per_part) for p in parts]
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(work, jobs))
    dt = time.time() - t0

    # 简要对照：提取前后的长度、空文本比例、耗时
    stats = {"docs": n, "seconds": round(dt, 1), "workers": a.workers}
    wet_len, empty = [], {m: 0 for m in methods}
    lens = {m: [] for m in methods}
    ms = {m: [] for m in methods}
    for r in read_jsonl(a.out):
        wet_len.append(len(r["wet_text"]))
        for m in methods:
            t = r.get(f"text_{m}") or ""
            lens[m].append(len(t))
            ms[m].append(r.get(f"ms_{m}") or 0)
            empty[m] += len(t.strip()) < 200
    stats["wet_chars_median"] = median(wet_len)
    stats["wet_total_mchars"] = round(sum(wet_len) / 1e6, 1)
    for m in methods:
        stats[m] = {"chars_median": median(lens[m]), "total_mchars": round(sum(lens[m]) / 1e6, 1),
                    "lt200_frac": round(empty[m] / n, 4), "ms_median": median(ms[m]),
                    "kept_vs_wet_chars": round(sum(lens[m]) / max(1, sum(wet_len)), 4)}
    Path(a.out, "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    print(f"{n:,} 篇，用时 {dt:.0f}s（{a.workers} 进程）")
    print(f"CC WET：中位 {stats['wet_chars_median']:,} 字符，合计 {stats['wet_total_mchars']} M 字符")
    for m in methods:
        s = stats[m]
        print(f"{m:12}：中位 {s['chars_median']:,} 字符，合计 {s['total_mchars']} M 字符"
              f"（为 WET 的 {s['kept_vs_wet_chars']:.0%}）；不足 200 字符的 {s['lt200_frac']:.1%}；"
              f"单篇耗时中位 {s['ms_median']} ms")


if __name__ == "__main__":
    main()
