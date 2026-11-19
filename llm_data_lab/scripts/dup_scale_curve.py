#!/usr/bin/env python3
"""【教师课前运行，可选】更大范围的重复率规模曲线：1 → 64 个 WET 文件（约 2 千 → 130 万网页）。

为了在单机上几分钟跑完：
  * WET 文件流式下载、边读边算，不落盘；
  * 先做一个很轻的过滤：只保留 ≥8 个词的行（中文 ≥15 字），剩余正文 ≥100 词（中文 ≥200 字）的文档；
  * MinHash 与手册 4.3.3 节相同（5-gram shingle、K=128、16 段×8 行、估计 Jaccard ≥0.8 确认），
    只是把逐个 shingle 调 blake2b 换成 zlib.crc32，把 LSH 换成按段排序分组的向量化实现。
输出：summary.json（每个规模下的精确重复率与近似重复率，英文/中文分开）
用法：python scripts/dup_scale_curve.py --crawl CC-MAIN-2026-43 --n-files 64 --out work/05_dedup/scale_curve
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from warcio.archiveiterator import ArchiveIterator

BASE = "https://data.commoncrawl.org/"
PRIME = (1 << 31) - 1
_rng = np.random.default_rng(0)
A = _rng.integers(1, PRIME, 128, dtype=np.uint64)
B = _rng.integers(0, PRIME, 128, dtype=np.uint64)
CJK = re.compile(r"[一-鿿]")


def light_filter(text, zh):
    keep = []
    for l in text.split("\n"):
        l = l.strip()
        if (zh and len(CJK.findall(l)) >= 15) or (not zh and len(l.split()) >= 8):
            keep.append(l)
    t = "\n".join(keep)
    ok = len(CJK.findall(t)) >= 200 if zh else len(t.split()) >= 100
    return t if ok else None


def sig_of(text, zh, n=5):
    units = [c for c in text if not c.isspace()] if zh else text.lower().split()
    sh = {" ".join(units[i:i + n]) for i in range(max(1, len(units) - n + 1))}
    base = np.fromiter((zlib.crc32(s.encode()) % PRIME for s in sh), dtype=np.uint64, count=len(sh))
    return ((A[:, None] * base[None, :] + B[:, None]) % PRIME).min(axis=1).astype(np.uint32)


def process(args):
    path, idx = args
    out = {"en": [], "zh": []}
    for k in range(5):
        try:
            with urllib.request.urlopen(BASE + path, timeout=120) as r:
                for rec in ArchiveIterator(r):
                    if rec.rec_type != "conversion":
                        continue
                    lang = (rec.rec_headers.get_header("WARC-Identified-Content-Language") or "").split(",")[0]
                    key = {"eng": "en", "zho": "zh"}.get(lang)
                    if not key:
                        continue
                    text = rec.content_stream().read().decode("utf-8", "replace")
                    t = light_filter(text, key == "zh")
                    if t is None:
                        continue
                    h = hashlib.md5(re.sub(r"\s+", " ", t).lower().encode()).digest()[:8]
                    out[key].append((idx, h, sig_of(t, key == "zh")))
            return out
        except Exception:                                            # noqa: BLE001
            out = {"en": [], "zh": []}
            time.sleep(5 * (k + 1))
    return out


def near_dup_removed(sigs, bands=16, rows=8, threshold=0.8):
    n = len(sigs)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for b in range(bands):
        band = np.ascontiguousarray(sigs[:, b * rows:(b + 1) * rows])
        keys = band.view(np.dtype((np.void, band.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            g = order[s:e]
            first = g[0]
            for o in g[1:]:
                if (sigs[first] == sigs[o]).mean() >= threshold:
                    ra, rb = find(first), find(o)
                    if ra != rb:
                        parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    return n - len(np.unique(roots))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default="CC-MAIN-2026-43")
    ap.add_argument("--n-files", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="work/05_dedup/scale_curve")
    a = ap.parse_args()
    with urllib.request.urlopen(f"{BASE}crawl-data/{a.crawl}/wet.paths.gz", timeout=120) as r:
        paths = gzip.decompress(r.read()).decode().split()
    chosen = random.Random(a.seed).sample(paths, a.n_files)
    t0 = time.time()
    data = {"en": [], "zh": []}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(process, [(p, i) for i, p in enumerate(chosen)])):
            for k in data:
                data[k].extend(res[k])
            print(f"  已处理 {i + 1}/{a.n_files} 个 WET 文件，en {len(data['en']):,}、zh {len(data['zh']):,}，{time.time() - t0:.0f}s", flush=True)
    summary = {"crawl": a.crawl, "n_files": a.n_files, "files": chosen, "curves": {}}
    for lang, rows in data.items():
        if not rows:
            continue
        rows.sort(key=lambda x: x[0])
        idx = np.array([r[0] for r in rows])
        hashes = [r[1] for r in rows]
        sigs = np.stack([r[2] for r in rows])
        curve = []
        k = 1
        while k <= a.n_files:
            sel = np.flatnonzero(idx < k)
            hs = [hashes[i] for i in sel]
            exact = len(hs) - len(set(hs))
            near = near_dup_removed(sigs[sel])
            curve.append({"files": k, "docs": int(len(sel)), "exact_frac": round(exact / len(sel), 4),
                          "near_frac": round(near / len(sel), 4)})
            print(f"  [{lang}] {k:3d} 个文件 {len(sel):8,} 篇：精确 {exact / len(sel):.2%}，近似 {near / len(sel):.2%}", flush=True)
            k = k * 2 if k * 2 <= a.n_files or k == a.n_files else a.n_files
            if curve[-1]["files"] == a.n_files:
                break
        summary["curves"][lang] = curve
    summary["seconds"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(f"完成：{a.out}/summary.json，用时 {summary['seconds']}s")


if __name__ == "__main__":
    main()
