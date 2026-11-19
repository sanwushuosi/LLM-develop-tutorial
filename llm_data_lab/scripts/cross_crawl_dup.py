#!/usr/bin/env python3
"""【教师课前运行，可选】跨快照重复：同一个网址，隔一个月再抓，内容变了多少？

做法：从本次样本（如 CC-MAIN-2026-43）的目标语言文档里随机取 M 个网址，到上一个快照（如 CC-MAIN-2026-39）里
查同一网址（CDXJ 索引：先下载 cluster.idx 二分查找，再用 HTTP Range 只取需要的索引块与 WARC 记录），
两边都用 trafilatura 提取正文，比较：完全相同 / 近似相同（5-gram Jaccard ≥ 0.8）/ 明显变化 / 上一个快照没有。
结论用于说明：把几十个快照合在一起时，稳定页面会被重复收录几十次——这是 FineWeb 全局去重删掉 90% 的原因。
用法：python scripts/cross_crawl_dup.py --sample work/02_extract --clean work/03_clean/clean_text_trafilatura \
          --prev CC-MAIN-2026-39 --n 1500 --out work/05_dedup/cross_crawl
"""
import argparse
import bisect
import gzip
import io
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import surt
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labkit.extract import extract_trafilatura  # noqa: E402
from labkit.io import read_jsonl  # noqa: E402

BASE = "https://data.commoncrawl.org/"


def get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:                                            # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def load_cluster(prev, cache_dir):
    p = Path(cache_dir) / f"{prev}.cluster.idx"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(f"{BASE}cc-index/collections/{prev}/indexes/cluster.idx"))
    keys, meta = [], []
    for line in p.read_text("utf-8").splitlines():
        head, fn, off, ln, _ = line.split("\t")
        keys.append(head.split(" ")[0])
        meta.append((fn, int(off), int(ln)))
    return keys, meta


def shingles(t, n=5):
    w = re.sub(r"\s+", " ", t).strip().lower().split(" ")
    if len(w) < n:
        return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def lookup(url, prev, keys, meta):
    key = surt.surt(url)
    i = max(0, bisect.bisect_right(keys, key) - 1)
    fn, off, ln = meta[i]
    block = gzip.decompress(get(f"{BASE}cc-index/collections/{prev}/indexes/{fn}", (off, off + ln - 1)))
    best = None
    for line in block.decode("utf-8", "replace").splitlines():
        k, ts, js = line.split(" ", 2)
        if k != key:
            continue
        d = json.loads(js)
        if d.get("status") == "200" and "html" in (d.get("mime-detected") or d.get("mime") or ""):
            best = d
    return best


def fetch_text(d):
    off, ln = int(d["offset"]), int(d["length"])
    raw = get(BASE + d["filename"], (off, off + ln - 1))
    for rec in ArchiveIterator(io.BytesIO(raw)):
        if rec.rec_type == "response":
            body = rec.content_stream().read()
            return extract_trafilatura(body.decode("utf-8", "replace"))
    return ""


def one(args):
    r, prev, keys, meta = args
    try:
        d = lookup(r["url"], prev, keys, meta)
        if not d:
            return {"id": r["id"], "status": "absent"}
        t_prev = fetch_text(d)
    except Exception as e:                                           # noqa: BLE001
        return {"id": r["id"], "status": "error", "err": type(e).__name__}
    a, b = shingles(r["text"]), shingles(t_prev)
    j = len(a & b) / max(1, len(a | b))
    same = re.sub(r"\s+", " ", r["text"]).strip() == re.sub(r"\s+", " ", t_prev).strip()
    st = "identical" if same else ("near" if j >= 0.8 else ("partial" if j >= 0.3 else "changed"))
    return {"id": r["id"], "url": r["url"], "status": st, "jaccard": round(j, 4), "prev_ts": d.get("timestamp"),
            "chars_now": len(r["text"]), "chars_prev": len(t_prev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="work/02_extract")
    ap.add_argument("--clean", default="work/03_clean/clean_text_trafilatura")
    ap.add_argument("--prev", default="CC-MAIN-2026-39")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cache", default="data/cc_index_cache")
    ap.add_argument("--out", default="work/05_dedup/cross_crawl")
    a = ap.parse_args()
    target = {r["id"]: r["lang"] for r in read_jsonl(a.clean, fields=["id", "lang", "reasons"]) if r["reasons"] != ["D01"]}
    pool = [{"id": r["id"], "url": r["url"], "text": r["text_trafilatura"]}
            for r in read_jsonl(a.sample, fields=["id", "url", "text_trafilatura"])
            if r["id"] in target and len((r["text_trafilatura"] or "").split()) >= 50]
    random.Random(0).shuffle(pool)
    pool = pool[:a.n]
    t0 = time.time()
    keys, meta = load_cluster(a.prev, a.cache)
    print(f"{a.prev} cluster.idx：{len(keys):,} 个索引块；查询 {len(pool)} 个网址…", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(one, [(r, a.prev, keys, meta) for r in pool]))
    from collections import Counter
    c = Counter(r["status"] for r in res)
    found = sum(v for k, v in c.items() if k not in ("absent", "error"))
    summary = {"prev": a.prev, "queried": len(pool), "counts": dict(c), "found_in_prev": found,
               "found_frac": round(found / max(1, len(pool) - c["error"]), 4),
               "among_found": {k: round(c[k] / max(1, found), 4) for k in ("identical", "near", "partial", "changed")},
               "seconds": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n", "utf-8")
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
