#!/usr/bin/env python3
"""【教师课前运行】从一个 Common Crawl 快照中随机抽样网页，生成实验课用的数据切片。

做法：
  1. 从 warc.paths.gz 里按随机种子选 N 个 WARC 文件（分布在不同 segment，近似全快照的随机样本）；
  2. 逐个流式读取（不落盘整个 1GB 文件），对 text/html 的 200 响应按固定比例抽样，保存原始 HTML；
  3. 再流式读取同名 WET 文件，取出这些记录对应的 CC 官方纯文本与语言标签，便于和自己的提取结果对比。

输出：<out>/part-XXXXX.jsonl.gz，每行一个网页：
  id, url, host, date, crawl, warc_file, charset, html, html_bytes, truncated, wet_text, cc_lang

默认参数（24 个文件 × 约 5.5%）约得到 2.8 万个网页、约 1 GB（gzip 后）。
用法：
  python scripts/prepare_cc_sample.py --crawl CC-MAIN-2026-43 --n-files 24 --rate 0.055 --out data/cc_sample
数据来自互联网，只当作文本读取，绝不执行或在浏览器中渲染其中的脚本。
"""
import argparse
import gzip
import hashlib
import io
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from warcio.archiveiterator import ArchiveIterator

BASE = "https://data.commoncrawl.org/"


def open_url(url, retries=5):
    for k in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=120)
        except Exception as e:                                   # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  重试 {url[-40:]}：{e}", file=sys.stderr)
            time.sleep(5 * (k + 1))


def keep(record_id, rate, seed):
    """确定性抽样：同样的种子与比例，任何时候都抽到同一批记录。"""
    h = hashlib.sha1(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < rate


def charset_of(http_headers, body):
    ct = (http_headers.get_header("Content-Type") or "").lower()
    if "charset=" in ct:
        return ct.split("charset=")[-1].split(";")[0].strip().strip("\"'") or "utf-8"
    head = body[:4096].decode("ascii", "ignore").lower()
    i = head.find("charset=")
    if i >= 0:
        cs = head[i + 8:i + 40].strip("\"' ").split("\"")[0].split("'")[0].split(";")[0].split(" ")[0].split("/")[0]
        return cs.strip("\"'>") or "utf-8"
    return "utf-8"


def decode(body, cs):
    for enc in (cs, "utf-8", "gb18030", "latin-1"):
        try:
            return body.decode(enc), enc
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace"), "utf-8"


def process(warc_path, crawl, rate, seed, out_dir, idx):
    out = Path(out_dir) / f"part-{idx:05d}.jsonl.gz"
    if out.exists():
        return str(out), -1, 0.0
    t0 = time.time()
    rows = {}
    with open_url(BASE + warc_path) as resp:
        for rec in ArchiveIterator(resp):
            if rec.rec_type != "response":
                continue
            rid = rec.rec_headers.get_header("WARC-Record-ID")
            if not keep(rid, rate, seed):
                continue
            hh = rec.http_headers
            ct = (hh.get_header("Content-Type") or "").split(";")[0].strip().lower()
            if ct != "text/html" or hh.get_statuscode() != "200":
                continue
            body = rec.content_stream().read()
            cs = charset_of(hh, body)
            html, enc = decode(body, cs)
            url = rec.rec_headers.get_header("WARC-Target-URI")
            rows[rid] = {
                "id": rid.strip("<>").replace("urn:uuid:", ""),
                "url": url,
                "host": urlsplit(url).hostname or "",
                "date": rec.rec_headers.get_header("WARC-Date"),
                "crawl": crawl,
                "warc_file": warc_path,
                "charset": enc,
                "html": html,
                "html_bytes": len(body),
                "truncated": rec.rec_headers.get_header("WARC-Truncated") or "",
                "wet_text": None,
                "cc_lang": None,
            }
    # 同名 WET：WARC-Refers-To 指向 WARC 响应记录的 ID
    wet_path = warc_path.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")
    with open_url(BASE + wet_path) as resp:
        for rec in ArchiveIterator(resp):
            if rec.rec_type != "conversion":
                continue
            ref = rec.rec_headers.get_header("WARC-Refers-To")
            if ref in rows:
                rows[ref]["wet_text"] = rec.content_stream().read().decode("utf-8", "replace")
                lang = rec.rec_headers.get_header("WARC-Identified-Content-Language") or ""
                rows[ref]["cc_lang"] = lang.split(",")[0] or None
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(out)
    return str(out), len(rows), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default="CC-MAIN-2026-43")
    ap.add_argument("--n-files", type=int, default=24)
    ap.add_argument("--rate", type=float, default=0.055)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/cc_sample")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open_url(f"{BASE}crawl-data/{a.crawl}/warc.paths.gz") as r:
        paths = gzip.decompress(r.read()).decode().split()
    chosen = sorted(random.Random(a.seed).sample(paths, a.n_files))
    (Path(a.out) / "manifest.json").write_text(json.dumps(
        {"crawl": a.crawl, "n_files": a.n_files, "rate": a.rate, "seed": a.seed,
         "warc_files": chosen, "total_warc_files_in_crawl": len(paths)}, indent=1), "utf-8")
    print(f"{a.crawl} 共 {len(paths)} 个 WARC 文件，随机选 {a.n_files} 个，抽样比例 {a.rate}")
    total = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, p, a.crawl, a.rate, a.seed, a.out, i) for i, p in enumerate(chosen)]
        for fu in as_completed(futs):
            path, n, dt = fu.result()
            total += max(n, 0)
            print(f"  {Path(path).name}: {'已存在，跳过' if n < 0 else f'{n} 个网页，{dt:.0f}s'}", flush=True)
    print(f"完成：{total} 个网页 → {a.out}")


if __name__ == "__main__":
    main()
