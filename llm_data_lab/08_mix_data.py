#!/usr/bin/env python3
"""第 8 步（1/4）：准备配比实验的数据池。

五个领域，每个领域按文档切成训练集与验证集（按 id 哈希分，确定性），写成 UTF-8 字节流：
  web_en  第 3 步清洗 + 第 5 步去重后的英文网页
  web_zh  同上，中文网页（样本里只有几百篇——这就是“中文高质量数据稀缺”在小尺度上的样子）
  pdf     第 7 步修复后的 FinePDFs 文本（英文 + 中文）
  code    Python 标准库源码（data/extra/code.jsonl.gz）
  math    FineMath-4plus 抽样（data/extra/math.jsonl.gz）
每个领域训练集最多 --max-train-mb，验证集约 --val-frac。
输出：data/mix_pool/<领域>/{train.bin,val.bin}、data/mix_pool/meta.json
用法：python 08_mix_data.py [--clean work/03_clean/clean_text_trafilatura] [--max-train-mb 4]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from labkit.io import read_jsonl


def split_write(docs, out_dir, max_train_bytes, val_frac):
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = bytearray(), bytearray()
    n_tr = n_va = 0
    for doc_id, text in sorted(docs, key=lambda x: hashlib.md5(x[0].encode()).hexdigest()):
        b = text.strip().encode("utf-8") + b"\n\n"
        h = int(hashlib.md5(("split:" + doc_id).encode()).hexdigest()[:8], 16) / 2**32
        if h < val_frac:
            if len(val) < max_train_bytes * val_frac * 1.5:
                val += b
                n_va += 1
        elif len(train) < max_train_bytes:
            train += b
            n_tr += 1
    np.frombuffer(bytes(train), dtype=np.uint8).tofile(out_dir / "train.bin")
    np.frombuffer(bytes(val), dtype=np.uint8).tofile(out_dir / "val.bin")
    return {"train_docs": n_tr, "val_docs": n_va, "train_bytes": len(train), "val_bytes": len(val)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="work/03_clean/clean_text_trafilatura")
    ap.add_argument("--keep-ids", default="", help="第 5 步的 keep_ids.txt；默认自动找 work/05_dedup/<clean 名>/keep_ids.txt")
    ap.add_argument("--pdf", default="work/07_pdf/pdf_fixed.jsonl.gz")
    ap.add_argument("--extra", default="data/extra")
    ap.add_argument("--max-train-mb", type=float, default=4)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--out", default="data/mix_pool")
    a = ap.parse_args()
    kp = Path(a.keep_ids or f"work/05_dedup/{Path(a.clean).name}/keep_ids.txt")
    keep = set(kp.read_text("utf-8").split()) if kp.exists() else None
    web = {"en": [], "zh": []}
    for r in read_jsonl(a.clean, fields=["id", "lang", "status", "text"]):
        if r["status"] == "kept" and r["lang"] in web and (keep is None or r["id"] in keep):
            web[r["lang"]].append((r["id"], r["text"]))
    docs = {"web_en": web["en"], "web_zh": web["zh"],
            "pdf": [(r["id"], r["text"]) for r in read_jsonl(a.pdf, fields=["id", "text"]) if r["text"].strip()],
            "code": [(r["id"], r["text"]) for r in read_jsonl(Path(a.extra) / "code.jsonl.gz")],
            "math": [(r["id"], r["text"]) for r in read_jsonl(Path(a.extra) / "math.jsonl.gz")]}
    meta = {"source_clean": a.clean, "dedup_keep_ids": str(kp) if keep else None, "domains": {}}
    for d, ds in docs.items():
        meta["domains"][d] = split_write(ds, Path(a.out) / d, int(a.max_train_mb * 1e6), a.val_frac)
        m = meta["domains"][d]
        print(f"{d:7} 训练 {m['train_docs']:5,} 篇 {m['train_bytes'] / 1e6:5.2f} MB；验证 {m['val_docs']:4,} 篇 {m['val_bytes'] / 1e6:5.2f} MB")
    Path(a.out, "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
