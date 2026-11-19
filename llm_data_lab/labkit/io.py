"""读写 jsonl / jsonl.gz。一个目录可以包含多个 part-*.jsonl.gz 分片。"""
import gzip
import json
from pathlib import Path


def _open(path, mode):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def files_of(path):
    p = Path(path)
    if p.is_dir():
        return sorted(list(p.glob("*.jsonl.gz")) + list(p.glob("*.jsonl")))
    return [p]


def read_jsonl(path, limit=None, fields=None):
    """逐行读取。path 可以是文件或目录；fields 只保留部分字段以省内存。"""
    n = 0
    for f in files_of(path):
        with _open(f, "r") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if fields:
                    r = {k: r.get(k) for k in fields}
                yield r
                n += 1
                if limit and n >= limit:
                    return


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
