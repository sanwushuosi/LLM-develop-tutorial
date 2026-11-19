"""去重工具：精确去重、MinHash 近似去重（手册 4.3.3 节代码）、跨文档模板行、规模曲线。"""
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

from .book import lsh_candidates, minhash, shingles


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def norm_line(l):
    """行的规范化：小写、数字统一成 0、空白合并。这样“© 2024”和“© 2025”算同一行。"""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", l.strip().lower()))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def exact_clusters(texts):
    """返回 {哈希: [文档下标...]}，只保留大小 ≥ 2 的簇。"""
    groups = defaultdict(list)
    for i, t in enumerate(texts):
        groups[hashlib.md5(norm_text(t).encode()).hexdigest()].append(i)
    return {h: v for h, v in groups.items() if len(v) > 1}


def signatures(texts, n=5):
    """每篇文档的 MinHash 签名（K=128）。shingle 少于 1 个的文档给一个全 0 签名并在后面跳过。"""
    sigs = np.zeros((len(texts), 128), dtype=np.uint64)
    ok = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        sh = shingles(t, n)
        if len(sh) >= 1 and t.strip():
            sigs[i] = minhash(sh)
            ok[i] = True
    return sigs, ok


def near_clusters(sigs, ok, bands=16, threshold=0.8):
    """LSH 找候选对，再用签名估计的 Jaccard ≥ threshold 确认，并查集聚簇。"""
    idx = np.flatnonzero(ok)
    cand = lsh_candidates(sigs[idx], bands=bands)
    uf = UnionFind(len(sigs))
    confirmed = []
    for a, b in cand:
        ia, ib = int(idx[a]), int(idx[b])
        sim = float((sigs[ia] == sigs[ib]).mean())
        if sim >= threshold:
            uf.union(ia, ib)
            confirmed.append((ia, ib, sim))
    groups = defaultdict(list)
    for i in idx:
        groups[uf.find(int(i))].append(int(i))
    clusters = [v for v in groups.values() if len(v) > 1]
    return clusters, len(cand), confirmed


def boilerplate_lines(docs, min_docs=5, min_hosts=3, min_len=10):
    """跨文档高频行：出现在 ≥min_docs 篇、来自 ≥min_hosts 个不同站点的（规范化后的）行。
    docs: [(host, text), ...]。返回 [(规范化行, 出现篇数, 站点数, 原样示例)]，按篇数降序。"""
    cnt, hosts, example = Counter(), defaultdict(set), {}
    for host, text in docs:
        for l in set(x.strip() for x in text.split("\n") if len(x.strip()) >= min_len):
            k = norm_line(l)
            cnt[k] += 1
            if len(hosts[k]) < min_hosts + 1:
                hosts[k].add(host)
            example.setdefault(k, l)
    out = [(k, c, len(hosts[k]), example[k]) for k, c in cnt.items() if c >= min_docs and len(hosts[k]) >= min_hosts]
    return sorted(out, key=lambda x: -x[1])
