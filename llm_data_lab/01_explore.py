#!/usr/bin/env python3
"""第 1 步：看清原始数据长什么样（不做任何清洗）。

读取 data/cc_sample（原始 HTML + CC 官方 WET 文本），回答这些问题：
  * 样本里有哪些语言？中文占多少？
  * 一个网页的 HTML 有多大，其中真正的文字占多少？
  * 哪些站点、哪些顶级域名最多？
  * CC 的 WET 文本有多长？有多少篇完全重复？最大的重复簇是什么？
  * 哪些“行”在大量不同网页里反复出现（导航、页脚、验证页…）？
  * 邮箱、手机号等个人信息有多常见？
输出：work/01_explore/report.html（双击打开）与 summary.json。
用法：python 01_explore.py [--input data/cc_sample] [--limit 0]
"""
import argparse
import hashlib
import re
from collections import Counter

from labkit import report as R
from labkit.book import PII_PATTERNS
from labkit.io import read_jsonl

LANG_NAMES = {"eng": "英语", "zho": "中文", "rus": "俄语", "deu": "德语", "jpn": "日语", "fra": "法语",
              "spa": "西班牙语", "por": "葡萄牙语", "ita": "意大利语", "pol": "波兰语", "nld": "荷兰语",
              "kor": "韩语", "vie": "越南语", "tur": "土耳其语", "ces": "捷克语", "ind": "印尼语"}


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cc_sample")
    ap.add_argument("--out", default="work/01_explore")
    ap.add_argument("--limit", type=int, default=0, help="只读前 N 条（调试用），0 表示全部")
    a = ap.parse_args()

    n = 0
    langs, hosts, tlds, charsets = Counter(), Counter(), Counter(), Counter()
    html_kb, wet_chars, ratio = [], [], []
    doc_hash = Counter()
    hash_example = {}
    line_cnt, line_text = Counter(), {}
    pii_docs = Counter()
    truncated = empty_wet = 0
    for r in read_jsonl(a.input, limit=a.limit or None,
                        fields=["url", "host", "cc_lang", "html_bytes", "charset", "truncated", "wet_text"]):
        n += 1
        wet = r["wet_text"] or ""
        langs[r["cc_lang"] or "未知"] += 1
        hosts[r["host"]] += 1
        tlds[(r["host"] or "").rsplit(".", 1)[-1]] += 1
        charsets[(r["charset"] or "").lower()] += 1
        truncated += bool(r["truncated"])
        html_kb.append(r["html_bytes"] / 1024)
        wet_chars.append(len(wet))
        ratio.append(len(wet.encode("utf-8")) / max(1, r["html_bytes"]))
        if not wet.strip():
            empty_wet += 1
        h = hashlib.md5(norm(wet).encode()).hexdigest()
        doc_hash[h] += 1
        if h not in hash_example:
            hash_example[h] = (r["url"], wet[:120].replace("\n", " ⏎ "))
        for line in set(l.strip() for l in wet.split("\n")[:400] if len(l.strip()) >= 15):
            k = hashlib.md5(line.encode()).digest()[:8]
            line_cnt[k] += 1
            if line_cnt[k] == 2:
                line_text[k] = line
        for tag, pat in PII_PATTERNS.items():
            if pat.search(wet):
                pii_docs[tag] += 1
    if n == 0:
        raise SystemExit(f"{a.input} 里没有读到数据")

    dup_docs = sum(c - 1 for c in doc_hash.values())
    top_clusters = [(c, *hash_example[h]) for h, c in doc_hash.most_common(8) if c > 1]
    multi_line = [(line_text[k], c) for k, c in line_cnt.most_common(30) if c >= 3]
    lines_total = sum(line_cnt.values())
    lines_in_3plus = sum(c for c in line_cnt.values() if c >= 3)
    host_ge10 = sum(c for c in hosts.values() if c >= 10)
    med = lambda xs: sorted(xs)[len(xs) // 2]                                   # noqa: E731

    summary = {
        "docs": n, "distinct_hosts": len(hosts), "docs_per_host": round(n / len(hosts), 3),
        "docs_from_hosts_with_10plus": host_ge10,
        "lang_top": langs.most_common(12),
        "zh_share": round(langs.get("zho", 0) / n, 4), "en_share": round(langs.get("eng", 0) / n, 4),
        "html_kb_median": round(med(html_kb), 1), "html_total_gb": round(sum(html_kb) / 1024 / 1024, 3),
        "wet_chars_median": med(wet_chars), "wet_total_mchars": round(sum(wet_chars) / 1e6, 1),
        "text_to_html_bytes_median": round(med(ratio), 4),
        "empty_wet": empty_wet, "truncated": truncated,
        "exact_dup_docs": dup_docs, "exact_dup_frac": round(dup_docs / n, 4),
        "largest_dup_cluster": top_clusters[0][0] if top_clusters else 1,
        "lines_15plus_chars": lines_total, "line_occurrences_in_3plus_docs_frac": round(lines_in_3plus / max(1, lines_total), 4),
        "pii_doc_frac": {k: round(v / n, 4) for k, v in pii_docs.items()},
        "charset_top": charsets.most_common(6),
    }
    R.save_json(summary, f"{a.out}/summary.json")

    S = []
    S.append(R.cards([("网页数", f"{n:,}"), ("不同站点数", f"{len(hosts):,}"),
                      ("HTML 总量（解压后）", f"{summary['html_total_gb']:.2f} GB"),
                      ("WET 文本总量", f"{summary['wet_total_mchars']:.0f} M 字符"),
                      ("文本 / HTML 字节比（中位数）", f"{summary['text_to_html_bytes_median']:.1%}"),
                      ("完全重复的网页", f"{summary['exact_dup_frac']:.2%}"),
                      ("中文网页占比", f"{summary['zh_share']:.1%}")]))
    S.append("<h2>1. 语言分布（CC 官方 CLD2 标签）</h2>")
    S.append(R.barh([(f"{LANG_NAMES.get(k, k)} {k}", v) for k, v in langs.most_common(15)], "网页数最多的 15 种语言"))
    S.append("<h2>2. 体积：HTML 很大，文字很少</h2><div class='grid2'>")
    S.append(R.hist({"HTML 大小": html_kb}, "每个网页的 HTML 大小", "KB（对数坐标）", log_x=True))
    S.append(R.hist({"WET 文本": [max(1, x) for x in wet_chars]}, "每个网页的 WET 文本长度", "字符数（对数坐标）", log_x=True))
    S.append("</div>")
    S.append(f"<div class='note'>HTML 中位数 {summary['html_kb_median']:.0f} KB，而 WET 文本只占 HTML 字节的约 "
             f"{summary['text_to_html_bytes_median']:.1%}：抓下来的绝大部分字节是标签、脚本和样式。</div>")
    S.append("<h2>3. 站点与顶级域名</h2><div class='grid2'>")
    S.append(R.barh(hosts.most_common(15), "网页数最多的 15 个站点"))
    S.append(R.barh(tlds.most_common(15), "顶级域名 Top 15"))
    S.append("</div>")
    S.append(f"<div class='note'>{len(hosts):,} 个不同站点，平均每站 {summary['docs_per_host']:.2f} 篇："
             "一次随机抽样几乎覆盖的都是不同站点，所以“同站模板页”在小样本里不多，在全量数据里却非常多。</div>")
    S.append("<h2>4. 完全重复：最大的几个重复簇</h2>")
    S.append(R.table([(c, u, t) for c, u, t in top_clusters], ["重复次数", "其中一个 URL", "文本开头"], num_cols=(0,)))
    S.append("<h2>5. 在 ≥3 个不同网页里出现的行（跨文档模板行）</h2>")
    S.append(f"<p>长度 ≥15 字符的行（同一网页内先去重）共 {lines_total:,} 次出现，其中 "
             f"<b>{summary['line_occurrences_in_3plus_docs_frac']:.1%}</b> 出现在 ≥3 个网页里。最常见的 30 行：</p>")
    S.append(R.table([(c, t[:100]) for t, c in multi_line], ["出现在多少个网页", "行内容"], num_cols=(0,)))
    S.append("<h2>6. 个人信息出现率（WET 文本，手册 4.2.5 节的正则）</h2>")
    S.append(R.table([(k, v, f"{v / n:.1%}") for k, v in pii_docs.most_common()], ["类型", "网页数", "占比"], num_cols=(1, 2)))
    S.append("<h2>7. 编码与截断</h2>")
    S.append(R.table([(k or "（未声明）", v) for k, v in charsets.most_common(8)], ["charset", "网页数"], num_cols=(1,)))
    S.append(f"<p>被 CC 截断（超过抓取上限）的网页：{truncated}；WET 文本为空的网页：{empty_wet}。</p>")
    out = R.page("第 1 步：原始数据概览（CC-MAIN 样本）", S, f"{a.out}/report.html")

    print(f"网页 {n:,}，站点 {len(hosts):,}；英文 {summary['en_share']:.1%}，中文 {summary['zh_share']:.1%}")
    print(f"HTML 中位 {summary['html_kb_median']:.0f} KB，文本/HTML 字节比中位 {summary['text_to_html_bytes_median']:.1%}")
    print(f"完全重复网页 {summary['exact_dup_frac']:.2%}（最大簇 {summary['largest_dup_cluster']} 篇）；"
          f"出现在 ≥3 个网页的行占 {summary['line_occurrences_in_3plus_docs_frac']:.1%}")
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
