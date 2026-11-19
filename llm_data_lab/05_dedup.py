#!/usr/bin/env python3
"""第 5 步：去重。看清“从源头拿到的数据里有多少重复”，以及重复长什么样。

四件事：
  1. 精确重复：规范化（小写、合并空白）后完全相同的文档；
  2. 近似重复：手册 4.3.3 节的 MinHash（K=128）+ LSH（16 段×8 行），估计 Jaccard ≥ 0.8 视为重复；
  3. 跨文档模板行：在 ≥5 篇、≥3 个站点里出现的行（导航、页脚、验证提示……），写成文件供第 3 步删除；
  4. 规模曲线：样本从 1 个 WARC 文件逐步增大到全部，重复率怎样变化。
清洗前（目标语言的提取文本）和清洗后（第 3 步保留的文档）各算一遍，对照看。
用法：python 05_dedup.py --run work/03_clean/clean_text_trafilatura
输出：work/05_dedup/<run 名>/report.html、summary.json、boilerplate_lines.txt、keep_ids.txt
"""
import argparse
import difflib
import json
import time
from pathlib import Path

import numpy as np

from labkit import report as R
from labkit.dedup import boilerplate_lines, exact_clusters, near_clusters, signatures
from labkit.io import files_of, read_jsonl


def load_before(extract_dir, lang_of):
    docs = []
    for pi, f in enumerate(files_of(extract_dir)):
        for r in read_jsonl(f, fields=["id", "url", "host", "text_trafilatura"]):
            if r["id"] in lang_of and (r["text_trafilatura"] or "").strip():
                docs.append({"id": r["id"], "url": r["url"], "host": r["host"], "lang": lang_of[r["id"]],
                             "text": r["text_trafilatura"], "part": pi})
    return docs


def load_after(run_dir):
    docs = []
    for pi, f in enumerate(files_of(run_dir)):
        for r in read_jsonl(f, fields=["id", "url", "host", "lang", "status", "text"]):
            if r["status"] == "kept":
                docs.append({"id": r["id"], "url": r["url"], "host": r["host"], "lang": r["lang"],
                             "text": r["text"], "part": pi})
    return docs


def analyse(docs, name):
    t0 = time.time()
    texts = [d["text"] for d in docs]
    ex = exact_clusters(texts)
    ex_removed = sum(len(v) - 1 for v in ex.values())
    sigs, ok = signatures(texts)
    near, n_cand, confirmed = near_clusters(sigs, ok)
    near_removed = sum(len(v) - 1 for v in near)
    print(f"  [{name}] {len(docs):,} 篇：精确重复可删 {ex_removed:,}（{ex_removed / len(docs):.2%}），"
          f"近似重复（含精确）可删 {near_removed:,}（{near_removed / len(docs):.2%}），"
          f"LSH 候选对 {n_cand:,}，最大簇 {max((len(v) for v in near), default=1)}；{time.time() - t0:.0f}s")
    return {"docs": len(docs), "exact_removed": ex_removed, "near_removed": near_removed,
            "lsh_candidates": n_cand, "confirmed_pairs": len(confirmed),
            "largest_cluster": max((len(v) for v in near), default=1)}, ex, near, sigs, ok, confirmed


def word_diff(a, b, limit=1500):
    """把两段文本按词对齐，标出不同之处：红=只在 A，绿=只在 B。"""
    wa, wb = a[:limit].split(), b[:limit].split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        sa, sb = R.esc(" ".join(wa[i1:i2])), R.esc(" ".join(wb[j1:j2]))
        if op == "equal":
            out_a.append(sa)
            out_b.append(sb)
        else:
            if sa:
                out_a.append(f"<span style='background:#ffd6d6'>{sa}</span>")
            if sb:
                out_b.append(f"<span style='background:#d4f5dc'>{sb}</span>")
    return " ".join(out_a), " ".join(out_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/03_clean/clean_text_trafilatura")
    ap.add_argument("--extract", default="work/02_extract")
    ap.add_argument("--boiler-min-docs", type=int, default=5)
    ap.add_argument("--boiler-min-hosts", type=int, default=3)
    a = ap.parse_args()
    out = Path("work/05_dedup") / Path(a.run).name
    out.mkdir(parents=True, exist_ok=True)

    lang_of = {r["id"]: r["lang"] for r in read_jsonl(a.run, fields=["id", "lang", "reasons"]) if r["reasons"] != ["D01"]}
    before = load_before(a.extract, lang_of)
    after = load_after(a.run)
    print("去重统计：")
    sb, ex_b, near_b, sigs_b, ok_b, conf_b = analyse(before, "清洗前")
    sa, ex_a, near_a, sigs_a, ok_a, conf_a = analyse(after, "清洗后")

    # 规模曲线：取前 k 个分片（每个分片来自一个随机 WARC 文件），清洗前、清洗后各算一条
    def scale_curve(docs, sigs, ok):
        parts = np.array([d["part"] for d in docs])
        n_parts = int(parts.max()) + 1
        res = []
        for k in sorted(set([1, 2, 4, 8, 12, 16, n_parts])):
            if k > n_parts:
                continue
            sel = np.flatnonzero(parts < k)
            cl, _, _ = near_clusters(sigs[sel], ok[sel])
            removed = sum(len(v) - 1 for v in cl)
            res.append((k, len(sel), removed, removed / len(sel)))
        return res
    curve = scale_curve(before, sigs_b, ok_b)
    curve_a = scale_curve(after, sigs_a, ok_a)
    print("  规模曲线（清洗前）：", ", ".join(f"{k} 个文件 {n:,} 篇 → {f:.2%}" for k, n, _, f in curve))
    print("  规模曲线（清洗后）：", ", ".join(f"{k} 个文件 {n:,} 篇 → {f:.2%}" for k, n, _, f in curve_a))

    # 跨文档模板行（在清洗前的提取文本上统计）
    bl = boilerplate_lines([(d["host"], d["text"]) for d in before], a.boiler_min_docs, a.boiler_min_hosts)
    (out / "boilerplate_lines.txt").write_text("\n".join(k for k, *_ in bl) + "\n", "utf-8")
    bl_set = {k for k, *_ in bl}
    from labkit.dedup import norm_line
    tot_chars = bl_chars = 0
    for d in before:
        for l in d["text"].split("\n"):
            s = l.strip()
            tot_chars += len(s)
            if len(s) >= 10 and norm_line(s) in bl_set:
                bl_chars += len(s)
    print(f"  跨文档模板行：{len(bl):,} 种，占清洗前正文字符的 {bl_chars / max(1, tot_chars):.2%} → {out / 'boilerplate_lines.txt'}")

    # 去重后保留的 id（清洗后的数据，每个近似重复簇保留第一篇）
    drop = set()
    for cl in near_a:
        for i in sorted(cl)[1:]:
            drop.add(i)
    keep_ids = [d["id"] for i, d in enumerate(after) if i not in drop]
    (out / "keep_ids.txt").write_text("\n".join(keep_ids) + "\n", "utf-8")

    summary = {"before": sb, "after": sa,
               "scale_curve_before": [dict(files=k, docs=n, removed=r, frac=round(f, 4)) for k, n, r, f in curve],
               "scale_curve_after": [dict(files=k, docs=n, removed=r, frac=round(f, 4)) for k, n, r, f in curve_a],
               "boilerplate_line_types": len(bl), "boilerplate_char_frac_before": round(bl_chars / max(1, tot_chars), 4),
               "keep_after_dedup": len(keep_ids)}
    R.save_json(summary, out / "summary.json")

    # ── 报告 ──
    S = [R.cards([("清洗前篇数", f"{sb['docs']:,}"), ("清洗前·精确重复", f"{sb['exact_removed'] / sb['docs']:.2%}"),
                  ("清洗前·近似重复", f"{sb['near_removed'] / sb['docs']:.2%}"),
                  ("清洗后篇数", f"{sa['docs']:,}"), ("清洗后·近似重复", f"{sa['near_removed'] / sa['docs']:.2%}"),
                  ("跨文档模板行占正文字符", f"{summary['boilerplate_char_frac_before']:.2%}")])]
    S.append("<h2>1. 重复率随样本规模增长</h2>")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for c, lab, col in ((curve, "清洗前（含验证页等垃圾）", R.COLORS[1]), (curve_a, "清洗后", R.COLORS[0])):
        ax.plot([k for k, *_ in c], [f * 100 for *_, f in c], "o-", color=col, label=lab)
    ax.set_xlabel("样本包含的 WARC 文件数（每个文件约 1 千个网页）")
    ax.set_ylabel("近似重复可删比例（%）")
    ax.set_title("样本越大，能找到的重复越多")
    ax.legend(fontsize=8)
    S.append(R.fig_to_img(fig))
    S.append("<div class='note'>重复是“成对”出现的：文档数翻倍，可能互相重复的文档对数约翻四倍。"
             "一个分片里看到的重复率远低于全量数据的真实重复率——去重必须在足够大的范围上做，"
             "但范围也不是越大越好（FineWeb：对 96 个快照做全局去重反而更差，逐快照去重更好）。</div>")
    for title, docs_, clusters_ in (("2. 最大的近似重复簇（清洗前）", before, near_b),
                                    ("2'. 清洗后仍然存在的近似重复簇——多数是“跨站模板页”，规则没拦住", after, near_a)):
        S.append(f"<h2>{title}</h2>")
        rows = []
        for cl in sorted(clusters_, key=len, reverse=True)[:12]:
            d0 = docs_[cl[0]]
            hosts = {docs_[i]["host"] for i in cl}
            rows.append((len(cl), len(hosts), d0["url"], d0["text"][:110].replace("\n", " ⏎ ")))
        S.append(R.table(rows, ["簇大小", "涉及站点数", "其中一个 URL", "文本开头"], num_cols=(0, 1)))
    S.append("<div class='note'>清洗后的重复簇是很好的质检线索：同一模板出现在几十个不同站点上（cookie 提示、"
             "CMS 默认示例页、站点不存在的报错页、成人站警告……），说明它们是一类系统性问题，"
             "应当补一条规则，而不只是删掉这几篇。这正是第 6 步 Agent 质检要做的事。</div>")
    S.append("<h2>3. 近似重复长什么样：逐词对比（红=只在 A，绿=只在 B）</h2>")
    shown = 0
    for ia, ib, sim in sorted(conf_b, key=lambda x: x[2]):
        A, B = before[ia], before[ib]
        if A["text"] == B["text"] or sim > 0.97:
            continue
        da, db = word_diff(A["text"], B["text"])
        S.append(f"<p><b>估计相似度 {sim:.2f}</b>｜A：{R.esc(A['url'])}<br>B：{R.esc(B['url'])}</p>"
                 f"<div class='grid2'><div class='col' style='font-size:12.5px;border:1px solid #ddd;padding:6px'>{da}</div>"
                 f"<div class='col' style='font-size:12.5px;border:1px solid #ddd;padding:6px'>{db}</div></div>")
        shown += 1
        if shown >= 5:
            break
    S.append("<h2>4. 跨文档模板行 Top 40（清洗前）</h2>")
    S.append(f"<p>共 {len(bl):,} 种（≥{a.boiler_min_docs} 篇、≥{a.boiler_min_hosts} 个站点）。已写入 "
             f"<code>{R.esc(out / 'boilerplate_lines.txt')}</code>，可用 <code>03_clean.py --boilerplate</code> 在清洗时删除。</p>")
    S.append(R.table([(c, h if h <= a.boiler_min_hosts else f"≥{a.boiler_min_hosts + 1}", ex[:100]) for _, c, h, ex in bl[:40]],
                     ["出现篇数", "站点数", "行（原样示例）"], num_cols=(0,)))
    R.page("第 5 步：去重", S, out / "report.html")
    print(f"报告：{out / 'report.html'}；去重后保留 {len(keep_ids):,} 篇 → {out / 'keep_ids.txt'}")


if __name__ == "__main__":
    main()
