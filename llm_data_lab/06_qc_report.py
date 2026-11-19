#!/usr/bin/env python3
"""第 6 步（3/3）：汇总质检结果。

  python 06_qc_report.py --pack work/06_qc/<名字>
  python 06_qc_report.py --pack work/06_qc/<迭代后> --compare work/06_qc/<迭代前>    # 前后对比
读取 <pack>/results.jsonl（API 模式或 Agent 模式产出的都可以），输出：
  * bad 比例及 95% 置信区间（样本少时区间很宽——这本身就是要学的一课）；
  * 各规则命中次数与例子（证据原文高亮）；
  * 模型给出的“可以写成清洗规则”的建议，按规则归类——这是你改进 03_clean 的线索。
输出：<pack>/report.html
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from labkit import report as R
from labkit.io import read_jsonl
from labkit.qc import wilson


def load(pack):
    pack = Path(pack)
    samples = {s["id"]: s for s in read_jsonl(pack / "samples.jsonl")}
    res = {}
    bad_lines = 0
    for line in open(pack / "results.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            res[r["sample_id"]] = r
        except (json.JSONDecodeError, KeyError):
            bad_lines += 1
    return samples, res, bad_lines


def stats(samples, res):
    judged = [r for sid, r in res.items() if sid in samples and r.get("verdict") in ("good", "bad")]
    n = len(judged)
    k = sum(r["verdict"] == "bad" for r in judged)
    rules = Counter(v.get("rule_id") for r in judged for v in r.get("violations", []))
    return n, k, rules, judged


def highlight(text, ev):
    t = R.esc(text)
    for e in ev:
        e = (e or "").strip()
        if len(e) >= 6 and R.esc(e) in t:
            t = t.replace(R.esc(e), f"<mark>{R.esc(e)}</mark>", 1)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    samples, res, bad_lines = load(a.pack)
    n, k, rules, judged = stats(samples, res)
    meta = json.loads(Path(a.pack, "pack.json").read_text("utf-8"))
    lo, hi = wilson(k, n)
    missing = len(samples) - len([s for s in samples if s in res])
    print(f"{a.pack}：已判 {n} 条（缺 {missing} 条，无法解析 {bad_lines} 行）；bad {k} 条 = {k / max(1, n):.1%}"
          f"（95% 置信区间 {lo:.1%}～{hi:.1%}）")
    for rid, c in rules.most_common():
        print(f"  {rid}: {c}")
    S = [R.cards([("样本（" + meta["status"] + "）", f"{len(samples)}"), ("已判", f"{n}"),
                  ("bad 比例", f"{k / max(1, n):.1%}"), ("95% 置信区间", f"{lo:.0%}～{hi:.0%}")])]
    if meta["status"] == "removed":
        S.append("<div class='note'>这批样本是被清洗规则<b>删掉</b>的文档：判为 good 的比例就是规则的“误杀率”估计。</div>")
    else:
        S.append("<div class='note'>这批样本是清洗后<b>保留</b>的文档：判为 bad 的比例就是“漏网之鱼”的估计。"
                 "样本只有几十条时置信区间很宽，结论要谨慎——想把区间缩窄一半，样本要多 4 倍。</div>")
    if meta.get("paired_with"):
        s0, r0, _ = load(meta["paired_with"])
        v0 = {sid: r.get("verdict") for sid, r in r0.items()}
        rm = meta.get("removed_now", [])
        trans = Counter()
        for sid, r in res.items():
            if sid in samples and sid in v0:
                trans[(v0[sid], r.get("verdict"))] += 1
        rm_prev = Counter(v0.get(x["id"]) for x in rm)
        S.append("<h2>配对比较：同一批文档，上一轮 → 本轮</h2>")
        S.append(R.table([("上一轮 bad → 本轮被清洗删掉", rm_prev.get("bad", 0)),
                          ("上一轮 good → 本轮被清洗删掉（新规则的误杀？）", rm_prev.get("good", 0)),
                          ("bad → bad（仍未解决）", trans[("bad", "bad")]), ("bad → good（修好了）", trans[("bad", "good")]),
                          ("good → bad（判定不一致或改坏了）", trans[("good", "bad")]), ("good → good", trans[("good", "good")])],
                         ["变化", "篇数"], num_cols=(1,)))
        S.append("<div class='note'>配对比较看的是同一批文档的变化，比两次独立抽样的 bad 比例更能说明“新规则到底解决了什么、误伤了什么”。"
                 "good → bad 的条目要人工看：可能是判定员前后不一致（判定本身也有噪声）。</div>")
        print(f"配对：上一轮 bad 中被删 {rm_prev.get('bad', 0)}、修好 {trans[('bad', 'good')]}、仍 bad {trans[('bad', 'bad')]}；"
              f"上一轮 good 中被删 {rm_prev.get('good', 0)}、变 bad {trans[('good', 'bad')]}")
    if a.compare:
        s2, r2, _ = load(a.compare)
        n2, k2, rules2, _ = stats(s2, r2)
        lo2, hi2 = wilson(k2, n2)
        S.append("<h2>前后对比</h2>")
        allr = sorted(set(rules) | set(rules2))
        S.append(R.table([("bad 比例", f"{k2 / max(1, n2):.1%}（{lo2:.0%}～{hi2:.0%}）", f"{k / max(1, n):.1%}（{lo:.0%}～{hi:.0%}）")] +
                         [(rid, rules2.get(rid, 0), rules.get(rid, 0)) for rid in allr],
                         ["", f"对照：{Path(a.compare).name}（n={n2}）", f"本次：{Path(a.pack).name}（n={n}）"]))
        print(f"对照 {a.compare}：bad {k2}/{n2} = {k2 / max(1, n2):.1%}")
    S.append("<h2>各规则命中</h2>")
    S.append(R.barh(rules.most_common(), "命中次数（一条样本可命中多条规则）", "样本数", color=R.COLORS[4]))
    hints = defaultdict(list)
    for r in judged:
        for v in r.get("violations", []):
            if r.get("rule_hint"):
                hints[v.get("rule_id")].append(r["rule_hint"])
    S.append("<h2>模型建议的清洗规则（按命中规则归类）</h2>")
    for rid, hs in sorted(hints.items(), key=lambda x: -len(x[1])):
        S.append(f"<h3>{R.esc(rid)}（{len(hs)} 条）</h3><ul>" + "".join(f"<li>{R.esc(h)}</li>" for h in hs[:12]) + "</ul>")
    S.append("<h2>逐条结果（bad 在前；黄色高亮为证据原文）</h2>")
    order = sorted(judged, key=lambda r: r["verdict"] != "bad")
    for r in order:
        s = samples[r["sample_id"]]
        ev = [v.get("evidence", "") for v in r.get("violations", [])]
        viol = "；".join(f"<b>{R.esc(v.get('rule_id'))}</b> {R.esc(v.get('reason', ''))}" for v in r.get("violations", []))
        color = "#9b1c1c" if r["verdict"] == "bad" else "#1b6b34"
        S.append(f"<details style='border:1px solid #dde;border-radius:6px;margin:8px 0;padding:6px 10px'>"
                 f"<summary><span style='color:{color};font-weight:bold'>{r['verdict']}</span> "
                 f"#{s['k']:03d} [{R.esc(s['lang'])}] {R.esc(s['url'])}</summary>"
                 f"<p>{viol}</p><p style='color:#555'>规则建议：{R.esc(r.get('rule_hint', ''))}</p>"
                 f"<div style='white-space:pre-wrap;font-size:12.5px;max-height:420px;overflow:auto;background:#fafbfc;"
                 f"border:1px solid #eee;padding:6px'>{highlight(s['text'], ev)}</div></details>")
    out = Path(a.pack) / "report.html"
    R.page(f"第 6 步：质检结果 · {Path(a.pack).name}", S, out)
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
