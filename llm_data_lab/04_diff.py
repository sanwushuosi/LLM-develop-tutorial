#!/usr/bin/env python3
"""第 4 步（下）：逐条对比，看清每一行是被谁删掉的。

对每篇选中的文档，并排显示三栏：
  ① CC 官方 WET 文本：灰色删除线 = 正文提取（trafilatura）时没有保留的行
  ② 提取后的文本：红底 = 清洗时删掉的行，行尾标出规则号（L01…L05）
  ③ 最终保留的文本（被整篇丢弃时显示丢弃原因）
并列出这篇文档的质量信号，超出阈值的标红。
页面是一个自包含的 HTML 文件，所有网页文字都已转义，不会执行其中的任何代码。
用法：
  python 04_diff.py --run work/03_clean/clean_text_trafilatura --n 20                  # 随机 20 篇
  python 04_diff.py --run work/03_clean/clean_text_trafilatura --status removed --rule D14 --n 15
  python 04_diff.py --run work/03_clean/clean_text_trafilatura --ids <id1>,<id2>
输出：work/04_diff/<run 名>_<筛选条件>.html
"""
import argparse
import random
import re
from pathlib import Path

from labkit import report as R
from labkit.clean import RULE_NAMES
from labkit.io import read_jsonl

CSS = R.CSS + """
.doc{border:1px solid #cddcee;border-radius:8px;margin:18px 0;padding:8px 12px}
.doc summary{cursor:pointer;font-weight:bold;color:#0b5aa6}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px}
.col{border:1px solid #e1e6ec;border-radius:6px;padding:6px;max-height:520px;overflow:auto;font-size:12.5px;
     white-space:pre-wrap;word-break:break-word;background:#fcfdff}
.col h4{margin:0 0 6px 0;font-size:13px;color:#333;position:sticky;top:-6px;background:#fcfdff}
.gone{color:#9aa3ab;text-decoration:line-through}
.del{background:#ffe1e1;color:#8a1f1f}
.tag{display:inline-block;background:#c0392b;color:#fff;border-radius:3px;font-size:10px;padding:0 4px;margin-left:4px}
.kept{color:#1a1a1a}
.badge{display:inline-block;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:4px}
.k{background:#d8f1df;color:#1b6b34}.r{background:#fde2e1;color:#9b1c1c}
.sig td.bad{background:#fde2e1}
"""

SIG_SHOW = ["n_units", "frac_lines_end_punct", "frac_short_lines", "dup_line_frac", "dup_line_char_frac",
            "symbol_ratio", "alpha_word_frac", "stop_hits", "zh_func_ratio", "top2_char_frac",
            "dup_short_char_frac", "dup_long_char_frac", "mojibake_hits"]


def norm(l):
    return re.sub(r"\s+", " ", l).strip()


def render_doc(i, ex, cl):
    wet = ex.get("wet_text") or ""
    src = ex.get("text_trafilatura") or ""
    src_lines = set(norm(l) for l in src.split("\n") if l.strip())
    col1 = []
    for l in wet.split("\n"):
        if not l.strip():
            col1.append("")
        elif norm(l) in src_lines:
            col1.append(f'<span class="kept">{R.esc(l)}</span>')
        else:
            col1.append(f'<span class="gone">{R.esc(l)}</span>')
    col2 = []
    ops = cl.get("line_ops") or [[1, ""]] * len(src.split("\n"))
    for l, (ok, why) in zip(src.split("\n"), ops):
        if ok:
            col2.append(R.esc(l))
        else:
            col2.append(f'<span class="del">{R.esc(l)}</span><span class="tag" title="{R.esc(RULE_NAMES.get(why, ""))}">{why}</span>')
    if cl["status"] == "kept":
        col3 = R.esc(cl["text"])
        badge = '<span class="badge k">保留</span>'
    else:
        col3 = "<b>整篇丢弃</b>：" + "；".join(f"{r} {R.esc(RULE_NAMES.get(r, ''))}" for r in cl["reasons"])
        badge = '<span class="badge r">丢弃</span>' + "".join(
            f'<span class="badge r">{r} {R.esc(RULE_NAMES.get(r, ""))}</span>' for r in cl["reasons"])
    sig = cl.get("signals") or {}
    sig_rows = "".join(f"<tr><td>{k}</td><td>{R.esc(sig.get(k))}</td></tr>" for k in SIG_SHOW if sig.get(k) is not None)
    n_del = sum(1 for ok, _ in ops if not ok)
    return f"""<details class="doc" id="d{i}" {'open' if i < 3 else ''}>
<summary>#{i} {badge} [{R.esc(cl['lang'])} {cl['lang_score']}] {R.esc(cl.get('url') or '')}</summary>
<p style="font-size:12px;color:#555">id={R.esc(cl['id'])}｜WET {len(wet):,} 字符 → 提取 {len(src):,} 字符 → 保留 {cl['n_chars_after']:,} 字符｜清洗删除 {n_del} 行</p>
<div class="cols">
<div class="col"><h4>① CC WET（灰色删除线 = 提取时丢掉）</h4>{'<br>'.join(col1)}</div>
<div class="col"><h4>② 提取结果（红底 = 清洗删除，标签为规则号）</h4>{'<br>'.join(col2)}</div>
<div class="col"><h4>③ 最终结果</h4>{col3}</div>
</div>
<details><summary style="font-weight:normal;font-size:12px">质量信号</summary><table class="sig">{sig_rows}</table></details>
</details>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--extract", default="work/02_extract")
    ap.add_argument("--status", default="all", choices=["all", "kept", "removed"])
    ap.add_argument("--rule", default="", help="只看命中某条规则的文档，如 D14；或某条行级规则，如 L03")
    ap.add_argument("--lang", default="", help="只看某种语言，如 zh")
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    want = set(x for x in a.ids.split(",") if x)
    pool = []
    for r in read_jsonl(a.run):
        if r["reasons"] == ["D01"]:
            continue                                 # 非目标语言不展示
        if want and r["id"] not in want:
            continue
        if not want:
            if a.status != "all" and r["status"] != a.status:
                continue
            if a.lang and r["lang"] != a.lang:
                continue
            if a.rule:
                in_doc = a.rule in r["reasons"]
                in_line = any((not ok) and why == a.rule for ok, why in (r.get("line_ops") or []))
                if not (in_doc or in_line):
                    continue
        pool.append(r)
    if not want:
        random.Random(a.seed).shuffle(pool)
        pool = pool[:a.n]
    ids = {r["id"] for r in pool}
    ex = {r["id"]: r for r in read_jsonl(a.extract, fields=["id", "wet_text", "text_trafilatura"]) if r["id"] in ids}
    docs = "".join(render_doc(i, ex[r["id"]], r) for i, r in enumerate(pool) if r["id"] in ex)
    run = Path(a.run).name
    cond = "ids" if want else ("_".join(x for x in (a.status if a.status != "all" else "", a.rule, a.lang) if x) or "random")
    out = Path("work/04_diff") / f"{run}_{cond}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    legend = ("<p>筛选：" + R.esc(f"run={run} status={a.status} rule={a.rule or '-'} lang={a.lang or '-'}") +
              f"，共 {len(pool)} 篇。点击标题展开/折叠；把鼠标停在红色规则标签上可看到规则名。</p>")
    out.write_text(f"<!doctype html><html lang='zh'><meta charset='utf-8'><title>diff {R.esc(run)}</title>"
                   f"<style>{CSS}</style><body><h1>清洗前后逐条对比</h1>{legend}{docs}</body></html>", "utf-8")
    print(f"{len(pool)} 篇 → {out}")


if __name__ == "__main__":
    main()
