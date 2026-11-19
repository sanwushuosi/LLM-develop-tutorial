#!/usr/bin/env python3
"""第 4 步（上）：清洗前后的统计看板。

把“清洗前”（目标语言的提取结果）和一个或多个清洗版本放在一起比较：
  漏斗（篇数、字符数）、各规则命中、长度分布、行末标点、重复行、符号比例、停用词、语言与站点分布、个人信息。
用法：
  python 04_report.py --runs work/03_clean/book_wet_text work/03_clean/clean_text_trafilatura
输出：work/04_report/report.html
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from labkit import report as R
from labkit import signals
from labkit.clean import RULE_NAMES, load_config
from labkit.io import read_jsonl

KEYS = [("n_units", "长度（英文词数 / 中文字数）", True, None),
        ("frac_lines_end_punct", "以标点结尾的行占比", False, (0, 1)),
        ("frac_short_lines", "短行占比", False, (0, 1)),
        ("dup_line_frac", "文档内重复行占比", False, (0, 0.6)),
        ("symbol_ratio", "符号字符占比", False, (0, 0.5)),
        ("alpha_word_frac", "含字母的词占比（英文）", False, (0.4, 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="work/02_extract")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", default="configs/clean.yaml", help="用于确定“目标语言”与 n-gram 设置")
    ap.add_argument("--out", default="work/04_report/report.html")
    a = ap.parse_args()
    cfg = load_config(a.config)
    langs = set(cfg.get("languages") or [])

    # 清洗前：用第一个完整流水线版本里的语言识别结果确定目标语言文档，再在提取文本上算信号
    first = a.runs[-1]
    lang_of = {r["id"]: r["lang"] for r in read_jsonl(first, fields=["id", "lang"])}
    before = {k: [] for k, *_ in KEYS}
    before_hosts, before_lang = Counter(), Counter()
    n_before = c_before = 0
    for r in read_jsonl(a.extract, fields=["id", "host", "text_trafilatura"]):
        lg = lang_of.get(r["id"])
        if langs and lg not in langs:
            continue
        t = r["text_trafilatura"] or ""
        if not t.strip():
            continue
        n_before += 1
        c_before += len(t)
        before_lang[lg] += 1
        before_hosts[(r["host"] or "").rsplit(".", 1)[-1]] += 1
        s = signals.compute(t, lg)
        for k, *_ in KEYS:
            before[k].append(s.get(k))

    S = []
    funnel_rows, after_sets = [], {}
    for run in a.runs:
        summ = json.loads(Path(run, "summary.json").read_text("utf-8"))
        name = summ["tag"]
        vals = {k: [] for k, *_ in KEYS}
        kept_lang = Counter()
        for r in read_jsonl(run, fields=["lang", "status", "text"]):
            if r["status"] != "kept":
                continue
            kept_lang[r["lang"]] += 1
            s = signals.compute(r["text"], r["lang"])
            for k, *_ in KEYS:
                vals[k].append(s.get(k))
        after_sets[name] = vals
        funnel_rows.append((name, summ["source"], f"{summ['docs_target_lang']:,}", f"{summ['docs_kept']:,}",
                            f"{summ['keep_frac_of_target_lang']:.1%}", f"{summ['char_keep_frac_of_target_lang']:.1%}",
                            ", ".join(f"{k} {v}" for k, v in sorted(kept_lang.items()))))
        hits = [(f"{k} {RULE_NAMES.get(k, '')}", v) for k, v in summ["doc_rule_hit_frac_of_target_lang"].items()][:15]
        S.append(f"<h2>规则命中：{R.esc(name)}</h2>")
        S.append(R.barh(hits, f"{name}：各规则命中的文档占比（一篇可命中多条）", "占目标语言文档的比例", color=R.COLORS[1]))
        if summ.get("line_rule_hits"):
            S.append(R.table([(k, RULE_NAMES.get(k, ""), f"{v:,}") for k, v in summ["line_rule_hits"].items()],
                             ["行级规则", "含义", "删除的行数"], num_cols=(2,)))
    head = [R.cards([("清洗前（目标语言、正文非空）", f"{n_before:,} 篇"), ("清洗前字符", f"{c_before / 1e6:.1f} M")]),
            "<h2>漏斗：每个版本保留了多少</h2>",
            R.table(funnel_rows, ["版本", "清洗所用文本", "目标语言篇数", "保留篇数", "篇数保留率", "字符保留率", "各语言保留"]),
            "<div class='note'>同一套规则，用在 CC WET 上和用在 trafilatura 提取结果上，保留率可以差好几倍："
            "阈值必须和上游的提取方式一起校准。</div>"]
    S = head + S
    S.append("<h2>分布：清洗前 vs 清洗后</h2><div class='grid2'>")
    for k, title, logx, rng in KEYS:
        series = {"清洗前": before[k]}
        for name, vals in after_sets.items():
            series[f"清洗后·{name}"] = vals[k]
        S.append(R.hist(series, title, title, log_x=logx, range_=rng))
    S.append("</div>")
    out = R.page("第 4 步：清洗前后的统计对比", S, a.out)
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
