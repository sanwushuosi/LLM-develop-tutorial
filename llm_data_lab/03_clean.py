#!/usr/bin/env python3
"""第 3 步：清洗。语言识别 → 行级清洗 → 文档级过滤 → 隐私脱敏。

三种常用跑法（建议都跑一遍，对比结果）：
  python 03_clean.py --mode book --source wet_text            # 手册 4.2.3 的 4 条规则，直接用在 CC WET 上
  python 03_clean.py --mode book --source text_trafilatura    # 同样的规则，用在 trafilatura 提取结果上
  python 03_clean.py --config configs/clean.yaml              # 完整流水线（默认）
改完 configs/clean.yaml 后用 --tag 起个新名字再跑，便于前后对比，例如 --tag v2_relax_punct。
加入第 5 步统计出的跨文档模板行：--boilerplate work/05_dedup/<tag>/boilerplate_lines.txt
输出：work/03_clean/<tag>/part-*.jsonl.gz 与 summary.json（漏斗、各规则命中数）。
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from labkit.clean import RULE_NAMES, Cleaner, load_config
from labkit.io import files_of, read_jsonl, write_jsonl


def work(args):
    src, dst, cfg, boiler, source, mode = args
    cl = Cleaner(cfg, boiler)
    rows = []
    for rec in read_jsonl(src):
        rows.append(cl(rec, source_field=source, mode=mode))
    write_jsonl(dst, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="work/02_extract")
    ap.add_argument("--config", default="configs/clean.yaml")
    ap.add_argument("--source", default="text_trafilatura", help="用哪个字段做清洗：text_trafilatura / text_resiliparse / wet_text")
    ap.add_argument("--mode", default="full", choices=["full", "book"])
    ap.add_argument("--boilerplate", default="", help="跨文档模板行文件（第 5 步生成），每行一条")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    boiler = []
    if a.boilerplate:
        boiler = [l.rstrip("\n") for l in open(a.boilerplate, encoding="utf-8") if l.strip()]
    tag = a.tag or (f"book_{a.source}" if a.mode == "book" else f"{Path(a.config).stem}_{a.source}")
    out = Path("work/03_clean") / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(str(p), str(out / p.name), cfg, boiler, a.source, a.mode) for p in files_of(a.input)]
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(work, jobs))

    # 汇总：漏斗与各规则的命中
    lang, status, rules, line_rules = Counter(), Counter(), Counter(), Counter()
    chars_in = chars_out = chars_in_target = 0
    lang_kept = Counter()
    for r in read_jsonl(out):
        lang[r["lang"]] += 1
        status[r["status"]] += 1
        rules.update(r["reasons"])
        chars_in += r["n_chars_before"]
        if r["reasons"] != ["D01"]:
            chars_in_target += r["n_chars_before"]
        chars_out += r["n_chars_after"]
        if r["status"] == "kept":
            lang_kept[r["lang"]] += 1
        for ok, why in (r.get("line_ops") or []):
            if not ok:
                line_rules[why] += 1
    target = n - rules.get("D01", 0)
    summary = {
        "tag": tag, "source": a.source, "mode": a.mode, "config": a.config, "boilerplate_lines": len(boiler),
        "docs_in": n, "docs_target_lang": target, "docs_kept": status["kept"],
        "keep_frac_of_target_lang": round(status["kept"] / max(1, target), 4),
        "kept_by_lang": dict(lang_kept), "lang_detected_top": lang.most_common(10),
        "chars_in": chars_in, "chars_in_target_lang": chars_in_target, "chars_out": chars_out,
        "char_keep_frac_of_target_lang": round(chars_out / max(1, chars_in_target), 4),
        "doc_rule_hits": {k: v for k, v in rules.most_common()},
        "doc_rule_hit_frac_of_target_lang": {k: round(v / max(1, target), 4) for k, v in rules.most_common() if k != "D01"},
        "line_rule_hits": dict(line_rules.most_common()),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(f"[{tag}] 输入 {n:,} 篇；目标语言（{','.join(cfg.get('languages') or ['全部'])}）{target:,} 篇；"
          f"保留 {status['kept']:,} 篇（{summary['keep_frac_of_target_lang']:.1%}），"
          f"字符保留 {summary['char_keep_frac_of_target_lang']:.1%}；用时 {summary['seconds']}s")
    print("  各语言保留：", dict(lang_kept))
    print("  文档级规则命中（占目标语言文档，一篇可命中多条）：")
    for k, v in rules.most_common():
        if k != "D01":
            print(f"    {k:10} {RULE_NAMES.get(k, ''):18} {v:6,}  {v / max(1, target):6.1%}")
    if line_rules:
        print("  行级规则删除的行数：", {f"{k} {RULE_NAMES.get(k, '')}": v for k, v in line_rules.most_common()})


if __name__ == "__main__":
    main()
