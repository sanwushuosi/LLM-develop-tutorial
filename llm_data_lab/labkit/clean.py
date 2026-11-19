"""可配置的清洗流水线：语言识别 → 行级清洗 → 文档级过滤 → 隐私脱敏。

设计要点（对应理论课“数据怎么处理”一节）：
  * 每一行被删、每一篇被丢，都记下是哪条规则干的——否则无法统计每条规则的剔除率，也无法做 diff 复核；
  * 阈值全部放在 configs/clean.yaml 里，改阈值不用改代码；
  * mode="book" 时只用手册 4.2.3 节的 4 条规则（labkit.book.apply_rules），作为基线对照。
"""
import re
from collections import Counter

import yaml

from . import book, custom_rules, langid, signals

# 规则编号 → 中文名（报告与 diff 页面都用这张表）
RULE_NAMES = {
    "D01": "语言不在目标集合或置信度低", "D02": "太短", "D03": "太长", "D04": "平均词长异常",
    "D05": "符号比例过高", "D06": "项目符号行过多", "D07": "省略号结尾的行过多", "D08": "含字母的词太少",
    "D09": "停用词/虚词太少", "D10": "重复行过多", "D11": "重复段落过多", "D12": "高频 n-gram 过于集中",
    "D13": "重复 n-gram 覆盖过多", "D14": "以标点结尾的行太少", "D15": "短行过多", "D16": "含 lorem ipsum",
    "D17": "乱码/编码损坏", "D18": "URL 命中黑名单", "D19": "清洗后正文为空",
    "L01": "行：模板词（版权、登录、cookie…）", "L02": "行：跨文档高频模板行", "L03": "行：太短且无句末标点",
    "L04": "行：纯符号/无文字", "L05": "行：文档内重复",
    "B-长度": "手册规则：长度", "B-符号比例": "手册规则：符号比例", "B-行级特征": "手册规则：行级特征",
    "B-重复程度": "手册规则：重复程度",
}
RULE_NAMES.update(custom_rules.NAMES)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_line(l):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", l.strip().lower()))


def _has_text(l):
    return bool(re.search(r"[^\W\d_]", l))


class Cleaner:
    def __init__(self, cfg, boilerplate_lines=None):
        self.cfg = cfg
        lf = cfg.get("line_filters", {})
        kws = [k.lower() for k in lf.get("boilerplate_keywords", [])]
        self.kw_re = re.compile("|".join(re.escape(k) for k in kws)) if kws else None
        self.boiler = set(boilerplate_lines or [])
        bl = cfg.get("doc_filters", {}).get("url_blocklist_tokens", [])
        self.url_block = set(bl)
        on = set(cfg.get("custom_rules") or [])
        self.custom_doc = {k: f for k, f in custom_rules.DOC_RULES.items() if k in on}
        self.custom_line = {k: f for k, f in custom_rules.LINE_RULES.items() if k in on}

    # ── 行级 ─────────────────────────────────────────────────────────
    def clean_lines(self, text, lang):
        lf = self.cfg.get("line_filters", {})
        zh = signals.is_zh(lang)
        min_units = lf.get("min_line_units_zh" if zh else "min_line_units", 0)
        seen = set()
        ops = []                       # 与原文逐行对齐：[保留?, 规则号或 ""]
        kept = []
        for raw in text.split("\n"):
            l = raw.strip()
            reason = ""
            if not l:
                ops.append([1, ""])
                kept.append(raw)
                continue
            n_units = len(signals.units_of(l, lang))
            if self.kw_re and n_units <= lf.get("boilerplate_max_units", 12) * (3 if zh else 1) \
                    and self.kw_re.search(l.lower()):
                reason = "L01"
            elif self.boiler and _norm_line(l) in self.boiler:
                reason = "L02"
            elif lf.get("drop_symbol_lines", True) and not _has_text(l):
                reason = "L04"
            elif n_units < min_units and not l.endswith(signals.END_PUNCT):
                reason = "L03"
            elif lf.get("drop_duplicate_lines", True) and len(l) >= 10 and l in seen:
                reason = "L05"
            else:
                for code, fn in self.custom_line.items():
                    if fn(l, lang):
                        reason = code
                        break
            seen.add(l)
            ops.append([0 if reason else 1, reason])
            if not reason:
                kept.append(raw)
        out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        return out, ops

    # ── 文档级 ───────────────────────────────────────────────────────
    def doc_reasons(self, text, lang, url=""):
        df = self.cfg.get("doc_filters", {})
        zh = signals.is_zh(lang)
        dup_n = df.get("dup_ngram_n_zh" if zh else "dup_ngram_n", [5, 10])
        s = signals.compute(text, lang, dup_n=dup_n)
        r = []
        if not text.strip():
            return ["D19"], s
        lo, hi = df.get("units_range_zh" if zh else "units_range", [50, 100000])
        if s["n_units"] < lo:
            r.append("D02")
        if s["n_units"] > hi:
            r.append("D03")
        if not zh and s["mean_word_len"] is not None:
            a, b = df.get("mean_word_len", [3, 10])
            if not a <= s["mean_word_len"] <= b:
                r.append("D04")
        if s["symbol_ratio"] > df.get("max_symbol_ratio", 1.0) or \
                s["hash_ellipsis_per_word"] > df.get("max_hash_ellipsis_per_word", 1.0):
            r.append("D05")
        if s["frac_bullet_lines"] > df.get("max_bullet_lines", 1.0):
            r.append("D06")
        if s["frac_ellipsis_lines"] > df.get("max_ellipsis_lines", 1.0):
            r.append("D07")
        if not zh and s["alpha_word_frac"] < df.get("min_alpha_word_frac", 0.0):
            r.append("D08")
        if zh:
            if s["zh_func_ratio"] < df.get("min_zh_func_ratio", 0.0):
                r.append("D09")
        elif s["stop_hits"] < df.get("min_stop_hits", 0):
            r.append("D09")
        if s["dup_line_frac"] > df.get("max_dup_line_frac", 1.0) or \
                s["dup_line_char_frac"] > df.get("max_dup_line_char_frac", 1.0):
            r.append("D10")
        if s["dup_para_frac"] > df.get("max_dup_para_frac", 1.0):
            r.append("D11")
        t2, t3, t4 = df.get("max_top_ngram_char_frac", [1, 1, 1])
        if s["top2_char_frac"] > t2 or s["top3_char_frac"] > t3 or s["top4_char_frac"] > t4:
            r.append("D12")
        d_short, d_long = df.get("max_dup_ngram_char_frac", [1, 1])
        if s["dup_short_char_frac"] > d_short or s["dup_long_char_frac"] > d_long:
            r.append("D13")
        if s["frac_lines_end_punct"] < df.get("min_lines_end_punct", 0.0):
            r.append("D14")
        if s["frac_short_lines"] > df.get("max_short_lines", 1.0):
            r.append("D15")
        if df.get("drop_lorem_ipsum", True) and s["has_lorem_ipsum"]:
            r.append("D16")
        if s["mojibake_hits"] > df.get("max_mojibake_hits", 10**9):
            r.append("D17")
        if self.url_block and url:
            host = re.sub(r"^https?://", "", url.lower()).split("/")[0]
            if set(re.split(r"[.\-]", host)) & self.url_block:
                r.append("D18")
        for code, fn in self.custom_doc.items():
            if fn(text, lang, url, s):
                r.append(code)
        return r, s

    # ── 一篇文档的完整处理 ─────────────────────────────────────────────
    def __call__(self, rec, source_field="text_trafilatura", mode="full"):
        text = rec.get(source_field) or ""
        lang, score = langid.top1(text) if text.strip() else ("unknown", 0.0)
        out = {"id": rec["id"], "url": rec.get("url"), "host": rec.get("host"), "lang": lang,
               "lang_score": score, "n_chars_before": len(text)}
        langs = self.cfg.get("languages")
        if langs and (lang not in langs or score < self.cfg.get("lid_min_score", 0.0)):
            out.update(status="removed", reasons=["D01"], text="", n_chars_after=0, line_ops=None)
            return out
        if mode == "book":
            failed = book.apply_rules(text)
            out.update(status="removed" if failed else "kept", reasons=["B-" + f for f in failed],
                       text="" if failed else text, n_chars_after=0 if failed else len(text), line_ops=None)
            return out
        cleaned, ops = self.clean_lines(text, lang)
        reasons, s = self.doc_reasons(cleaned, lang, rec.get("url") or "")
        if self.cfg.get("pii_mask", True) and not reasons:
            cleaned, counts = book.mask_pii(cleaned)
            out["pii_masked"] = {k: v for k, v in counts.items() if v}
        out.update(status="removed" if reasons else "kept", reasons=reasons,
                   text="" if reasons else cleaned, n_chars_after=0 if reasons else len(cleaned),
                   line_ops=ops, signals={k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
        return out


def summarize(results):
    """统计每条规则的命中数（一篇文档可能同时命中多条）。"""
    by_rule = Counter()
    line_rule = Counter()
    for r in results:
        by_rule.update(r["reasons"])
        for ok, why in (r.get("line_ops") or []):
            if not ok:
                line_rule[why] += 1
    return by_rule, line_rule
