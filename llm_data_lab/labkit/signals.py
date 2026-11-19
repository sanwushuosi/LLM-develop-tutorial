"""逐文档质量信号。

把“这篇文档长什么样”变成一组可统计、可画分布、可设阈值的数字。
信号大多来自公开工作，阈值见 configs/clean.yaml：
  Gopher（Rae et al., 2021, 附录 A）：词数、平均词长、符号/词比、项目符号行、省略号行、字母词比例、停用词、
                                     重复行/段落、top-n-gram 与重复 n-gram 字符占比
  C4（Raffel et al., 2020）：行末标点、javascript / lorem ipsum / cookie 一类模板词
  FineWeb（Penedo et al., 2024）：以标点结尾的行占比、重复行字符占比、短行占比
中文没有空格分词：长度、n-gram 都按“字”计（与手册 4.2.3 节一致），停用词改用高频虚词。
"""
import re
from collections import Counter

from .book import PII_PATTERNS

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ]+")
END_PUNCT = tuple(".!?。！？…\"”'’」』)）:;：；")
BULLETS = ("•", "-", "*", "·", "»", "‣", "▪", "◦", "●", "–", "—", "→", "√")
EN_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}                 # Gopher 的 8 个停用词
ZH_FUNC = set("的了是在和有我不这也就人都一上中个们来到说为对以要会能可他她它着过与及而或被把从")
MOJIBAKE = re.compile(r"Ã[\x80-\xbf¡-¿]|â€|ï»¿|锟斤拷|�")


def is_zh(lang):
    return (lang or "").startswith("zh")


def units_of(text, lang):
    """长度与 n-gram 的计数单位：中文按字（去空白），其他语言按空格切词。"""
    if is_zh(lang):
        return [c for c in text if not c.isspace()]
    return text.split()


def _top_ngram_char_frac(units, n):
    if len(units) < n:
        return 0.0
    grams = Counter(tuple(units[i:i + n]) for i in range(len(units) - n + 1))
    g, c = grams.most_common(1)[0]
    if c < 2:
        return 0.0
    tot = sum(len(u) for u in units)
    return sum(len(u) for u in g) * c / max(1, tot)


def _dup_ngram_char_frac(units, n):
    """被重复 n-gram 覆盖的字符占比（Gopher 的 duplicate n-gram character fraction）。"""
    if len(units) < n:
        return 0.0
    seen, covered = {}, [False] * len(units)
    for i in range(len(units) - n + 1):
        g = tuple(units[i:i + n])
        if g in seen:
            for j in range(i, i + n):
                covered[j] = True
            j0 = seen[g]
            for j in range(j0, j0 + n):
                covered[j] = True
        else:
            seen[g] = i
    tot = sum(len(u) for u in units)
    return sum(len(u) for u, c in zip(units, covered) if c) / max(1, tot)


def compute(text, lang="en", max_units=20000, dup_n=(5, 10)):
    """返回一个 dict。所有比例都在 0~1 之间，便于画分布与设阈值。
    dup_n：计算“重复 n-gram 字符占比”用的两个 n。英文按词取 (5, 10)（Gopher 原值）；
    中文按字计，5 个字只相当于两三个词，常用短语天然会重复，应取更大的 n（默认配置用 (10, 20)）。"""
    lines = [l.strip() for l in text.splitlines()]
    nonempty = [l for l in lines if l]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = units_of(text, lang)
    u = units[:max_units]
    n_chars = len(text)
    nonspace = re.sub(r"\s", "", text)
    s = {}
    s["n_chars"] = n_chars
    s["n_lines"] = len(nonempty)
    s["n_units"] = len(units)                                    # 英文=词数，中文=字数
    s["cjk_ratio"] = len(CJK.findall(text)) / max(1, len(nonspace))
    words = [w for w in units if LATIN_WORD.search(w)] if not is_zh(lang) else []
    s["mean_word_len"] = (sum(len(w) for w in units) / len(units)) if units and not is_zh(lang) else None
    s["alpha_word_frac"] = (sum(1 for w in units if re.search(r"[^\W\d_]", w)) / len(units)) if units else 0.0
    s["symbol_ratio"] = sum(1 for c in nonspace if not c.isalnum() and not CJK.match(c)) / max(1, len(nonspace))
    s["hash_ellipsis_per_word"] = (text.count("#") + text.count("...") + text.count("…")) / max(1, len(units))
    s["digit_ratio"] = sum(c.isdigit() for c in nonspace) / max(1, len(nonspace))
    if nonempty:
        s["frac_lines_end_punct"] = sum(l.endswith(END_PUNCT) for l in nonempty) / len(nonempty)
        s["frac_bullet_lines"] = sum(l.startswith(BULLETS) for l in nonempty) / len(nonempty)
        s["frac_ellipsis_lines"] = sum(l.endswith(("...", "…", "。。。")) for l in nonempty) / len(nonempty)
        s["frac_short_lines"] = sum(len(l) <= (10 if is_zh(lang) else 30) for l in nonempty) / len(nonempty)
        cnt = Counter(nonempty)
        dup = [l for l in nonempty if cnt[l] > 1]
        s["dup_line_frac"] = 1 - len(cnt) / len(nonempty)
        s["dup_line_char_frac"] = sum(len(l) for l in dup) / max(1, sum(len(l) for l in nonempty))
    else:
        for k in ("frac_lines_end_punct", "frac_bullet_lines", "frac_ellipsis_lines", "frac_short_lines",
                  "dup_line_frac", "dup_line_char_frac"):
            s[k] = 0.0
    pc = Counter(paras)
    s["dup_para_frac"] = (1 - len(pc) / len(paras)) if paras else 0.0
    s["top2_char_frac"] = _top_ngram_char_frac(u, 2)
    s["top3_char_frac"] = _top_ngram_char_frac(u, 3)
    s["top4_char_frac"] = _top_ngram_char_frac(u, 4)
    s["dup_short_char_frac"] = _dup_ngram_char_frac(u, dup_n[0])
    s["dup_long_char_frac"] = _dup_ngram_char_frac(u, dup_n[1])
    if is_zh(lang):
        s["stop_hits"] = None
        s["zh_func_ratio"] = sum(1 for c in u if c in ZH_FUNC) / max(1, len(u))
    else:
        low = {w.lower().strip(".,;:!?\"'()") for w in u}
        s["stop_hits"] = len(EN_STOP & low)
        s["zh_func_ratio"] = None
    s["mojibake_hits"] = len(MOJIBAKE.findall(text))
    low_text = text.lower()
    s["has_lorem_ipsum"] = "lorem ipsum" in low_text
    s["has_curly"] = "{" in text
    for tag, pat in PII_PATTERNS.items():
        s["pii_" + tag.strip("<>").lower()] = len(pat.findall(text))
    return s
