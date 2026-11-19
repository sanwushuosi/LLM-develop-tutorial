#!/usr/bin/env python3
"""第 7 步：文档数据（FinePDFs）与网页数据（Common Crawl）有什么不同？处理上要做哪些不同的事？

读取 data/finepdfs/*.parquet（FinePDFs 的英文 eng_Latn 与中文 cmn_Hani 小分片）和第 3 步的网页清洗结果，比较：
  1. 体量与结构：长度、页数、Markdown 表格 / 标题 / LaTeX 公式的出现率；
  2. 提取方式：Docling（直接读 PDF 文字层）与 RolmOCR（把页面当图片做 OCR）各占多少，两者的文本有何不同；
  3. PDF 特有的问题（用 page_ends 页边界定位）：
       · 页眉页脚：在多数页的开头或结尾反复出现的行；
       · 中文行内多余空格：PDF 按“行”排版，换行处被拼成空格（“亚 洲”）；
       · 跨页断句：上一页最后一句话在下一页开头才结束；
  4. 网页规则用在 PDF 上会怎样：直接套第 3 步的清洗配置，看哪些规则误伤；
  5. 一个最小的 PDF 专用修复（删页眉页脚、合并中文行内空格），修复前后对比。
输出：work/07_pdf/report.html、pdf_fixed.jsonl.gz（修复后的文本，可用于第 6 步质检与第 8 步配比）
用法：python 07_pdf_vs_web.py [--web work/03_clean/clean_text_trafilatura]
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from labkit import report as R
from labkit import signals
from labkit.clean import RULE_NAMES, Cleaner, load_config
from labkit.io import read_jsonl, write_jsonl

CJK_SPACE = re.compile(r"(?<=[一-鿿，。、；：？！“”（）])[ 　]+(?=[一-鿿，。、；：？！“”（）])")
HEADING = re.compile(r"^#{1,6} ", re.M)
TABLE = re.compile(r"^\|.*\|\s*$", re.M)
LATEX = re.compile(r"\$[^$\n]{1,120}\$|\\\(|\\\[|\\frac|\\sum")


def pages_of(text, ends):
    out, s = [], 0
    for e in ends:
        out.append(text[s:e])
        s = e
    if s < len(text):
        out.append(text[s:])
    return out


def norm_hf(l):
    return re.sub(r"\d+", "0", re.sub(r"\s+", " ", l.strip().lower()))


def header_footer_lines(pages, k=2, min_frac=0.5):
    """统计每页前 k 行与后 k 行，在 ≥ min_frac 的页里重复出现（数字归一化后）的行视为页眉页脚。"""
    if len(pages) < 4:
        return set()
    c = Counter()
    for p in pages:
        ls = [l for l in p.split("\n") if l.strip()]
        for l in set(ls[:k] + ls[-k:]):
            c[norm_hf(l)] += 1
    return {l for l, v in c.items() if v >= max(3, min_frac * len(pages)) and len(l) <= 120}


def fix_pdf(text, ends, lang):
    """最小的 PDF 专用修复：删页眉页脚、合并中文行内空格、跨页断句拼回。返回 (新文本, 统计)。"""
    pages = pages_of(text, ends)
    hf = header_footer_lines(pages)
    st = Counter()
    new_pages = []
    for p in pages:
        keep = []
        for l in p.split("\n"):
            if l.strip() and norm_hf(l) in hf:
                st["页眉页脚行"] += 1
                continue
            if re.fullmatch(r"\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*|\s*[０-９]{1,4}\s*", l) and l.strip():
                st["孤立页码行"] += 1
                continue
            keep.append(l)
        new_pages.append("\n".join(keep).strip("\n"))
    out = []
    for i, p in enumerate(new_pages):
        if out and p:
            prev = out[-1].rstrip()
            # 跨页断句：上一页最后一行是“正文长行”且没有句末标点，下一页以小写字母（英文）或汉字（中文）开头。
            # 只拼正文长行，避免把标题、表格、页码拼进下一段。
            last_line = prev.split("\n")[-1].strip()
            first = p.lstrip()[:1]
            zh = lang.startswith("zh")
            long_enough = len(last_line) >= (15 if zh else 40) and not last_line.startswith(("|", "#"))
            cont = first.islower() if not zh else bool(re.match(r"[一-鿿]", first))
            if long_enough and last_line[-1] not in ".!?。！？：:;；”\"|)" and cont:
                out[-1] = prev + ("" if zh else " ") + p.lstrip()
                st["跨页断句拼接"] += 1
                continue
        out.append(p)
    text2 = "\n\n".join(x for x in out if x.strip())
    if lang.startswith("zh"):
        text2, n = CJK_SPACE.subn("", text2)
        st["中文行内多余空格"] += n
    return text2, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/finepdfs")
    ap.add_argument("--web", default="work/03_clean/clean_text_trafilatura")
    ap.add_argument("--config", default="configs/clean.yaml")
    ap.add_argument("--out", default="work/07_pdf")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in sorted(Path(a.pdf_dir).glob("*.parquet")):
        df = pd.read_parquet(f, columns=["id", "url", "text", "language", "extractor", "page_ends", "is_truncated",
                                         "minhash_cluster_size", "duplicate_count"])
        frames.append(df)
    pdf = pd.concat(frames, ignore_index=True)
    pdf["lang"] = pdf["language"].map(lambda x: "zh" if x.startswith("cmn") or x.startswith("zho") else "en")
    pdf["n_pages"] = pdf["page_ends"].map(len)
    pdf["chars"] = pdf["text"].str.len()
    web = [r for r in read_jsonl(a.web, fields=["id", "url", "lang", "status", "text"]) if r["status"] == "kept"]

    # 1. 体量与结构
    rows = []
    for name, texts, lang in [("网页·en", [w["text"] for w in web if w["lang"] == "en"], "en"),
                              ("网页·zh", [w["text"] for w in web if w["lang"] == "zh"], "zh"),
                              ("PDF·en", list(pdf[pdf.lang == "en"].text), "en"),
                              ("PDF·zh", list(pdf[pdf.lang == "zh"].text), "zh")]:
        L = sorted(len(t) for t in texts)
        rows.append((name, len(texts), L[len(L) // 2], f"{sum(L) / 1e6:.1f}M",
                     f"{sum(bool(TABLE.search(t)) for t in texts) / len(texts):.0%}",
                     f"{sum(bool(HEADING.search(t)) for t in texts) / len(texts):.0%}",
                     f"{sum(bool(LATEX.search(t)) for t in texts) / len(texts):.0%}"))
    S = [R.cards([("PDF 文档", f"{len(pdf):,}"), ("PDF 中位页数", int(pdf.n_pages.median())),
                  ("PDF 中位字符", f"{int(pdf.chars.median()):,}"),
                  ("网页（清洗后）中位字符", f"{sorted(len(w['text']) for w in web)[len(web) // 2]:,}"),
                  ("走 OCR（RolmOCR）的 PDF", f"{(pdf.extractor == 'rolmOCR').mean():.0%}")])]
    S.append("<h2>1. 体量与结构</h2>")
    S.append(R.table(rows, ["数据", "篇数", "中位字符数", "总字符", "含 Markdown 表格", "含 # 标题", "含 LaTeX 公式"], num_cols=(1, 2)))
    S.append("<div class='grid2'>")
    S.append(R.hist({"网页（清洗后）": [len(w["text"]) for w in web], "PDF": list(pdf.chars)}, "每篇字符数", "字符数（对数坐标）", log_x=True))
    S.append(R.hist({"PDF 页数": list(pdf.n_pages)}, "每份 PDF 的页数", "页数（对数坐标）", log_x=True))
    S.append("</div>")
    S.append("<div class='note'>PDF 中位长度是网页的数倍到十几倍，并且带着表格、标题层级与公式：它是长文本与专业知识的主要来源，"
             "但处理难度在“版面”而不在“去模板”。</div>")

    # 2. 提取方式
    S.append("<h2>2. 提取方式：读文字层（Docling）还是 OCR（RolmOCR）</h2>")
    t = pdf.groupby(["lang", "extractor"], observed=True).agg(篇数=("id", "count"), 中位页数=("n_pages", "median"),
                                                              中位字符=("chars", "median"), 被截断=("is_truncated", "mean")).reset_index()
    S.append(R.table([(r.lang, r.extractor, r.篇数, int(r.中位页数), int(r.中位字符), f"{r.被截断:.0%}") for r in t.itertuples()],
                     ["语言", "提取器", "篇数", "中位页数", "中位字符", "原文件被 CC 截断"], num_cols=(2, 3, 4)))
    S.append("<div class='note'>FinePDFs 用一个 XGBoost 路由器决定每份 PDF 走哪条路：有可靠文字层的走便宜的 Docling（CPU），"
             "扫描件、文字层损坏或被截断后重新抓取的走 RolmOCR（GPU，贵得多）。路由本身就是一个质量—成本权衡。</div>")

    # 3. PDF 特有问题
    hf_docs = hf_lines_total = 0
    cjk_docs = cjk_spaces = 0
    cross = 0
    fixed_rows, stats_all = [], Counter()
    examples = []
    for r in pdf.itertuples():
        ends = list(r.page_ends)
        pages = pages_of(r.text, ends)
        hf = header_footer_lines(pages)
        if hf:
            hf_docs += 1
        if r.lang == "zh":
            n = len(CJK_SPACE.findall(r.text))
            if n >= 5:
                cjk_docs += 1
            cjk_spaces += n
        new, st = fix_pdf(r.text, ends, r.lang)
        stats_all.update(st)
        if st.get("页眉页脚行", 0) + st.get("中文行内多余空格", 0) >= 8 and len(examples) < 6:
            examples.append((r, new, st, hf))
        fixed_rows.append({"id": r.id, "url": r.url, "lang": r.lang, "extractor": r.extractor, "n_pages": r.n_pages,
                           "text_raw": r.text, "text": new, "fix_stats": dict(st)})
    write_jsonl(out / "pdf_fixed.jsonl.gz", fixed_rows)
    S.append("<h2>3. PDF 特有的问题（用 page_ends 页边界定位）</h2>")
    S.append(R.table([("检测到重复页眉/页脚的文档（≥4 页）", f"{hf_docs:,}", f"{hf_docs / len(pdf):.0%}"),
                      ("中文：行内多余空格 ≥5 处的文档", f"{cjk_docs:,}", f"{cjk_docs / max(1, (pdf.lang == 'zh').sum()):.0%}（占中文 PDF）"),
                      ("修复：删除的页眉页脚行", f"{stats_all['页眉页脚行']:,}", ""),
                      ("修复：删除的孤立页码行", f"{stats_all['孤立页码行']:,}", ""),
                      ("修复：拼回的跨页断句", f"{stats_all['跨页断句拼接']:,}", ""),
                      ("修复：合并的中文行内空格", f"{stats_all['中文行内多余空格']:,}", "")],
                     ["项目", "数量", "占比"], num_cols=(1,)))
    S.append("<h3>修复前后示例（红=删除，绿=修复后）</h3>")
    import difflib
    for r, new, st, hf in examples:
        raw = r.text[:2500]
        sm = difflib.SequenceMatcher(a=raw, b=new[:2500], autojunk=False)
        a_html, b_html = [], []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                a_html.append(R.esc(raw[i1:i2]))
                b_html.append(R.esc(new[j1:j2]))
            else:
                if i2 > i1:
                    a_html.append(f"<span style='background:#ffd6d6'>{R.esc(raw[i1:i2]) or '␣'}</span>")
                if j2 > j1:
                    b_html.append(f"<span style='background:#d4f5dc'>{R.esc(new[j1:j2])}</span>")
        S.append(f"<p><b>[{r.lang} · {r.extractor} · {r.n_pages} 页]</b> {R.esc(r.url)}<br>修复统计：{R.esc(dict(st))}；"
                 f"识别出的页眉页脚：{R.esc(list(hf)[:4])}</p>"
                 "<div class='grid2'>"
                 f"<div style='white-space:pre-wrap;font-size:12px;border:1px solid #ddd;padding:6px;max-height:380px;overflow:auto'>{''.join(a_html)}</div>"
                 f"<div style='white-space:pre-wrap;font-size:12px;border:1px solid #ddd;padding:6px;max-height:380px;overflow:auto'>{''.join(b_html)}</div></div>")

    # 4. 网页规则用在 PDF 上
    cfg = load_config(a.config)
    cl = Cleaner(cfg)
    hit = {"en": Counter(), "zh": Counter()}
    kept = Counter()
    tot = Counter()
    for r in fixed_rows:
        res = cl({"id": r["id"], "url": r["url"], "text_trafilatura": r["text"]})
        lg = r["lang"]
        tot[lg] += 1
        if res["reasons"] == ["D01"]:
            hit[lg]["D01"] += 1
            continue
        hit[lg].update(res["reasons"])
        kept[lg] += res["status"] == "kept"
    S.append("<h2>4. 直接套用网页清洗规则会怎样？</h2>")
    rows = []
    for rid in sorted(set(hit["en"]) | set(hit["zh"]), key=lambda k: -(hit["en"][k] + hit["zh"][k])):
        rows.append((rid, RULE_NAMES.get(rid, ""), f"{hit['en'][rid] / max(1, tot['en']):.0%}", f"{hit['zh'][rid] / max(1, tot['zh']):.0%}"))
    S.append(R.table(rows, ["规则", "含义", "PDF·en 命中", "PDF·zh 命中"]))
    S.append(f"<p>保留率：PDF·en {kept['en'] / max(1, tot['en']):.0%}，PDF·zh {kept['zh'] / max(1, tot['zh']):.0%}。"
             "对照网页的保留率（第 3 步）思考：被删掉的 PDF 真的是坏数据吗？用 04_diff 的思路抽几篇看看。</p>")
    S.append("<div class='note'>典型误伤：表格、目录、参考文献、公式让“以标点结尾的行”“短行比例”“符号比例”这类为网页设计的规则大量命中；"
             "OCR 文本的断行又让行级规则失真。文档数据需要按“页 / 版面块”而不是按“行”来清洗，并且阈值要单独校准。</div>")
    R.page("第 7 步：文档数据（FinePDFs）vs 网页数据（Common Crawl）", S, out / "report.html")
    print(f"PDF {len(pdf):,} 篇（en {(pdf.lang == 'en').sum()}，zh {(pdf.lang == 'zh').sum()}）；"
          f"OCR 占 {(pdf.extractor == 'rolmOCR').mean():.0%}；中位 {int(pdf.n_pages.median())} 页、{int(pdf.chars.median()):,} 字符")
    print(f"页眉页脚文档 {hf_docs}；中文行内空格文档 {cjk_docs}；修复统计 {dict(stats_all)}")
    print("网页规则用在 PDF 上的保留率：", {k: f"{kept[k] / max(1, tot[k]):.0%}" for k in tot})
    print("  命中最多的规则 en:", hit["en"].most_common(5), " zh:", hit["zh"].most_common(5))
    print(f"报告：{out / 'report.html'}")


if __name__ == "__main__":
    main()
