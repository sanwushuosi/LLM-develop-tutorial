"""HTML → 正文。

两种提取器：
  trafilatura   FineWeb / RefinedWeb 使用的启发式正文提取器，较准但较慢（实测约 40 ms/页）
  resiliparse   DCLM / Essential-Web 使用的 HTML→文本工具，很快（实测约 3 ms/页），保留更多导航文字
另有 CC 官方 WET 文本（数据里的 wet_text 字段），不区分正文与导航，可作为“不做正文提取”的对照。

安全提示：网页来自互联网，这里只把 HTML 当作字符串解析，不执行其中的脚本，也不要用浏览器打开原始 HTML。
"""
import time
from concurrent.futures import ProcessPoolExecutor

MAX_HTML_CHARS = 3_000_000          # 极端超长页面截断，避免个别页面拖慢整批


def extract_trafilatura(html):
    import trafilatura
    return trafilatura.extract(html[:MAX_HTML_CHARS], include_comments=False, include_tables=True,
                               deduplicate=False, favor_precision=False) or ""


def extract_resiliparse(html):
    from resiliparse.extract.html2text import extract_plain_text
    return extract_plain_text(html[:MAX_HTML_CHARS], main_content=True, alt_texts=False) or ""


EXTRACTORS = {"trafilatura": extract_trafilatura, "resiliparse": extract_resiliparse}


def _one(args):
    rec, methods = args
    out = {k: rec.get(k) for k in ("id", "url", "host", "date", "cc_lang", "html_bytes", "truncated")}
    out["wet_text"] = rec.get("wet_text") or ""
    for m in methods:
        t0 = time.time()
        try:
            out[f"text_{m}"] = EXTRACTORS[m](rec["html"])
        except Exception as e:                                       # noqa: BLE001
            out[f"text_{m}"] = ""
            out[f"error_{m}"] = type(e).__name__
        out[f"ms_{m}"] = round((time.time() - t0) * 1000, 1)
    return out


def extract_many(records, methods=("trafilatura",), workers=4, chunksize=16):
    """并行提取。records 是可迭代的原始记录（含 html 字段）。"""
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(_one, ((r, methods) for r in records), chunksize=chunksize)
