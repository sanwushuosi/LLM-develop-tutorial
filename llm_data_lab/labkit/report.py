"""生成自包含的 HTML 报告：图用 matplotlib 画成 PNG 内嵌（base64），不依赖任何外部资源，可直接双击打开。"""
import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
                                   "Noto Sans SC", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

COLORS = ["#1f6fb5", "#e67e22", "#2e8b57", "#8e44ad", "#c0392b", "#7f8c8d", "#16a085", "#d4ac0d"]

CSS = """
body{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif;margin:24px auto;max-width:1180px;color:#222;line-height:1.55}
h1{color:#0b5aa6;border-bottom:3px solid #0b5aa6;padding-bottom:6px}
h2{color:#0b5aa6;margin-top:34px;border-left:6px solid #0b5aa6;padding-left:10px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.card{background:#f3f7fc;border:1px solid #cddcee;border-radius:8px;padding:10px 16px;min-width:150px}
.card b{display:block;font-size:22px;color:#0b5aa6}
.card span{font-size:13px;color:#555}
table{border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{border:1px solid #ccd6e0;padding:4px 9px;text-align:left;vertical-align:top}
th{background:#e8f0fa}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:#fff7e6;border-left:4px solid #e67e22;padding:8px 12px;margin:10px 0;font-size:14px}
img{max-width:100%}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
code{background:#f2f2f2;padding:1px 4px;border-radius:3px}
"""


def esc(s):
    return html.escape(str(s), quote=True)


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def cards(items):
    return '<div class="cards">' + "".join(
        f'<div class="card"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in items) + "</div>"


def table(rows, header, num_cols=()):
    h = "".join(f"<th>{esc(c)}</th>" for c in header)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="num">{esc(c)}</td>' if i in num_cols else f"<td>{esc(c)}</td>"
            for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def hist(series_dict, title, xlabel, bins=40, log_x=False, range_=None):
    """series_dict: {名字: 数值列表}，多组叠画，便于“清洗前 vs 清洗后”对比。
    纵轴是“落在该区间的文档占本组的比例”，各组文档数不同也能直接比较；对数横轴下也不会失真。"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    allv = [v for vs in series_dict.values() for v in vs if v is not None]
    if not allv:
        plt.close(fig)
        return ""
    if log_x:
        allv = [max(1, v) for v in allv]
        edges = np.logspace(0, np.log10(max(allv) + 1), bins)
    else:
        edges = np.linspace(*(range_ or (min(allv), max(allv) + 1e-9)), bins)
    for i, (name, vs) in enumerate(series_dict.items()):
        vs = [max(1, v) if log_x else v for v in vs if v is not None]
        if vs:
            ax.hist(vs, bins=edges, alpha=0.55, weights=[1 / len(vs)] * len(vs), label=f"{name}（n={len(vs):,}）",
                    color=COLORS[i % len(COLORS)])
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("文档占比")
    ax.legend(fontsize=8)
    return fig_to_img(fig)


def barh(pairs, title, xlabel="文档数", color=None):
    pairs = list(pairs)[::-1]
    if not pairs:
        return ""
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(pairs) + 1.0))
    ax.barh([str(k)[:40] for k, _ in pairs], [v for _, v in pairs], color=color or COLORS[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for i, (_, v) in enumerate(pairs):
        ax.text(v, i, f" {v:,}" if isinstance(v, int) else f" {v:.1%}", va="center", fontsize=8)
    return fig_to_img(fig)


def page(title, sections, out_path):
    body = f"<h1>{esc(title)}</h1>" + "".join(sections)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(f"<!doctype html><html lang='zh'><meta charset='utf-8'><title>{esc(title)}</title>"
                              f"<style>{CSS}</style><body>{body}</body></html>", "utf-8")
    return out_path


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), "utf-8")
