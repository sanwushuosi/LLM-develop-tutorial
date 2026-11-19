"""你自己的清洗规则（实验第 6 步：根据 Agent 质检的发现补规则）。

两种规则，写好后在 configs/*.yaml 的 custom_rules 里列出要启用的编号即可，不用改 clean.py：
  文档级：fn(text, lang, url, signals) -> bool   返回 True 表示这篇文档应当丢弃
  行级：  fn(line, lang) -> bool                  返回 True 表示删掉这一行
signals 是 labkit.signals.compute() 算好的质量信号（dict），可以直接用。

下面的 X01、X02 是写好的例子：
  X01 来自第 5 步的重复簇：CMS 默认页、域名出售页、“站点不存在”报错页在几十个不相关站点上反复出现；
  X02 来自第 6 步第一轮质检：不合格样本大多是“正文本来就少，剩下的是导航、表单、站点通用文字”。
X03、X04 留给你写。
"""
import re

# ── X01：模板默认页 / 占位页 / 主机报错页（例子）──────────────────────────────
X01_PATTERNS = re.compile("|".join([
    r"this is an example page", r"hello world!\s*welcome to wordpress", r"sample page",
    r"the domain name .{0,40} is for sale", r"this domain is for sale", r"buy this domain",
    r"website coming soon", r"under construction", r"site is currently unavailable",
    r"没有找到站点", r"网站建设中", r"该域名正在出售", r"网站暂时无法访问",
    r"please wait while your request is being verified", r"verify you are human",
]), re.I)


def x01_template_page(text, lang, url, s):
    """短文档里命中模板短语即丢弃；长文档里偶尔提到这些短语不算。"""
    return s["n_units"] < (600 if lang == "zh" else 300) and bool(X01_PATTERNS.search(text[:3000]))


# ── X02：删完模板之后，完整的句子太少（例子，来自第一轮质检：W03 / W05 占不合格样本的大半）─────────
SENT_END = re.compile(r"[.!?。！？](?:[\"'”’」』)]*)(?=\s|$)")


def complete_sentences(text, lang):
    """数“像样的句子”：以句末标点结束、且不太短（英文 ≥ 8 个词，中文 ≥ 15 个字）。"""
    n = 0
    for line in text.split("\n"):
        # 英文句末标点后面跟空格才算断句；中文句号后面通常没有空格，直接断开
        for seg in re.split(r"(?<=[。！？])|(?<=[.!?])\s+", line.strip()):
            if not SENT_END.search(seg[-3:] + " "):
                continue
            if (lang == "zh" and len(re.findall(r"[一-鿿]", seg)) >= 15) or (lang != "zh" and len(seg.split()) >= 8):
                n += 1
    return n


def x02_few_sentences(text, lang, url, s):
    return complete_sentences(text, lang) < 2


# ── X03：你的行级规则（例如：以“阅读全文 / Read more / 点击查看”结尾的摘要行）──────────────
def x03_todo(line, lang):
    return False


# ── X04：你的规则（例如：中文 SEO 站群——同一前缀、没有句末标点的短行反复出现）──────────────
def x04_todo(text, lang, url, s):
    return False


DOC_RULES = {"X01": x01_template_page, "X02": x02_few_sentences, "X04": x04_todo}
LINE_RULES = {"X03": x03_todo}
NAMES = {"X01": "模板默认页 / 占位页 / 主机报错页", "X02": "完整句子少于 2 句", "X03": "行：（你的规则）",
         "X04": "（你的规则）"}
