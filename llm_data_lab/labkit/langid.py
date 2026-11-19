"""fastText 语言识别（lid.176.ftz，约 0.9 MB，覆盖 176 种语言）。

注意：fasttext 官方包的 predict() 在 numpy 2.x 下会报
"Unable to avoid copy while creating an array"，这里改用底层接口 model.f.predict 绕开。
"""
from pathlib import Path

_MODEL = None
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "lid.176.ftz"


def load(path=None):
    global _MODEL
    if _MODEL is None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None          # 屏蔽加载时的警告
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"找不到语言识别模型 {p}。请下载 https://dl.fbaipublicfiles.com/fasttext/"
                "supervised-models/lid.176.ftz 放到 data/models/ 下")
        _MODEL = fasttext.load_model(str(p))
    return _MODEL


def predict(text, k=1, max_chars=5000):
    """返回 [(语言, 置信度), ...]。语言为 ISO 639-1 代码，如 en、zh、ja。"""
    m = load()
    s = " ".join(text[:max_chars].split())
    if not s:
        return [("unknown", 0.0)]
    pairs = m.f.predict(s, k, 0.0, "replace")
    if not pairs:                                   # 极少数只含特殊字符的文本，模型不给任何标签
        return [("unknown", 0.0)]
    return [(lab.replace("__label__", ""), float(p)) for p, lab in pairs]


def top1(text):
    lang, score = predict(text, 1)[0]
    return lang, round(score, 4)
