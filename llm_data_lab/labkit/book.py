"""《现代大模型研发学习手册》第 4 章的原样代码（只删去了书中演示用的示例数据与 print）。

实验课里的各个脚本都从这里导入这些函数，你可以对照手册相应小节阅读、复制、修改。
每一段前面标注了出处小节；为了便于对照，函数体一字未改。
需要改进的地方（阈值、中文处理、更多规则）请写在 labkit/clean.py 或你自己的文件里，
不要直接改这个文件——保留原样，才能和改进版做对照。
"""
from collections import Counter

import numpy as np


# ────── 手册 4.2.3 规则过滤 ──────────────────────────────────────────────────
import re
from collections import Counter

def is_cjk_text(text):
    cjk = len(re.findall(r"[一-鿿]", text))
    return cjk > 0.3 * max(1, len(re.sub(r"\s", "", text)))

def doc_units(text):
    """英文按词切分；中文按字符切分（去掉空白）。"""
    if is_cjk_text(text):
        return [c for c in text if not c.isspace()]
    return text.split()

def rule_length(text):
    n = len(doc_units(text))
    return 50 <= n <= 100_000

def rule_symbols(text):
    s = re.sub(r"\s", "", text)
    if not s:
        return False
    symbol = sum(1 for c in s if not c.isalnum() and not "一" <= c <= "鿿")
    return symbol / len(s) < 0.25

def rule_lines(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    bullet = sum(l.startswith(("•", "-", "*", "·", "»")) for l in lines) / len(lines)
    ends = sum(l[-1] in ".!?。！？…\"”'" for l in lines) / len(lines)
    return bullet < 0.9 and ends > 0.2

def rule_repetition(text, n=3):
    units = doc_units(text)
    grams = Counter(tuple(units[i:i + n]) for i in range(len(units) - n + 1))
    if not grams:
        return False
    top = grams.most_common(1)[0][1] * n
    lines = [l for l in text.splitlines() if l.strip()]
    dup_lines = 1 - len(set(lines)) / max(1, len(lines))
    return top / max(1, len(units)) < 0.2 and dup_lines < 0.3

RULES = {"长度": rule_length, "符号比例": rule_symbols,
         "行级特征": rule_lines, "重复程度": rule_repetition}

def apply_rules(text):
    """返回未通过的规则名列表；空列表表示通过全部规则。"""
    return [name for name, rule in RULES.items() if not rule(text)]


# ────── 手册 4.2.5 隐私处理 ──────────────────────────────────────────────────
PII_PATTERNS = {
    "<EMAIL>": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "<PHONE>": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),            # 中国大陆手机号
    "<ID>": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),               # 18 位身份证号
    "<IP>": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
}

def mask_pii(text):
    counts = {}
    for tag, pat in PII_PATTERNS.items():
        text, n = pat.subn(tag, text)
        counts[tag] = n
    return text, counts


# ────── 手册 4.3.3 近似去重：MinHash 与局部敏感哈希 ──────────────────────────────────
import hashlib
import numpy as np

def shingles(text, n=5):
    """中文用字符 n-gram，英文用词 n-gram。"""
    units = [c for c in text if not c.isspace()] if is_cjk_text(text) else text.lower().split()
    return {" ".join(units[i:i + n]) for i in range(max(1, len(units) - n + 1))}

PRIME = (1 << 31) - 1                                # 31 位梅森素数
_rng = np.random.default_rng(0)
K = 128                                              # 签名长度
HASH_A = _rng.integers(1, PRIME, K, dtype=np.uint64)
HASH_B = _rng.integers(0, PRIME, K, dtype=np.uint64)

def minhash(sh):
    """先给每个 shingle 一个基础哈希，再用 K 个随机线性函数 (a·x+b) mod p 取最小值。"""
    base = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                                    "little") % PRIME for s in sh], dtype=np.uint64)
    # a、x 都小于 2^31，乘积小于 2^62，不会溢出 64 位无符号整数
    vals = (HASH_A[:, None] * base[None, :] + HASH_B[:, None]) % PRIME   # [K, |sh|]
    return vals.min(axis=1)

def lsh_candidates(sigs, bands=16):
    """把签名分成 bands 段，任一段相同即成为候选对。"""
    rows = sigs.shape[1] // bands
    buckets, pairs = {}, set()
    for doc_id, sig in enumerate(sigs):
        for b in range(bands):
            key = (b, sig[b * rows:(b + 1) * rows].tobytes())
            for other in buckets.get(key, []):
                pairs.add((other, doc_id))
            buckets.setdefault(key, []).append(doc_id)
    return pairs


# ────── 手册 4.3.5 污染控制 ──────────────────────────────────────────────────
def ngram_set(text, n=13):
    units = [c for c in text if not c.isspace()] if is_cjk_text(text) else text.lower().split()
    return {tuple(units[i:i + n]) for i in range(len(units) - n + 1)}

def contaminated(train_doc, eval_items, n=13):
    """返回与训练文档共享至少一个 n-gram 的评测条目编号。"""
    grams = ngram_set(train_doc, n)
    return [i for i, item in enumerate(eval_items) if grams & ngram_set(item, n)]


# ────── 手册 4.4.2 多语言与中文（温度采样） ──────────────────────────────────────────
def temperature_weights(sizes, alpha):
    w = {k: v ** alpha for k, v in sizes.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


# ────── 手册 4.4.4 数据量与质量的取舍（重复轮数） ───────────────────────────────────────
def epochs_per_source(sizes_b, weights, budget_b):
    """sizes_b：各来源去重后的 token 数（十亿）；weights：目标配比；budget_b：训练总 token 数。"""
    return {k: weights[k] * budget_b / sizes_b[k] for k in sizes_b}


# ────── 手册 4.6.3 按配比采样 ─────────────────────────────────────────────────
class MixtureSampler:
    """按权重从多个来源中确定性地抽取文档；状态可以保存与恢复。"""

    def __init__(self, sources, weights, seed=0):
        self.sources = sources                       # {来源名: 文档列表}
        self.names = list(sources)
        w = np.array([weights[n] for n in self.names], dtype=float)
        self.p = w / w.sum()
        self.seed = seed
        self.step = 0
        self.cursor = {n: 0 for n in self.names}     # 本轮读到的位置
        self.epoch = {n: 0 for n in self.names}      # 已完成的轮数

    def _order(self, name, epoch):
        # 每个来源、每一轮的打乱顺序只由（种子，来源，轮次）决定
        rng = np.random.default_rng([self.seed, self.names.index(name), epoch])
        return rng.permutation(len(self.sources[name]))

    def next(self):
        # 第几步选哪个来源，只由（种子，步数）决定，与是否中断过无关
        rng = np.random.default_rng([self.seed, 1_000_003, self.step])
        name = self.names[rng.choice(len(self.names), p=self.p)]
        order = self._order(name, self.epoch[name])
        doc = self.sources[name][order[self.cursor[name]]]
        self.cursor[name] += 1
        if self.cursor[name] == len(order):          # 一轮读完，进入下一轮
            self.cursor[name], self.epoch[name] = 0, self.epoch[name] + 1
        self.step += 1
        return name, doc

    def state_dict(self):
        return {"step": self.step, "cursor": dict(self.cursor), "epoch": dict(self.epoch)}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.cursor, self.epoch = dict(state["cursor"]), dict(state["epoch"])

