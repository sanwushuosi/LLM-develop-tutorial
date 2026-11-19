"""《现代大模型研发学习手册》第 2 章的 MiniGPT（原样代码，只删去演示与断言）。

配比实验与去重实验用它在 CPU 上训练几十万参数的字节级小模型。
"""
import torch
import torch.nn.functional as F


# ────── 手册 2.3 因果自注意力 ──────────────────────────────────────────────────
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def split(self, t):                       # [B, T, d] -> [B, h, T, dh]
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x):                     # x: [B, T, d]
        B, T, d = x.shape
        q, k, v = self.split(self.wq(x)), self.split(self.wk(x)), self.split(self.wv(x))
        scores = q @ k.transpose(-2, -1) / self.dh ** 0.5          # [B, h, T, T]
        future = torch.ones(T, T, dtype=torch.bool).triu(1)        # j > i 的位置
        scores = scores.masked_fill(future, float("-inf"))         # 屏蔽未来
        probs = scores.softmax(dim=-1)                             # 每行之和为 1
        out = (probs @ v).transpose(1, 2).reshape(B, T, d)         # 合并各头
        return self.wo(out)


# ────── 手册 2.3.4 RMSNorm 与 SwiGLU ──────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # 在 FP32 中计算均方根，再转回输入的数据类型
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)   # 激活分支
        self.w3 = nn.Linear(d, d_ff, bias=False)   # 门控分支
        self.w2 = nn.Linear(d_ff, d, bias=False)   # 降维

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ────── 手册 2.4.3 RoPE ──────────────────────────────────────────────────
def rope_angles(T, dh, base=10000.0):
    # θ_i = base^(-2i/dh)；角度必须在 FP32 中计算
    inv_freq = base ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)  # [dh/2]
    pos = torch.arange(T, dtype=torch.float32)                              # [T]
    ang = torch.outer(pos, inv_freq)                                        # [T, dh/2]
    return torch.cat([ang, ang], dim=-1)                                    # [T, dh]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)          # 第 i 维与第 i + dh/2 维配对
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, ang):                  # x: [..., T, dh]，ang: [T, dh]
    return x * ang.cos() + rotate_half(x) * ang.sin()


# ────── 手册 2.6.1 MiniGPT ───────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d, h, d_ff, use_rope=True):
        super().__init__()
        self.attn = CausalSelfAttention(d, h)
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.use_rope = use_rope

    def attention(self, x, ang):
        a = self.attn
        B, T, d = x.shape
        q, k, v = a.split(a.wq(x)), a.split(a.wk(x)), a.split(a.wv(x))
        if self.use_rope:                       # 只旋转查询和键，不旋转值
            q, k = apply_rope(q, ang[:T]), apply_rope(k, ang[:T])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return a.wo(out.transpose(1, 2).reshape(B, T, d))

    def forward(self, x, ang):
        x = x + self.attention(self.norm1(x), ang)
        return x + self.ffn(self.norm2(x))

class MiniGPT(nn.Module):
    def __init__(self, V, d=64, L=2, h=4, d_ff=172, max_T=256, use_rope=True):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, h, d_ff, use_rope) for _ in range(L))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight      # 嵌入与输出投影共享权重
        self.register_buffer("ang", rope_angles(max_T, d // h), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):                     # ids: [B, T] -> logits: [B, T, V]
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x, self.ang)
        return self.head(self.norm(x))

