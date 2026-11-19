#!/usr/bin/env python3
"""第 0 步：环境与数据自检。课前或上课第一件事运行：python 00_check.py"""
import importlib
import os
import platform
import sys
import time
from pathlib import Path

OK, BAD, WARN = "  ✔", "  ✘", "  !"
problems = 0


def check(cond, msg, fix="", warn=False):
    global problems
    if cond:
        print(f"{OK} {msg}")
    else:
        print(f"{WARN if warn else BAD} {msg}" + (f"\n      → {fix}" if fix else ""))
        if not warn:
            problems += 1


print(f"Python {sys.version.split()[0]}  {platform.system()} {platform.machine()}  CPU 核数 {os.cpu_count()}")
check(sys.version_info >= (3, 10), "Python ≥ 3.10", "请安装 Python 3.10 或更新版本")
for mod, pip in [("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("yaml", "pyyaml"),
                 ("matplotlib", "matplotlib"), ("trafilatura", "trafilatura"), ("fasttext", "fasttext-wheel"),
                 ("torch", "torch"), ("requests", "requests")]:
    try:
        m = importlib.import_module(mod)
        check(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception:                                                # noqa: BLE001
        check(False, f"缺少 {mod}", f"pip install {pip}")
for mod, pip, why in [("resiliparse", "resiliparse", "第 2 步对比提取器（可选）"), ("warcio", "warcio", "教师准备数据用（可选）")]:
    try:
        importlib.import_module(mod)
        check(True, f"{mod}（可选）")
    except Exception:                                                # noqa: BLE001
        check(False, f"缺少 {mod}（可选：{why}）", f"pip install {pip}", warn=True)

D = Path("data")
check((D / "cc_sample").exists() and any((D / "cc_sample").glob("part-*.jsonl.gz")),
      "data/cc_sample/ Common Crawl 样本", "从课程网盘下载数据包，解压到 llm_data_lab/data/")
check((D / "models" / "lid.176.ftz").exists(), "data/models/lid.176.ftz 语言识别模型",
      "下载 https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
check(any((D / "finepdfs").glob("*.parquet")), "data/finepdfs/*.parquet（第 7 步）", warn=True)
check((D / "extra" / "code.jsonl.gz").exists() and (D / "extra" / "math.jsonl.gz").exists(),
      "data/extra/ 代码与数学语料（第 8 步）", warn=True)
check(Path("work/02_extract").exists(), "work/02_extract 预提取结果（机器较慢时可直接用，跳过第 2 步）", warn=True)
check(Path("work/08_mix/runs.jsonl").exists(), "work/08_mix/runs.jsonl 教师预跑的配比蜂群（第 8 步）", warn=True)

try:
    from labkit import langid
    lang, sc = langid.top1("大语言模型的训练数据决定了它能学到什么。")
    check(lang == "zh", f"语言识别可用（示例判为 {lang} {sc}）")
except Exception as e:                                               # noqa: BLE001
    check(False, f"语言识别加载失败：{e}")
try:
    import torch
    from labkit.book_model import MiniGPT
    import torch.nn.functional as F
    torch.set_num_threads(2)
    m = MiniGPT(V=256, d=96, L=4, h=4, d_ff=256, max_T=128)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randint(0, 256, (32, 129))
    t0 = time.time()
    for _ in range(5):
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    ms = (time.time() - t0) / 5 * 1000
    check(True, f"小模型训练速度：{ms:.0f} ms/步（2 线程）→ 第 8 步一次 800 步训练约 {ms * 800 / 60000 + 0.3:.1f} 分钟")
except Exception as e:                                               # noqa: BLE001
    check(False, f"torch 训练测试失败：{e}")
if os.environ.get("QC_BASE_URL") and os.environ.get("QC_MODEL") and os.environ.get("QC_API_KEY"):
    check(True, f"质检接口环境变量已设置（模型 {os.environ['QC_MODEL']}）")
else:
    check(False, "未设置 QC_BASE_URL / QC_MODEL / QC_API_KEY（第 6 步 API 模式需要；用 Agent 模式可忽略）", warn=True)
print("\n全部必需项通过，可以开始。" if problems == 0 else f"\n有 {problems} 个必需项未通过，请按提示修复。")
