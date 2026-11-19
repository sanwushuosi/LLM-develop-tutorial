"""llm_data_lab 实验工具包。

  book.py        手册第 4 章原样代码（规则过滤、PII、MinHash、污染检测、温度采样、配比采样器）
  book_model.py  手册第 2 章原样代码（MiniGPT）
  io.py          读写 jsonl.gz
  extract.py     HTML → 正文（trafilatura / resiliparse），与 CC 官方 WET 对照
  langid.py      fastText 语言识别
  signals.py     逐文档质量信号（Gopher / C4 / FineWeb 一类的统计量）
  clean.py       可配置的清洗流水线（行级 + 文档级），记录每一行被删的原因
  dedup.py       精确去重、MinHash 近似去重、跨文档模板行、同站模板页
  report.py      清洗前后的统计看板（HTML）
  diffview.py    清洗前后逐条对比（HTML）
  qc.py          Agent / LLM 质检：抽样、拼 prompt、调用、汇总
  mixture.py     配比实验：数据池、训练、按领域评测、回归拟合
"""
