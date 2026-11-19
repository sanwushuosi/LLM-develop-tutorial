#!/usr/bin/env python3
"""第 6 步（2/3，API 模式）：用任意 OpenAI 兼容接口逐条质检。

先在终端里设置环境变量（不要把密钥写进代码或文件，也不要贴到聊天里）：
  export QC_BASE_URL="https://<课程提供的接口地址>/v1"     # 例如课程提供的混元 hy4 接口，或你自己的服务
  export QC_MODEL="<模型名>"
  export QC_API_KEY="<你的密钥>"
然后：
  python 06_qc_run.py --pack work/06_qc/<名字> [--workers 4]
已经判过的样本会跳过，中断后重跑即可续上。结果写入 <pack>/results.jsonl。
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from labkit.io import read_jsonl
from labkit.qc import call_openai_compatible, fill, parse_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    base, model, key = os.environ.get("QC_BASE_URL"), os.environ.get("QC_MODEL"), os.environ.get("QC_API_KEY")
    if not (base and model and key):
        sys.exit("请先设置环境变量 QC_BASE_URL、QC_MODEL、QC_API_KEY（见文件开头说明）")
    pack = Path(a.pack)
    rubric = (pack / "rubric.prompt").read_text("utf-8")
    res_path = pack / "results.jsonl"
    done = {json.loads(l)["sample_id"] for l in open(res_path, encoding="utf-8")} if res_path.exists() else set()
    todo = [s for s in read_jsonl(pack / "samples.jsonl") if s["id"] not in done]
    print(f"待判 {len(todo)} 条（已完成 {len(done)} 条），模型 {model}")

    def one(s):
        prompt = fill(rubric, {"id": s["id"], "lang": s["lang"], "url": s["url"], "truncated": s["truncated"],
                               "text": s["text"]})
        last = ""
        for _ in range(a.retries + 1):
            try:
                out = parse_json(call_openai_compatible(prompt, base, key, model))
                out["sample_id"] = s["id"]
                return out
            except Exception as e:                                    # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
        return {"sample_id": s["id"], "verdict": "error", "violations": [], "rule_hint": "", "error": last}

    with ThreadPoolExecutor(a.workers) as ex, open(res_path, "a", encoding="utf-8") as f:
        for i, out in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {out['sample_id'][:8]} → {out.get('verdict')}"
                  f" {[v.get('rule_id') for v in out.get('violations', [])]}")
    print(f"完成 → {res_path}；下一步：python 06_qc_report.py --pack {pack}")


if __name__ == "__main__":
    main()
