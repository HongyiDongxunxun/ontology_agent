#!/usr/bin/env python3
"""
评价关系校验工具 (Relation Verification CLI)

对已抽取的评价关系逐条复核, 判断每条关系是「评价」还是「事实/描述」。
事实类关系标记 is_evaluation=false 并附 fact_type, 评价类关系放行。

输入 (二选一):
  --input-relations: 关系 JSONL (run_relation_agent.py 或主管道的输出)
  --input-dir:       输入目录, 配合 --verify 从 input/*.json 直接跑
                    (抽取+校验一体: 先 Agent 1 抽取, 再本工具校验)

输出 (output/relation_verification/):
  {name}_verification.jsonl — 每行一句: 校验结论 + 标记后的关系
  {name}_report.md          — 可读报告 (事实/评价分布)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import config as cfg, build_config
from pipeline import LLMClient, RelationVerificationAgent, EvaluativeRelationAgent


def load_relation_jsonl(filepath: str) -> list[dict]:
    """读取关系 JSONL, 按句分组"""
    grouped: dict[str, dict] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("sentence_id", ""))
            if not sid:
                continue
            if sid not in grouped:
                grouped[sid] = {
                    "sentence_id": sid,
                    "sentence": item.get("sentence", ""),
                    "has_evaluation": item.get("has_evaluation", False),
                    "relations": [],
                }
            # 逐关系平铺或整句嵌套两种格式都兼容
            if "relations" in item and isinstance(item["relations"], list):
                grouped[sid]["relations"] = item["relations"]
                grouped[sid]["has_evaluation"] = item.get("has_evaluation", False)
            else:
                grouped[sid]["relations"].append({
                    k: v for k, v in item.items()
                    if k in ("subject", "object", "aspect", "opinion", "evidence")
                })
    return list(grouped.values())


def verify_sentences(sentences: list[dict], agent: RelationVerificationAgent,
                     concurrency: int) -> list[dict]:
    """逐句校验 (每句一次 LLM 调用)"""
    outputs: list[dict] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _verify_one(entry: dict) -> dict:
        result = agent.verify(entry["sentence_id"], entry["sentence"],
                              entry["relations"])
        return result.to_dict()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_verify_one, s): i for i, s in enumerate(sentences)}
        done = 0
        for fut in as_completed(futures):
            outputs.append((futures[fut], fut.result()))
            done += 1
            if done % 5 == 0 or done == len(sentences):
                print(f"  [进度] {done}/{len(sentences)} 句")
    outputs.sort(key=lambda x: x[0])
    return [o for _, o in outputs]


def generate_report(outputs: list[dict], elapsed: float) -> str:
    lines: list[str] = []
    total_rel = sum(o["total_relations"] for o in outputs)
    total_fact = sum(o["fact_count"] for o in outputs)
    total_eval = sum(o["evaluation_count"] for o in outputs)
    lines.append("# 评价关系校验报告")
    lines.append("")
    lines.append(f"> 句数: {len(outputs)} | 耗时: {elapsed:.0f}s | "
                 f"关系总数: {total_rel} | 事实(标记): {total_fact} | 评价(放行): {total_eval}")
    if total_rel:
        lines.append(f"> 事实占比: {total_fact / total_rel:.1%}")
    lines.append("")
    for i, out in enumerate(outputs, 1):
        if not out["verifications"]:
            continue
        lines.append("---")
        lines.append(f"## [{i}] {out['sentence_id']}")
        lines.append("")
        lines.append(f"**原文:** {out['sentence'][:200]}")
        lines.append("")
        for v in out["verifications"]:
            mark = "✓评价" if v["is_evaluation"] else "✗事实"
            ft = f" [{v.get('fact_type','')}]" if v.get("fact_type") else ""
            lines.append(f"- {mark}{ft} {v.get('subject','')} → "
                         f"{v.get('object','')} | "
                         f"opinion=\"{str(v.get('opinion',''))[:40]}\"")
            lines.append(f"  理由: {v['verdict_reason']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="评价关系校验工具")
    parser.add_argument("--live", action="store_true", help="调用真实 LLM API (必须)")
    parser.add_argument("--input-relations", type=str, default="",
                        help="关系 JSONL 文件路径")
    parser.add_argument("--output", type=str, default="",
                        help="输出目录 (默认 output/relation_verification/)")
    parser.add_argument("--limit", type=int, default=0, help="只校验前 N 句 (0=全部)")
    parser.add_argument("--concurrency", type=int, default=8, help="并发句数")
    args = parser.parse_args()

    build_config(live_mode=args.live, verbose=True)
    api_key = (cfg.llm.api_key_relation or cfg.llm.api_key_extraction
               or cfg.llm.api_key)
    if not api_key:
        print("[ERR] 未找到 API Key (DEEPSEEK_API_KEY_RELATION / _EXTRACTION / 通用)")
        return 1

    if not args.input_relations:
        print("[ERR] 请用 --input-relations 指定关系 JSONL 文件")
        return 1
    if not os.path.exists(args.input_relations):
        print(f"[ERR] 文件不存在: {args.input_relations}")
        return 1

    output_dir = args.output or str(cfg.output_dir / "relation_verification")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sentences = load_relation_jsonl(args.input_relations)
    # 只校验有关系的句子
    sentences = [s for s in sentences if s["relations"]]
    if args.limit and args.limit > 0:
        sentences = sentences[:args.limit]

    if not sentences:
        print("[INFO] 输入文件中没有待校验的关系")
        return 0

    llm = LLMClient(model=cfg.llm.model, api_key=api_key,
                    base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
                    max_tokens=cfg.llm.max_tokens, timeout=cfg.llm.timeout)
    agent = RelationVerificationAgent(llm)

    print("=" * 66)
    print(f"评价关系校验 — {len(sentences)} 句, 并发 {args.concurrency}")
    print(f"输入: {args.input_relations} | 输出: {output_dir}")
    print("=" * 66)

    t0 = time.time()
    outputs = verify_sentences(sentences, agent, args.concurrency)
    elapsed = time.time() - t0

    base = Path(args.input_relations).stem
    jsonl_path = Path(output_dir) / f"{base}_verification.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    report = generate_report(outputs, elapsed)
    report_path = Path(output_dir) / f"{base}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    total_rel = sum(o["total_relations"] for o in outputs)
    total_fact = sum(o["fact_count"] for o in outputs)
    total_eval = sum(o["evaluation_count"] for o in outputs)
    print()
    print("=" * 66)
    print(f"完成: {elapsed:.0f}s | 关系 {total_rel} 条 | "
          f"事实(标记) {total_fact} | 评价(放行) {total_eval}")
    print(f"JSONL: {jsonl_path}")
    print(f"报告: {report_path}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
