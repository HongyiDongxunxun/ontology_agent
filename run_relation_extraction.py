#!/usr/bin/env python3
"""
独立评价关系抽取 Agent (EvaluativeRelationAgent) 命令行工具

单独运行 Agent 1 (评价关系抽取), 不执行实体补充/分类/审查。
每句一次 LLM 调用, 输出评价关系与关系中的实体。

用途:
  1. 快速验证评价关系抽取质量 (配合 --sample 小批量试跑)
  2. 单独批量生产评价关系数据集
  3. 调试提示词 (关系优先策略 / 评价有效性过滤)

输入:
  input/reviewed_full_*.json (与主管道相同的输入格式)

输出 (output/relation_agent_standalone/):
  {name}_relations.jsonl  — 每行一句: {sentence_id, sentence, has_evaluation, entities, relations}
  {name}_report.md        — 可读审阅报告
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import config as cfg, build_config
from pipeline import LLMClient, EvaluativeRelationAgent


def reviewed_key_sort(key: str) -> tuple:
    """与 run.py 一致的键排序 (纯数字在前, non_X 在后)"""
    if key.isdigit():
        return (0, int(key), key)
    m = re.search(r"(\d+)$", key)
    if m:
        return (1, int(m.group(1)), key)
    return (2, 0, key)


def load_sentences(input_dir: str, limit: int = 0, sample: int = 0,
                   seed: int = 42) -> list[tuple[str, str]]:
    """加载输入目录下的全部句子 (全局唯一 sentence_id)"""
    files = sorted(glob.glob(os.path.join(input_dir, "reviewed_full_*.json")))
    if not files:
        return []
    sentences: list[tuple[str, str]] = []
    for f in files:
        data = json.load(open(f, "r", encoding="utf-8"))
        base = Path(f).stem
        for key in sorted(data.keys(), key=reviewed_key_sort):
            entry = data[key]
            full = f"{entry.get('previous_sentence','')} {entry.get('evaluative_sentence','')} {entry.get('next_sentence','')}".strip()
            sentences.append((f"{base}::{key}", full))
    if limit and limit > 0:
        sentences = sentences[:limit]
    if sample and sample > 0:
        random.seed(seed)
        sentences = random.sample(sentences, min(sample, len(sentences)))
    return sentences


def process_one(sentence_id: str, sentence: str, agent: EvaluativeRelationAgent) -> dict:
    """单句关系抽取"""
    result = agent.extract(sentence_id, sentence)
    return {
        "sentence_id": sentence_id,
        "sentence": sentence,
        "has_evaluation": result.has_evaluation,
        "entities": result.entities,
        "relations": [r.to_dict() for r in result.relations],
    }


def generate_report(outputs: list[dict], elapsed: float, total_sentences: int) -> str:
    """生成可读审阅报告"""
    lines: list[str] = []
    lines.append("# 独立评价关系抽取 Agent — 审阅报告")
    lines.append("")
    lines.append(f"> 句数: {len(outputs)}/{total_sentences} | 耗时: {elapsed:.0f}s | "
                 f"关系总数: {sum(len(o['relations']) for o in outputs)} | "
                 f"含评价句数: {sum(1 for o in outputs if o['has_evaluation'])}")
    lines.append("")
    lines.append("## 图例")
    lines.append("")
    lines.append("- 关系: `主体 → 客体(entity_id/名字) | aspect | opinion | evidence`")
    lines.append("")
    for i, out in enumerate(outputs, 1):
        lines.append("---")
        lines.append(f"## [{i}] {out['sentence_id']}")
        lines.append("")
        lines.append(f"**原文:** {out['sentence'][:300]}")
        lines.append("")
        lines.append(f"**has_evaluation: {out['has_evaluation']}** | "
                     f"实体 {len(out['entities'])} 个 | 关系 {len(out['relations'])} 条")
        lines.append("")
        if out["entities"]:
            for e in out["entities"]:
                lines.append(f"- 实体 [{e.get('entity_id','')}] {e.get('entity','')}")
            lines.append("")
        if out["relations"]:
            lines.append("**评价关系:**")
            lines.append("")
            for r in out["relations"]:
                aspect = r.get("aspect") or "-"
                lines.append(f"- {r['subject']} → **{r['object']}** | aspect={aspect} | "
                             f"opinion=\"{r['opinion']}\" | evid=\"{r['evidence'][:60]}\"")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立评价关系抽取 Agent (EvaluativeRelationAgent)")
    parser.add_argument("--live", action="store_true", help="调用真实 LLM API (必须)")
    parser.add_argument("--input", type=str, default="", help="输入目录 (默认 input/)")
    parser.add_argument("--output", type=str, default="",
                        help="输出目录 (默认 output/relation_agent_standalone/)")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 句 (0=全部)")
    parser.add_argument("--sample", type=int, default=0,
                        help="随机抽样 N 句 (0=不抽样, 适合快速试跑)")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--concurrency", type=int, default=8, help="并发句数")
    args = parser.parse_args()

    build_config(live_mode=args.live, verbose=True)

    # API Key: 优先 DEEPSEEK_API_KEY_RELATION, 回退 extraction, 再回退通用
    api_key = (cfg.llm.api_key_relation or cfg.llm.api_key_extraction
               or cfg.llm.api_key)
    if not api_key:
        print("[ERR] 未找到 API Key。请设置 DEEPSEEK_API_KEY_RELATION / "
              "DEEPSEEK_API_KEY_EXTRACTION / DEEPSEEK_API_KEY 环境变量")
        return 1

    input_dir = args.input or str(cfg.input_dir)
    output_dir = args.output or str(cfg.output_dir / "relation_agent_standalone")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(input_dir, args.limit, args.sample, args.seed)
    if not sentences:
        print(f"[ERR] 在 {input_dir} 中未找到 reviewed_full_*.json 输入文件")
        return 1

    llm = LLMClient(model=cfg.llm.model, api_key=api_key,
                    base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
                    max_tokens=cfg.llm.max_tokens, timeout=cfg.llm.timeout)
    agent = EvaluativeRelationAgent(llm)

    print("=" * 66)
    print(f"独立评价关系抽取 Agent — {len(sentences)} 句, 并发 {args.concurrency}")
    print(f"模型: {cfg.llm.model} | 输入: {input_dir} | 输出: {output_dir}")
    print("=" * 66)

    t0 = time.time()
    outputs: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(process_one, sid, stmt, agent): sid
                   for sid, stmt in sentences}
        done = 0
        for fut in as_completed(futures):
            try:
                outputs.append(fut.result())
            except Exception as e:
                print(f"  [FAIL] {futures[fut]}: {e}")
            done += 1
            if done % 5 == 0 or done == len(sentences):
                print(f"  [进度] {done}/{len(sentences)} 句")
    elapsed = time.time() - t0

    # 保持输入顺序输出
    order = {sid: i for i, (sid, _) in enumerate(sentences)}
    outputs.sort(key=lambda o: order.get(o["sentence_id"], 1 << 30))

    base_name = "relation_agent_sample" if args.sample else "relation_agent_all"
    jsonl_path = Path(output_dir) / f"{base_name}_relations.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    report = generate_report(outputs, elapsed, len(sentences))
    report_path = Path(output_dir) / f"{base_name}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    total_rel = sum(len(o["relations"]) for o in outputs)
    total_ent = sum(len(o["entities"]) for o in outputs)
    has_eval = sum(1 for o in outputs if o["has_evaluation"])
    print()
    print("=" * 66)
    print(f"完成: {elapsed:.0f}s | 关系 {total_rel} 条 | 实体 {total_ent} 个 | "
          f"含评价句 {has_eval}/{len(outputs)}")
    print(f"JSONL: {jsonl_path}")
    print(f"报告: {report_path}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
