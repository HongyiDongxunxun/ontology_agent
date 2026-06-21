#!/usr/bin/env python3
"""
V4.1 端到端文献知识挖掘系统 — 三Agent并行启动入口

Agent 1: 实体抽取 (高召回) → 中间结果写入 mid_data/
Agent 2: L1/L2/L3 精分类 + L4 MicroMapping
Agent 3: 图书馆学专业学长审查 + Likert 5点量表 → 最终 JSONL 写入 output/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import config as cfg, build_config
from pipeline import (
    LLMClient,
    DualAgentPipeline,
    DynamicTermDB,
    export_jsonl,
    export_summary_json,
    FinalEntityResult,
)

_print_lock = Lock()
_summary_lock = Lock()

RUN_MODE = "full"  # "full"=全量模式 | "test"=测试模式(随机10文件)
# ----------------------------------------------------------------


def load_reviewed_json(filepath: str) -> list[tuple[str, str]]:
    """读取 reviewed_full_*.json，返回 [(sentence_id, sentence_text), ...]"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sentences: list[tuple[str, str]] = []
    for key in sorted(raw.keys(), key=lambda k: int(k)):
        entry = raw[key]
        stmt = entry.get("evaluative_sentence", "")
        prev = entry.get("previous_sentence", "")
        nxt = entry.get("next_sentence", "")
        full = f"{prev} {stmt} {nxt}".strip()
        sentences.append((key, full))
    return sentences


def scan_input_files(input_dir: str, limit: int = 0) -> list[Path]:
    dirpath = Path(input_dir)
    if not dirpath.is_dir():
        return []
    import re as _re
    def _num_key(p: Path) -> int:
        m = _re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(dirpath.glob("reviewed_full_*.json"), key=_num_key)
    if limit and limit > 0:
        files = files[:limit]
    return files


def output_exists(fpath: Path, output_dir: str) -> bool:
    base_name = fpath.stem
    out_file = Path(output_dir) / f"{base_name}_result.jsonl"
    return out_file.exists()


def process_one_file(fpath: Path, g, output_dir: str, mid_data_dir: str, idx: int, total: int, dynamic_db: DynamicTermDB) -> dict:
    fname = fpath.name
    base_name = fpath.stem
    with _print_lock:
        print(f"[{idx}/{total}] {fname}  开始处理...")
    try:
        llm_extraction = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_extraction,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        llm_classification = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_classification,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        llm_reviewer = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_reviewer,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )

        pipeline = DualAgentPipeline(
            llm_extraction=llm_extraction,
            llm_classification=llm_classification,
            llm_reviewer=llm_reviewer,
            dynamic_term_db=dynamic_db,
            batch_size=cfg.pipeline.batch_size,
            mid_data_dir=mid_data_dir,
            verbose=False,
        )

        sentences = load_reviewed_json(str(fpath))
        results: list[FinalEntityResult] = pipeline.run(sentences, base_name)

        # 导出最终 JSONL
        jsonl_path = Path(output_dir) / f"{base_name}_result.jsonl"
        export_jsonl(results, str(jsonl_path))

        # 导出汇总统计
        summary_path = Path(output_dir) / f"{base_name}_summary.json"
        export_summary_json(results, base_name, str(summary_path))

        valid = sum(1 for r in results if r.valid_entity)
        invalid = sum(1 for r in results if not r.valid_entity)
        type_dist = dict(Counter(r.l3_type_code for r in results if r.valid_entity))
        scored = sum(1 for r in results if r.likert_confidence > 0)
        avg_likert = (
            sum(r.likert_confidence for r in results if r.likert_confidence > 0) / scored
            if scored > 0 else 0
        )

        summary = {
            "file": fname,
            "success": True,
            "statements": len(sentences),
            "entities": len(results),
            "valid_entities": valid,
            "invalid_entities": invalid,
            "type_distribution": type_dist,
            "likert_scored": scored,
            "likert_average": round(avg_likert, 2),
            "likert_distribution": {
                "1_完全不认同": sum(1 for r in results if r.likert_confidence == 1),
                "2_不认同": sum(1 for r in results if r.likert_confidence == 2),
                "3_不确定": sum(1 for r in results if r.likert_confidence == 3),
                "4_认同": sum(1 for r in results if r.likert_confidence == 4),
                "5_完全认同": sum(1 for r in results if r.likert_confidence == 5),
            },
        }
        with _print_lock:
            print(f"[{idx}/{total}] {fname}  OK  {len(sentences)}句 {len(results)}实体 ({valid}有效 {invalid}无效) Likert均分:{avg_likert:.1f} TermDB:{dynamic_db.size()}")
        return summary

    except Exception as e:
        import traceback
        with _print_lock:
            print(f"[{idx}/{total}] {fname}  FAIL  {e}")
        return {"file": fname, "success": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="V4.1 三Agent端到端文献知识挖掘系统")
    parser.add_argument("--live", action="store_true", help="启用真实 LLM API (2 层独立 Key)")
    parser.add_argument("--limit", type=int, default=0, help="限制处理文件数量 (0 = 全部)")
    parser.add_argument("--concurrency", type=int, default=40, help="并行处理文件数 (默认 40)")
    parser.add_argument("--mode", choices=["full", "test"], default=None,
                        help="运行模式: full=全量处理, test=随机抽取10个文件测试 (默认取 RUN_MODE)")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已有输出，强制重新处理")
    parser.add_argument("--dynamic-out", type=str, default="", help="动态术语库导出路径 (默认 output/dynamic_terms.json)")
    args = parser.parse_args()

    run_mode = args.mode if args.mode is not None else RUN_MODE

    g = build_config(live_mode=args.live, verbose=True)

    api_key_extraction = cfg.llm.api_key_extraction
    api_key_classification = cfg.llm.api_key_classification
    api_key_reviewer = cfg.llm.api_key_reviewer

    if args.live and (not api_key_extraction or not api_key_classification or not api_key_reviewer):
        print("[错误] --live 模式需要 3 个 API Key (抽取 + 分类 + 审查)，请在环境变量中配置")
        return 1

    files = scan_input_files(str(g.input_dir), args.limit)
    if not files:
        print("[批量] input/ 目录下无 JSON 文件，请确认路径")
        return 1

    if run_mode == "test":
        sample_size = min(10, len(files))
        random.seed(42)
        files = random.sample(files, sample_size)
        print(f"[测试模式] 随机抽取 {sample_size} 个文件进行处理")

    total_files = len(files)
    output_dir = str(g.output_dir)
    mid_data_dir = str(g.mid_data_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(mid_data_dir).mkdir(parents=True, exist_ok=True)

    if not args.no_skip:
        skipped = 0
        remaining = []
        for fpath in files:
            if output_exists(fpath, output_dir):
                skipped += 1
            else:
                remaining.append(fpath)
        if skipped > 0:
            print(f"[跳过] 已有输出: {skipped} 个，待处理: {len(remaining)} 个")
        files = remaining

    if not files:
        print("[完成] 所有文件已处理完毕")
        return 0

    total = len(files)
    concurrency = args.concurrency
    mode_label = "测试模式 (随机10文件)" if run_mode == "test" else "全量模式"

    print(f"\n{'='*70}")
    print(f"  三Agent并行批量处理 [{mode_label}] — {total} 个文件 (并发 {concurrency}, 总计 {total_files} 个)")
    print(f"  输出目录: {output_dir}/")
    print(f"  中间数据: {mid_data_dir}/")
    print(f"{'='*70}\n")

    aggregated: list[dict] = []
    completed = 0
    start_time = time.time()

    dynamic_db = DynamicTermDB()
    print(f"[TermDB] 动态术语底库就绪 (初始为空)")
    print(f"{'='*70}\n")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for i, fpath in enumerate(files, 1):
            future = executor.submit(
                process_one_file, fpath, g, output_dir, mid_data_dir, i, total, dynamic_db
            )
            futures[future] = i

        for future in as_completed(futures):
            summary = future.result()
            with _summary_lock:
                aggregated.append(summary)
                completed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            successes = sum(1 for s in aggregated if s.get("success"))
            failures = sum(1 for s in aggregated if not s.get("success"))
            print(f"  [进度] {completed}/{total} | "
                  f"成功:{successes} 失败:{failures} | "
                  f"耗时:{elapsed:.0f}s 剩余:{eta:.0f}s")

    elapsed = time.time() - start_time
    successes = [r for r in aggregated if r.get("success")]
    failures = [r for r in aggregated if not r.get("success")]

    total_statements = sum(r.get("statements", 0) for r in successes)
    total_entities = sum(r.get("entities", 0) for r in successes)
    total_valid = sum(r.get("valid_entities", 0) for r in successes)
    total_invalid = sum(r.get("invalid_entities", 0) for r in successes)
    total_likert_scored = sum(r.get("likert_scored", 0) for r in successes)

    type_dist: dict[str, int] = {}
    likert_dist: dict[str, int] = {}
    for r in successes:
        for t, c in r.get("type_distribution", {}).items():
            type_dist[t] = type_dist.get(t, 0) + c
        for k, c in r.get("likert_distribution", {}).items():
            likert_dist[k] = likert_dist.get(k, 0) + c

    print(f"\n{'='*70}")
    print(f"  批量处理完成 — 汇总报告")
    print(f"{'='*70}")
    print(f"  总文件数: {total}")
    print(f"  成功: {len(successes)} / 失败: {len(failures)}")
    if total:
        print(f"  耗时: {elapsed:.0f}s  (平均 {elapsed/total:.1f}s/文件)")
    print(f"  总评价句: {total_statements}")
    print(f"  总实体: {total_entities}  (有效: {total_valid}, 无效: {total_invalid})")
    print(f"  Likert评分: {total_likert_scored} 条已评分")
    if likert_dist:
        print(f"  Likert 分布:")
        for k, v in sorted(likert_dist.items()):
            pct = v / total_likert_scored * 100 if total_likert_scored else 0
            print(f"    {k}: {v}  ({pct:.1f}%)")
    if type_dist:
        print(f"  实体类型分布:")
        for t, c in sorted(type_dist.items(), key=lambda x: -x[1])[:20]:
            print(f"    {t}: {c}")
    if failures:
        print(f"\n  失败文件:")
        for f in failures[:15]:
            print(f"    {f['file']}: {f.get('error', '?')}")
        if len(failures) > 15:
            print(f"    ... 及另外 {len(failures) - 15} 个")

    # 动态术语库统计
    db_stats = dynamic_db.get_stats()
    print(f"\n{'='*70}")
    print(f"  动态术语底库 (DynamicTermDB) 统计")
    print(f"{'='*70}")
    print(f"  累计术语: {db_stats['total_terms']}")
    print(f"  新增: {db_stats['added']}  去重跳过: {db_stats['skipped_duplicates']}")

    # 导出动态术语库
    dynamic_out = args.dynamic_out or os.path.join(output_dir, "dynamic_terms.json")
    dynamic_db.export(dynamic_out)
    print(f"  已导出: {Path(dynamic_out).resolve()}")

    print(f"{'='*70}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
