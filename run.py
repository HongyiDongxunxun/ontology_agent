#!/usr/bin/env python3
"""
V4.2 端到端文献知识挖掘系统 — 三Agent并行启动入口

Agent 1: 实体抽取 (高召回) → 中间结果写入 mid_data/
Agent 2: L1/L2/L3 精分类 + L4 MicroMapping
Agent 3: 图书馆学专业学长审查 + Likert 5点量表 → 最终 JSONL 写入 output/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import config as cfg, build_config
from pipeline import (
    LLMClient,
    AgentPipeline,
    DynamicTermDB,
    export_jsonl,
    export_summary_json,
    FinalEntityResult,
)

# ── 评估模式导入 ──
try:
    from eval import (
        GoldStandard,
        compute_metrics,
        normalize_pipeline_output,
        generate_report,
        print_summary,
    )
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

_print_lock = Lock()
_summary_lock = Lock()

RUN_MODE = "test"  # "full"=全量模式 | "test"=测试模式(随机10文件)


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

    def _num_key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    files = sorted(dirpath.glob("reviewed_full_*.json"), key=_num_key)
    if limit and limit > 0:
        files = files[:limit]
    return files


def output_exists(fpath: Path, output_dir: str) -> bool:
    out_file = Path(output_dir) / f"{fpath.stem}_result.jsonl"
    return out_file.exists()


def process_one_file(
    fpath: Path,
    output_dir: str,
    mid_data_dir: str,
    idx: int,
    total: int,
    dynamic_db: DynamicTermDB,
) -> dict:
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
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_extraction,
            reasoning_effort=cfg.llm.reasoning_effort,
        )
        llm_classification = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_classification,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_classification,
            reasoning_effort=cfg.llm.reasoning_effort,
        )
        llm_reviewer = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_reviewer,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_reviewer,
            reasoning_effort=cfg.llm.reasoning_effort,
        )

        pipeline = AgentPipeline(
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
            print(
                f"[{idx}/{total}] {fname}  OK  {len(sentences)}句 {len(results)}实体 "
                f"({valid}有效 {invalid}无效) Likert均分:{avg_likert:.1f} TermDB:{dynamic_db.size()}"
            )
        return summary

    except Exception as e:
        with _print_lock:
            print(f"[{idx}/{total}] {fname}  FAIL  {e}")
        return {"file": fname, "success": False, "error": str(e)}


def run_eval_mode(args) -> int:
    """
    评估模式: 加载标注数据, 对标注句子运行 pipeline, 计算 F1 指标。
    不依赖 DynamicTermDB 累积, 每个句子独立评估。
    """
    print(f"\n{'='*70}")
    print(f"  评估模式 (Evaluation Mode)")
    print(f"  标注数据: {args.gold}")
    print(f"  报告输出: {args.eval_output}")
    print(f"{'='*70}\n")

    # 1. 加载标注数据
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[错误] 标注文件不存在: {gold_path}")
        print(f"[提示] 请先使用 eval/annotation_tool.py 创建标注数据:")
        print(f"       python -m eval.annotation_tool -i output/ -g {args.gold}")
        return 1

    gold_standard = GoldStandard.from_jsonl(str(gold_path))
    if gold_standard.sentence_count() == 0:
        print(f"[错误] 标注文件为空: {gold_path}")
        return 1

    print(f"[评估] 已加载标注: {gold_standard.sentence_count()} 句, "
          f"{gold_standard.entity_count()} 个实体")
    gs_stats = gold_standard.get_stats()
    print(f"[评估] 有效实体: {gs_stats['valid_entities']}, "
          f"无效实体: {gs_stats['invalid_entities']}")
    print(f"[评估] L1 分布: {gs_stats['l1_distribution']}")

    # 2. 检查 API Key
    api_key_extraction = cfg.llm.api_key_extraction
    api_key_classification = cfg.llm.api_key_classification
    api_key_reviewer = cfg.llm.api_key_reviewer

    if not (api_key_extraction or api_key_classification or api_key_reviewer):
        print("[错误] 评估模式需要配置 LLM API Key, 请在环境变量中设置")
        return 1

    # 3. 构建输入: 从标注的句子中提取 (sentence_id, sentence_text)
    sentences: list[tuple[str, str]] = []
    for sid in sorted(gold_standard._sentences.keys()):
        gs = gold_standard.get_sentence(sid)
        sentences.append((sid, gs.sentence))

    print(f"[评估] 共 {len(sentences)} 句待评估\n")

    # 4. 加载动态术语库 (用于 L4 匹配 + RAG few-shot)
    dynamic_db = None
    try:
        terms_path = args.dynamic_terms or cfg.dynamic_terms_path
        if terms_path and Path(terms_path).exists():
            print(f"[评估] 加载动态术语库: {terms_path}")
            max_terms = getattr(args, 'max_terms', 0)
            if max_terms > 0:
                print(f"[评估] 限制加载: {max_terms:,} 条")
            dynamic_db = DynamicTermDB.from_json(terms_path, max_terms=max_terms)
            print(f"[评估] 术语库: {dynamic_db.size():,} 条")
        else:
            print(f"[评估] 未找到动态术语库, 跳过预加载")
    except Exception as e:
        print(f"[评估] 术语库加载失败: {e}")

    # 5. 配置 thinking + voting + RAG
    cfg.llm.enable_thinking_classification = getattr(args, 'thinking', True)
    cfg.pipeline.enable_voting = getattr(args, 'voting', True)
    cfg.pipeline.enable_rag = getattr(args, 'rag', False)  # RAG needs gold_flat

    print(f"[评估] Thinking: {cfg.llm.enable_thinking_classification} | "
          f"Voting: {cfg.pipeline.enable_voting} | "
          f"RAG: {cfg.pipeline.enable_rag}")
    print(f"[评估] Voting rounds: {cfg.pipeline.voting_rounds} | "
          f"Temperature: {cfg.pipeline.voting_temperature}")

    # 6. 运行 pipeline
    llm_extraction = LLMClient(
        model=cfg.llm.model,
        api_key=api_key_extraction,
        base_url=cfg.llm.base_url,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout=cfg.llm.timeout,
        enable_thinking=cfg.llm.enable_thinking_extraction,
        reasoning_effort=cfg.llm.reasoning_effort,
    )
    llm_classification = LLMClient(
        model=cfg.llm.model,
        api_key=api_key_classification,
        base_url=cfg.llm.base_url,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout=cfg.llm.timeout,
        enable_thinking=cfg.llm.enable_thinking_classification,
        reasoning_effort=cfg.llm.reasoning_effort,
    )
    llm_reviewer = LLMClient(
        model=cfg.llm.model,
        api_key=api_key_reviewer,
        base_url=cfg.llm.base_url,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout=cfg.llm.timeout,
        enable_thinking=cfg.llm.enable_thinking_reviewer,
        reasoning_effort=cfg.llm.reasoning_effort,
    )

    # 构建 AgentPipeline (带投票+RAG+动态术语库)
    pipeline = AgentPipeline(
        llm_extraction=llm_extraction,
        llm_classification=llm_classification,
        llm_reviewer=llm_reviewer,
        dynamic_term_db=dynamic_db,
        batch_size=cfg.pipeline.batch_size,
        mid_data_dir=str(cfg.mid_data_dir),
        verbose=False,
    )
    # 注入 voting/rag 配置到 classification agent
    pipeline.classification_agent.enable_voting = cfg.pipeline.enable_voting
    pipeline.classification_agent.voting_rounds = cfg.pipeline.voting_rounds
    pipeline.classification_agent.voting_temperature = cfg.pipeline.voting_temperature
    pipeline.classification_agent.enable_rag = cfg.pipeline.enable_rag
    pipeline.classification_agent.rag_k_examples = cfg.pipeline.rag_k_examples

    print("[评估] 开始运行 pipeline...")
    results = pipeline.run(sentences, "eval_baseline")
    print(f"[评估] pipeline 完成, 共 {len(results)} 个实体\n")

    # 5. 保存评估结果 JSONL (方便人工检查)
    eval_jsonl_path = Path(args.eval_output).with_suffix(".jsonl")
    export_jsonl(results, str(eval_jsonl_path))
    print(f"[评估] 预测结果已保存: {eval_jsonl_path}")

    # 6. 计算指标
    print(f"\n{'='*70}")
    print(f"  计算评估指标...")
    print(f"{'='*70}\n")

    predictions = normalize_pipeline_output([r.to_dict() for r in results])
    eval_result = compute_metrics(
        predictions, gold_standard, match_mode="exact", detailed=True
    )

    # 7. 终端摘要
    print_summary(eval_result)

    # 8. 生成 Markdown 报告
    report = generate_report(
        eval_result,
        gold_standard=gold_standard,
        title="Ontology_Agent 基线评估报告",
        output_path=args.eval_output,
    )

    print(f"\n[评估] 评估完成! 报告: {args.eval_output}")
    print(f"[评估] 预测详情: {eval_jsonl_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="V4.2 三Agent端到端文献知识挖掘系统")
    parser.add_argument("--live", action="store_true", help="启用真实 LLM API (3 层独立 Key)")
    parser.add_argument("--limit", type=int, default=0, help="限制处理文件数量 (0 = 全部)")
    parser.add_argument("--concurrency", type=int, default=40, help="并行处理文件数 (默认 40)")
    parser.add_argument(
        "--mode", choices=["full", "test"], default=None,
        help="运行模式: full=全量处理, test=随机抽取10个文件测试 (默认取 RUN_MODE)",
    )
    parser.add_argument("--no-skip", action="store_true", help="不跳过已有输出，强制重新处理")
    parser.add_argument(
        "--dynamic-out", type=str, default="",
        help="动态术语库导出路径 (默认 output/dynamic_terms.json)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="评估模式: 加载标注数据, 运行 pipeline, 计算 F1 指标并生成报告",
    )
    parser.add_argument(
        "--gold", type=str, default="eval/gold_data.jsonl",
        help="标注数据文件路径 (用于 --eval 模式, 默认: eval/gold_data.jsonl)",
    )
    parser.add_argument(
        "--eval-output", type=str, default="eval/report.md",
        help="评估报告输出路径 (默认: eval/report.md)",
    )
    parser.add_argument(
        "--thinking", action="store_true", default=True,
        help="启用 DeepSeek thinking mode (默认: 开启)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="禁用 thinking mode",
    )
    parser.add_argument(
        "--voting", action="store_true", default=True,
        help="启用 Self-Consistency 投票机制 (默认: 开启)",
    )
    parser.add_argument(
        "--no-voting", action="store_true",
        help="禁用投票机制",
    )
    parser.add_argument(
        "--rag", action="store_true", default=False,
        help="启用 RAG few-shot 增强 (需要 gold_flat 数据)",
    )
    parser.add_argument(
        "--dynamic-terms", type=str, default="",
        help="动态术语库 JSON 路径 (用于 L4 匹配 + few-shot 检索)",
    )
    parser.add_argument(
        "--max-terms", type=int, default=0,
        help="动态术语库最大加载条数 (0=全部, 建议 100000)",
    )
    args = parser.parse_args()

    # ── 处理 --no-* 标志 ──
    if args.no_thinking:
        args.thinking = False
    if args.no_voting:
        args.voting = False
    # 默认 dynamic_terms 路径
    if not args.dynamic_terms:
        default_terms = Path("dynamic_terms.json")
        if default_terms.exists():
            args.dynamic_terms = str(default_terms)

    # ── 评估模式 ──
    if args.eval:
        if not EVAL_AVAILABLE:
            print("[错误] eval 包未安装或导入失败，无法使用 --eval 模式")
            return 1
        return run_eval_mode(args)

    run_mode = args.mode if args.mode is not None else RUN_MODE

    run_cfg = build_config(live_mode=args.live, verbose=True)

    api_key_extraction = cfg.llm.api_key_extraction
    api_key_classification = cfg.llm.api_key_classification
    api_key_reviewer = cfg.llm.api_key_reviewer

    if args.live and (not api_key_extraction or not api_key_classification or not api_key_reviewer):
        print("[错误] --live 模式需要 3 个 API Key (抽取 + 分类 + 审查)，请在环境变量中配置")
        return 1

    files = scan_input_files(str(run_cfg.input_dir), args.limit)
    if not files:
        print("[批量] input/ 目录下无 JSON 文件，请确认路径")
        return 1

    if run_mode == "test":
        sample_size = min(10, len(files))
        random.seed(42)
        files = random.sample(files, sample_size)
        print(f"[测试模式] 随机抽取 {sample_size} 个文件进行处理")

    total_files = len(files)
    output_dir = str(run_cfg.output_dir)
    mid_data_dir = str(run_cfg.mid_data_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(mid_data_dir).mkdir(parents=True, exist_ok=True)

    if not args.no_skip:
        skipped = 0
        remaining: list[Path] = []
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
                process_one_file, fpath, output_dir, mid_data_dir, i, total, dynamic_db,
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
            print(
                f"  [进度] {completed}/{total} | "
                f"成功:{successes} 失败:{failures} | "
                f"耗时:{elapsed:.0f}s 剩余:{eta:.0f}s"
            )

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
    dynamic_out = args.dynamic_out or str(Path(output_dir) / "dynamic_terms.json")
    dynamic_db.export(dynamic_out)
    print(f"  已导出: {Path(dynamic_out).resolve()}")

    print(f"{'='*70}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
