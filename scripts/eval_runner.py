#!/usr/bin/env python3
"""
Evaluation Runner: Load gold_flat → convert to GoldStandard → run pipeline → compute F1.

This script:
1. Loads all gold_flat JSONL files from AcademicEvaluation_exam/baseline/gold_flat/
2. Converts them to GoldStandard format (deriving L1/L2 from L3 type_code via taxonomy)
3. Runs the 3-agent pipeline on the sentences
4. Computes F1 metrics and generates a report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

# Add project root to path and change working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import config as cfg, build_config
from pipeline import (
    LLMClient,
    AgentPipeline,
    DynamicTermDB,
    export_jsonl,
    FinalEntityResult,
)
from pipeline.taxonomy import get_l1l2_from_l3, get_l2_label, get_l3_label

# Import eval module
try:
    from eval import (
        GoldStandard,
        GoldEntity,
        GoldSentence,
        compute_metrics,
        normalize_pipeline_output,
        generate_report,
        print_summary,
    )
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False
    import traceback
    traceback.print_exc()
    print("[ERROR] eval module not available")
    sys.exit(1)


def load_all_gold_flat(gold_dir: str) -> dict[str, dict[str, list[dict]]]:
    """
    Load all gold_flat JSONL files.

    Returns: {source_file_base: {sentence_id: [entity_dict, ...]}}
    where entity_dict = {entity, type (L3), sentence}
    """
    gold_path = Path(gold_dir)
    if not gold_path.is_dir():
        print(f"[ERROR] gold_flat directory not found: {gold_dir}")
        return {}

    # Group by source_file
    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for fpath in sorted(gold_path.glob("*.jsonl")):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source_file = entry.get("source_file", "")
                # source_file looks like "reviewed_full_10009_result.jsonl"
                # Strip "_result.jsonl" to get base name
                base_name = source_file.replace("_result.jsonl", "").replace("_result_gold_flat.jsonl", "")
                if not base_name:
                    continue

                sentence_id = entry.get("sentence_id", "")
                sentence = entry.get("sentence", "")
                entity = entry.get("entity", "")
                etype = entry.get("type", "")  # This is L3 type_code

                by_source[base_name][sentence_id].append({
                    "entity": entity,
                    "type": etype,
                    "sentence": sentence,
                })

    return {k: dict(v) for k, v in by_source.items()}


def convert_to_gold_standard(
    grouped_data: dict[str, dict[str, list[dict]]],
) -> GoldStandard:
    """
    Convert grouped gold_flat data to GoldStandard format.

    For each entity:
    - mention = entity name
    - type (L3) → derive L1, L2 from taxonomy
    - All entities in gold_flat are valid_entity=True
    """
    gs = GoldStandard(name="gold_flat_eval")

    for base_name, sentences in grouped_data.items():
        for sentence_id, entities in sentences.items():
            gold_entities = []
            sentence_text = entities[0]["sentence"] if entities else ""

            for ent in entities:
                entity_name = ent["entity"]
                l3_code = ent["type"]
                l1, l2 = get_l1l2_from_l3(l3_code)

                # Handle unknown types - map to "other" with derived L1
                if not l1:
                    # Try to guess L1 from context
                    print(f"[WARN] Unknown L3 type: '{l3_code}' for entity '{entity_name}' in {base_name}:{sentence_id}")
                    continue

                gold_entities.append(GoldEntity(
                    mention=entity_name,
                    l1=l1,
                    l2=l2,
                    l3_type_code=l3_code,
                    valid_entity=True,
                    normalized_name=entity_name,
                ))

            # Use compound sentence_id: base_name::sentence_id
            compound_id = f"{base_name}::{sentence_id}"
            gs.add_sentence(GoldSentence(
                sentence_id=compound_id,
                sentence=sentence_text,
                gold_entities=gold_entities,
            ))

    return gs


def prepare_sentences_for_pipeline(
    grouped_data: dict[str, dict[str, list[dict]]],
) -> dict[str, list[tuple[str, str]]]:
    """
    Prepare sentence lists for pipeline input, organized by source file.

    Returns: {base_name: [(sentence_id, sentence_text), ...]}
    """
    result: dict[str, list[tuple[str, str]]] = {}

    for base_name, sentences in grouped_data.items():
        sent_list: list[tuple[str, str]] = []
        for sentence_id in sorted(sentences.keys(), key=lambda x: int(x) if x.isdigit() else x):
            sent_text = sentences[sentence_id][0]["sentence"] if sentences[sentence_id] else ""
            sent_list.append((sentence_id, sent_text))
        result[base_name] = sent_list

    return result


def run_evaluation(
    grouped_data: dict[str, dict[str, list[dict]]],
    output_dir: str,
    concurrency: int = 10,
    limit_files: int = 0,
    dry_run: bool = False,
) -> dict:
    """Run full evaluation pipeline."""

    # ── 1. Convert to GoldStandard ──
    print("\n[Step 1] Converting gold_flat to GoldStandard...")
    gold_standard = convert_to_gold_standard(grouped_data)
    print(f"  GoldStandard: {gold_standard.sentence_count()} sentences, {gold_standard.entity_count()} entities")

    gs_stats = gold_standard.get_stats()
    print(f"  Valid entities: {gs_stats['valid_entities']}")
    print(f"  L1 distribution: {gs_stats['l1_distribution']}")
    print(f"  L3 distribution (top 15):")
    l3_dist = sorted(gs_stats['l3_distribution'].items(), key=lambda x: -x[1])
    for code, count in l3_dist[:15]:
        label = get_l3_label(code)
        print(f"    {code} ({label}): {count}")

    if dry_run:
        print("\n[Dry run] Skipping pipeline execution.")
        return {"gold_standard": gold_standard}

    # ── 2. Prepare sentences ──
    print(f"\n[Step 2] Preparing sentences for pipeline...")
    sentences_by_file = prepare_sentences_for_pipeline(grouped_data)
    base_names = sorted(sentences_by_file.keys())
    if limit_files > 0:
        base_names = base_names[:limit_files]
    total_sentences = sum(len(sentences_by_file[bn]) for bn in base_names)
    print(f"  Files to process: {len(base_names)}")
    print(f"  Total sentences: {total_sentences}")

    # ── 3. Check API Keys ──
    api_key_extraction = cfg.llm.api_key_extraction
    api_key_classification = cfg.llm.api_key_classification
    api_key_reviewer = cfg.llm.api_key_reviewer

    if not (api_key_extraction or api_key_classification or api_key_reviewer):
        print("\n[ERROR] No API keys configured!")
        print("  Set at least one of: DEEPSEEK_API_KEY, DEEPSEEK_API_KEY_EXTRACTION, etc.")
        print("  Export them as environment variables and re-run.")
        return {"error": "no_api_keys"}

    # ── 4. Run Pipeline (concurrent) ──
    print(f"\n[Step 3] Running pipeline (concurrency={concurrency})...")

    dynamic_db = DynamicTermDB()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Build RAG index (shared, read-only after build) ──
    shared_rag_db = None
    if cfg.pipeline.enable_rag:
        from pipeline.rag import build_rag_index
        gold_flat = "D:/code/AcademicEvaluation_exam/baseline/gold_flat"
        cache = str(output_path / "rag_cache")
        shared_rag_db = build_rag_index(gold_flat, cache)
        print(f"  RAG index: {shared_rag_db.size()} examples ready")

    all_predictions: dict[str, list[dict]] = {}
    all_results: list[FinalEntityResult] = []
    start_time = time.time()

    _pred_lock = Lock()
    _print_lock = Lock()

    def process_one_file(bn: str, idx: int, total: int):
        """Process a single file - runs in thread."""
        sentences = sentences_by_file[bn]
        mid_data_dir = str(output_path / "mid_data" / bn)

        # Each thread gets its own LLM clients
        llm_ext = LLMClient(
            model=cfg.llm.model,
            api_key=api_key_extraction,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_extraction,
            reasoning_effort=cfg.llm.reasoning_effort,
        )
        llm_cls = LLMClient(
            model=cfg.llm.model,
            api_key=api_key_classification,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_classification,
            reasoning_effort=cfg.llm.reasoning_effort,
        )
        llm_rev = LLMClient(
            model=cfg.llm.model,
            api_key=api_key_reviewer,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            enable_thinking=cfg.llm.enable_thinking_reviewer,
            reasoning_effort=cfg.llm.reasoning_effort,
        )

        # Voting: per-thread voter (optional)
        voter = None
        if cfg.pipeline.enable_voting:
            from pipeline.voting import ExtractionVoter
            voter = ExtractionVoter(
                llm=llm_ext,
                rounds=cfg.pipeline.voting_rounds,
                threshold=cfg.pipeline.voting_threshold,
                temperature=cfg.pipeline.voting_temperature,
            )

        pipeline = AgentPipeline(
            llm_extraction=llm_ext,
            llm_classification=llm_cls,
            llm_reviewer=llm_rev,
            dynamic_term_db=dynamic_db,
            rag_db=shared_rag_db,
            voter=voter,
            batch_size=cfg.pipeline.batch_size,
            mid_data_dir=mid_data_dir,
            verbose=False,
        )

        with _print_lock:
            print(f"\n  [{idx}/{total}] Processing {bn} ({len(sentences)} sentences)...")

        try:
            results = pipeline.run(sentences, bn)
            with _pred_lock:
                all_results.extend(results)
                for r in results:
                    compound_id = f"{bn}::{r.sentence_id}"
                    all_predictions.setdefault(compound_id, []).append(r.to_dict())

            # Export individual file output
            jsonl_path = output_path / f"{bn}_result.jsonl"
            export_jsonl(results, str(jsonl_path))

            with _print_lock:
                print(f"  [{idx}/{total}] {bn} OK ({len(results)} entities, TermDB:{dynamic_db.size()})")
            return {"file": bn, "success": True, "entities": len(results)}

        except Exception as e:
            with _print_lock:
                print(f"  [{idx}/{total}] {bn} FAIL: {e}")
            import traceback
            traceback.print_exc()
            return {"file": bn, "success": False, "error": str(e)}

    total = len(base_names)
    completed_count = 0
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for i, bn in enumerate(base_names, 1):
            future = executor.submit(process_one_file, bn, i, total)
            futures[future] = (i, bn)

        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            if result.get("success"):
                success_count += 1
            else:
                fail_count += 1
            elapsed_t = time.time() - start_time
            eta = (elapsed_t / completed_count) * (total - completed_count) if completed_count > 0 else 0
            print(
                f"  [Progress] {completed_count}/{total} | "
                f"OK:{success_count} FAIL:{fail_count} | "
                f"Elapsed:{elapsed_t:.0f}s ETA:{eta:.0f}s | TermDB:{dynamic_db.size()}"
            )

    elapsed = time.time() - start_time
    print(f"\n[Pipeline Complete] {total} files in {elapsed:.0f}s ({elapsed/total:.1f}s/file)")

    # ── 5. Compute Metrics ──
    print(f"\n[Step 4] Computing metrics...")
    print(f"  Predictions: {len(all_predictions)} sentences, {len(all_results)} entities")

    eval_result = compute_metrics(all_predictions, gold_standard, match_mode="exact", detailed=True)

    # ── 6. Print Summary ──
    print_summary(eval_result)

    # ── 7. Generate Report ──
    report_path = output_path / "eval_report.md"
    generate_report(
        eval_result,
        gold_standard=gold_standard,
        title="Ontology_Agent_2 评估报告 (Gold Flat 200 Files)",
        output_path=str(report_path),
    )

    # ── 8. Save detailed results ──
    metrics_path = output_path / "eval_metrics.json"
    metrics_data = {
        "extraction": {
            "precision": eval_result.extraction_precision,
            "recall": eval_result.extraction_recall,
            "f1": eval_result.extraction_f1,
            "tp": eval_result.extraction_tp,
            "fp": eval_result.extraction_fp,
            "fn": eval_result.extraction_fn,
        },
        "classification": {
            "l1_accuracy": eval_result.l1_accuracy,
            "l2_accuracy": eval_result.l2_accuracy,
            "l3_accuracy": eval_result.l3_accuracy,
            "l1_correct": eval_result.l1_correct,
            "l2_correct": eval_result.l2_correct,
            "l3_correct": eval_result.l3_correct,
            "total_matched": eval_result.l1_total,
        },
        "strict": {
            "precision": eval_result.strict_precision,
            "recall": eval_result.strict_recall,
            "f1": eval_result.strict_f1,
            "tp": eval_result.strict_tp,
            "fp": eval_result.strict_fp,
            "fn": eval_result.strict_fn,
        },
        "validity_accuracy": eval_result.validity_accuracy,
        "total_predicted": eval_result.total_predicted,
        "total_gold": eval_result.total_gold,
        "total_sentences": eval_result.total_sentences,
        "per_type": {
            t: {
                "f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "strict_f1": m.strict_f1,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
            }
            for t, m in eval_result.per_type.items()
        },
        "elapsed_seconds": elapsed,
        "concurrency": concurrency,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Metrics] Saved to {metrics_path}")

    # Save gold_standard for reference
    gs_path = output_path / "gold_standard.jsonl"
    gold_standard.to_jsonl(str(gs_path))

    return {
        "eval_result": eval_result,
        "gold_standard": gold_standard,
        "predictions": all_predictions,
        "metrics": metrics_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Ontology Agent 2 Evaluation Runner")
    parser.add_argument("--gold-dir", type=str,
                        default="D:/code/AcademicEvaluation_exam/baseline/gold_flat",
                        help="Path to gold_flat directory")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "output" / "eval"),
                        help="Output directory for results")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of files to process concurrently (default: 1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to process (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load and validate data, don't run pipeline")
    parser.add_argument("--live", action="store_true",
                        help="Enable live LLM API calls")
    args = parser.parse_args()

    # Configure
    build_config(live_mode=args.live, verbose=True)

    print("=" * 70)
    print("  Ontology Agent 2 — Evaluation Runner")
    print(f"  Gold dir: {args.gold_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Limit: {args.limit if args.limit > 0 else 'all'}")
    print("=" * 70)

    # Load gold data
    print("\n[Step 0] Loading gold_flat data...")
    grouped = load_all_gold_flat(args.gold_dir)

    total_files = len(grouped)
    total_sentences = sum(len(sents) for sents in grouped.values())
    total_entities = sum(
        sum(len(entities) for entities in sents.values())
        for sents in grouped.values()
    )
    print(f"  Files: {total_files}")
    print(f"  Sentences: {total_sentences}")
    print(f"  Entity annotations: {total_entities}")

    if args.limit > 0:
        # Only keep first N files
        keys = sorted(grouped.keys())[:args.limit]
        grouped = {k: grouped[k] for k in keys}
        print(f"  Limited to {len(grouped)} files")

    # Run evaluation
    result = run_evaluation(
        grouped_data=grouped,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        limit_files=0,  # Already limited above
        dry_run=args.dry_run,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
