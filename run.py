#!/usr/bin/env python3
"""
V4.3 端到端文献知识挖掘系统 — 三Agent并行启动入口 (eval mode support)
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from config import config as cfg, build_config
from pipeline import (LLMClient, DualAgentPipeline, DynamicTermDB,
                       export_jsonl, export_summary_json, export_relation_jsonl, FinalEntityResult)
try:
    from eval import (GoldStandard, compute_metrics, normalize_pipeline_output,
                       generate_report, print_summary)
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

_print_lock = Lock(); _summary_lock = Lock()
RUN_MODE = "full"

def reviewed_key_sort(key: str) -> tuple[int, int, str]:
    if key.isdigit():
        return (0, int(key), key)
    import re
    match = re.search(r"(\d+)$", key)
    if match:
        return (1, int(match.group(1)), key)
    return (2, 0, key)

def load_reviewed_json(filepath: str) -> list[tuple[str, str]]:
    with open(filepath, "r", encoding="utf-8") as f: raw = json.load(f)
    sentences: list[tuple[str, str]] = []
    for key in sorted(raw.keys(), key=reviewed_key_sort):
        entry = raw[key]
        stmt = entry.get("evaluative_sentence", "")
        prev = entry.get("previous_sentence", ""); nxt = entry.get("next_sentence", "")
        full = f"{prev} {stmt} {nxt}".strip()
        sentences.append((key, full))
    return sentences

def scan_input_files(input_dir: str, limit: int = 0) -> list[Path]:
    dirpath = Path(input_dir)
    if not dirpath.is_dir(): return []
    import re as _re
    def _num_key(p): m = _re.search(r"(\d+)", p.stem); return int(m.group(1)) if m else 0
    files = sorted(dirpath.glob("reviewed_full_*.json"), key=_num_key)
    if limit and limit > 0: files = files[:limit]
    return files

def output_exists(fpath: Path, output_dir: str) -> bool:
    return Path(output_dir, f"{fpath.stem}_result.jsonl").exists()

def process_one_file(fpath: Path, g, output_dir: str, mid_data_dir: str,
                     idx: int, total: int, dynamic_db: DynamicTermDB) -> dict:
    fname = fpath.name; base_name = fpath.stem
    with _print_lock: print(f"[{idx}/{total}] {fname}  开始处理...")
    try:
        llm_extraction = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_extraction,
            base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_extraction)
        llm_classification = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_classification,
            base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_classification)
        llm_reviewer = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_reviewer,
            base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_reviewer)
        pipeline = DualAgentPipeline(llm_extraction=llm_extraction,
            llm_classification=llm_classification, llm_reviewer=llm_reviewer,
            dynamic_term_db=dynamic_db, batch_size=cfg.pipeline.batch_size,
            mid_data_dir=mid_data_dir, verbose=False)
        # Inject voting/rag config
        pipeline.classification_agent.enable_voting = cfg.pipeline.enable_voting
        pipeline.classification_agent.voting_rounds = cfg.pipeline.voting_rounds
        pipeline.classification_agent.voting_temperature = cfg.pipeline.voting_temperature
        pipeline.classification_agent.enable_rag = cfg.pipeline.enable_rag
        pipeline.classification_agent.rag_k_examples = cfg.pipeline.rag_k_examples
        sentences = load_reviewed_json(str(fpath))
        results: list[FinalEntityResult]
        relations: list[dict]
        results, relations = pipeline.run(sentences, base_name)
        export_jsonl(results, str(Path(output_dir) / f"{base_name}_result.jsonl"))
        export_summary_json(results, base_name, str(Path(output_dir) / f"{base_name}_summary.json"))
        # V4.4: 导出评价关系 (来自 Agent 1)
        if relations:
            rel_dir = str(Path(output_dir).parent / "evaluative_relation")
            rel_file = str(Path(rel_dir) / f"relation_{base_name.replace('reviewed_', '')}.jsonl")
            Path(rel_dir).mkdir(parents=True, exist_ok=True)
            export_relation_jsonl(relations, rel_file)
        valid = sum(1 for r in results if r.valid_entity)
        invalid = sum(1 for r in results if not r.valid_entity)
        type_dist = dict(Counter(r.l3_type_code for r in results if r.valid_entity))
        scored = sum(1 for r in results if r.likert_confidence > 0)
        avg_likert = (sum(r.likert_confidence for r in results if r.likert_confidence > 0) / scored) if scored > 0 else 0
        summary = {"file": fname, "success": True, "statements": len(sentences),
            "entities": len(results), "valid_entities": valid, "invalid_entities": invalid,
            "type_distribution": type_dist, "likert_scored": scored, "likert_average": round(avg_likert, 2),
            "likert_distribution": {f"{i}_": sum(1 for r in results if r.likert_confidence == i) for i in range(1,6)}}
        with _print_lock:
            print(f"[{idx}/{total}] {fname}  OK  {len(sentences)}句 {len(results)}实体 "
                  f"({valid}有效 {invalid}无效) Likert均分:{avg_likert:.1f} TermDB:{dynamic_db.size()}")
        return summary
    except Exception as e:
        with _print_lock: print(f"[{idx}/{total}] {fname}  FAIL  {e}")
        return {"file": fname, "success": False, "error": str(e)}

# ── Eval Mode ──
def run_eval_mode(args) -> int:
    print(f"\n{'='*70}\n  评估模式 (Eval Mode)\n  标注数据: {args.gold}\n  报告输出: {args.eval_output}\n{'='*70}\n")
    gold_path = Path(args.gold)
    if not gold_path.exists(): print(f"[ERR] gold file not found: {gold_path}"); return 1
    gold_standard = GoldStandard.from_jsonl(str(gold_path))
    if gold_standard.sentence_count() == 0: print(f"[ERR] empty gold: {gold_path}"); return 1
    print(f"[Eval] loaded: {gold_standard.sentence_count()} sentences, {gold_standard.entity_count()} entities")
    print(f"[Eval] L1 dist: {gold_standard.l1_distribution()}")

    api_keys = [cfg.llm.api_key_extraction, cfg.llm.api_key_classification, cfg.llm.api_key_reviewer]
    if not any(api_keys): print("[ERR] need API keys"); return 1

    sentences = [(sid, gold_standard.get_sentence(sid).sentence)
                 for sid in sorted(gold_standard._sentences.keys())]
    print(f"[Eval] {len(sentences)} sentences to evaluate\n")

    # Load dynamic terms
    dynamic_db = None
    terms_path = args.dynamic_terms or cfg.dynamic_terms_path
    if terms_path and Path(terms_path).exists():
        max_terms = getattr(args, 'max_terms', 0)
        print(f"[Eval] loading terms: {terms_path}" + (f" (max {max_terms:,})" if max_terms else ""))
        dynamic_db = DynamicTermDB.from_json(terms_path, max_terms=max_terms)
        print(f"[Eval] terms: {dynamic_db.size():,}")
    else:
        print("[Eval] no dynamic terms, skipping")

    # Config thinking/voting/RAG
    cfg.llm.enable_thinking_classification = getattr(args, 'thinking', False)
    cfg.pipeline.enable_voting = getattr(args, 'voting', False)
    cfg.pipeline.enable_rag = getattr(args, 'rag', False)
    print(f"[Eval] Thinking={cfg.llm.enable_thinking_classification} | "
          f"Voting={cfg.pipeline.enable_voting} | RAG={cfg.pipeline.enable_rag}")

    # Build LLM clients
    llm_ext = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_extraction,
        base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_extraction)
    llm_cls = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_classification,
        base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_classification)
    llm_rev = LLMClient(model=cfg.llm.model, api_key=cfg.llm.api_key_reviewer,
        base_url=cfg.llm.base_url, temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens, enable_thinking=cfg.llm.enable_thinking_reviewer)

    pipeline = DualAgentPipeline(llm_extraction=llm_ext, llm_classification=llm_cls,
        llm_reviewer=llm_rev, dynamic_term_db=dynamic_db,
        batch_size=cfg.pipeline.batch_size, mid_data_dir=str(cfg.mid_data_dir), verbose=False)
    pipeline.classification_agent.enable_voting = cfg.pipeline.enable_voting
    pipeline.classification_agent.voting_rounds = cfg.pipeline.voting_rounds
    pipeline.classification_agent.voting_temperature = cfg.pipeline.voting_temperature
    pipeline.classification_agent.enable_rag = cfg.pipeline.enable_rag
    pipeline.classification_agent.rag_k_examples = cfg.pipeline.rag_k_examples

    print("[Eval] running pipeline...")
    results, relations = pipeline.run(sentences, "eval_latest")
    print(f"[Eval] pipeline done: {len(results)} entities, {len(relations)} relations\n")

    eval_jsonl = Path(args.eval_output).with_suffix(".jsonl")
    export_jsonl(results, str(eval_jsonl))
    print(f"[Eval] predictions saved: {eval_jsonl}")

    predictions = normalize_pipeline_output([r.to_dict() for r in results])
    eval_result = compute_metrics(predictions, gold_standard, match_mode="exact", detailed=True)
    print_summary(eval_result)
    generate_report(eval_result, gold_standard=gold_standard,
                    title="Ontology_Agent V4.3 Eval Report", output_path=args.eval_output)
    print(f"\n[Eval] done! Report: {args.eval_output}")
    print(f"[Eval] Predictions: {eval_jsonl}")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="V4.3 三Agent端到端文献知识挖掘系统")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--mode", choices=["full", "test"], default=None)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--dynamic-out", type=str, default="")
    parser.add_argument("--eval", action="store_true", help="评估模式")
    parser.add_argument("--gold", type=str, default="eval/gold_data.jsonl", help="标注数据路径")
    parser.add_argument("--eval-output", type=str, default="eval/report.md", help="评估报告路径")
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--voting", action="store_true", default=False)
    parser.add_argument("--no-voting", action="store_true")
    parser.add_argument("--rag", action="store_true", default=False)
    parser.add_argument("--dynamic-terms", type=str, default="")
    parser.add_argument("--max-terms", type=int, default=0)
    args = parser.parse_args()

    if args.no_thinking: args.thinking = False
    if args.no_voting: args.voting = False
    if not args.dynamic_terms:
        p = Path("dynamic_terms.json")
        if p.exists(): args.dynamic_terms = str(p)

    if args.eval:
        if not EVAL_AVAILABLE: print("[ERR] eval package not available"); return 1
        return run_eval_mode(args)

    run_mode = args.mode if args.mode is not None else RUN_MODE
    g = build_config(live_mode=args.live, verbose=True)
    api_keys = [cfg.llm.api_key_extraction, cfg.llm.api_key_classification, cfg.llm.api_key_reviewer]
    if args.live and not all(api_keys):
        print("[ERR] --live needs 3 API keys"); return 1
    files = scan_input_files(str(g.input_dir), args.limit)
    if not files: print("[Batch] no input files"); return 1
    if run_mode == "test":
        sample_size = min(10, len(files)); random.seed(42)
        files = random.sample(files, sample_size)
        print(f"[测试模式] 随机抽取 {sample_size} 个文件进行处理")

    total_files = len(files)
    output_dir = str(g.output_dir / "entities")
    mid_data_dir = str(g.mid_data_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(mid_data_dir).mkdir(parents=True, exist_ok=True)
    if not args.no_skip:
        remaining = [fp for fp in files if not output_exists(fp, output_dir)]
        skipped = len(files) - len(remaining); files = remaining
        if skipped: print(f"[Skip] {skipped} already done, {len(files)} to go")
    if not files: print("[Done] all done"); return 0
    total = len(files); concurrency = args.concurrency
    label = "Test(10)" if run_mode == "test" else "Full"
    print(f"\n{'='*70}\n  Pipeline [{label}] — {total} files (x{concurrency})\n{'='*70}\n")
    aggregated, completed, start = [], 0, time.time(); dynamic_db = DynamicTermDB()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(process_one_file, fp, g, output_dir, mid_data_dir, i, total, dynamic_db): i for i, fp in enumerate(files, 1)}
        for fut in as_completed(futures):
            s = fut.result()
            with _summary_lock: aggregated.append(s); completed += 1
            elapsed = time.time() - start
            eta = (elapsed / completed) * (total - completed) if completed else 0
            print(f"  [Progress] {completed}/{total} | OK:{sum(1 for x in aggregated if x.get('success'))} "
                  f"FAIL:{sum(1 for x in aggregated if not x.get('success'))} | {elapsed:.0f}s ETA:{eta:.0f}s")
    elapsed = time.time() - start
    succ = [r for r in aggregated if r.get("success")]; fail = [r for r in aggregated if not r.get("success")]
    print(f"\n{'='*70}\n  Done — {len(succ)}/{len(fail)} success/fail\n{'='*70}")
    print(f"  Files: {total} | Time: {elapsed:.0f}s | Sentences: {sum(r.get('statements',0) for r in succ)}")
    print(f"  Entities: {sum(r.get('entities',0) for r in succ)}")
    db_stats = dynamic_db.get_stats()
    print(f"\n{'='*70}")
    print(f"  动态术语底库 (DynamicTermDB) 统计")
    print(f"{'='*70}")
    print(f"  累计术语: {db_stats['total_terms']}")
    print(f"  新增: {db_stats['added']}  去重跳过: {db_stats['skipped_duplicates']}")

    # 导出动态术语库
    dynamic_out = args.dynamic_out or os.path.join(str(g.output_dir), "dynamic_terms.json")
    dynamic_db.export(dynamic_out)
    print(f"  已导出: {Path(dynamic_out).resolve()}")

    print(f"{'='*70}")

    return 0 if not fail else 1

if __name__ == "__main__":
    sys.exit(main())
