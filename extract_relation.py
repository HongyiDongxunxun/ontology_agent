#!/usr/bin/env python3
"""
Run evaluation relation extraction over entity JSONL results.

Input:
    output/entities/reviewed_full_{num}_result.jsonl

Output:
    output/evaluative_relation/relation_full_{num}.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import config as cfg, build_config
from pipeline import EvaluativeRelationAgent, LLMClient, SentenceRelationOutput


_print_lock = Lock()


def _num_key(path: Path) -> int:
    match = re.search(r"reviewed_full_(\d+)_result\.jsonl$", path.name)
    return int(match.group(1)) if match else 0


def scan_entity_result_files(input_dir: str, limit: int = 0) -> list[Path]:
    dirpath = Path(input_dir)
    if not dirpath.is_dir():
        return []
    files = sorted(dirpath.glob("reviewed_full_*_result.jsonl"), key=_num_key)
    if limit and limit > 0:
        files = files[:limit]
    return files


def relation_output_path(entity_file: Path, output_dir: str) -> Path:
    match = re.search(r"reviewed_full_(\d+)_result\.jsonl$", entity_file.name)
    if not match:
        return Path(output_dir) / f"{entity_file.stem}_relations.jsonl"
    return Path(output_dir) / f"relation_full_{match.group(1)}.jsonl"


def load_entity_jsonl(filepath: str) -> list[tuple[str, str, list[dict]]]:
    grouped: dict[str, dict] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sentence_id = str(item.get("sentence_id", "")).strip()
            if not sentence_id:
                continue
            if sentence_id not in grouped:
                grouped[sentence_id] = {
                    "sentence": item.get("sentence", ""),
                    "entities": [],
                }
            grouped[sentence_id]["entities"].append(item)

    def sentence_key(value: str) -> tuple[int, str]:
        return (int(value), value) if value.isdigit() else (10**9, value)

    return [
        (sid, grouped[sid]["sentence"], grouped[sid]["entities"])
        for sid in sorted(grouped.keys(), key=sentence_key)
    ]


def export_relation_jsonl(results: list[SentenceRelationOutput], filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    print(f"[export] relation JSONL -> {path.resolve()} ({len(results)} lines)")


def process_one_file(
    fpath: Path,
    output_dir: str,
    idx: int,
    total: int,
    no_skip: bool,
) -> dict:
    out_path = relation_output_path(fpath, output_dir)
    if out_path.exists() and not no_skip:
        with _print_lock:
            print(f"[{idx}/{total}] {fpath.name} skipped: {out_path.name} exists")
        return {"file": fpath.name, "success": True, "skipped": True}

    with _print_lock:
        print(f"[{idx}/{total}] {fpath.name} relation extraction started")

    try:
        llm = LLMClient(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key_reviewer or cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
        )
        agent = EvaluativeRelationAgent(llm)

        sentences = load_entity_jsonl(str(fpath))
        results: list[SentenceRelationOutput] = []
        total_sentences = len(sentences)
        for i, (sentence_id, sentence, entities) in enumerate(sentences, 1):
            results.append(agent.extract(sentence_id, sentence, entities))
            if i % 10 == 0 or i == total_sentences:
                with _print_lock:
                    print(f"  [{fpath.stem}] {i}/{total_sentences} sentences")

        export_relation_jsonl(results, str(out_path))

        relation_count = sum(len(result.relations) for result in results)
        evaluated_sentences = sum(1 for result in results if result.has_evaluation)
        with _print_lock:
            print(
                f"[{idx}/{total}] {fpath.name} OK: "
                f"{evaluated_sentences}/{total_sentences} sentences, "
                f"{relation_count} relations"
            )
        return {
            "file": fpath.name,
            "success": True,
            "sentences": total_sentences,
            "evaluated_sentences": evaluated_sentences,
            "relations": relation_count,
        }
    except Exception as exc:
        with _print_lock:
            print(f"[{idx}/{total}] {fpath.name} FAIL: {exc}")
        return {"file": fpath.name, "success": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract evaluation relations from entity JSONL files")
    parser.add_argument("--live", action="store_true", help="enable real LLM API mode")
    parser.add_argument("--limit", type=int, default=0, help="limit number of files (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel file count")
    parser.add_argument("--no-skip", action="store_true", help="overwrite existing relation outputs")
    parser.add_argument("--input-dir", type=str, default="", help="entity JSONL directory")
    parser.add_argument("--output-dir", type=str, default="", help="relation JSONL directory")
    args = parser.parse_args()

    build_config(live_mode=args.live, verbose=True)
    if args.live and not (cfg.llm.api_key_reviewer or cfg.llm.api_key):
        print("[error] --live requires DEEPSEEK_API_KEY_REVIEWER or DEEPSEEK_API_KEY")
        return 1

    root_output = cfg.output_dir
    input_dir = args.input_dir or str(root_output / "entities")
    output_dir = args.output_dir or str(root_output / "evaluative_relation")

    files = scan_entity_result_files(input_dir, args.limit)
    if not files and not args.input_dir:
        legacy_input_dir = str(root_output)
        files = scan_entity_result_files(legacy_input_dir, args.limit)
        if files:
            print(f"[compat] using legacy entity result directory: {legacy_input_dir}")
            input_dir = legacy_input_dir

    if not files:
        print(f"[batch] no reviewed_full_*_result.jsonl files found in {input_dir}")
        return 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total = len(files)
    start = time.time()
    print("=" * 70)
    print(f"Evaluation relation extraction: {total} files, concurrency {args.concurrency}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    aggregated: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_one_file, fpath, output_dir, i, total, args.no_skip): fpath
            for i, fpath in enumerate(files, 1)
        }
        for future in as_completed(futures):
            aggregated.append(future.result())

    successes = [item for item in aggregated if item.get("success")]
    failures = [item for item in aggregated if not item.get("success")]
    elapsed = time.time() - start
    print("=" * 70)
    print(f"Done in {elapsed:.0f}s. success={len(successes)} failure={len(failures)}")
    print(f"Sentences: {sum(item.get('sentences', 0) for item in successes)}")
    print(f"Evaluated sentences: {sum(item.get('evaluated_sentences', 0) for item in successes)}")
    print(f"Relations: {sum(item.get('relations', 0) for item in successes)}")
    if failures:
        print("Failures:")
        for item in failures[:20]:
            print(f"  {item['file']}: {item.get('error', '?')}")
    print("=" * 70)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
