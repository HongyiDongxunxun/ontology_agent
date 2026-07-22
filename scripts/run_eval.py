#!/usr/bin/env python3
"""
Wrapper: reads api.txt keys, sets env vars, then runs eval pipeline directly.
Usage: python scripts/run_eval.py --limit 5 --concurrency 20 [--dry-run] [--live]
"""
import os, sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Read API keys from api.txt ──
with open(PROJECT_ROOT / "api.txt") as f:
    keys = [l.strip() for l in f if l.strip()]

os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
os.environ["LLM_MODEL"] = "deepseek-chat"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

print(f"[run_eval] API keys loaded, model={os.environ['LLM_MODEL']}")

# ── Parse args ──
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--concurrency", type=int, default=20)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--live", action="store_true")
args = parser.parse_args()

# ── Import and run ──
from config import build_config
build_config(live_mode=args.live, verbose=True)

from scripts.eval_runner import (
    load_all_gold_flat, run_evaluation
)

gold_dir = "D:/code/AcademicEvaluation_exam/baseline/gold_flat"
output_dir = str(PROJECT_ROOT / "output" / "eval")

# Load data
print("Loading gold_flat data...")
grouped = load_all_gold_flat(gold_dir)
print(f"Loaded {len(grouped)} files")

if args.limit > 0:
    keys_sorted = sorted(grouped.keys())[:args.limit]
    grouped = {k: grouped[k] for k in keys_sorted}
    print(f"Limited to {len(grouped)} files")

# Run
result = run_evaluation(
    grouped_data=grouped,
    output_dir=output_dir,
    concurrency=args.concurrency,
    limit_files=0,
    dry_run=args.dry_run,
)

if "error" in result:
    print(f"ERROR: {result['error']}")
    sys.exit(1)
else:
    print("Evaluation complete!")
    if "metrics" in result:
        m = result["metrics"]
        print(f"Extraction F1: {m['extraction']['f1']:.2%}")
        print(f"Strict F1: {m['strict']['f1']:.2%}")
        print(f"L3 Accuracy: {m['classification']['l3_accuracy']:.2%}")
    sys.exit(0)
