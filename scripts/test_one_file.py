#!/usr/bin/env python3
"""Quick test: Run 1 file through the pipeline and evaluate."""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "api.txt") as f:
    keys = [l.strip() for l in f if l.strip()]
os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
os.environ["LLM_MODEL"] = "deepseek-chat"

from scripts.eval_runner import load_all_gold_flat, run_evaluation

grouped = load_all_gold_flat("D:/code/AcademicEvaluation_exam/baseline/gold_flat")
ks = sorted(grouped.keys())[:1]
grouped = {k: grouped[k] for k in ks}

print("Running live eval on 1 file...")
result = run_evaluation(grouped, str(PROJECT_ROOT / "output" / "eval"), concurrency=1, dry_run=False)
print("Done!")
if "metrics" in result:
    m = result["metrics"]
    print(f"Extraction F1: {m['extraction']['f1']}")
    print(f"Strict F1: {m['strict']['f1']}")
    print(f"L3 Accuracy: {m['classification']['l3_accuracy']}")
