#!/usr/bin/env python3
"""run_eval.py — Load API keys from api.txt, run evaluation with RAG+terms"""
import os, sys
from pathlib import Path

# Load API keys
api_file = Path(__file__).parent / "api.txt"
if api_file.exists():
    keys = [k.strip() for k in api_file.read_text().strip().split("\n") if k.strip()]
    if len(keys) >= 3:
        os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
        os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
        os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
        print("[run_eval] API keys loaded from api.txt")
os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Patch argv
if "--eval" not in sys.argv: sys.argv.append("--eval")
if "--live" not in sys.argv: sys.argv.append("--live")

gold_path = Path(__file__).parent / "output" / "eval" / "gold_standard.jsonl"
terms_path = Path(__file__).parent / "dynamic_terms.json"

if "--gold" not in " ".join(sys.argv) and gold_path.exists():
    sys.argv.extend(["--gold", str(gold_path)])
if "--dynamic-terms" not in " ".join(sys.argv) and terms_path.exists():
    sys.argv.extend(["--dynamic-terms", str(terms_path)])
else:
    # Try ontology_agent_2's dynamic_terms.json
    alt_terms = Path("D:/code/AcademicEvaluation/ontology_agent_2/dynamic_terms.json")
    if alt_terms.exists():
        sys.argv.extend(["--dynamic-terms", str(alt_terms)])

if "--eval-output" not in " ".join(sys.argv):
    sys.argv.extend(["--eval-output", str(Path(__file__).parent / "eval" / "report_latest.md")])

# Enable RAG (no thinking/voting for speed)
if "--rag" not in " ".join(sys.argv): sys.argv.append("--rag")
if "--max-terms" not in " ".join(sys.argv): sys.argv.extend(["--max-terms", "50000"])

print(f"[run_eval] argv: {' '.join(sys.argv)}")

sys.path.insert(0, str(Path(__file__).parent))
from run import main
if __name__ == "__main__":
    sys.exit(main())
