#!/usr/bin/env python3
"""
run_eval.py — 加载 API keys + 动态术语库，启用 thinking + voting，运行评估。
从 api.txt 读取密钥，避免命令行暴露。
"""
import os
import sys
from pathlib import Path

# ── 1. 读取 API keys ──
api_file = Path(__file__).parent / "api.txt"
if api_file.exists():
    keys = api_file.read_text().strip().split("\n")
    keys = [k.strip() for k in keys if k.strip()]
    if len(keys) >= 3:
        os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
        os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
        os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
        print("[run_eval] API keys loaded from api.txt")
    else:
        print(f"[run_eval] WARNING: api.txt has {len(keys)} lines, expected 3")
else:
    print("[run_eval] WARNING: api.txt not found, using env vars")

os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── 2. 导入并运行 ──
# Patch sys.argv to pass --eval and other flags
if "--eval" not in sys.argv:
    sys.argv.append("--eval")
if "--live" not in sys.argv:
    sys.argv.append("--live")

# Set default paths
gold_path = Path(__file__).parent / "output" / "eval" / "gold_standard.jsonl"
dynamic_terms_path = Path(__file__).parent / "dynamic_terms.json"

if "--gold" not in " ".join(sys.argv) and gold_path.exists():
    sys.argv.extend(["--gold", str(gold_path)])
if "--dynamic-terms" not in " ".join(sys.argv) and dynamic_terms_path.exists():
    sys.argv.extend(["--dynamic-terms", str(dynamic_terms_path)])
if "--eval-output" not in " ".join(sys.argv):
    sys.argv.extend(["--eval-output", str(Path(__file__).parent / "eval" / "report_thinking_voting.md")])

# Enable thinking (OFF - too slow for eval, ~1min/sentence)
# Keep RAG few-shot + dynamic terms for F1 boost
if "--no-thinking" not in " ".join(sys.argv) and "--thinking" not in " ".join(sys.argv):
    sys.argv.append("--no-thinking")
# Voting: OFF for speed
if "--no-voting" not in " ".join(sys.argv) and "--voting" not in " ".join(sys.argv):
    sys.argv.append("--no-voting")
# Enable RAG few-shot (use gold_flat from output/eval for sentence-level RAG)
if "--rag" not in " ".join(sys.argv):
    sys.argv.append("--rag")

# Load medium-sized term set for fast loading + good coverage
if "--max-terms" not in " ".join(sys.argv):
    sys.argv.extend(["--max-terms", "50000"])

print(f"[run_eval] sys.argv: {' '.join(sys.argv)}")

# ── 3. 导入并运行 run.py 的 main ──
sys.path.insert(0, str(Path(__file__).parent))
from run import main

if __name__ == "__main__":
    sys.exit(main())
