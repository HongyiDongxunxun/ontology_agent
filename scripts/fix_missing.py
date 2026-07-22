"""Process the 3 missing files that failed due to API balance."""
import os, sys, json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Read API keys
with open(PROJECT_ROOT / "api.txt") as f:
    keys = [l.strip() for l in f if l.strip()]
os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
os.environ["LLM_MODEL"] = "deepseek-chat"

from config import build_config
build_config(live_mode=True, verbose=True)

from pipeline import LLMClient, AgentPipeline, DynamicTermDB, export_jsonl

# Missing files
MISSING = ["reviewed_full_8888", "reviewed_full_920", "reviewed_full_997"]

# Load gold_flat data for these files
from scripts.eval_runner import load_all_gold_flat
all_grouped = load_all_gold_flat("D:/code/AcademicEvaluation_exam/baseline/gold_flat")
grouped = {k: v for k, v in all_grouped.items() if k in MISSING}

print(f"Processing {len(grouped)} missing files...")

output_dir = str(PROJECT_ROOT / "output" / "eval")
dynamic_db = DynamicTermDB()

for bn in sorted(grouped.keys()):
    sentences_data = grouped[bn]
    sentences = [(sid, sentences_data[sid][0]["sentence"]) for sid in sorted(sentences_data.keys(), key=lambda x: int(x) if x.isdigit() else x)]

    print(f"\nProcessing {bn} ({len(sentences)} sentences)...")

    llm_ext = LLMClient(
        model="deepseek-chat", api_key=keys[0], base_url="https://api.deepseek.com",
        temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False,
    )
    llm_cls = LLMClient(
        model="deepseek-chat", api_key=keys[1], base_url="https://api.deepseek.com",
        temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False,
    )
    llm_rev = LLMClient(
        model="deepseek-chat", api_key=keys[2], base_url="https://api.deepseek.com",
        temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False,
    )

    pipeline = AgentPipeline(
        llm_extraction=llm_ext, llm_classification=llm_cls, llm_reviewer=llm_rev,
        dynamic_term_db=dynamic_db, batch_size=12,
        mid_data_dir=str(output_dir + "/mid_data/" + bn),
        verbose=False,
    )

    try:
        results = pipeline.run(sentences, bn)
        jsonl_path = Path(output_dir) / f"{bn}_result.jsonl"
        export_jsonl(results, str(jsonl_path))
        valid = sum(1 for r in results if r.valid_entity)
        print(f"  OK: {len(results)} entities ({valid} valid)")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
