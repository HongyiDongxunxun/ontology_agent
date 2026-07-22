"""Process a single file."""
import os, sys, json
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

from config import build_config
build_config(live_mode=True, verbose=True)

from pipeline import LLMClient, AgentPipeline, DynamicTermDB, export_jsonl
from scripts.eval_runner import load_all_gold_flat

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("base_name")
args = parser.parse_args()

bn = args.base_name
all_grouped = load_all_gold_flat("D:/code/AcademicEvaluation_exam/baseline/gold_flat")
sentences_data = all_grouped[bn]
sentences = [(sid, sentences_data[sid][0]["sentence"]) for sid in sorted(sentences_data.keys(), key=lambda x: int(x) if x.isdigit() else x)]
print(f"Processing {bn} ({len(sentences)} sentences)", flush=True)

dynamic_db = DynamicTermDB()
output_dir = str(PROJECT_ROOT / "output" / "eval")

llm_ext = LLMClient(model="deepseek-chat", api_key=keys[0], base_url="https://api.deepseek.com", temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False)
llm_cls = LLMClient(model="deepseek-chat", api_key=keys[1], base_url="https://api.deepseek.com", temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False)
llm_rev = LLMClient(model="deepseek-chat", api_key=keys[2], base_url="https://api.deepseek.com", temperature=0.0, max_tokens=8192, timeout=180, enable_thinking=False)

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
    print(f"DONE: {len(results)} entities ({valid} valid)", flush=True)
except Exception as e:
    print(f"FAIL: {e}", flush=True)
    import traceback
    traceback.print_exc()
