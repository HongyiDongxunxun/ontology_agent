#!/usr/bin/env python3
"""V4.4 端到端测试 — DualAgentPipeline 完整流程 (Agent 1+2+3)"""
import json, os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载 API keys
keys = [k.strip() for k in open("api.txt").read().strip().split("\n") if k.strip()]
os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

from pipeline import LLMClient, DualAgentPipeline, DynamicTermDB, export_jsonl, export_summary_json, export_relation_jsonl
from pathlib import Path

# 构建 3 个 LLM 客户端
llm_ext = LLMClient(model="deepseek-chat", api_key=keys[0],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)
llm_cls = LLMClient(model="deepseek-chat", api_key=keys[1],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)
llm_rev = LLMClient(model="deepseek-chat", api_key=keys[2],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)

# 加载测试数据, 取前 3 句
with open("input/reviewed_full_9.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_items = list(data.items())[:3]
sentences = []
for sid, entry in test_items:
    full = f"{entry.get('previous_sentence','')} {entry.get('evaluative_sentence','')} {entry.get('next_sentence','')}".strip()
    sentences.append((sid, full))

print(f"{'='*70}")
print(f"V4.4 端到端测试 — DualAgentPipeline (Agent 1+2+3) — {len(sentences)} 句")
print(f"{'='*70}\n")

# 运行管道
pipeline = DualAgentPipeline(
    llm_extraction=llm_ext, llm_classification=llm_cls, llm_reviewer=llm_rev,
    dynamic_term_db=DynamicTermDB(), batch_size=12, mid_data_dir="mid_data", verbose=True
)

t_start = time.time()
results, relations = pipeline.run(sentences, "test_e2e")
elapsed = time.time() - t_start

# 输出结果
print(f"\n{'='*70}")
print(f"端到端完成! 耗时: {elapsed:.0f}s")
print(f"{'='*70}")
print(f"实体结果: {len(results)} 条")
valid = sum(1 for r in results if r.valid_entity)
invalid = sum(1 for r in results if not r.valid_entity)
print(f"  有效: {valid}  无效: {invalid}")
print(f"  Likert 平均分: {sum(r.likert_confidence for r in results if r.likert_confidence > 0) / max(sum(1 for r in results if r.likert_confidence > 0), 1):.1f}")
print(f"评价关系: {len(relations)} 条")

print(f"\n--- 实体分类结果 ---")
for r in results[:15]:
    print(f"  [{r.sentence_id}][{r.entity_id}] {r.entity} → {r.l1}/{r.l2}/{r.l3_type_code} | valid={r.valid_entity} | Likert={r.likert_confidence}")

print(f"\n--- 评价关系 (来自 Agent 1) ---")
for r in relations:
    xtra = f" obj_text={r.get('object_text','')}" if r.get('object') == '_missing_entity' else ''
    print(f"  [{r['sentence_id']}] {r['subject']} --[{r['polarity']}]--> {r['object']} | \"{r['evidence'][:60]}\"{xtra}")

# 验证关系引用
print(f"\n🔍 验证:")
entity_ids_by_sentence = {}
for r in results:
    sid = r.sentence_id
    if sid not in entity_ids_by_sentence:
        entity_ids_by_sentence[sid] = set()
    entity_ids_by_sentence[sid].add(r.entity_id)

all_valid = True
for rel in relations:
    sid = rel['sentence_id']
    obj = rel['object']
    eids = entity_ids_by_sentence.get(sid, set())
    if obj == '_missing_entity':
        print(f"  ✅ [{sid}] _missing_entity → object_text={rel.get('object_text','')}")
    elif obj in eids:
        print(f"  ✅ [{sid}] {rel['subject']} → {obj} (有效)")
    else:
        print(f"  ❌ [{sid}] {rel['subject']} → {obj} 不在有效 entity_id 集合 {eids} 中!")
        all_valid = False

if all_valid:
    print(f"\n✅ 所有评价关系引用有效! V4.4 端到端测试通过!")
else:
    print(f"\n❌ 存在无效引用")

# 导出示意
export_jsonl(results, "output/test_e2e_result.jsonl")
print(f"\n已导出: output/test_e2e_result.jsonl")
