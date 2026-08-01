#!/usr/bin/env python3
"""V4.4 合并 Agent 1 测试 — 验证实体抽取 + 评价关系识别"""
import json, os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载 API keys
keys = [k.strip() for k in open("api.txt").read().strip().split("\n") if k.strip()]
os.environ["DEEPSEEK_API_KEY"] = keys[0]
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
print(f"[Setup] Loaded {len(keys)} API keys from api.txt")

from pipeline import LLMClient, EntityExtractionAgent

agent = EntityExtractionAgent(LLMClient(
    model="deepseek-chat", api_key=keys[0],
    base_url="https://api.deepseek.com",
    temperature=0.0, max_tokens=4096,
))

# 加载测试数据
with open("input/reviewed_full_9.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 取前 6 句测试
test_items = list(data.items())[:6]
print(f"\n{'='*70}")
print(f"V4.4 Agent 1 合并测试 — {len(test_items)} 句")
print(f"{'='*70}\n")

total_entities = 0
total_relations = 0
sentences_with_eval = 0
results_store = []

t_start = time.time()
for i, (sid, entry) in enumerate(test_items):
    full_text = f"{entry.get('previous_sentence','')} {entry.get('evaluative_sentence','')} {entry.get('next_sentence','')}".strip()
    result = agent.extract(full_text, sentence_id=sid)
    results_store.append(result)

    n_e, n_r = len(result.entities), len(result.relations)
    total_entities += n_e
    total_relations += n_r
    if result.has_evaluation:
        sentences_with_eval += 1

    print(f"--- 句 {i+1} (ID={sid}) ---")
    print(f"  文本: {entry.get('evaluative_sentence','')[:100]}...")
    print(f"  实体: {n_e} | 评价关系: {n_r} | has_evaluation: {result.has_evaluation}")
    for e in result.entities:
        print(f"    [{e.entity_id}] {e.mention} → L1={e.candidate_l1} L3={e.candidate_l3}")
    for r in result.relations:
        xtra = f" obj_text={r.object_text}" if r.object == "_missing_entity" else ""
        print(f"    REL: {r.subject} --[{r.polarity}]--> {r.object} | evid=\"{r.evidence[:50]}\"{xtra}")
    if not result.relations:
        print(f"    (无评价关系)")
    print()

elapsed = time.time() - t_start

print(f"{'='*70}")
print(f"汇总: {total_entities} 实体, {total_relations} 关系, "
      f"{sentences_with_eval}/{len(test_items)} 句有评价")
print(f"耗时: {elapsed:.1f}s (平均 {elapsed/len(test_items):.1f}s/句)")
print(f"{'='*70}")

# 验证: relation object 是否引用了有效 entity_id
print("\n🔍 验证 relation-object 引用有效性:")
all_ok = True
for i, result in enumerate(results_store):
    eids = {e.entity_id for e in result.entities}
    for r in result.relations:
        if r.object == "_missing_entity":
            print(f"  句{i+1}: object=_missing_entity (object_text={r.object_text}) — OK")
        elif r.object not in eids:
            print(f"  ❌ 句{i+1}: object={r.object} 不是有效 entity_id! 有效ID: {eids}")
            all_ok = False
        elif r.subject not in eids and r.subject not in ("_paper_author", "_unknown") and not r.subject.startswith("_cite["):
            print(f"  ⚠️ 句{i+1}: subject={r.subject} 类型未知 (非entity_id/特殊标记)")
        else:
            print(f"  句{i+1}: {r.subject} → {r.object} — ✅ 有效引用")

if all_ok:
    print("\n✅ 所有 relation object 引用有效!")
else:
    print("\n❌ 存在无效引用, 需要修复")

print("\n✅ V4.4 Agent 1 合并测试完成!")
