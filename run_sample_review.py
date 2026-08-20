#!/usr/bin/env python3
"""随机采样 50 句运行完整四Agent管道, 输出人工审阅报告"""
import json, os, sys, io, random, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

keys = [k.strip() for k in open("api.txt").read().strip().split("\n") if k.strip()]
os.environ["DEEPSEEK_API_KEY_EXTRACTION"] = keys[0]
os.environ["DEEPSEEK_API_KEY_CLASSIFICATION"] = keys[1]
os.environ["DEEPSEEK_API_KEY_REVIEWER"] = keys[2]
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

from pipeline import LLMClient, DualAgentPipeline, DynamicTermDB, export_jsonl, export_relation_jsonl

SAMPLE_SIZE = 50
SEED = 42

# ── 加载全部句子 ──
import glob, re

def reviewed_key_sort(key: str) -> tuple:
    """与 run.py 一致的键排序 (纯数字在前, non_X 在后)"""
    if key.isdigit():
        return (0, int(key), key)
    m = re.search(r"(\d+)$", key)
    if m:
        return (1, int(m.group(1)), key)
    return (2, 0, key)

all_sentences = []
for f in sorted(glob.glob("input/reviewed_full_*.json")):
    data = json.load(open(f, "r", encoding="utf-8"))
    base = Path(f).stem  # e.g. reviewed_full_9
    for key in sorted(data.keys(), key=reviewed_key_sort):
        entry = data[key]
        full = f"{entry.get('previous_sentence','')} {entry.get('evaluative_sentence','')} {entry.get('next_sentence','')}".strip()
        # 用全局唯一 sentence_id
        all_sentences.append((f"{base}::{key}", full))

print(f"[Data] 总句数: {len(all_sentences)}, 采样 {SAMPLE_SIZE} 句 (seed={SEED})")
random.seed(SEED)
sampled = random.sample(all_sentences, min(SAMPLE_SIZE, len(all_sentences)))

# ── 构建管道 ──
llm_ext = LLMClient(model="deepseek-chat", api_key=keys[0],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)
llm_cls = LLMClient(model="deepseek-chat", api_key=keys[1],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)
llm_rev = LLMClient(model="deepseek-chat", api_key=keys[2],
    base_url="https://api.deepseek.com", temperature=0.0, max_tokens=4096)

pipeline = DualAgentPipeline(
    llm_extraction=llm_ext, llm_classification=llm_cls, llm_reviewer=llm_rev,
    dynamic_term_db=DynamicTermDB(), batch_size=12,
    mid_data_dir="mid_data", verbose=False)

t0 = time.time()
results, relations = pipeline.run(sampled, "sample_review_50")
elapsed = time.time() - t0

# ── 输出机器可读文件 ──
export_jsonl(results, "output/sample_review_50_result.jsonl")
if relations:
    export_relation_jsonl(relations, "output/sample_review_50_relations.jsonl")

# ── 生成人工审阅 Markdown 报告 ──
# 按句分组
entities_by_sid = {}
for r in results:
    entities_by_sid.setdefault(r.sentence_id, []).append(r)

relations_by_sid = {}
for rel in relations:
    relations_by_sid.setdefault(rel["sentence_id"], []).append(rel)

lines = []
lines.append("# 四Agent管道人工审阅报告 — 随机50句")
lines.append("")
lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 采样: {len(sampled)} 句 (seed={SEED}) | 耗时: {elapsed:.0f}s")
lines.append(f"> 实体总数: {len(results)} | 评价关系总数: {len(relations)}")
lines.append("")
lines.append("## 图例")
lines.append("")
lines.append("- **实体**: `[实体ID] 实体名 → L1/L2/L3 | Likert分 | 审查意见`")
lines.append("- **关系**: `评价主体 → 客体(实体名) | aspect=评价方面 | opinion=评价表达`")
lines.append("")

for i, (sid, stmt) in enumerate(sampled, 1):
    ents = entities_by_sid.get(sid, [])
    rels = relations_by_sid.get(sid, [])
    has_eval = pipeline.last_has_evaluation.get(sid, False)

    lines.append(f"---")
    lines.append(f"## [{i}] {sid}")
    lines.append("")
    lines.append(f"**原文:** {stmt[:300]}")
    lines.append("")
    lines.append(f"**has_evaluation: {has_eval}** | 实体 {len(ents)} 个 | 关系 {len(rels)} 条")
    lines.append("")

    # 实体表
    id_to_name = {}
    if ents:
        lines.append("| 实体ID | 实体 | L1/L2/L3 | 有效 | Likert | 审查意见 |")
        lines.append("|--------|------|----------|------|--------|----------|")
        for e in ents:
            id_to_name[e.entity_id] = e.entity
            l = f"{e.l1}/{e.l2}/{e.l3_type_code}" if e.valid_entity else f"无效({e.invalid_reason[:20]})"
            lines.append(
                f"| {e.entity_id} | {e.entity} | {l} | {'✓' if e.valid_entity else '✗'} "
                f"| {e.likert_confidence} | {e.reviewer_comment[:60]} |")
        lines.append("")

    # 关系
    if rels:
        lines.append("**评价关系:**")
        lines.append("")
        for rel in rels:
            obj_name = id_to_name.get(rel["object"], rel.get("object_text", rel["object"]))
            aspect = rel.get("aspect") or "-"
            lines.append(
                f"- {rel['subject']} → **{obj_name}** | aspect={aspect} | "
                f"opinion=\"{rel['opinion']}\" | evid=\"{rel['evidence'][:50]}\"")
        lines.append("")

report = "\n".join(lines)
out_path = "output/sample_review_50_report.md"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

# ── 终端摘要 ──
print(f"\n[Done] 耗时 {elapsed:.0f}s | {len(results)} 实体 | {len(relations)} 关系")
print(f"[Done] 人工审阅报告: {out_path}")
print(f"[Done] 实体JSONL: output/sample_review_50_result.jsonl")
print(f"[Done] 关系JSONL: output/sample_review_50_relations.jsonl")
print(f"[Done] TermDB 术语数: {pipeline.dynamic_term_db.size()}")
