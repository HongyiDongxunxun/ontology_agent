"""
pipeline.dual_agent_pipeline — 三Agent主管道编排
Agent 1 抽取 → 写 mid_data/ → Agent 2 分类 → Agent 3 审查(Likert) → 写最终 JSONL
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from .llm import LLMClient
from .entity_extraction_agent import (
    EntityExtractionAgent,
    ExtractedEntity,
    SentenceExtractionOutput,
)
from .classification_agent import (
    ClassificationAgent,
    FinalEntityResult,
)
from .reviewer_agent import (
    ReviewerAgent,
    ReviewResult,
)
from .evaluative_relation_agent import (
    EvaluativeRelationAgent,
    _normalize_name,
)
from .dynamic_term_db import DynamicTermDB


class DualAgentPipeline:
    """四Agent管道: Agent 1 评价关系 → Agent 2 实体抽取 → Agent 3 分类 → Agent 4 审查(Likert)"""

    def __init__(
        self,
        llm_extraction: Optional[LLMClient] = None,
        llm_classification: Optional[LLMClient] = None,
        llm_reviewer: Optional[LLMClient] = None,
        llm_relation: Optional[LLMClient] = None,
        dynamic_term_db: Optional[DynamicTermDB] = None,
        batch_size: int = 12,
        mid_data_dir: str = "",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.llm_extraction = llm_extraction or LLMClient()
        self.llm_classification = llm_classification or LLMClient()
        self.llm_reviewer = llm_reviewer or LLMClient()
        self.dynamic_term_db = dynamic_term_db
        self.batch_size = batch_size
        self.mid_data_dir = mid_data_dir

        self.evaluative_relation_agent = EvaluativeRelationAgent(
            llm_relation or self.llm_extraction
        )
        self.extraction_agent = EntityExtractionAgent(self.llm_extraction)
        self.classification_agent = ClassificationAgent(
            llm=self.llm_classification,
            dynamic_term_db=self.dynamic_term_db,
            batch_size=self.batch_size,
        )
        self.reviewer_agent = ReviewerAgent(
            llm=self.llm_reviewer,
            batch_size=self.batch_size,
        )

    def run(
        self, sentences: list[tuple[str, str]], base_name: str
    ) -> tuple[list[FinalEntityResult], list[dict]]:
        """
        执行四Agent管道:
        - Agent 1: 评价关系抽取 (EvaluativeRelationAgent)
        - Agent 2: 实体抽取补充 (EntityExtractionAgent, 接收上游已知实体)
        - Agent 3: 分类 (ClassificationAgent)
        - Agent 4: 审查 Likert (ReviewerAgent)
        - sentences: list of (sentence_id, sentence_text)
        - base_name: 文件基础名 (如 "reviewed_full_1")
        - 返回: (所有 FinalEntityResult, 评价关系列表)
        """
        self._log(f"[Pipeline] 开始处理 {base_name}, 共 {len(sentences)} 句")

        all_extractions: list[SentenceExtractionOutput] = []
        all_relations: list[dict] = []
        relations_by_sid: dict[str, list[dict]] = {}
        # 每句的 has_evaluation 判定 (含情形B: 有评价但无合法关系), 供评估使用
        self.last_has_evaluation: dict[str, bool] = {}

        total_s = len(sentences)
        total_entities = 0
        total_relations_count = 0
        unresolved_count = 0

        # ── Agent 1: 评价关系抽取 + Agent 2: 实体抽取补充 ──
        for idx, (sid, stmt) in enumerate(sentences, 1):
            # Agent 1: 评价关系抽取
            rel_output = self.evaluative_relation_agent.extract(sid, stmt)
            self.last_has_evaluation[sid] = rel_output.has_evaluation

            # Agent 2: 实体抽取（补充，接收上游已知实体）
            known_entities = rel_output.entities
            extraction = self.extraction_agent.extract(
                stmt, sentence_id=sid, known_entities=known_entities
            )

            # ── 关系 object ID 重映射: Agent 1 短ID → Agent 2 完整ID ──
            # Agent 1 的 entities 用短ID (e1/e2), relations 引用短ID;
            # Agent 2 的 entities 用完整ID ({sid}_eN)。此处按实体名匹配回填。
            rel_entities_by_id = {
                str(e.get("entity_id", "")): e for e in rel_output.entities
            }
            for rel in rel_output.relations:
                rel_dict = {"sentence_id": sid, "sentence": stmt, **rel.to_dict()}
                obj = rel.object
                if obj != "_missing_entity" and obj in rel_entities_by_id:
                    rel_ent = rel_entities_by_id[obj]
                    name = rel_ent.get("entity") or rel_ent.get("normalized_name") or ""
                    matched_eid = self._find_entity_id(
                        extraction.entities, name, rel_ent.get("normalized_name", "")
                    )
                    if matched_eid:
                        rel_dict["object"] = matched_eid
                    else:
                        # 回抽保障: Agent 2 漏抽了关系引用的实体 → 程序化补入
                        new_eid = f"{sid}_e{len(extraction.entities) + 1}"
                        extraction.entities.append(ExtractedEntity(
                            entity_id=new_eid,
                            mention=name,
                            normalized_name=rel_ent.get("normalized_name") or name,
                            evidence=rel_ent.get("evidence", ""),
                            candidate_l3="",
                            candidate_l1=[],
                            is_specific_entity=True,
                            confidence=0.0,
                            uncertainty="来自评价关系抽取的补抽实体",
                        ))
                        rel_dict["object"] = new_eid
                elif obj != "_missing_entity":
                    # LLM 引用了不存在的实体ID → 降级为 _missing_entity
                    rel_dict["object"] = "_missing_entity"
                    rel_dict["object_text"] = rel_dict.get("object_text") or obj
                    unresolved_count += 1
                # obj == "_missing_entity" 时保留 LLM 提供的 object_text, 原样透传
                all_relations.append(rel_dict)
                relations_by_sid.setdefault(sid, []).append(rel_dict)
            total_relations_count += len(rel_output.relations)

            all_extractions.append(extraction)
            total_entities += len(extraction.entities)

            if idx % 5 == 0 or idx == total_s:
                print(f"  [{base_name}] 抽取 [{idx}/{total_s}] 句, "
                      f"累计 {total_entities} 实体, {total_relations_count} 评价关系")

        if unresolved_count:
            print(f"  [{base_name}] 警告: {unresolved_count} 条关系无法解析 object, 已降级为 _missing_entity")

        # ── 写中间结果到 mid_data/ ──
        if self.mid_data_dir:
            mid_path = Path(self.mid_data_dir)
            mid_path.mkdir(parents=True, exist_ok=True)
            mid_file = mid_path / f"{base_name}_extracted.json"
            mid_data = {
                "base_name": base_name,
                "stage": "agent1_relation_agent2_entity_extraction",
                "extractions": [
                    {
                        **e.to_dict(),
                        "has_evaluation": bool(relations_by_sid.get(e.sentence_id)),
                        "relations": relations_by_sid.get(e.sentence_id, []),
                    }
                    for e in all_extractions
                ],
            }
            with open(mid_file, "w", encoding="utf-8") as f:
                json.dump(mid_data, f, ensure_ascii=False, indent=2)
            print(f"  [{base_name}] 中间结果已写入: mid_data/{base_name}_extracted.json")

        # ── Agent 3: Classification ──
        print(f"  [{base_name}] Agent 3 分类中...")
        all_results: list[FinalEntityResult] = []
        total_ex = len(all_extractions)
        for i, extraction in enumerate(all_extractions, 1):
            results = self.classification_agent.classify(extraction)
            all_results.extend(results)
            if i % 5 == 0 or i == total_ex:
                print(f"  [{base_name}] 分类 [{i}/{total_ex}] 句, 累计 {len(all_results)} 条结果")

        valid_count = sum(1 for r in all_results if r.valid_entity)
        invalid_count = sum(1 for r in all_results if not r.valid_entity)
        print(f"\n[Agent 3] 完成: {valid_count} 有效, {invalid_count} 无效")

        # ── Agent 4: Library Science Review + Likert ──
        print(f"  [{base_name}] Agent 4 审查中...")
        for i, extraction in enumerate(all_extractions, 1):
            sentence_results = [
                r for r in all_results if r.sentence_id == extraction.sentence_id
            ]
            if not sentence_results:
                continue
            reviews = self.reviewer_agent.review(extraction.sentence, sentence_results)
            for review in reviews:
                for result in sentence_results:
                    if result.entity_id == review.entity_id:
                        result.likert_confidence = review.likert_score
                        result.reviewer_comment = review.reviewer_comment
                        result.suggested_correction = review.suggested_correction
                        break
            if i % 5 == 0 or i == total_ex:
                print(f"  [{base_name}] 审查 [{i}/{total_ex}] 句")

        scored = sum(1 for r in all_results if r.likert_confidence > 0)
        avg_score = (
            sum(r.likert_confidence for r in all_results if r.likert_confidence > 0) / scored
            if scored > 0 else 0
        )
        print(f"  [{base_name}] Agent 4 完成: {scored}/{len(all_results)} 条已评分, 平均 Likert: {avg_score:.2f}")

        # ── 动态术语底库: 收集 Likert 5 高置信实体 ──
        if self.dynamic_term_db:
            new_terms = 0
            for r in all_results:
                if r.likert_confidence == 5 and r.valid_entity:
                    added = self.dynamic_term_db.add(r.entity, r.l3_type_code)
                    if added:
                        new_terms += 1
            if new_terms > 0:
                print(f"  [{base_name}] TermDB +{new_terms} 新术语 (总计 {self.dynamic_term_db.size()})")

        return all_results, all_relations

    @staticmethod
    def _find_entity_id(
        entities: list[ExtractedEntity],
        name: str,
        normalized_name: str = "",
    ) -> str:
        """在 Agent 2 实体列表中按名字查找实体ID (精确 → 包含匹配)"""
        target = _normalize_name(name)
        alt = _normalize_name(normalized_name)
        for e in entities:
            if _normalize_name(e.mention) in (target, alt) and target:
                return e.entity_id
            if _normalize_name(e.normalized_name) in (target, alt) and target:
                return e.entity_id
        # 包含匹配: 任一方向互为子串
        for e in entities:
            en = _normalize_name(e.mention)
            if target and (target in en or en in target):
                return e.entity_id
        return ""

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ===========================================================================
# Output Helpers
# ===========================================================================


def export_jsonl(
    results: list[FinalEntityResult],
    filepath: str,
) -> None:
    """导出最终 JSONL 结果"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    print(f"[导出] JSONL -> {path.resolve()}  ({len(results)} 行)")


def export_summary_json(
    results: list[FinalEntityResult],
    base_name: str,
    filepath: str,
) -> None:
    """导出汇总统计 JSON"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    type_dist: dict[str, int] = {}
    l1_dist: dict[str, int] = {}
    for r in results:
        if r.valid_entity:
            type_dist[r.l3_type_code] = type_dist.get(r.l3_type_code, 0) + 1
            l1_dist[r.l1] = l1_dist.get(r.l1, 0) + 1

    summary = {
        "base_name": base_name,
        "total_entities": len(results),
        "valid_entities": sum(1 for r in results if r.valid_entity),
        "invalid_entities": sum(1 for r in results if not r.valid_entity),
        "l1_distribution": dict(
            sorted(l1_dist.items(), key=lambda x: -x[1])
        ),
        "l3_type_distribution": dict(
            sorted(type_dist.items(), key=lambda x: -x[1])
        ),
        "likert_distribution": {
            "1_完全不认同": sum(1 for r in results if r.likert_confidence == 1),
            "2_不认同": sum(1 for r in results if r.likert_confidence == 2),
            "3_不确定": sum(1 for r in results if r.likert_confidence == 3),
            "4_认同": sum(1 for r in results if r.likert_confidence == 4),
            "5_完全认同": sum(1 for r in results if r.likert_confidence == 5),
        },
        "likert_average": round(
            sum(r.likert_confidence for r in results if r.likert_confidence > 0)
            / max(sum(1 for r in results if r.likert_confidence > 0), 1),
            2,
        ),
        "dynamic_term_db_size": len(results) if results else 0,  # placeholder, updated in run.py
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[导出] 汇总 -> {path.resolve()}")


def export_relation_jsonl(
    relations: list[dict],
    filepath: str,
) -> None:
    """导出评价关系 JSONL (V4.4: 来自 Agent 1)"""
    if not relations:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rel in relations:
            f.write(json.dumps(rel, ensure_ascii=False) + "\n")
    print(f"[导出] 评价关系 JSONL -> {path.resolve()}  ({len(relations)} 条关系)")
