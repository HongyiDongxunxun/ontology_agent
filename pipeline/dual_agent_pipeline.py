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
    EvaluativeRelation,
    _normalize_name,
)
from .relation_verification_agent import RelationVerificationAgent
from .dynamic_term_db import DynamicTermDB


class DualAgentPipeline:
    """五Agent管道: Agent 1 评价关系(仅关系,主客体为原文短语)
    → Agent 2 实体抽取(接收必抽短语,统一负责实体识别+ID+分类)
    → Agent 3 分类 → Agent 4 审查(Likert) → Agent 5 关系校验(过滤事实类)"""

    def __init__(
        self,
        llm_extraction: Optional[LLMClient] = None,
        llm_classification: Optional[LLMClient] = None,
        llm_reviewer: Optional[LLMClient] = None,
        llm_relation: Optional[LLMClient] = None,
        llm_verification: Optional[LLMClient] = None,
        dynamic_term_db: Optional[DynamicTermDB] = None,
        batch_size: int = 12,
        mid_data_dir: str = "",
        enable_verification: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.llm_extraction = llm_extraction or LLMClient()
        self.llm_classification = llm_classification or LLMClient()
        self.llm_reviewer = llm_reviewer or LLMClient()
        self.dynamic_term_db = dynamic_term_db
        self.batch_size = batch_size
        self.mid_data_dir = mid_data_dir
        self.enable_verification = enable_verification
        # 校验输出快照 (供 run.py 导出 verification JSONL)
        self.last_verification_outputs: list[dict] = []

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
        self.verification_agent = RelationVerificationAgent(
            llm_verification or self.llm_extraction
        )

    def run(
        self, sentences: list[tuple[str, str]], base_name: str
    ) -> tuple[list[FinalEntityResult], list[dict]]:
        """
        执行五Agent管道:
        - Agent 1: 评价关系抽取 (EvaluativeRelationAgent) — 仅输出 relations,
          subject/object 为原文精确短语 (不再输出 entities)
        - Agent 2: 实体抽取 (EntityExtractionAgent, 接收必抽短语) — 统一负责
          实体识别、ID 分配与分类; pipeline 把 Agent 1 的 object 原文与 Agent 2
          的 normalized_name/mention 做一次性匹配, 回填 entity_id 到 relation
        - Agent 3: 分类 (ClassificationAgent)
        - Agent 4: 审查 Likert (ReviewerAgent)
        - Agent 5: 关系校验 (RelationVerificationAgent, 标记并过滤事实类)
        - sentences: list of (sentence_id, sentence_text)
        - base_name: 文件基础名 (如 "reviewed_full_1")
        - 返回: (所有 FinalEntityResult, 过滤后的评价关系列表)
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

        # ── Agent 1: 评价关系抽取(仅 relations, 主客体为原文短语) ──
        # ── Agent 2: 实体抽取(接收必抽短语, 统一负责实体识别+ID+分类) ──
        for idx, (sid, stmt) in enumerate(sentences, 1):
            # Agent 1: 评价关系抽取 — 输出 relations, subject/object 为原文短语
            rel_output = self.evaluative_relation_agent.extract(sid, stmt)
            self.last_has_evaluation[sid] = rel_output.has_evaluation

            # 从 relations 提取必抽短语 (object 原文 + subject 中的人物名)
            # 占位符 (_paper_author / _cite[N] / _unknown / _missing_entity) 跳过
            required_mentions = self._extract_required_mentions(rel_output.relations)

            # Agent 2: 实体抽取 — 接收必抽短语, 统一负责实体识别/ID/分类
            extraction = self.extraction_agent.extract(
                stmt, sentence_id=sid, required_mentions=required_mentions
            )

            # ── 一次性匹配: relation 的 object/subject 原文 → Agent 2 的 entity_id ──
            # 构建 normalized_name / mention → entity_id 索引 (双向, 不去重, 先到先得)
            ent_index: dict[str, str] = {}
            for e in extraction.entities:
                for key in (_normalize_name(e.normalized_name), _normalize_name(e.mention)):
                    if key and key not in ent_index:
                        ent_index[key] = e.entity_id

            for rel in rel_output.relations:
                rel_dict = {"sentence_id": sid, "sentence": stmt, **rel.to_dict()}
                # ── object 回填 ──
                obj = rel.object
                if obj != "_missing_entity" and not self._is_placeholder(obj):
                    matched_eid = ent_index.get(_normalize_name(obj), "")
                    if matched_eid:
                        rel_dict["object"] = matched_eid
                    else:
                        # Agent 2 未抽到该评价对象 → 保留原文 + 标记未匹配
                        rel_dict["object_unmatched"] = True
                        unresolved_count += 1
                # _missing_entity / 占位符 原样透传

                # ── subject 回填 (仅当 subject 是具体人物名, 非占位符) ──
                subj = rel.subject
                if not self._is_placeholder(subj):
                    matched_eid = ent_index.get(_normalize_name(subj), "")
                    if matched_eid:
                        rel_dict["subject"] = matched_eid
                    # subject 不匹配时不标记 (subject 可能本来就不是实体)

                all_relations.append(rel_dict)
                relations_by_sid.setdefault(sid, []).append(rel_dict)
            total_relations_count += len(rel_output.relations)

            all_extractions.append(extraction)
            total_entities += len(extraction.entities)

            if idx % 5 == 0 or idx == total_s:
                print(f"  [{base_name}] 抽取 [{idx}/{total_s}] 句, "
                      f"累计 {total_entities} 实体, {total_relations_count} 评价关系")

        if unresolved_count:
            print(f"  [{base_name}] 警告: {unresolved_count} 条关系 object 未匹配到实体, 已标记 object_unmatched")

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

        # ── Agent 5: Relation Verification (校验评价关系, 默认开启) ──
        # 逐句复核 Agent 1 抽取的评价关系: 事实类标记 is_evaluation=false 并过滤
        # 校验详情保存在 self.last_verification_outputs 供 run.py 导出审计
        self.last_verification_outputs = []
        if self.enable_verification and all_relations:
            print(f"  [{base_name}] Agent 5 关系校验中...")
            relations_by_sid_v: dict[str, list[dict]] = {}
            for rel in all_relations:
                relations_by_sid_v.setdefault(rel["sentence_id"], []).append(rel)

            total_eval = 0
            total_fact = 0
            filtered_relations: list[dict] = []
            for sid, sid_rels in relations_by_sid_v.items():
                sentence_text = sid_rels[0].get("sentence", "") if sid_rels else ""
                result = self.verification_agent.verify(sid, sentence_text, sid_rels)
                self.last_verification_outputs.append(result.to_dict())
                total_eval += result.evaluation_count
                total_fact += result.fact_count
                # 按 relation_id 建立校验映射, 回填字段到原关系
                verdict_map = {v.relation_id: v for v in result.verifications}
                for i, rel in enumerate(sid_rels):
                    rid = f"r{i + 1}"
                    v = verdict_map.get(rid)
                    if v:
                        rel["is_evaluation"] = v.is_evaluation
                        rel["verdict_reason"] = v.verdict_reason
                        if not v.is_evaluation:
                            rel["fact_type"] = v.fact_type
                        if v.is_evaluation:
                            filtered_relations.append(rel)
                    else:
                        # 校验未匹配 → 默认放行
                        rel["is_evaluation"] = True
                        rel["verdict_reason"] = "校验未匹配, 默认放行"
                        filtered_relations.append(rel)
            total_orig = len(all_relations)
            all_relations = filtered_relations
            print(f"  [{base_name}] Agent 5 完成: 评价 {total_eval} / 事实 {total_fact} "
                  f"(原 {total_orig} → 过滤后 {len(all_relations)})")

        return all_results, all_relations

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        """判断 subject/object 是否为占位符 (非具体原文短语)。

        占位符集合:
        - _paper_author, _unknown, _missing_entity
        - _cite[N] (N 为具体引文编号, 如 _cite[3])
        """
        if not value:
            return True
        if value.startswith("_"):
            return True
        return False

    @staticmethod
    def _extract_required_mentions(
        relations: list[EvaluativeRelation],
    ) -> list[str]:
        """从 Agent 1 的 relations 提取必抽短语 (供 Agent 2 作为必抽提示)。

        提取范围:
        - 每条 relation 的 object (若不是占位符 / _missing_entity)
        - 每条 relation 的 subject (若不是占位符, 即具体人物名)

        返回去重保序的字符串列表; 占位符 (_paper_author / _cite[N] /
        _unknown / _missing_entity) 被过滤, 不传入 Agent 2。
        """
        mentions: list[str] = []
        seen: set[str] = set()
        for rel in relations:
            # object 原文 (跳过占位符与 _missing_entity)
            obj = rel.object or ""
            if obj and obj != "_missing_entity" and not obj.startswith("_"):
                if obj not in seen:
                    seen.add(obj)
                    mentions.append(obj)
            # subject 具体人物名 (跳过占位符)
            subj = rel.subject or ""
            if subj and not subj.startswith("_"):
                if subj not in seen:
                    seen.add(subj)
                    mentions.append(subj)
        return mentions

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
    """导出评价关系 JSONL (V4.4: 来自 Agent 1, V5.1: 已过滤事实类)"""
    if not relations:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rel in relations:
            f.write(json.dumps(rel, ensure_ascii=False) + "\n")
    print(f"[导出] 评价关系 JSONL -> {path.resolve()}  ({len(relations)} 条关系)")


def export_verification_jsonl(
    verification_outputs: list[dict],
    filepath: str,
) -> None:
    """导出关系校验详情 JSONL (含 is_evaluation=false 的事实类, 供审计)"""
    if not verification_outputs:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for out in verification_outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[导出] 关系校验 -> {path.resolve()}  ({len(verification_outputs)} 句)")
