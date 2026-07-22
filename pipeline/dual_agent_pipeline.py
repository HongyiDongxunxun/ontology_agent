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
from .dynamic_term_db import DynamicTermDB
from .rag import RAGExampleDB
from .voting import ExtractionVoter


class AgentPipeline:
    """三Agent管道: Agent 1 抽取 → Agent 2 分类 → Agent 3 审查(Likert)"""

    def __init__(
        self,
        llm_extraction: Optional[LLMClient] = None,
        llm_classification: Optional[LLMClient] = None,
        llm_reviewer: Optional[LLMClient] = None,
        dynamic_term_db: Optional[DynamicTermDB] = None,
        rag_db: Optional["RAGExampleDB"] = None,
        voter: Optional["ExtractionVoter"] = None,
        batch_size: int = 12,
        mid_data_dir: str = "",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.llm_extraction = llm_extraction or LLMClient()
        self.llm_classification = llm_classification or LLMClient()
        self.llm_reviewer = llm_reviewer or LLMClient()
        self.dynamic_term_db = dynamic_term_db
        self.rag_db = rag_db
        self.voter = voter
        self.batch_size = batch_size
        self.mid_data_dir = mid_data_dir

        self.extraction_agent = EntityExtractionAgent(self.llm_extraction)
        self.classification_agent = ClassificationAgent(
            llm=self.llm_classification,
            dynamic_term_db=self.dynamic_term_db,
            batch_size=self.batch_size,
            enable_voting=getattr(self, '_enable_voting', False),
            voting_rounds=getattr(self, '_voting_rounds', 3),
            voting_temperature=getattr(self, '_voting_temperature', 0.3),
            enable_rag=getattr(self, '_enable_rag', False),
            rag_k_examples=getattr(self, '_rag_k_examples', 5),
        )
        self.reviewer_agent = ReviewerAgent(
            llm=self.llm_reviewer,
            batch_size=self.batch_size,
        )

    def run(
        self, sentences: list[tuple[str, str]], base_name: str
    ) -> list[FinalEntityResult]:
        """
        执行双Agent管道:
        - sentences: list of (sentence_id, sentence_text)
        - base_name: 文件基础名 (如 "reviewed_full_1")
        - 返回: 所有 FinalEntityResult
        """
        self._log(f"[AgentPipeline] 开始处理 {base_name}, 共 {len(sentences)} 句")

        all_extractions: list[SentenceExtractionOutput] = []

        # ── Agent 1: Entity Extraction (with optional RAG + Voting) ──
        total_s = len(sentences)
        rag_hits = 0
        voted_count = 0
        for idx, (sid, stmt) in enumerate(sentences, 1):
            # RAG: 检索相似 few-shot 示例
            few_shot = ""
            if self.rag_db and self.rag_db.is_ready():
                few_shot = self.rag_db.build_few_shot_text(stmt, k=5, max_examples=5)
                if few_shot:
                    rag_hits += 1

            # Voting: 判断是否需要投票
            use_voting = False
            if self.voter and self.voter.rounds > 1:
                use_voting = self.voter.should_use_voting(stmt)

            if use_voting:
                extraction = self.voter.extract_with_voting(stmt, sid, enable=True)
                voted_count += 1
            else:
                extraction = self.extraction_agent.extract(
                    stmt, sentence_id=sid, few_shot_text=few_shot,
                )

            all_extractions.append(extraction)
            if idx % 5 == 0 or idx == total_s:
                print(f"  [{base_name}] 抽取 [{idx}/{total_s}] 句, 累计 {sum(len(e.entities) for e in all_extractions)} 实体"
                      + (f" (RAG:{rag_hits} vote:{voted_count})" if (rag_hits or voted_count) else ""))

        # ── 写中间结果到 mid_data/ ──
        if self.mid_data_dir:
            mid_path = Path(self.mid_data_dir)
            mid_path.mkdir(parents=True, exist_ok=True)
            mid_file = mid_path / f"{base_name}_extracted.json"
            mid_data = {
                "base_name": base_name,
                "stage": "agent1_extraction",
                "extractions": [e.to_dict() for e in all_extractions],
            }
            with open(mid_file, "w", encoding="utf-8") as f:
                json.dump(mid_data, f, ensure_ascii=False, indent=2)
            print(f"  [{base_name}] 中间结果已写入: mid_data/{base_name}_extracted.json")

        # ── Agent 2: Classification ──
        print(f"  [{base_name}] Agent 2 分类中...")
        all_results: list[FinalEntityResult] = []
        total_ex = len(all_extractions)
        for i, extraction in enumerate(all_extractions, 1):
            results = self.classification_agent.classify(extraction)
            all_results.extend(results)
            if i % 5 == 0 or i == total_ex:
                print(f"  [{base_name}] 分类 [{i}/{total_ex}] 句, 累计 {len(all_results)} 条结果")

        valid_count = sum(1 for r in all_results if r.valid_entity)
        invalid_count = sum(1 for r in all_results if not r.valid_entity)
        print(f"\n[Agent 2] 完成: {valid_count} 有效, {invalid_count} 无效")

        # ── Agent 3: Library Science Review + Likert ──
        print(f"  [{base_name}] Agent 3 审查中...")
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
        print(f"  [{base_name}] Agent 3 完成: {scored}/{len(all_results)} 条已评分, 平均 Likert: {avg_score:.2f}")

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

        return all_results

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
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[导出] 汇总 -> {path.resolve()}")
