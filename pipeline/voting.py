"""
pipeline.voting — Self-Consistency 投票机制 (可选功能)
对 Agent 1 抽取结果进行多轮投票，提升 Precision 和鲁棒性。

使用方式:
    from pipeline.voting import ExtractionVoter
    voter = ExtractionVoter(extraction_agent, rounds=3, threshold=2)
    output = voter.extract_with_voting(statement, sentence_id)
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .entity_extraction_agent import (
    EntityExtractionAgent,
    ExtractedEntity,
    SentenceExtractionOutput,
)
from .llm import LLMClient


class ExtractionVoter:
    """
    Agent 1 抽取结果的多轮投票器。

    对同一句子运行 k 轮抽取 (不同 temperature)，
    对每轮结果做实体级别的多数投票。
    仅保留 >= threshold 轮都抽出的实体。

    使用场景:
    - 复杂句子 (实体数 > 5)
    - 低置信度抽取 (Agent 1 confidence < 0.8)
    - 需要高 Precision 的场景
    """

    def __init__(
        self,
        extraction_agent: Optional[EntityExtractionAgent] = None,
        llm: Optional[LLMClient] = None,
        rounds: int = 3,
        threshold: int = 2,
        temperature: float = 0.3,
    ):
        """
        Args:
            extraction_agent: Agent 1 实体抽取实例
            llm: LLM 客户端 (若未提供 extraction_agent)
            rounds: 投票轮数 (默认 3)
            threshold: 最少同意轮数 (默认 2, 即 majority >= 2/3)
            temperature: 投票轮使用的 temperature (> 0 保证多样性)
        """
        if extraction_agent is not None:
            self.agent = extraction_agent
        else:
            self.agent = EntityExtractionAgent(llm or LLMClient())

        self.rounds = rounds
        self.threshold = min(threshold, rounds)
        self.temperature = temperature

    def extract_with_voting(
        self,
        statement: str,
        sentence_id: str = "",
        enable: bool = True,
    ) -> SentenceExtractionOutput:
        """
        带投票的实体抽取。

        Args:
            statement: 输入句子
            sentence_id: 句子 ID
            enable: 是否启用投票 (False 时退化为单轮抽取)

        Returns:
            SentenceExtractionOutput (已投票去重)
        """
        if not enable or self.rounds <= 1:
            return self.agent.extract(statement, sentence_id)

        # ── 多轮抽取 ──
        all_rounds: list[list[ExtractedEntity]] = []
        original_temp = self.agent.llm.temperature

        for r in range(self.rounds):
            # Set temperature for diversity
            self.agent.llm.temperature = self.temperature + r * 0.1

            output = self.agent.extract(statement, sentence_id)
            all_rounds.append(output.entities)

        # Restore temperature
        self.agent.llm.temperature = original_temp

        # ── 投票 ──
        voted_entities = self._vote(all_rounds, sentence_id)

        return SentenceExtractionOutput(
            sentence_id=sentence_id,
            sentence=statement,
            entities=voted_entities,
        )

    def _vote(
        self,
        all_rounds: list[list[ExtractedEntity]],
        sentence_id: str,
    ) -> list[ExtractedEntity]:
        """
        对多轮抽取结果做实体级别投票。

        匹配规则: normalized_name 精确匹配 (大小写不敏感)
        保留条件: 出现轮数 >= threshold
        """
        if not all_rounds:
            return []

        # Count entity occurrences across rounds
        entity_votes: dict[str, dict] = {}  # key -> {entity, rounds, mentions, l3_candidates}

        for round_idx, entities in enumerate(all_rounds):
            for e in entities:
                key = e.normalized_name.lower().strip() or e.mention.lower().strip()
                if not key or len(key) < 2:
                    continue

                if key not in entity_votes:
                    entity_votes[key] = {
                        "entity": e,
                        "rounds": set(),
                        "mentions": [],
                        "candidate_l3_list": [],
                        "candidate_l1_list": [],
                        "evidences": [],
                        "confidences": [],
                    }

                v = entity_votes[key]
                v["rounds"].add(round_idx)
                v["mentions"].append(e.mention)
                v["candidate_l3_list"].append(e.candidate_l3)
                v["evidences"].append(e.evidence)
                v["confidences"].append(e.confidence)

                for l1 in (e.candidate_l1 or []):
                    if l1 not in v["candidate_l1_list"]:
                        v["candidate_l1_list"].append(l1)

        # Filter by threshold
        voted: list[ExtractedEntity] = []
        for i, (key, v) in enumerate(entity_votes.items(), 1):
            if len(v["rounds"]) >= self.threshold:
                # Use the most common mention
                mention_counter = Counter(v["mentions"])
                best_mention = mention_counter.most_common(1)[0][0]

                # Use the most common candidate_l3
                l3_counter = Counter(v["candidate_l3_list"])
                best_l3 = l3_counter.most_common(1)[0][0] if l3_counter else ""

                # Best evidence (longest is often most informative)
                best_evidence = max(v["evidences"], key=len) if v["evidences"] else ""

                # Average confidence
                avg_confidence = (
                    sum(v["confidences"]) / len(v["confidences"])
                    if v["confidences"] else 0.0
                )

                eid = f"{sentence_id}_e{i}"
                voted.append(ExtractedEntity(
                    entity_id=eid,
                    mention=best_mention,
                    normalized_name=v["entity"].normalized_name or best_mention,
                    candidate_l3=best_l3,
                    candidate_l1=v["candidate_l1_list"],
                    evidence=best_evidence,
                    is_specific_entity=v["entity"].is_specific_entity,
                    confidence=avg_confidence,
                    uncertainty=f"voted_{len(v['rounds'])}/{self.rounds}",
                ))

        return voted

    def should_use_voting(
        self,
        statement: str,
        existing_count: Optional[int] = None,
    ) -> bool:
        """
        启发式判断是否需要对当前句子启用投票。

        触发条件:
        - 句子长度 > 150 字符
        - 句子包含多个数字/专有名词
        - (可选) 上一轮抽取实体数 > 5
        """
        if existing_count is not None and existing_count > 5:
            return True
        if len(statement) > 150:
            return True
        # Check for many proper nouns (capitalized or in brackets)
        import re
        brackets = len(re.findall(r'[《「『（(].*?[》」』）)]', statement))
        if brackets > 2:
            return True
        return False
