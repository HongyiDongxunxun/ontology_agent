"""
pipeline.classification_agent — Agent 2: L1/L2/L3 分类 + L4 规则匹配
对齐: 实体类型分类体系_opencode版.md — LangGPT 风格提示词
读取 Agent 1 的中间结果，验证实体有效性，标注 L1/L2/L3，L4 为规则匹配。
"""

from __future__ import annotations

from typing import Optional

from .llm import LLMClient
from .dynamic_term_db import DynamicTermDB
from .taxonomy import (
    TAXONOMY_HIERARCHY,
    L2_LABELS,
    L3_LABELS,
    get_l1l2_from_l3,
    get_l2_label,
    get_l3_label,
)
from .entity_extraction_agent import ExtractedEntity, SentenceExtractionOutput

# ===========================================================================
# Agent 2 分类 Prompt — LangGPT 风格
# ===========================================================================

CLASSIFICATION_PROMPT = (
    "# Role\n"
    "你是一个学术文献实体精分类与验证专家。\n\n"
    "## Profile\n"
    "- 专长: 验证实体有效性，完成 L1/L2/L3 层级的精准标注\n"
    "- 能力: 在给定分类体系中对候选实体精确归类，识别并拒绝无效实体\n"
    "- 原则: 判断优先看被评价对象的**功能/身份/语境**，而非实体名称本身\n\n"
    "## Rules\n"
    "### 有效性验证 (先验证，后分类):\n"
    "**有效实体 (valid_entity=true)** — 具体可唯一识别:\n"
    "- 具体人名: 潘光旦 ✓, 罗香林 ✓, 巴巴拉·奎恩特 ✓\n"
    "- 具体机构名: 哥伦比亚大学 ✓, 沈阳师范大学图书馆 ✓, 鲍克公司 ✓\n"
    "- 具体文献/系统/理论/方法名: 《中图法》✓, CDWS分词系统 ✓, 学习迁移理论 ✓\n"
    "- 具体事件/会议/运动: Information与Intelligence的争论 ✓, 第二届全国灰色文献年会 ✓\n"
    "**无效实体 (valid_entity=false)** — 必填 invalid_reason:\n"
    "- 泛称身份类别: 科学家 ✗, 学者 ✗, 教授 ✗, 图书馆员 ✗, 作者 ✗\n"
    "- 泛称机构类别: 大学图书馆 ✗, 高校 ✗, 研究机构 ✗, 公共图书馆 ✗\n"
    "- 泛称文献/工具类别: 某论文 ✗, 相关文献 ✗, 统计软件 ✗\n"
    "- 通用词/虚义动词: 比较 ✗, 进行 ✗, 通过 ✗, 基于 ✗\n"
    "- 泛化评价用语: 重要意义 ✗, 研究成果 ✗\n"
    "- 无学术语义计量词: 篇数 ✗, 比例 ✗\n"
    "判定标准: 该实体能否指向一个**具体可唯一识别**的对象?\n"
    "若无效 → 仅填 valid_entity 和 invalid_reason，其余字段留空。\n\n"
    "### 逐级分类 (L1 → L2 → L3):\n"
    "1. **L1 判定** (4选1): Agent | Artifact | Abstract | Event — 基于被评价对象的**功能/语境**\n"
    "2. **L2 判定**: 在选定 L1 下选择二级类别 (见 Background 分类体系)\n"
    "3. **L3 判定**: 在选定 L2 下选择叶子类型 code + label\n"
    "4. 标注 `other` 时必须填写 `other_suggestion` 描述建议类型，不可留空\n"
    "5. Entity 带缩写/代称时输出保留全名和简称\n\n"
    "## Workflow\n"
    "1. 读取每个候选实体及其原句\n"
    "2. 有效性验证 → 无效则填 invalid_reason, 跳过后续步骤\n"
    "3. 判定 L1 (Agent|Artifact|Abstract|Event)\n"
    "4. 在 L1 下选定 L2\n"
    "5. 在 L2 下选定 L3 (type_code + label)\n"
    "6. 按 OutputFormat 输出 JSON 数组\n\n"
    "## Background\n"
    "### 分类体系 (L1 → L2 → L3)\n"
    "{taxonomy_text}\n\n"
    "## OutputFormat\n"
    "严格 JSON 数组，不含 markdown 代码块。每个元素对应一个实体。\n"
    "无效实体: 仅填 entity_id, valid_entity, invalid_reason，其余字段留空。\n"
    '[\n'
    '  {{\n'
    '    "entity_id": "<原始entity_id>",\n'
    '    "valid_entity": true,\n'
    '    "invalid_reason": "",\n'
    '    "l1": "Agent",\n'
    '    "l2": "Person",\n'
    '    "l2_label": "个人",\n'
    '    "l3_type_code": "scholar",\n'
    '    "l3_label": "知识生产者",\n'
    '    "evidence": "分类证据原文片段",\n'
    '    "reason": "简短分类理由 (<=30字)",\n'
    '    "other_suggestion": ""\n'
    '  }}\n'
    ']\n\n'
    "## Examples\n"
    "### 例1 — scholar vs practitioner 区分\n"
    "实体: 罗香林 | 原句: 建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。\n"
    "分类: l1=Agent, l2=Person, l3_type_code=scholar, reason=以研究/知识生产为主要职能\n"
    "规则: scholar 产出是知识本身(论文/著作)；practitioner 产出是专业服务(咨询/管理)。同一人物以研究身份被评价→scholar。\n\n"
    "### 例2 — book vs journal_article 区分\n"
    "实体: 新版《图书馆学概论》 | 原句: 新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果。\n"
    "分类: l1=Artifact, l2=Discursive, l3_type_code=book, reason=独立出版物发行,有独立书名\n"
    "规则: 专著有独立 ISBN 和书名，整本出版；期刊论文有卷期页码属于某期刊。\n\n"
    "### 例3 — policy(Artifact) vs policy_initiative(Event) 区分\n"
    "实体A: HR1858 | 原句: HR1858很好地平衡了反盗与利用信息制作新数据库产品需求之间的关系。\n"
    "分类: l1=Artifact, l2=Normative, l3_type_code=policy, reason=政策文件本身,是制品\n"
    "实体B: 文献检索与利用课 | 原句: 教育部下发了在高校开设「文献检索与利用课」的红头文件,大大推动了其发展。\n"
    "分类: l1=Event, l2=Event, l3_type_code=policy_initiative, reason=政策实施过程中的制度性行动事件\n"
    "规则: policy 是政策文件本身(制品, Artifact)；policy_initiative 是政策实施事件(事件, Event)。判断语境评价的是文件内容还是实施行动。\n\n"
    "### 例4 — concept vs definition 区分\n"
    "实体: 知识流动 | 原句: Szulanski认为知识流动是在特定环境下知识从来源方到接收方的传递过程。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=definition, reason=对概念边界的特定界定方式,含有提出者\n"
    "规则: concept 是命名单元本身；definition 是对命名单元边界的特定界定(通常含提出者)。同一概念可有多个 definition。\n\n"
    "### 例5 — Organization 子类型选择 (research vs service vs governance)\n"
    "实体A: 哥伦比亚大学 | 分类: l3_type_code=research, reason=以知识生产为主要职能的大学\n"
    "实体B: 沈阳师范大学图书馆 | 分类: l3_type_code=service, reason=以专业服务为主要职能的图书馆\n"
    "实体C: 广州市档案局 | 分类: l3_type_code=governance, reason=有行政/治理职能的政府机构\n"
    "规则: 以核心职能判断。大学/研究院→research; 图书馆/档案馆→service; 学会/协会→professional; 政府部门→governance; 出版社→publishing。\n\n"
    "### 例6 — conference_paper(Artifact) vs conference_meeting(Event) 区分\n"
    "实体: 第二届全国灰色文献年会 | 原句: 本次会议在2018年第一次全国灰色文献年会的基础上进行了深入探讨。\n"
    "分类: l1=Event, l2=Event, l3_type_code=conference_meeting, reason=具体召开的学术会议事件\n"
    "规则: conference_paper 是会议论文(制品, Artifact)；conference_meeting 是会议召开本身(事件, Event)。\n\n"
    "## Input\n"
    "{entities_text}\n\n"
    "# 原始评价句\n"
    "{sentence}"
)

# ===========================================================================
# 最终输出数据模型
# ===========================================================================


class FinalEntityResult:
    """Agent 2 输出的单条最终分类结果 (对应 JSONL 一行)"""

    def __init__(
        self,
        sentence_id: str = "",
        entity_id: str = "",
        sentence: str = "",
        entity: str = "",
        normalized_name: str = "",
        valid_entity: bool = True,
        invalid_reason: str = "",
        l1: str = "",
        l2: str = "",
        l2_label: str = "",
        l3_type_code: str = "",
        l3_label: str = "",
        l4_matched_term: str = "",
        l4_term_type: Optional[str] = None,
        evidence: str = "",
        reason: str = "",
        other_suggestion: str = "",
        likert_confidence: int = 0,
        reviewer_comment: str = "",
        suggested_correction: str = "",
    ):
        self.sentence_id = sentence_id
        self.entity_id = entity_id
        self.sentence = sentence
        self.entity = entity
        self.normalized_name = normalized_name
        self.valid_entity = valid_entity
        self.invalid_reason = invalid_reason
        self.l1 = l1
        self.l2 = l2
        self.l2_label = l2_label
        self.l3_type_code = l3_type_code
        self.l3_label = l3_label
        self.l4_matched_term = l4_matched_term
        self.l4_term_type = l4_term_type
        self.evidence = evidence
        self.reason = reason
        self.other_suggestion = other_suggestion
        self.likert_confidence = likert_confidence
        self.reviewer_comment = reviewer_comment
        self.suggested_correction = suggested_correction

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "entity_id": self.entity_id,
            "sentence": self.sentence,
            "entity": self.entity,
            "normalized_name": self.normalized_name,
            "valid_entity": self.valid_entity,
            "invalid_reason": self.invalid_reason,
            "l1": self.l1,
            "l2": self.l2,
            "l2_label": self.l2_label,
            "l3_type_code": self.l3_type_code,
            "l3_label": self.l3_label,
            "l4_matched_term": self.l4_matched_term,
            "l4_term_type": self.l4_term_type,
            "evidence": self.evidence,
            "reason": self.reason,
            "other_suggestion": self.other_suggestion,
            "likert_confidence": self.likert_confidence,
            "reviewer_comment": self.reviewer_comment,
            "suggested_correction": self.suggested_correction,
        }


# ===========================================================================
# Agent 2: Classification Agent
# ===========================================================================


class ClassificationAgent:
    """
    Agent 2 — L1/L2/L3 精分类 + L4 规则匹配
    读取 Agent 1 中间结果，LLM 完成 L1/L2/L3 分类，L4 由 _match_l4_rules() 做字符串规则匹配。
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        dynamic_term_db: Optional[DynamicTermDB] = None,
        batch_size: int = 12,
    ):
        self.llm = llm or LLMClient()
        self.dynamic_term_db = dynamic_term_db
        self.batch_size = batch_size

    def _build_taxonomy_text(self) -> str:
        lines: list[str] = []
        for l1, l2_map in TAXONOMY_HIERARCHY.items():
            lines.append(f"**L1 = {l1}**")
            for l2, l3_list in l2_map.items():
                l2_label = L2_LABELS.get(l2, l2)
                l3_details = ", ".join(
                    f"{code}({L3_LABELS.get(code, code)})" for code in l3_list
                )
                lines.append(f"  L2 = {l2} ({l2_label}) → L3: {l3_details}")
        return "\n".join(lines)

    def _match_l4_rules(self, entity_name: str, normalized_name: str) -> tuple[str, Optional[str]]:
        """规则匹配 L4: 与 DynamicTermDB 做精确/包含字符串匹配 (不依赖 LLM)"""
        if self.dynamic_term_db is None:
            return ("", None)
        terms = self.dynamic_term_db.get_micro_terms()
        if not terms:
            return ("", None)

        candidates = []
        if entity_name:
            candidates.append(entity_name)
        if normalized_name and normalized_name != entity_name:
            candidates.append(normalized_name)

        for candidate in candidates:
            cl = candidate.lower().strip()
            if not cl or len(cl) < 2:
                continue
            for term, type_code in terms:
                tl = term.lower().strip()
                if not tl or len(tl) < 2:
                    continue
                if cl == tl:
                    return (term, type_code)
            for term, type_code in terms:
                tl = term.lower().strip()
                if not tl or len(tl) < 2:
                    continue
                if tl in cl or (len(tl) >= 3 and cl in tl):
                    return (term, type_code)
        return ("", None)

    def classify(
        self,
        extraction: SentenceExtractionOutput,
    ) -> list[FinalEntityResult]:
        """对单句的抽取结果进行精分类"""
        entities = extraction.entities
        if not entities:
            return []

        results: list[FinalEntityResult] = []

        for batch_start in range(0, len(entities), self.batch_size):
            batch = entities[batch_start : batch_start + self.batch_size]
            batch_results = self._classify_batch(
                batch, extraction.sentence, extraction.sentence_id
            )
            results.extend(batch_results)

        return results

    def _classify_batch(
        self,
        entities: list[ExtractedEntity],
        sentence: str,
        sentence_id: str,
    ) -> list[FinalEntityResult]:
        taxonomy_text = self._build_taxonomy_text()

        entities_lines: list[str] = []
        for i, e in enumerate(entities):
            info = (
                f"[{i}] entity_id={e.entity_id}, mention=\"{e.mention}\", "
                f"candidate_l1={e.candidate_l1}, candidate_l3={e.candidate_l3}, "
                f"evidence=\"{e.evidence}\""
            )
            entities_lines.append(info)
        entities_text = "\n".join(entities_lines)

        prompt = CLASSIFICATION_PROMPT.format(
            taxonomy_text=taxonomy_text,
            entities_text=entities_text,
            sentence=sentence,
        )

        data = self.llm.call_json(prompt, [])
        if not isinstance(data, list):
            return self._fallback_results(entities, sentence, sentence_id)

        results: list[FinalEntityResult] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            idx = i
            entity_ref = entities[idx] if idx < len(entities) else None
            if entity_ref is None:
                continue

            result = FinalEntityResult(
                sentence_id=sentence_id,
                entity_id=item.get("entity_id") or entity_ref.entity_id,
                sentence=sentence,
                entity=entity_ref.mention,
                normalized_name=entity_ref.normalized_name,
                valid_entity=item.get("valid_entity", True),
                invalid_reason=item.get("invalid_reason", ""),
                l1=item.get("l1", ""),
                l2=item.get("l2", ""),
                l2_label=item.get("l2_label") or get_l2_label(item.get("l2", "")),
                l3_type_code=item.get("l3_type_code", ""),
                l3_label=item.get("l3_label") or get_l3_label(item.get("l3_type_code", "")),
                l4_matched_term="",
                l4_term_type=None,
                evidence=item.get("evidence") or entity_ref.evidence,
                reason=item.get("reason", ""),
                other_suggestion=item.get("other_suggestion", ""),
            )
            # 规则匹配 L4 (不依赖 LLM)
            if result.valid_entity:
                matched, matched_type = self._match_l4_rules(
                    entity_ref.mention, entity_ref.normalized_name
                )
                if matched:
                    result.l4_matched_term = matched
                    result.l4_term_type = matched_type
            results.append(result)

        for remaining in entities[len(results):]:
            matched, matched_type = self._match_l4_rules(
                remaining.mention, remaining.normalized_name
            )
            results.append(
                FinalEntityResult(
                    sentence_id=sentence_id,
                    entity_id=remaining.entity_id,
                    sentence=sentence,
                    entity=remaining.mention,
                    normalized_name=remaining.normalized_name,
                    valid_entity=True,
                    l1="",
                    l2="",
                    l3_type_code=remaining.candidate_l3,
                    evidence=remaining.evidence,
                    l4_matched_term=matched,
                    l4_term_type=matched_type,
                )
            )

        return results

    def _fallback_results(
        self,
        entities: list[ExtractedEntity],
        sentence: str,
        sentence_id: str,
    ) -> list[FinalEntityResult]:
        """LLM 调用失败时的兜底输出，L4 仍用规则匹配"""
        results: list[FinalEntityResult] = []
        for e in entities:
            l1, l2 = get_l1l2_from_l3(e.candidate_l3)
            matched, matched_type = self._match_l4_rules(e.mention, e.normalized_name)
            results.append(
                FinalEntityResult(
                    sentence_id=sentence_id,
                    entity_id=e.entity_id,
                    sentence=sentence,
                    entity=e.mention,
                    normalized_name=e.normalized_name,
                    valid_entity=e.is_specific_entity,
                    l1=l1,
                    l2=l2,
                    l2_label=get_l2_label(l2),
                    l3_type_code=e.candidate_l3,
                    l3_label=get_l3_label(e.candidate_l3),
                    evidence=e.evidence,
                    l4_matched_term=matched,
                    l4_term_type=matched_type,
                )
            )
        return results
