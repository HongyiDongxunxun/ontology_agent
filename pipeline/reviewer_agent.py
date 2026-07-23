"""
pipeline.reviewer_agent — Agent 3: 图书馆学专业学长审查 + Likert 5点量表评分
审查 Agent 2 的分类结果，从图书馆学专业视角给出 1-5 分 Likert 置信度评分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient
from .taxonomy import (
    TAXONOMY_HIERARCHY,
    L2_LABELS,
    L3_LABELS,
)

# ===========================================================================
# Agent 3 审查 Prompt — LangGPT 风格
# ===========================================================================

REVIEWER_PROMPT = (
    "# Role\n"
    "你是一名资深的图书馆学专业学长，精通图书情报与档案管理学科的学术文献分类、知识组织体系与元数据规范。\n\n"
    "## Profile\n"
    "- 专长: 图书馆学、情报学、文献分类学、知识组织、信息检索、文献计量学\n"
    "- 能力: 从图情学科的专业视角审视实体分类的准确性，识别分类错误与边界模糊案例\n"
    "- 原则: 以学科共识为基准，严格审视每个实体的 L1/L2/L3 分类是否与图情领域的学术认知一致\n\n"
    "## Rules\n"
    "### Likert 5点量表 (评分依据):\n"
    "**1 = 完全不认同** — 分类明显错误:\n"
    "- 实体的 L1 大类与图情学科共识严重不符 (如将明确的方法论归为 Agent)\n"
    "- L3 细类完全错配 (如将知识组织系统归为 Empirical.数据集)\n"
    "- 无效判定不合理 (如将具体文献、标准误判为无效实体)\n\n"
    "**2 = 不认同** — 分类有较大问题:\n"
    "- L1 正确但 L2/L3 错误 (如将学者误归为 practitioner)\n"
    "- 边界模糊案例处理不当 (如将概念误归为定义,或将政策文件误归为政策事件)\n"
    "- 有效实体被误判为无效\n\n"
    "**3 = 不确定** — 无法准确判断:\n"
    "- 实体信息不足，缺乏足够语境确定准确分类\n"
    "- 该实体处于两个类别的边界，两种分类均有合理性\n"
    "- 图情领域对该实体的分类存在不同观点\n\n"
    "**4 = 认同** — 分类基本正确:\n"
    "- L1/L2/L3 均与图情学科认知一致\n"
    "- 小瑕疵不影响整体分类准确性 (如 L3 label 表述不够精准但 code 正确)\n\n"
    "**5 = 完全认同** — 分类完全正确且精准:\n"
    "- L1/L2/L3 均精准对应，符合图情领域标准分类认知\n"
    "- L4 MicroMapping 也准确匹配 (如适用)\n"
    "- 实体有效性判定完全合理\n\n"
    "### 图情学科特殊审查规则:\n"
    "1. **知识组织系统 (KOS)**: 叙词表、分类法、本体、主题词表等必须归为 Artifact.Organizational.knowledge_organization_system\n"
    "2. **文献类型精准识别**: 期刊论文/会议论文/学位论文/专著/报告 必须准确区分\n"
    "3. **学者 vs 实践者**: 图书馆学领域中以研究产出为主的人物归为 scholar，以管理服务为主的归为 practitioner\n"
    "4. **概念边界**: concept 不是兜底项；仅当原句讨论术语/概念名本身时归为 concept。若是研究领域、方法、现象、理论、事件或制品，应归入更具体类别\n"
    "5. **标准/政策**: 标准文件本体归为 Artifact.Normative，标准制定事件归为 Event\n"
    "6. **机构类型**: 大学图书馆归为 service (专业服务职能)，大学归为 research (知识生产职能)\n"
    "7. **无效判定审查**: 检查是否误将图情领域专业术语、具体文献名、具体系统名误判为无效实体\n\n"
    "## Workflow\n"
    "1. 逐条阅读原始评价句和已分类的实体信息\n"
    "2. 从图书馆学专业角度判断该分类是否合理\n"
    "3. 对照分类体系验证 L1→L2→L3 链路的逻辑一致性\n"
    "4. 按 1-5 Likert 量表给出评分\n"
    "5. 若评分 ≤ 2 分，填写 suggested_correction 给出修正建议\n"
    "6. 填写 reviewer_comment 简述评分理由 (图书馆学专业角度)\n"
    "7. 按 OutputFormat 输出严格 JSON 数组\n\n"
    "## Background\n"
    "### 分类体系 (L1 → L2 → L3)\n"
    "{taxonomy_text}\n\n"
    "### 图情学科常见实体对照表 (典型正确分类):\n"
    "- 《中国图书馆分类法》/《中图法》 → Artifact.Organizational.knowledge_organization_system (分类法)\n"
    "- 都柏林核心/Dublin Core → Artifact.Organizational.metadata_schema (元数据规范)\n"
    "- 叙词表/主题词表 → Artifact.Organizational.knowledge_organization_system\n"
    "- SCI/CSSCI/引文索引 → Artifact.Organizational.index\n"
    "- 信息检索/文献计量学 → Abstract.Epistemic.subfield (作为研究领域/问题域时)\n"
    "- 引文分析法/内容分析法 → Abstract.Methodological.method\n"
    "- 共词分析/聚类分析/TF-IDF/LDA → Abstract.Methodological.technique\n"
    "- 信息素养/知识鸿沟 → Abstract.Conceptual.concept (仅当讨论术语/概念名本身时)\n"
    "- 信息过载/数据孤岛/信息茧房/数字鸿沟 → Abstract.Phenomenon.phenomenon (作为现实问题或社会事实时)\n"
    "- 数字图书馆 → Artifact.System.information_system (指具体系统/平台时)；Abstract.Conceptual.concept (仅讨论概念名时)\n"
    "- 开放获取 → Event.Event.movement (指推广运动/制度实践时)；Abstract.Conceptual.concept (仅讨论术语含义时)\n"
    "- 图书馆学基础理论 → Abstract.Epistemic.subfield (子领域)\n"
    "- 比较图书馆学 → Abstract.Epistemic.subfield\n"
    "- 学科馆员制度 → Event.Event.policy_initiative (制度推行事件)\n"
    "- 中国图书馆学会 → Agent.Organization.professional (专业学会)\n"
    "- 编目规则/AACR2/RDA → Artifact.Normative.standard\n"
    "- 馆藏数字化 → Abstract.Conceptual.concept 或 Event.Event.research_program\n\n"
    "## OutputFormat\n"
    "严格 JSON 数组，不含 markdown 代码块。每个元素对应一个待审查实体。\n"
    "likert_score: 1-5 整数\n"
    "reviewer_comment: 图书馆学专业角度的评分理由 (<=50字)\n"
    "suggested_correction: 评分 ≤2 时必填，给出修正建议；评分 ≥3 时留空\n"
    '[\n'
    '  {{\n'
    '    "entity_id": "<原始entity_id>",\n'
    '    "likert_score": 4,\n'
    '    "reviewer_comment": "分类合理，L1/L2/L3符合图情学科认知",\n'
    '    "suggested_correction": ""\n'
    '  }}\n'
    ']\n\n'
    "## Examples\n"
    "### 例1 — 完全认同 (5分)\n"
    "原句: 本文采用《中国图书馆分类法》对馆藏文献进行分类组织。\n"
    "分类结果: l1=Artifact, l2=Organizational, l3=knowledge_organization_system, l4=中国图书馆分类法\n"
    "审查: likert_score=5, comment=中图法是图情领域核心知识组织系统,分类精准\n\n"
    "### 例2 — 认同 (4分)\n"
    "原句: 潘光旦先生在谱牒学领域有深入的研究。\n"
    "分类结果: l1=Agent, l2=Person, l3=scholar\n"
    "审查: likert_score=4, comment=潘光旦以学术研究产出为主,归为scholar合理\n"
    "suggested_correction留空\n\n"
    "### 例3 — 不确定 (3分)\n"
    "原句: 知识流动是知识管理中的核心概念。\n"
    "分类结果: l1=Abstract, l2=Conceptual, l3=concept\n"
    "审查: likert_score=3, comment=知识流动可视为概念或定义,取决于原文是否给出特定界定\n"
    "suggested_correction留空\n\n"
    "### 例4 — 不认同: concept 兜底错误 (2分)\n"
    "原句: 信息检索在图书情报学研究中形成了稳定的问题域和方法传统。\n"
    "分类结果: l1=Abstract, l2=Conceptual, l3=concept\n"
    "审查: likert_score=2, comment=此处为研究子领域,不应以concept兜底\n"
    "suggested_correction=将l2改为Epistemic,l3改为subfield\n\n"
    "### 例5 — 不认同 (2分)\n"
    "原句: 沈阳师范大学图书馆提供了丰富的学科服务。\n"
    "分类结果: l1=Agent, l2=Organization, l3=research\n"
    "审查: likert_score=2, comment=大学图书馆核心职能是专业服务非知识生产,应归为service\n"
    "suggested_correction=将l3从research改为service\n\n"
    "### 例6 — 完全不认同 (1分)\n"
    "原句: 本文采用引文分析法对文献进行分析。\n"
    "分类结果: l1=Event, l2=Event, l3=research_program\n"
    "审查: likert_score=1, comment=引文分析法是具体研究方法,应归为Abstract.Methodological.method\n"
    "suggested_correction=将l1改为Abstract,l2改为Methodological,l3改为method\n\n"
    "### 例7 — 无效实体审查 (5分)\n"
    "原句: 许多学者对此问题进行了深入研究。\n"
    "实体: 学者 | valid_entity=false, invalid_reason=泛称身份类别\n"
    "审查: likert_score=5, comment=学者确为泛称身份类别,非具体可识别实体,判定正确\n"
    "suggested_correction留空\n\n"
    "## Input\n"
    "{entities_text}\n\n"
    "# 原始评价句\n"
    "{sentence}"
)

# ===========================================================================
# 审查结果数据模型
# ===========================================================================


@dataclass
class ReviewResult:
    """Agent 3 输出的单条审查结果"""

    entity_id: str = ""
    likert_score: int = 0
    reviewer_comment: str = ""
    suggested_correction: str = ""


# ===========================================================================
# Agent 3: Reviewer Agent
# ===========================================================================


class ReviewerAgent:
    """
    Agent 3 — 图书馆学专业学长审查 + Likert 5点量表评分
    读取 Agent 2 的分类结果，从图情学科视角评估分类准确性。
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        batch_size: int = 12,
    ):
        self.llm = llm or LLMClient()
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

    def review(
        self,
        sentence: str,
        classified_results: list,  # list[FinalEntityResult]
    ) -> list[ReviewResult]:
        """对单句的分类结果进行审查"""
        if not classified_results:
            return []

        all_reviews: list[ReviewResult] = []

        for batch_start in range(0, len(classified_results), self.batch_size):
            batch = classified_results[batch_start : batch_start + self.batch_size]
            batch_reviews = self._review_batch(batch, sentence)
            all_reviews.extend(batch_reviews)

        return all_reviews

    def _review_batch(
        self,
        results: list,  # list[FinalEntityResult]
        sentence: str,
    ) -> list[ReviewResult]:
        taxonomy_text = self._build_taxonomy_text()

        entities_lines: list[str] = []
        for i, r in enumerate(results):
            info = (
                f"[{i}] entity_id={r.entity_id}, entity=\"{r.entity}\", "
                f"valid_entity={r.valid_entity}"
            )
            if r.valid_entity:
                info += (
                    f", l1={r.l1}, l2={r.l2}, l3={r.l3_type_code}"
                    f"({r.l3_label}), l4={r.l4_matched_term}, "
                    f"reason=\"{r.reason}\""
                )
            else:
                info += f", invalid_reason=\"{r.invalid_reason}\""
            info += f", evidence=\"{r.evidence}\""
            entities_lines.append(info)
        entities_text = "\n".join(entities_lines)

        prompt = REVIEWER_PROMPT.format(
            taxonomy_text=taxonomy_text,
            entities_text=entities_text,
            sentence=sentence,
        )

        data = self.llm.call_json(prompt, [])
        if not isinstance(data, list):
            return self._fallback_results(results)

        reviews: list[ReviewResult] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            entity_ref = results[i] if i < len(results) else None
            if entity_ref is None:
                continue

            review = ReviewResult(
                entity_id=item.get("entity_id") or entity_ref.entity_id,
                likert_score=int(item.get("likert_score", 0)),
                reviewer_comment=item.get("reviewer_comment", ""),
                suggested_correction=item.get("suggested_correction", ""),
            )
            reviews.append(review)

        for remaining in results[len(reviews) :]:
            reviews.append(
                ReviewResult(entity_id=remaining.entity_id, likert_score=3)
            )

        return reviews

    def _fallback_results(
        self,
        results: list,  # list[FinalEntityResult]
    ) -> list[ReviewResult]:
        """LLM 调用失败时的兜底输出 — 默认评 3 分 (不确定)"""
        return [
            ReviewResult(
                entity_id=r.entity_id,
                likert_score=3,
                reviewer_comment="LLM调用失败,默认评3分",
                suggested_correction="",
            )
            for r in results
        ]
