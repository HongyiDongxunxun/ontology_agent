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
    "- 单纯时间表达: 2005年 ✗, 20世纪90年代 ✗。除非原文明确表示一个发展阶段，否则不标 stage\n"
    "- 单纯地名/行政区名: 深圳 ✗, 青岛 ✗。除非指向具体政府机构/组织主体，否则不标 governance\n"
    "判定标准: 该实体能否指向一个**具体可唯一识别**的对象?\n"
    "若无效 → 仅填 valid_entity 和 invalid_reason，其余字段留空。\n\n"
    "### 逐级分类 (L1 → L2 → L3):\n"
    "1. **L1 判定** (4选1): Agent | Artifact | Abstract | Event — 基于被评价对象的**功能/语境**\n"
    "2. **L2 判定**: 在选定 L1 下选择二级类别 (见 Background 分类体系)\n"
    "3. **L3 判定**: 在选定 L2 下选择叶子类型 code + label\n"
    "4. 标注 `other` 时必须填写 `other_suggestion` 描述建议类型，不可留空\n"
    "5. Entity 带缩写/代称时输出保留全名和简称\n\n"
    "### `concept` 严格边界 — 禁止当作兜底项:\n"
    "`concept` 只表示学术术语/命名单元本身，不是 Abstract 的默认兜底类型。\n"
    "若准备标为 concept，必须同时满足: (1) 原句把对象作为术语/概念名讨论；(2) 没有更具体的 L3 可用；(3) reason 中说明为何不是更具体类别。\n"
    "标 concept 前必须优先排除:\n"
    "- 操作程序/分析/测量/检索/聚类/回归/实验/方法 (含具体操作手段) → method\n"
    "- 由概念/命题/变量及其关系组成的解释或分析构造 (理论/概念模型/分析框架) → theoretical_construct；单一待检验命题 → hypothesis\n"
    "- 学科/领域/方向/问题域 → research_field；研究传统/学派 → school_of_thought；研究范式/立场/视角 → research_orientation\n"
    "- 可观察的状态/行为模式/社会事实 (含增长、衰退、扩散等带时间方向的变化) → phenomenon\n"
    "- 运动/倡议/改革/计划/项目/争论/发展阶段 → Event\n"
    "- 具体系统、数据库、工具、标准、文献、知识组织系统 → Artifact\n"
    "边界提示: 信息检索/文献计量学/图书情报学 → research_field；引文分析法/内容分析法/共词分析/聚类分析/LDA → method；信息过载/数据孤岛 → phenomenon；开放获取若作为推广运动 → movement；计算机可执行模型与算法 (GRU4REC/PageRank) → algorithm。\n\n"
    "### `method` / `algorithm` / `instrument` / `phenomenon` 严格边界:\n"
    "- `method` 覆盖研究设计、采集、处理、分析各环节的可操作程序与具体手段 (共词分析、聚类分析、LDA 主题建模、回归方法、问卷调查法)。技术领域/技术体系不得标 method。\n"
    "- `algorithm` 仅限计算机可执行的计算逻辑，有明确输入/处理步骤/输出 (PageRank、KNN、TF-IDF、各类神经网络模型)。method 是研究操作程序，algorithm 是计算机执行的计算步骤。\n"
    "- `instrument` 仅限测量工具、问卷、量表、评价指标或采集数据的工具。计算机、机器人、VR/AR设备等普通设备不得标 instrument；可运行的软件/平台/系统 → software_system。\n"
    "- `phenomenon` 覆盖可观察的状态、行为模式或变化表现 (数据孤岛、信息茧房，及带时间方向的增长/衰退/扩散/转变)。业务职能、服务模式、技术概念不得因难分类而标 phenomenon。\n\n"
    "### 技术类实体临时规则 — 当前本体无 Technology:\n"
    "- 5G/5G技术/第五代移动通信系统: 若强调技术标准/通信制式 → Artifact.Normative.standard；若泛指技术体系或应用概念 → Abstract.Conceptual.concept。\n"
    "- 人工智能技术/大数据/云计算/VR/AR: 若作为技术领域、技术体系或应用概念 → Abstract.Conceptual.concept；若是具体可运行系统/平台/软件 → Artifact.System.software_system。\n"
    "- 电子计算机/机器人/智能设备: 若只是设备种类且无具体名称，优先判 invalid；若原句作为技术概念讨论 → Abstract.Conceptual.concept；只有用于测量/采集数据才可标 instrument。\n"
    "- 图书馆自动化、档案管理自动化、保存文件遗产等业务职能/理念 → Abstract.Conceptual.concept；只有强调历史时段推进才考虑 stage。\n\n"
    "### `other` 使用规则:\n"
    "各 L2 分组下无法归入具体 L3 (如政策倡导者、科普作家等其他个人，或混合职能机构) 时标 other，other_suggestion 必填建议类型描述；不得用 other 逃避精细分类——先用 concept/paper 等通用兜底类型，仍无法归类才用 other。\n\n"
    "## Workflow\n"
    "1. 读取每个候选实体及其原句\n"
    "2. 有效性验证 → 无效则填 invalid_reason, 跳过后续步骤\n"
    "3. 判定 L1 (Agent|Artifact|Abstract|Event)\n"
    "4. 在 L1 下选定 L2\n"
    "5. 在 L2 下选定 L3 (type_code + label)\n"
    "6. 按 OutputFormat 输出紧凑 JSON 对象\n\n"
    "## Background\n"
    "### 分类体系 (L1 → L2 → L3)\n"
    "{taxonomy_text}\n\n"
    "## OutputFormat\n"
    "严格 JSON 对象，不含 markdown 代码块。数组顺序 = 输入实体顺序，长度必须等于输入实体数。\n"
    "根对象: {{\"c\": [ ... ]}}\n"
    "每个元素是**位置固定的紧凑数组**，不要写字段名:\n"
    "  [\"<l3_type_code>\", \"<reason ≤12字>\"]                  有效实体\n"
    "  [\"!\", \"<invalid_reason ≤12字>\"]                        无效实体\n"
    "  [\"other\", \"<reason ≤12字>\", \"<l1>\", \"<l2>\", \"<other_suggestion>\"]  仅当 l3_type_code=other\n"
    "- 除 other 分支外，不输出 entity_id / l1 / l2 / l2_label / l3_label / evidence，"
    "这些由下游按 l3_type_code 自动推导或继承\n"
    "- reason 必须控制在 12 字内，只写决定性依据\n\n"
    "## Examples\n"
    "### 例1 — scholar vs practitioner 区分\n"
    "实体: 罗香林 | 原句: 建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。\n"
    "分类: [\"scholar\",\"以研究/知识生产为主要职能\"]\n"
    "规则: scholar 产出是知识本身(论文/著作)；practitioner 产出是专业服务(咨询/管理)。同一人物以研究身份被评价→scholar。\n\n"
    "### 例2 — book vs journal_article 区分\n"
    "实体: 新版《图书馆学概论》 | 原句: 新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果。\n"
    "分类: [\"book\",\"独立出版物发行,有独立书名\"]\n"
    "规则: 专著有独立 ISBN 和书名，整本出版；期刊论文有卷期页码属于某期刊。\n\n"
    "### 例3 — policy(Artifact) vs policy_initiative(Event) 区分\n"
    "实体A: HR1858 | 原句: HR1858很好地平衡了反盗与利用信息制作新数据库产品需求之间的关系。\n"
    "分类: [\"policy\",\"政策文件本身,是制品\"]\n"
    "实体B: 文献检索与利用课 | 原句: 教育部下发了在高校开设「文献检索与利用课」的红头文件,大大推动了其发展。\n"
    "分类: [\"policy_initiative\",\"政策实施过程中的制度性行动事件\"]\n"
    "规则: policy 是政策文件本身(制品, Artifact)；policy_initiative 是政策实施事件(事件, Event)。判断语境评价的是文件内容还是实施行动。\n\n"
    "### 例4 — concept vs definition 区分\n"
    "实体: 知识流动 | 原句: Szulanski认为知识流动是在特定环境下知识从来源方到接收方的传递过程。\n"
    "分类: [\"definition\",\"对概念边界的特定界定方式,含有提出者\"]\n"
    "规则: concept 是命名单元本身；definition 是对命名单元边界的特定界定(通常含提出者)。同一概念可有多个 definition。\n\n"
    "### 例5 — concept vs research_field/method/phenomenon 区分\n"
    "实体A: 信息检索 | 原句: 信息检索在图书情报学研究中形成了稳定的问题域和方法传统。\n"
    "分类: [\"research_field\",\"作为研究领域被评价\"]\n"
    "实体B: 引文分析法 | 原句: 本文采用引文分析法揭示知识流动路径。\n"
    "分类: [\"method\",\"完整研究方法\"]\n"
    "实体C: 信息过载 | 原句: 网络环境中的信息过载加重了用户筛选负担。\n"
    "分类: [\"phenomenon\",\"现实信息问题,非术语本身\"]\n"
    "实体D: 信息素养 | 原句: 「信息素养」这一概念强调个体识别、获取和评价信息的能力。\n"
    "分类: [\"concept\",\"原句讨论概念名本身\"]\n"
    "规则: concept 只收术语/概念名本身；若原句评价研究领域、方法或现实现象，必须归入更具体类别。\n\n"
    "### 例6 — 技术类实体边界 (无 Technology 时)\n"
    "实体A: 5G技术 | 原句: 5G技术为智慧图书馆提供高速低时延连接。\n"
    "分类: [\"concept\",\"技术体系概念,非算法\"]\n"
    "实体B: 第五代移动通信系统 | 原句: 第五代移动通信系统作为通信制式标准支撑移动互联。\n"
    "分类: [\"standard\",\"通信制式/技术标准\"]\n"
    "实体C: VR/AR | 原句: VR/AR被用于构建沉浸式阅读场景。\n"
    "分类: [\"concept\",\"应用技术概念,非研究技术手段\"]\n"
    "实体D: 电子计算机 | 原句: 电子计算机推动了图书馆自动化的发展。\n"
    "分类: [\"concept\",\"技术概念,非测量工具\"]\n"
    "规则: 技术领域/技术体系不标 method/algorithm；普通设备不标 instrument；5G 不标 algorithm。\n\n"
    "### 例7 — 时间/地名过滤\n"
    "实体A: 2005年 | 原句: 2005年档案信息化建设进一步推进。\n"
    "分类: [\"!\",\"单纯时间表达,非发展阶段实体\"]\n"
    "实体B: 深圳 | 原句: 深圳和青岛的数字图书馆建设进展较快。\n"
    "分类: [\"!\",\"单纯地名,非具体治理机构\"]\n"
    "规则: 年份不是 stage；城市名不是 governance，除非原句明确指城市政府/行政主体。\n\n"
    "### 例8 — Organization 子类型选择 (research vs service vs governance)\n"
    "实体A: 哥伦比亚大学 | 分类: [\"research\",\"以知识生产为主要职能的大学\"]\n"
    "实体B: 沈阳师范大学图书馆 | 分类: [\"service\",\"以专业服务为主要职能的图书馆\"]\n"
    "实体C: 广州市档案局 | 分类: [\"governance\",\"有行政/治理职能的政府机构\"]\n"
    "规则: 以核心职能判断。大学/研究院→research; 图书馆/档案馆→service; 学会/协会→professional; 政府部门→governance; 出版社→publishing。\n\n"
    "### 例9 — conference_paper(Artifact) vs conference_meeting(Event) 区分\n"
    "实体: 第二届全国灰色文献年会 | 原句: 本次会议在2018年第一次全国灰色文献年会的基础上进行了深入探讨。\n"
    "分类: [\"conference_meeting\",\"具体召开的学术会议事件\"]\n"
    "规则: conference_paper 是会议论文(制品, Artifact)；conference_meeting 是会议召开本身(事件, Event)。\n\n"
    "## Few-Shot 参考 (来自动态术语库的高置信度实体类型)\n"
    "以下是从已确认术语库中检索到的与当前实体相似的参考分类，仅供参考:\n"
    "{few_shot_hints}\n\n"
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
    V4.3: + RAG few-shot hints + Self-Consistency voting
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        dynamic_term_db: Optional[DynamicTermDB] = None,
        batch_size: int = 12,
        enable_voting: bool = False,
        voting_rounds: int = 3,
        voting_temperature: float = 0.3,
        enable_rag: bool = False,
        rag_k_examples: int = 5,
    ):
        self.llm = llm or LLMClient()
        self.dynamic_term_db = dynamic_term_db
        self.batch_size = batch_size
        self.enable_voting = enable_voting
        self.voting_rounds = voting_rounds
        self.voting_temperature = voting_temperature
        self.enable_rag = enable_rag
        self.rag_k_examples = rag_k_examples

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

        # ── RAG Few-Shot: 从动态术语库检索相似实体 ──
        few_shot_hints = ""
        if self.enable_rag and self.dynamic_term_db and self.dynamic_term_db.size() > 0:
            hints_parts: list[str] = []
            for e in entities:
                hint = self.dynamic_term_db.get_few_shot_hints(e.mention, k=self.rag_k_examples)
                if hint:
                    hints_parts.append(f"实体[{e.mention}]的参考:\n{hint}")
            if hints_parts:
                few_shot_hints = "\n\n".join(hints_parts)
        if not few_shot_hints:
            few_shot_hints = "(暂无相似参考)"

        prompt = CLASSIFICATION_PROMPT.format(
            taxonomy_text=taxonomy_text,
            entities_text=entities_text,
            sentence=sentence,
            few_shot_hints=few_shot_hints,
        )

        schema_hint = (
            '{"c":[["<l3_type_code>","<reason <=12字>"],["!","<invalid_reason <=12字>"]]}'
        )

        # ── Voting or single call ──
        if self.enable_voting:
            data, _all, _agr = self.llm.call_json_voting(
                prompt, [], self.voting_rounds, self.voting_temperature, schema_hint
            )
        else:
            data = self.llm.call_json(prompt, [], schema_hint)

        raw = None
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("c")
        if not isinstance(raw, list):
            return self._fallback_results(entities, sentence, sentence_id)

        results: list[FinalEntityResult] = []
        for idx, entity_ref in enumerate(entities):
            item = raw[idx] if idx < len(raw) else None
            entity_id = entity_ref.entity_id
            valid_entity = bool(entity_ref.is_specific_entity)
            invalid_reason = ""
            l3_code = ""
            reason = ""
            other_suggestion = ""
            l1 = l2 = ""
            l2_label = l3_label = ""
            evidence = entity_ref.evidence

            if isinstance(item, dict):
                # 兼容旧格式
                entity_id = item.get("entity_id") or entity_ref.entity_id
                valid_entity = bool(item.get("valid_entity", True))
                invalid_reason = item.get("invalid_reason", "")
                l3_code = item.get("l3_type_code", "")
                reason = item.get("reason", "")
                other_suggestion = item.get("other_suggestion", "")
                l1 = item.get("l1", "")
                l2 = item.get("l2", "")
                l2_label = item.get("l2_label") or get_l2_label(l2)
                l3_label = item.get("l3_label") or get_l3_label(l3_code)
                evidence = item.get("evidence") or entity_ref.evidence
            elif isinstance(item, (list, tuple)):
                # 紧凑格式: [l3_type_code, reason] | ["!", invalid_reason]
                vals = list(item)
                code = str(vals[0]).strip() if len(vals) > 0 and vals[0] is not None else ""
                note = str(vals[1]).strip() if len(vals) > 1 and vals[1] is not None else ""
                if code == "!":
                    valid_entity = False
                    invalid_reason = note
                elif code == "other":
                    # other 无既有 L3 可推导, 需由 LLM 直接给出 L1/L2
                    l3_code = "other"
                    reason = note
                    l1 = str(vals[2]).strip() if len(vals) > 2 and vals[2] else ""
                    l2 = str(vals[3]).strip() if len(vals) > 3 and vals[3] else ""
                    other_suggestion = (
                        str(vals[4]).strip() if len(vals) > 4 and vals[4] else ""
                    )
                else:
                    # 空值表示沿用 Agent 2 的候选 L3
                    l3_code = code or entity_ref.candidate_l3
                    reason = note

            if valid_entity:
                if not l1 or not l2:
                    l1, l2 = get_l1l2_from_l3(l3_code)
                l2_label = l2_label or get_l2_label(l2)
                l3_label = l3_label or get_l3_label(l3_code)

            result = FinalEntityResult(
                sentence_id=sentence_id,
                entity_id=entity_id,
                sentence=sentence,
                entity=entity_ref.mention,
                normalized_name=entity_ref.normalized_name,
                valid_entity=valid_entity,
                invalid_reason=invalid_reason,
                l1=l1,
                l2=l2,
                l2_label=l2_label,
                l3_type_code=l3_code,
                l3_label=l3_label,
                l4_matched_term="",
                l4_term_type=None,
                evidence=evidence,
                reason=reason,
                other_suggestion=other_suggestion,
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
