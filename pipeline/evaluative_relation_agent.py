"""
pipeline.evaluative_relation_agent -- evaluation relation extraction (Agent 1).

Reads one sentence and returns evaluation subject/object relations
plus entities extracted from those relations. Entities are passed to the
next agent for supplementary entity extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient


EVALUATIVE_RELATION_PROMPT = """
## Academic Evaluation Object Rules (增强规则)
本任务采用"先找评价关系，后抽取实体"的关系优先策略：
1. 先识别句中的评价触发、opinion、aspect 和 evidence。
2. 再通过 Object Resolution 确定每条评价真正指向的 object 文本。
3. 最后把这些 object 文本作为必须进入 entities 的候选实体，生成实体列表并用 entity_id 回填 relation.object。
4. 若某个短语只是 aspect，不要放入 entities；若某个短语是被评价 object，即使它不是传统命名实体，也应作为 Entity 抽取。
5. entities 必须覆盖所有可解析的 relation.object。不要先因为实体列表缺失而把关系 object 写成 _missing_entity。

请明确区分四类成分：
1. Entity: 文本中具有独立语义、可作为知识图谱节点的对象。
2. Evaluation Object: 评价关系中被评价的核心对象，通常来自 Entity，并在 relation.object 中填写对应 entity_id。
3. Evaluation Aspect: 评价对象的某个评价维度，不是 Entity。
4. Opinion: 评价表达或评价内容。

实体抽取不要只按传统 NER。学术评价文本中的研究对象、领域主题和复合研究对象也应抽取为 Entity，例如：
- 农村图书馆研究
- 农村图书馆问题
- 档案信息化建设
- 数字图书馆建设
- 中西部地区研究
- 数字化建设
- 基础理论研究

最小评价对象原则：
若一个名词短语能够整体接受评价词修饰，优先抽取完整短语，而不是拆出内部成分。
- "中西部地区研究不足" -> Entity: 中西部地区研究；不要抽取"中西部地区"。
- "农村图书馆事业发展良好" -> Entity: 农村图书馆事业发展；不要只抽"农村图书馆"。
- "数字信息资源建设存在不足" -> Entity: 数字信息资源建设；不要只抽"数字信息资源"。

当名词短语后接"研究、建设、发展、问题、实践、应用、水平、能力、体系"，且整体构成被评价的研究对象或主题对象时，优先整体抽取。

以下通常不是 Entity，除非原文把它们作为独立研究对象或术语本身讨论：
作者分布、研究水平、研究质量、理论基础、应用效果、区域分布。
它们在评价关系中通常应放入 aspect 字段。

评价关系不要按"实体 + 评价词"机械抽取，而应识别：
subject = 评价主体
object = 被评价的核心对象 entity_id
aspect = 评价方面；没有则为 null
opinion = 评价表达
evidence = 支持该评价的最小原文片段

## 评价对象回溯（Object Resolution）：
先识别 aspect 与 opinion，再判断"这个评价是在评价哪个实体"。不要因为 aspect 不是实体，就直接输出 _missing_entity。
1. 优先绑定已有实体：若 aspect 属于某个已抽取实体的属性、组成部分、发展情况、研究维度或评价维度，object 必须绑定该实体。
2. Aspect 不是 Object：aspect 表示评价维度，object 表示真正被评价的对象。例如"作者分布不合理"若句子讨论"农村图书馆研究"，object=农村图书馆研究，aspect=作者分布。
3. 寻找 aspect 所属对象：当 aspect 出现时，优先向左寻找其所属对象。
   - "数字图書館建設的發展速度較快" -> object=数字图書館建設, aspect=發展速度, opinion=較快。
   - "法明頓計畫在協調布局方面堪稱典範" -> object=法明頓計畫, aspect=協調布局, opinion=堪稱典範。
4. 允许跨短语回溯：object 不一定紧邻 aspect。例如"近年来，档案信息化建设取得快速发展，其理论研究仍存在不足"中，"理论研究/不足"应回溯到"档案信息化建设"。
5. 仅当当前句不存在任何可作为评价对象的实体、aspect 无法归属于任何实体、且上下文无法确定评价对象时，才使用 _missing_entity。
6. Entity 优先原则：多个候选实体时，选择最直接被评价、语义距离最近、且能够完整支撑 aspect 的实体。不要选择地名、时间、修饰语。
7. Aspect 属于 object，不是独立 object。例如"研究水平偏低"：object=农村图书馆研究，aspect=研究水平，opinion=偏低；不要 object=研究水平。

## Positive Examples（正确例子）：
句子1："我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。"
Entity 只抽取"农村图书馆研究"，不要抽取"作者分布"或"研究水平"。
Relations:
[
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"作者分布","opinion":"不合理","evidence":"作者分布不合理"}},
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"研究水平","opinion":"偏低","evidence":"研究水平偏低"}}
]

句子2："中西部地区研究不足。"
Entity: 中西部地区研究
Relation: object=<中西部地区研究的entity_id>, aspect=null, opinion=不足。

关系输出字段必须使用 subject、object、aspect、opinion、evidence。不要输出 polarity。

## Negative Examples（非评价关系）：
以下情况不要抽取评价关系，has_evaluation=false，relations=[]。

例1：事实描述（无评价）
句子：
“用户服务平台提供了按资源类型、标题、作者、关键词等多种检索途径查找资源的功能。”

不要抽取：
object=用户服务平台
opinion=提供多种检索途径

原因：
“提供、支持、包含、具有”等描述功能或事实，不代表评价。


例2：定义说明（无评价）
句子：
“结构化是指将获取的知识内容加以归纳和整理，使之条理化、纲领化。”

不要抽取评价关系。

原因：
“是指”属于概念定义。


例3：方法流程描述（无评价）
句子：
“本文采用文献计量法对相关研究进行分析。”

不要抽取评价关系。

原因：
“采用、使用、利用”表示研究方法，不表示评价。


例4：分类枚举（无评价）
句子：
“研究内容主要包括理论研究、应用研究和实践研究。”

不要抽取评价关系。

原因：
“包括、分为、涉及”属于分类描述。


例5：功能/能力陈述（无评价）
句子：
“该系统能够实现文献检索、数据分析和结果展示。”

不要抽取评价关系。

原因：
“能够实现”描述功能，不等于“功能好”或“效果显著”。


例6：避免将Aspect误认为Object
句子：
“农村图书馆研究水平偏低。”

错误：
object=研究水平

正确：
object=农村图书馆研究
aspect=研究水平
opinion=偏低

原因：
研究水平是评价方面，不是被评价对象。

注意还有以下几类也是Negative Examples：

例7：“发展起来、出现、形成、产生”描述变化过程，不代表评价
句子：
自动文摘研究逐渐发展起来。

不要抽取评价关系。

例8：“主要集中于”描述研究分布，不表示好坏评价。
句子：
目前的研究主要集中于文摘生成方法。

不要抽取评价关系。

例9：“实现、完成、构建、提出”描述研究工作，不表示效果评价。
句子：
该模型实现了自动摘要生成。

不要抽取评价关系。

## OutputFormat
严格 JSON，不含 markdown 代码块。entities 和 relations 数组无内容则为空数组 []。
{{
  "has_evaluation": true,
  "entities": [
    {{
      "entity_id": "e1",
      "entity": "<原文精确短语>",
      "normalized_name": "<规范化实体名>",
      "evidence": "<原文证据片段>"
    }}
  ],
  "relations": [
    {{
      "subject": "_paper_author|_cite[N]|_unknown",
      "object": "<entity_id>",
      "aspect": "<评价方面或 null>",
      "opinion": "<评价表达>",
      "evidence": "<评价依据文本片段>"
    }}
  ]
}}

## Input
{statement}
""".strip()


@dataclass
class EvaluativeRelation:
    subject: str
    object: str
    aspect: Optional[str]
    opinion: str
    evidence: str
    object_text: str = ""

    def to_dict(self) -> dict:
        data = {
            "subject": self.subject,
            "object": self.object,
            "aspect": self.aspect,
            "opinion": self.opinion,
            "evidence": self.evidence,
        }
        if self.object == "_missing_entity" and self.object_text:
            data["object_text"] = self.object_text
        return data


@dataclass
class SentenceRelationOutput:
    sentence_id: str
    sentence: str
    has_evaluation: bool
    relations: list[EvaluativeRelation] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "has_evaluation": self.has_evaluation,
            "relations": [r.to_dict() for r in self.relations],
            "entities": self.entities,
        }


class EvaluativeRelationAgent:
    """Agent 1: evaluation relation extraction + entity extraction."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, sentence_id: str, sentence: str) -> SentenceRelationOutput:
        if not sentence:
            return SentenceRelationOutput(sentence_id, sentence, False, [], [])

        prompt = EVALUATIVE_RELATION_PROMPT.replace("{statement}", sentence)
        data = self.llm.call_json(
            prompt,
            {"has_evaluation": False, "relations": [], "entities": []},
        )
        return self._parse_output(sentence_id, sentence, data)

    @staticmethod
    def _parse_output(
        sentence_id: str,
        sentence: str,
        data: object,
    ) -> SentenceRelationOutput:
        if not isinstance(data, dict):
            return SentenceRelationOutput(sentence_id, sentence, False, [], [])

        # ── 解析实体 ──
        raw_entities = data.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        parsed_entities: list[dict] = []
        entity_ids: set[str] = set()
        for i, item in enumerate(raw_entities):
            if not isinstance(item, dict):
                continue
            eid = str(item.get("entity_id", f"e{i + 1}")).strip()
            entity_name = str(item.get("entity", "")).strip()
            if not entity_name:
                continue
            entity_ids.add(eid)
            parsed_entities.append({
                "entity_id": eid,
                "entity": entity_name,
                "normalized_name": str(item.get("normalized_name", entity_name)).strip(),
                "evidence": str(item.get("evidence", "")).strip(),
            })

        # ── 解析评价关系 ──
        raw_relations = data.get("relations", [])
        if not isinstance(raw_relations, list):
            raw_relations = []

        relations: list[EvaluativeRelation] = []
        for item in raw_relations:
            if not isinstance(item, dict):
                continue

            subject = str(item.get("subject", "")).strip()
            obj = str(item.get("object", "")).strip()
            aspect_value = item.get("aspect")
            aspect = None if aspect_value is None else str(aspect_value).strip()
            opinion = str(item.get("opinion", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            object_text = str(item.get("object_text", "")).strip()

            if not subject or not obj or not opinion or not evidence:
                continue
            if obj != "_missing_entity" and obj not in entity_ids:
                obj = "_missing_entity"
                if not object_text:
                    object_text = str(item.get("object", "")).strip()

            relations.append(
                EvaluativeRelation(
                    subject=subject,
                    object=obj,
                    aspect=aspect,
                    opinion=opinion,
                    evidence=evidence,
                    object_text=object_text,
                )
            )

        return SentenceRelationOutput(
            sentence_id=sentence_id,
            sentence=sentence,
            has_evaluation=bool(relations),
            relations=relations,
            entities=parsed_entities,
        )