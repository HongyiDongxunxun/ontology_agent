"""
pipeline.evaluative_relation_agent -- evaluation relation extraction.

Reads one sentence plus its extracted entity list and returns evaluation
subject/object relations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient


EVALUATIVE_RELATION_PROMPT = """
你是一名学术评价信息抽取专家。

你的任务是根据一个句子及其已抽取出的实体列表，识别句子中的所有评价关系（Evaluation Relation）。

一、任务说明

请判断该句是否包含评价（Evaluation）。

评价是指作者、引用文献作者或其他评价主体，对某一对象作出的正向、负向或中性的判断、概括、评价、比较或总结。

评价通常包含但不限于以下表达：

重要
有效
丰富
较高
较低
明显
成熟
完善
不足
较好
优于
落后
快
慢
提高
降低
具有……价值
存在……问题
有待……
……能力较强
……效果较好

如果整个句子只是客观事实描述，没有评价，则返回空结果。

二、评价主体判定规则

评价主体（Evaluation Subject）是作出评价的人或来源，不是执行动作的主体（Action Agent）。

例如：

错误：

国家档案局出台了政策。

国家档案局属于行为主体，不属于评价主体。

正确：

档案信息化建设的步伐越走越快。

评价主体为：

_paper_author

评价主体仅允许以下四种类型：
（1）显性实体

如果句中明确指出某人、某机构进行了评价，则评价主体填写对应实体ID。

例如：

张三认为……

主体：

1_e3

（2）本文作者

如果句中没有显式评价主体，评价实际来自当前论文作者，则填写：

_paper_author

例如：

现有研究仍存在不足。

档案信息化建设的步伐越走越快。

均属于：

_paper_author

（3）引用文献作者

如果评价属于引用文献，而不是当前论文作者，则填写：

_cite[12]

其中数字对应句中的引用编号。

例如：

[12]指出……

返回：

_cite[12]

如果同时引用多个文献：

[3,5]

返回：

_cite[3][5]

（4）无法确定

若依据当前句无法确定评价主体，则填写：

_unknown

三、评价客体判定规则

评价客体（Evaluation Object）必须是被评价的对象。

评价客体必须优先对应提供的实体列表中的实体。

请输出对应实体ID。

例如：

entity_id:
1_e6
entity:
档案信息化建设

则返回：

1_e6

不要重新创建实体。

若评价对象不存在于实体列表中

请返回：

"_missing_entity"

同时增加字段：

object_text

填写原始文本。

例如：

{
    "object":"_missing_entity",
    "object_text":"相关研究成果"
}

不要自行创造新的entity_id。

四、多评价关系

一个句子可能包含多个评价关系。

例如：

方法A计算效率较高，鲁棒性较好，但泛化能力不足。

应返回三条评价关系。

五、评价方面与评价内容

请识别评价方面（aspect）和评价内容（opinion）。

aspect 是评价对象的具体维度，例如作者分布、研究水平、研究质量、理论基础、应用效果、区域分布。
aspect 不是实体，不要把 aspect 当作 object。
如果没有明确评价方面，aspect 返回 null。

opinion 是原文中的评价表达，例如不合理、偏低、不足、步伐越走越快、取得了一定成绩。

例如：

作者分布不合理

aspect = 作者分布
opinion = 不合理

研究水平偏低

aspect = 研究水平
opinion = 偏低

中西部地区研究不足

aspect = null
opinion = 不足

六、评价依据

请输出支持该评价的最小文本片段（Evidence）。

要求：

尽可能短
能完整表达评价
不要输出整句话

例如：

档案信息化建设的步伐越走越快

而不是整段。

七、输出格式

输出JSON，不允许输出任何解释。

格式如下：

{
  "has_evaluation": true,
  "relations": [
    {
      "subject": "_paper_author",
      "object": "1_e6",
      "aspect": null,
      "opinion": "步伐越走越快",
      "evidence": "档案信息化建设的步伐越走越快"
    }
  ]
}

若没有评价：

{
  "has_evaluation": false,
  "relations": []
}

八、特别注意
评价主体不是行为主体。
评价客体不是动作对象，而是被评价对象。
一个句子可能没有评价。
一个句子可能有多个评价关系。
不允许虚构实体ID。
优先使用提供的实体ID作为评价客体。
当评价对象不存在于实体列表时，使用"_missing_entity"。
当评价主体属于本文作者时，统一使用"_paper_author"。
当评价主体属于引用文献时，统一使用"_cite[引用编号]"。
除JSON外，不输出任何额外内容。

九、输入

句子：
{sentence}

实体列表：
{entities_text}
""".strip()

EVALUATIVE_RELATION_PROMPT = (
    EVALUATIVE_RELATION_PROMPT
    + """

十、学术评价对象与评价方面增强规则

请明确区分：
Entity: 具有独立语义、可作为知识图谱节点的对象。
Evaluation Object: 评价关系中被评价的核心对象，必须优先使用实体列表中的 entity_id。
Evaluation Aspect: 评价对象的评价维度，不是实体。
Opinion: 评价表达或评价内容。

不要直接寻找“实体 + 评价词”，而要寻找：
评价主体 subject、评价对象 object、评价方面 aspect（可选）、评价内容 opinion。

以下通常是 aspect，不应作为 object：
作者分布、研究水平、研究质量、理论基础、应用效果、区域分布。

若句子评价的是研究对象的某个维度，应把核心研究对象放入 object，把维度放入 aspect。
例如“农村图书馆研究存在作者分布不合理、研究水平偏低的问题”：
object = 农村图书馆研究对应的 entity_id
aspect = 作者分布
opinion = 不合理
aspect = 研究水平
opinion = 偏低

若没有明确评价方面，则 aspect = null。
例如“档案信息化建设的步伐越走越快”：
object = 档案信息化建设对应的 entity_id
aspect = null
opinion = 步伐越走越快

输出格式以本节为准，必须使用：
{
  "has_evaluation": true,
  "relations": [
    {
      "subject": "_paper_author",
      "object": "entity_id|_missing_entity",
      "aspect": "评价方面或null",
      "opinion": "评价表达",
      "evidence": "最小评价证据",
      "object_text": "仅当 object 为 _missing_entity 时填写"
    }
  ]
}

不要输出 polarity。
"""
).strip()


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

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "has_evaluation": self.has_evaluation,
            "relations": [r.to_dict() for r in self.relations],
        }


class EvaluativeRelationAgent:
    """Agent for evaluation relation extraction."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(
        self,
        sentence_id: str,
        sentence: str,
        entities: list[dict],
    ) -> SentenceRelationOutput:
        if not sentence:
            return SentenceRelationOutput(sentence_id, sentence, False, [])

        entity_ids = {
            str(item.get("entity_id", "")).strip()
            for item in entities
            if str(item.get("entity_id", "")).strip()
        }
        prompt = (
            EVALUATIVE_RELATION_PROMPT
            .replace("{sentence}", sentence)
            .replace("{entities_text}", self._format_entities(entities))
        )
        data = self.llm.call_json(
            prompt,
            {"has_evaluation": False, "relations": []},
        )
        return self._parse_output(sentence_id, sentence, data, entity_ids)

    @staticmethod
    def _format_entities(entities: list[dict]) -> str:
        compact_entities = []
        for item in entities:
            compact_entities.append(
                {
                    "entity_id": item.get("entity_id", ""),
                    "entity": item.get("entity", ""),
                    "normalized_name": item.get("normalized_name", ""),
                    "valid_entity": item.get("valid_entity", True),
                    "l1": item.get("l1", ""),
                    "l2": item.get("l2", ""),
                    "l3_type_code": item.get("l3_type_code", ""),
                    "evidence": item.get("evidence", ""),
                }
            )
        return json.dumps(compact_entities, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_output(
        sentence_id: str,
        sentence: str,
        data: object,
        entity_ids: set[str],
    ) -> SentenceRelationOutput:
        if not isinstance(data, dict):
            return SentenceRelationOutput(sentence_id, sentence, False, [])

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
        )
