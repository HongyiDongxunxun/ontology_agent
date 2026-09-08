"""
pipeline.evaluative_relation_agent -- evaluation relation extraction (Agent 1).

Reads one sentence and returns evaluation subject/object relations.
Subject/object are emitted as raw text spans (not entity_ids); entity
recognition is delegated to Agent 2 (EntityExtractionAgent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient


def _normalize_name(name: str) -> str:
    """规范化实体名用于匹配 (去空格/全角括号/书名号)"""
    name = (name or "").strip()
    for a, b in (("（", "("), ("）", ")"), ("《", ""), ("》", "")):
        name = name.replace(a, b)
    return name


EVALUATIVE_RELATION_PROMPT = """

# Task

你是一个通用的评价关系抽取专家。

你的任务是从给定文本中识别评价关系，输出评价关系的主客体。
主客体直接使用原文精确短语，不要引用实体编号，也不要输出独立的实体列表。
实体的识别与分类由下游 Agent 完成，你只需保证 subject/object 是原文中出现的精确短语。

核心流程：

评价识别
→ 确定评价主体、评价对象、评价方面、评价内容
→ Object Resolution（确定真正被评价的对象）
→ 判断对象是否符合评价客体要求
→ 输出 Relation（subject/object 用原文短语）

# 一、基本定义

### Entity

文本中具有独立语义、可以作为知识图谱节点的对象。

### Evaluation Object

评价关系中真正被评价的对象。

注意：
Entity 与 Evaluation Object 不完全等价。
一个对象可以是 Entity，但未必允许作为 Evaluation Object。

### Aspect

评价对象的某个评价维度或方面，不是独立的 Evaluation Object。

### Opinion

对评价对象作出的评价性表达。

### Evidence

能够直接支持该评价关系的最小原文片段。

# 二、评价识别

只有文本真正表达了对某个对象的评价、判断、比较、价值判断、优劣判断、效果判断、问题判断或改进判断时，才建立评价关系。

不要因为出现“较高、明显、具有、取得、存在、发展”等词就机械判断为评价，必须结合完整语义判断。

以下通常不是评价：

* 事实、事件、过程描述
* 定义、分类、说明
* 研究或工作行为
* 方法或技术使用
* 功能描述
* 单纯属性陈述
* 统计结果或数据罗列
* 纯粹的研究计划

例如：

“该系统支持全文检索。”
→ 不是评价。

“本文采用该方法进行分析。”
→ 不是评价。

“该项目具有较高的社会影响力。”
→ 是评价。

# 三、Evaluation Object

评价对象必须是文本中被明确指称、能够独立作为评价对象的实体或研究对象。

允许的评价对象包括但不限于：

* 具体人物或群体
* 机构、组织、团队
* 国家、地区或其他明确地域实体
* 出版物、期刊、论文、图书等文献对象
* 项目、计划、工程、倡议等
* 明确的研究对象、研究主题或复合对象

不要因为对象属于抽象领域就自动排除；关键是判断它在当前文本中是否被作为一个独立对象进行评价。

但以下通常不作为 Evaluation Object：

* 单纯的评价方面（如“质量”“水平”“效果”）
* 单纯的评价词
* 无独立指称的代词或泛指表达
* 仅作为背景修饰的时间、地点等成分
* 仅表示方法、技术、模型、算法、工具或功能，而任务本身不允许其作为评价对象的对象

如果项目提供了明确的评价客体类型要求，以项目要求为最高优先级。

# 四、Object Resolution

确定评价对象时，不要简单选择距离评价词最近的名词。

应优先判断：

1. 该评价实际针对谁/什么；
2. Aspect 属于哪个对象；
3. 是否存在前文可以回溯的对象；
4. 哪个候选对象能够完整解释当前评价。

例如：

“某机构的研究水平较高。”

object = 某机构
aspect = 研究水平
opinion = 较高

不要：

object = 研究水平

例如：

“某地区在数字化建设方面发展较快。”

object = 某地区
aspect = 数字化建设
opinion = 发展较快

不要：

object = 数字化建设

如果评价对象在句中没有直接出现，但可以根据上下文明确确定，则回溯到对应对象。

只有无法确定评价对象时，才使用：

object = _missing_entity

# 五、Subject

subject 表示实际作出评价的主体，而不是执行某个动作的主体。

允许：

* _paper_author：当前论文作者
* _cite[N]：第 N 篇被引文献发出的评价
* 具体人物名称：当评价主体是明确人物时
* _unknown：无法确定主体

注意：

1. _cite[...] 中只能填写引文编号，不得填写人物姓名、作者名或缩写。
2. N 是占位符，必须替换为原文中出现的真实数字（如 [3] → _cite[3]），不得输出 _cite[N]。
3. 如果无法从原文确定具体引文编号（如只看到“有研究指出”“相关文献认为”而无编号），不得使用 _cite[?]，应降级为 _unknown。
4. 不得创造原文中不存在的引文编号。

# 六、Aspect 与 Opinion

### Aspect

填写评价对象被评价的具体方面；如果没有明确方面，则为 null。

### Opinion

填写原文中的完整评价表达。

不要只抽取“具有、存在、取得、实现、提出、表现为”等功能性词语。

例如：

“某论文具有较高的理论价值。”

正确：
aspect = 理论价值
opinion = 具有较高的理论价值

错误：
opinion = 具有

# 七、多个评价关系

一个句子可能包含多个评价关系。

如果一个句子同时评价多个方面或多个对象，应分别输出多个 Relation。

# 八、输出要求

严格输出 JSON，不要输出解释，不要输出 Markdown。
不要输出 entities 字段；实体识别由下游 Agent 完成。

如果不存在评价：

{
"has_evaluation": false,
"relations": []
}

如果存在评价但评价对象不符合任务要求：

{
"has_evaluation": true,
"relations": []
}

如果存在合法评价关系：

{
"has_evaluation": true,
"relations": [
{
"subject": "_paper_author|_cite[<真实编号>]|<人物名字>|_unknown",
"object": "<原文精确短语，不要填 entity_id>",
"aspect": "<评价方面或 null>",
"opinion": "<完整评价表达>",
"evidence": "<最小评价证据片段>"
}
]
}

# Examples

## Example 1：合法评价

句子：
“某机构在人才培养方面表现突出。”

结果：

{
"has_evaluation": true,
"relations": [
{
"subject": "_paper_author",
"object": "某机构",
"aspect": "人才培养",
"opinion": "表现突出",
"evidence": "某机构在人才培养方面表现突出"
}
]
}

## Example 2：研究对象作为评价对象

句子：
“农村图书馆研究取得了一定成绩。”

结果：

{
"has_evaluation": true,
"relations": [
{
"subject": "_paper_author",
"object": "农村图书馆研究",
"aspect": null,
"opinion": "取得了一定成绩",
"evidence": "农村图书馆研究取得了一定成绩"
}
]
}

## Example 3：Aspect 不得成为 Object

句子：
“某机构的研究水平较高。”

正确：
object = 某机构
aspect = 研究水平
opinion = 较高

错误：
object = 研究水平

## Example 4：事实描述，不是评价

句子：
“该系统支持全文检索和批量下载功能。”

结果：

{
"has_evaluation": false,
"relations": []
}

## Example 5：研究行为，不是评价

句子：
“本文采用该方法对相关文献进行了分析。”

结果：

{
"has_evaluation": false,
"relations": []
}

## Example 6：有评价，但对象不符合任务要求

句子：
“该方法具有较高的准确率。”

如果当前任务规定“方法”不能作为 Evaluation Object，则：

{
"has_evaluation": true,
"relations": []
}

## Example 7：引用主体

句子：
“文献[3]认为，该机构具有较强的创新能力。”

结果中的 relation：

{
"subject": "_cite[3]",
"object": "该机构",
"aspect": "创新能力",
"opinion": "具有较强的创新能力",
"evidence": "该机构具有较强的创新能力"
}

## Example 8：无编号引用主体

句子：
“相关文献认为，该机构具有较强的创新能力。”

错误（不得使用 _cite[?]、_cite[N] 或猜测编号）：
subject = "_cite[?]"
subject = "_cite[N]"
subject = "_cite[1]"

正确：
subject = "_unknown"

## Example 9：作者名不得填入 _cite

句子：
“许晓东等认为，该机构具有较强的创新能力。”

错误：
subject = "_cite[许晓东等]"
subject = "_cite[Xiaodong Xu]"

正确：
subject = "许晓东等"

# Input

{statement}
""".strip()



@dataclass
class EvaluativeRelation:
    """评价关系。subject/object 均为原文精确短语或占位符。

    - subject: 占位符(_paper_author / _cite[N] / _unknown) 或具体人物名
    - object: 原文精确短语；当 LLM 无法确定时为 _missing_entity，
              此时 object_text 保留 LLM 原始输出便于审计。
    """
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
    """单句评价关系抽取输出（仅 relations，不含 entities）。

    实体识别由下游 EntityExtractionAgent 完成。本类只关心评价关系，
    其中 subject/object 直接携带原文短语，供 pipeline 后处理与下游
    抽取的实体做名称匹配回填 entity_id。
    """
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
    """Agent 1: evaluation relation extraction only.

    subject/object 使用原文精确短语；实体识别交由 Agent 2。
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, sentence_id: str, sentence: str) -> SentenceRelationOutput:
        if not sentence:
            return SentenceRelationOutput(sentence_id, sentence, False, [])

        prompt = EVALUATIVE_RELATION_PROMPT.replace("{statement}", sentence)
        data = self.llm.call_json(
            prompt,
            {"has_evaluation": False, "relations": []},
        )
        return self._parse_output(sentence_id, sentence, data)

    @staticmethod
    def _parse_output(
        sentence_id: str,
        sentence: str,
        data: object,
    ) -> SentenceRelationOutput:
        if not isinstance(data, dict):
            return SentenceRelationOutput(sentence_id, sentence, False, [])

        # ── 解析评价关系 ──
        # object 直接保留 LLM 输出的原文短语；若 LLM 输出 _missing_entity,
        # 同时附带 object_text 时保留 object_text 便于审计。
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

            # 必填字段缺失 → 丢弃该 relation
            if not subject or not obj or not opinion or not evidence:
                continue
            # _missing_entity 必须附带 object_text, 否则无法回溯原文
            if obj == "_missing_entity" and not object_text:
                object_text = obj  # 至少保留占位符文本

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

        # has_evaluation 表示"句中是否存在真正评价"，独立于 relations 是否为空。
        # 情形B（有评价但客体非法）relations 为空但 has_evaluation 仍为 true，
        # 因此尊重模型返回的 has_evaluation；同时若存在合法 relation 则必为 true。
        model_has_eval = bool(data.get("has_evaluation", False))
        has_evaluation = bool(relations) or model_has_eval

        return SentenceRelationOutput(
            sentence_id=sentence_id,
            sentence=sentence,
            has_evaluation=has_evaluation,
            relations=relations,
        )