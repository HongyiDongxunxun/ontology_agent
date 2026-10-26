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
from .compact_output import derive_evidence


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

严格输出 JSON，不要输出解释，不要输出 Markdown，不要使用代码块。
不要输出 entities 字段；实体识别由下游 Agent 完成。
不要输出 evidence 字段；证据片段由下游按原句自动截取。

根对象固定为：

{"h": <0|1>, "r": [ ... ]}

- h: 1 = 句中存在真正的评价（即使 r 为空）；0 = 不存在评价。
- 无评价时只输出：{"h": 0}
- 有评价但评价对象不符合任务要求：{"h": 1, "r": []}
- r 中每条关系是**位置固定的紧凑数组**，不要写字段名：
  [object, opinion, aspect?, subject?]
  1. object: 原文精确短语，不要填 entity_id
  2. opinion: 原文完整评价表达
  3. aspect: 有明确评价方面时放第 3 位；没有则省略
  4. subject: 仅当 subject 不是 _paper_author 时才放第 4 位；
     此时若 aspect 为空，第 3 位写 null

# Examples

## Example 1：合法评价

句子：
“某机构在人才培养方面表现突出。”

结果：

{"h":1,"r":[["某机构","表现突出","人才培养"]]}

## Example 2：研究对象作为评价对象

句子：
“农村图书馆研究取得了一定成绩。”

结果：

{"h":1,"r":[["农村图书馆研究","取得了一定成绩"]]}

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

{"h":0}

## Example 5：研究行为，不是评价

句子：
“本文采用该方法对相关文献进行了分析。”

结果：

{"h":0}

## Example 6：有评价，但对象不符合任务要求

句子：
“该方法具有较高的准确率。”

如果当前任务规定“方法”不能作为 Evaluation Object，则：

{"h":1,"r":[]}

## Example 7：引用主体

句子：
“文献[3]认为，该机构具有较强的创新能力。”

结果中的 relation：

{"h":1,"r":[["该机构","具有较强的创新能力","创新能力","_cite[3]"]]}

## Example 8：无编号引用主体

句子：
“相关文献认为，该机构具有较强的创新能力。”

错误（不得使用 _cite[?]、_cite[N] 或猜测编号）：
subject = "_cite[?]"
subject = "_cite[N]"
subject = "_cite[1]"

正确：
{"h":1,"r":[["该机构","具有较强的创新能力","创新能力","_unknown"]]}

## Example 9：作者名不得填入 _cite

句子：
“许晓东等认为，该机构具有较强的创新能力。”

错误：
subject = "_cite[许晓东等]"
subject = "_cite[Xiaodong Xu]"

正确：
{"h":1,"r":[["该机构","具有较强的创新能力","创新能力","许晓东等"]]}

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
        if isinstance(data, list):
            # 兼容旧格式: 关系数组
            raw_relations, model_has_eval = data, bool(data)
        elif isinstance(data, dict):
            raw_relations = data.get("r", data.get("relations", []))
            model_has_eval = bool(data.get("h", data.get("has_evaluation", False)))
        else:
            return SentenceRelationOutput(sentence_id, sentence, False, [])

        if not isinstance(raw_relations, list):
            raw_relations = []

        relations: list[EvaluativeRelation] = []
        for item in raw_relations:
            object_text = ""
            if isinstance(item, dict):
                # 兼容旧格式: 具名对象
                subject = str(item.get("subject", "")).strip()
                obj = str(item.get("object", "")).strip()
                aspect_value = item.get("aspect")
                aspect = None if aspect_value is None else str(aspect_value).strip()
                opinion = str(item.get("opinion", "")).strip()
                evidence = str(item.get("evidence", "")).strip()
                object_text = str(item.get("object_text", "")).strip()
            elif isinstance(item, (list, tuple)):
                # 紧凑格式: [object, opinion, aspect?, subject?]
                vals = list(item)
                obj = str(vals[0]).strip() if len(vals) > 0 and vals[0] else ""
                opinion = str(vals[1]).strip() if len(vals) > 1 and vals[1] else ""
                aspect_value = vals[2] if len(vals) > 2 else None
                aspect = None if aspect_value in (None, "") else str(aspect_value).strip()
                subject = str(vals[3]).strip() if len(vals) > 3 and vals[3] else "_paper_author"
                evidence = ""
            else:
                continue

            # 必填字段缺失 → 丢弃该 relation（旧格式仍要求 evidence 非空）
            if not subject or not obj or not opinion:
                continue
            if isinstance(item, dict) and not evidence:
                continue

            # 紧凑格式不输出 evidence，由原句回填
            if not evidence:
                evidence = derive_evidence(sentence, obj, opinion) or opinion or obj

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
        has_evaluation = bool(relations) or model_has_eval

        return SentenceRelationOutput(
            sentence_id=sentence_id,
            sentence=sentence,
            has_evaluation=has_evaluation,
            relations=relations,
        )