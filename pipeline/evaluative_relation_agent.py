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

本任务采用"先找评价关系，后抽取实体"的关系优先策略。核心流程如下：

评价成立
    ↓
Object Resolution（确定被评价对象）
    ↓
Object 类型检查（object 是否属于合法评价客体类型？）
    ↓
┌───────────────┐
│ 合法？        │
└───────────────┘
 ↓             ↓
是             否
↓              ↓
生成 Relation   丢弃 Relation

分步说明：
1. 判断句子是否成立评价（存在真正的评价表达，而非事实/定义/方法/过程/功能/属性描述）。若不成立评价 → has_evaluation=false，relations=[]。
2. 识别句中的 subject、aspect、opinion、evidence。
3. 通过 Object Resolution 确定每条评价真正指向的 object 候选文本。
4. 对 object 做类型检查：判断它是否属于允许的评价客体类型。
5. 若合法 → 将 object 作为 Entity 补充进 entities，生成 entity_id 并回填 relation.object，生成 Relation。
6. 若非法 → 丢弃该 Relation（该 object 如符合 Entity 抽取规则，仍可出现在 entities 中，但不能作为 relation.object）。
7. 若某个短语只是 aspect，不要放入 entities；若某个短语是被评价 object，即使它不是传统命名实体，也应作为 Entity 抽取。
8. entities 必须覆盖所有可解析的 relation.object。不要先因为实体列表缺失而把关系 object 写成 _missing_entity。

## Entity 与 Evaluation Object 的定义区分

请明确区分 Entity 和 Evaluation Object，两者不完全等价。

### Entity（实体）
文本中具有独立语义、可作为知识图谱节点的对象。Entity 的范围较宽，可以包括：
- 人物：学者、研究人员、作者
- 组织：机构、组织、团队、实验室、院系
- 地理/政治实体：国家、地区、区域
- 出版物：期刊、会议、出版物
- 文献：论文、著作、图书、文献
- 项目：项目、计划、工程、倡议
- 概念、术语
- 方法、技术
- 理论、模型、算法
- 工具、平台、系统
- 数据、数据集、语料库
- 研究主题、研究领域
- 其他具有独立语义的对象

### Evaluation Object（评价客体）
评价关系中被评价的核心对象，但必须属于以下允许类型，否则不生成评价关系。

允许作为 Evaluation Object 的类型（封闭清单，仅限以下七类，不得扩展）：
- 学者、研究人员、作者等具体人物
- 机构、组织、团队、实验室、院系等具体组织
- 国家、地区、区域等具体地理/政治实体
- 期刊、会议、出版物
- 论文、著作、图书、文献
- 项目、计划、工程、倡议
- 研究主题、研究领域、复合研究对象（如"农村图书馆研究""档案信息化建设""数字图书馆建设""中西部地区研究"等，通常以"研究/建设/发展/问题/实践/应用/体系"等结尾）

以上七类之外的对象，一律不得作为 Evaluation Object，即使它具有明确独立指称、看起来像可被评价的具体实体。不要以"其他具体实体"为由放宽客体范围。

禁止作为 Evaluation Object 的类型（即使有评价色彩也不抽取）：
- 概念、术语
- 方法、技术
- 理论、模型、算法
- 工具、平台、系统
- 数据、数据集、语料库
- 评价维度、评价方面（Aspect）

核心原则：
"可以被识别为 Entity" 不等于 "可以作为 relation.object"。
Entity 范围较宽，Evaluation Object 范围较窄。
注意：研究主题、研究领域、复合研究对象既是合法 Entity，也可作为合法 Evaluation Object；但概念、方法、技术、理论、模型、算法、工具、平台、系统、数据等仍不能作为 Evaluation Object。

## Entity 抽取规则

实体抽取不要只按传统 NER。学术评价文本中的研究对象、领域主题和复合研究对象也应抽取为 Entity，例如：
- 农村图书馆研究
- 农村图书馆问题
- 档案信息化建设
- 数字图书馆建设
- 中西部地区研究
- 数字化建设
- 基础理论研究

注意：以上对象既可作为 Entity，也可作为合法 Evaluation Object（研究主题/研究领域/复合研究对象属于允许客体类型）。例如"农村图书馆研究"既是 Entity，也可作为评价客体。但如果对象属于概念、方法、技术、理论、模型、算法等禁止类型，则仍不能作为 Evaluation Object。

最小评价对象原则（适用于 Entity 抽取）：
若一个名词短语能够整体接受评价词修饰，优先抽取完整短语，而不是拆出内部成分。
- "中西部地区研究不足" -> Entity: 中西部地区研究；不要抽取"中西部地区"。
- "农村图书馆事业发展良好" -> Entity: 农村图书馆事业发展；不要只抽"农村图书馆"。
- "数字信息资源建设存在不足" -> Entity: 数字信息资源建设；不要只抽"数字信息资源"。

当名词短语后接"研究、建设、发展、问题、实践、应用、水平、能力、体系"，且整体构成被评价的研究对象或主题对象时，优先整体抽取。

以下通常不是 Entity，除非原文把它们作为独立研究对象或术语本身讨论：
作者分布、研究水平、研究质量、理论基础、应用效果、区域分布。
它们在评价关系中通常应放入 aspect 字段。

## 评价关系抽取规则

评价关系不要按"实体 + 评价词"机械抽取，而应识别：
subject = 评价主体
object = 被评价的核心对象 entity_id
aspect = 评价方面；没有则为 null
opinion = 评价表达
evidence = 支持该评价的最小原文片段

## Object Resolution（评价对象回溯）

Object Resolution 只负责回答："这条评价语义指向哪个候选对象？"
先识别 aspect 与 opinion，再判断"这个评价是在评价哪个实体"。不要因为 aspect 不是实体，就直接输出 _missing_entity。
1. 优先绑定已有实体：若 aspect 属于某个已抽取实体的属性、组成部分、发展情况、研究维度或评价维度，object 必须绑定该实体。
2. Aspect 不是 Object：aspect 表示评价维度，object 表示真正被评价的对象。例如"作者分布不合理"若句子讨论"某机构"，object=某机构，aspect=作者分布。
3. 寻找 aspect 所属对象：当 aspect 出现时，优先向左寻找其所属对象。
   - "数字图书馆建设的发展速度较快" -> object 候选=数字图书馆建设, aspect=发展速度, opinion=较快。
   - "法明顿计画在协调布局方面堪称典范" -> object 候选=法明顿计画, aspect=协调布局, opinion=堪称典范。
4. 允许跨短语回溯：object 不一定紧邻 aspect。例如"近年来，档案信息化建设取得快速发展，其理论研究仍存在不足"中，"理论研究/不足"应回溯到"档案信息化建设"。
5. 仅当当前句不存在任何可作为评价对象的实体、aspect 无法归属于任何实体、且上下文无法确定评价对象时，才使用 _missing_entity。
6. Entity 优先原则：多个候选实体时，选择最直接被评价、语义距离最近、且能够完整支撑 aspect 的实体。不要选择仅作背景修饰的时间状语、地点状语或其他修饰语。
   注意：国家、地区、区域本身是合法的评价客体类型。只有当地名仅作为背景状语（如"在中西部地区，……研究不足"中的"中西部地区"只是地点背景）时才不选它；若地名/地区本身正是被评价对象（如"中西部地区在数字化建设方面较为落后"），则应选择该地区作为 object。
7. Aspect 属于 object，不是独立 object。例如"研究水平偏低"：若句子讨论"某机构"，object=某机构，aspect=研究水平，opinion=偏低；不要 object=研究水平。

注意：Object Resolution 仅确定候选对象。确定候选对象后，必须做 Object 类型检查（评价客体资格过滤）：合法则生成 Relation，非法则丢弃 Relation。此外，若整句根本不是评价（事实/定义/方法/过程/功能/属性描述），则 has_evaluation=false 且不生成任何 Relation。

## Evaluation Object Eligibility Filtering（评价客体资格过滤）

Evaluation Object Eligibility Filtering 负责回答："这个对象是否有资格成为 Evaluation Object？"

过滤流程：
Step 1 — 识别评价对象候选 object（来自 Object Resolution 的结果）。
Step 2 — 判断 object 是否属于"允许作为 Evaluation Object 的类型"（参见上方类型列表）。
Step 3 — 如果属于允许类型 → 继续下一阶段（Evaluation Validity Filtering），并将 object 补充为 Entity（如尚未在 entities 中）。
Step 4 — 如果属于禁止类型 → 丢弃整条评价关系。该 object 如符合 Entity 抽取规则，仍可出现在 entities 中，但不能作为 relation.object。
Step 5 — 如果只是因为对象尚未被识别为 Entity，但从语义上明确属于允许类型 → 允许将 object 补充为 Entity，再进入下一阶段。

禁止作为 Evaluation Object 的对象（即使存在评价表达也不生成 relation）：
学术概念、术语、方法、技术、模型、算法、理论、工具、平台、系统、数据集、语料库。
（注意：研究主题、研究领域、复合研究对象已列为允许客体类型，不在此禁止清单中。）

示例（客体资格不合法，丢弃 relation）：
- "开源大模型性能优异。" → 不要输出 object=开源大模型。原因：开源大模型属于技术/概念对象。
- "深度学习方法具有较高准确率。" → 不要输出 object=深度学习方法。原因：方法不是允许的评价客体。
- "Transformer模型表现良好。" → 不要输出 object=Transformer模型。原因：模型不是允许的评价客体。

重要约束：
- 不要因为某个短语具有明显评价表达，就自动认为它是合法评价客体。
- 必须先通过 Evaluation Object Eligibility Filtering。
- 禁止类型的对象即使存在评价表达，也不生成评价关系。

独立指称约束（不扩大类型范围，仅增加指称要求）：
允许七类中的对象必须具有独立、明确的指称。泛称、虚指以及仅作为背景修饰成分的表达不能作为 Object。
- "某地区的研究水平较高。" → 若"某地区"只是泛指、无法确定具体地区，则不能作为 Object。
- "广东省在数字化建设方面较为领先。" → "广东省"具有明确独立指称，可以作为 Object。
- "该期刊的影响力较高。" → 若"该期刊"能通过上下文确定具体期刊，则 Object 回溯到该具体期刊实体；若无法确定，则不作为 Object。

## Evaluation Validity Filtering（评价有效性过滤）

Evaluation Validity Filtering 负责回答："这句话是真正的评价，还是仅仅在陈述事实/定义/方法/过程/功能/属性？"
不要仅因为存在评价词，就认为存在评价关系；也不要仅因为出现某个谓词就机械判定为非评价。

输出关系前，必须先强制回答一个问题：
这句话是在"评价"一个对象，还是在"描述"这个对象发生了什么？

核心原则（强约束）：
动作、过程、研究行为、方法使用、功能实现、定义、分类、统计结果本身，都不是评价；只有当文本明确表达对合法评价客体的价值、优劣、重要性、效果、问题、程度、可行性、认可度或建议性判断时，才生成评价关系。
若句子只是描述"做了什么/发生了什么/是什么/包含什么/分布如何"，即使涉及合法客体，也不生成评价关系。

完整语义判断原则：
Evaluation Validity Filtering 必须判断完整语义，而不是根据固定谓词机械判断。"具有、存在、表现为、体现出、取得、实现"等词本身既可能是事实/属性描述，也可能引出真正评价。应根据其后内容判断是否存在评价性判断。
- "某机构具有创新能力。" → "创新能力"是客观属性陈述，无评价级差，可视为事实/属性描述，不必生成评价。
- "某机构具有较强的创新能力。" → "较强的"引出评价级差，有明确评价，应正常抽取。
- "某论文具有较高的理论价值。" → "较高的"引出评价级差，有明确评价，应正常抽取。

以下情况属于"根本没有评价"（情形A）：has_evaluation=false，relations=[]。

1. 事实描述
   例如："系统支持文献检索功能。" 描述功能，不代表评价。
2. 定义说明
   例如："结构化是指将获取的知识内容加以归纳和整理。" 属于概念定义。
3. 方法描述
   例如："本文采用文献计量法进行分析。" 描述研究方法。
4. 过程描述
   例如："自动文摘研究逐渐发展起来。" 描述发展过程，不代表好坏评价。
5. 工作描述
   例如："该模型实现了自动摘要生成。" 描述完成某项工作。
6. 属性陈述
   例如："文献之间具有重复性和差异性。" 描述客观属性，不代表价值判断。

## 候选评价关系的统一判断顺序

每发现一个候选评价关系，必须依次执行以下检查，任一步不通过即删除该关系：

Step 1 — 句子是否存在评价性判断？
   不要把评价检测理解为固定评价词匹配。评价可以由显式评价词、评价短语或完整评价性判断表达；即使没有典型形容词，也可能存在评价。
   以下都可以是评价表达：优秀、突出、不足、问题、较高、较低、值得探讨、效果显著、受到广泛认可、值得进一步推广、堪称典范、难以满足实际需求、为……提供了重要支撑、具有较高价值等。
   判断标准：句子是否对对象作出了带级差或价值取向的判断，而不仅仅是陈述客观属性或事实。若没有任何评价性判断 → 删除关系。
Step 2 — 确定评价对象。若 object 属于概念、方法、技术、模型、算法、理论、工具、平台、系统、数据集、语料库等禁止类型 → 删除关系。
Step 3 — 判断该表达是否只是事实、定义、功能、方法、过程、属性描述？若是 → 删除关系。
Step 4 — 只有通过以上全部检查，才生成 relation。

## Opinion 完整性规则

opinion 必须保留能够完整表达评价意义的最小连续文本片段，不得只抽取语法功能词、连接性谓词或空泛动词。

以下词如果单独出现、不能表达评价，不得作为完整 opinion：具有、存在、表现为、体现出、取得、实现、成为。
当这些词后接评价性内容时，opinion 应包含完整片段（谓词 + 评价内容），而不是只截取谓词。

- "某论文具有较高的理论价值。" → aspect=理论价值，opinion=具有较高的理论价值（正确）；opinion=具有（错误）。
- "该项目取得了较好的社会效益。" → aspect=社会效益，opinion=取得了较好的社会效益（正确）；opinion=取得（错误）。

## has_evaluation 语义定义

has_evaluation 表示"句中是否存在真正的评价表达"，它独立于 relations 是否为空。请严格按以下三种情形赋值：

情形A：句子根本没有评价（只是事实/定义/方法/过程/功能/属性描述）。
- has_evaluation = false
- relations = []
- 例："本文采用文献计量法进行分析。"、"文献之间具有重复性和差异性。"

情形B：句子存在真正的评价表达，但被评价对象属于禁止类型（概念/术语/方法/技术/模型/算法/理论/工具/平台/系统/数据集/语料库），因此没有合法 relation。
- has_evaluation = true
- relations = []
- 例："开源大模型性能优异。"、"深度学习方法具有较高准确率。"、"Transformer模型表现良好。"
- 说明：句中确有评价（性能优异、准确率高、表现良好），只是客体属于禁止类型，故 relations 为空，但 has_evaluation 仍为 true。
- 注意：研究主题、研究领域、复合研究对象（如"农村图书馆研究"）已是合法客体，不属于情形B，应正常生成 relation（情形C）。

情形C：句子存在真正的评价表达，且被评价对象属于合法客体类型。
- has_evaluation = true
- relations = [至少一条合法关系]

核心区分：
- "根本没有评价" → has_evaluation=false。
- "有评价但客体非法" → has_evaluation=true 且 relations=[]。
- 不要把情形B误标为 has_evaluation=false。

## Positive Examples（正确例子）

句子1：
"南京大学图书馆在数字资源建设方面表现突出。"
Entity: 南京大学图书馆
Relations:
[
  {{"subject":"_paper_author","object":"<南京大学图书馆的entity_id>","aspect":"数字资源建设","opinion":"表现突出","evidence":"南京大学图书馆在数字资源建设方面表现突出"}}
]

句子2：
"该论文的理论贡献较为突出。"
Entity: 该论文（如能识别出具体论文名，则使用具体论文名）
Relations:
[
  {{"subject":"_paper_author","object":"<该论文的entity_id>","aspect":"理论贡献","opinion":"较为突出","evidence":"该论文的理论贡献较为突出"}}
]

句子3：
"Smith教授的研究成果受到广泛认可。"
Entity: Smith教授
Relations:
[
  {{"subject":"_paper_author","object":"<Smith教授的entity_id>","aspect":"研究成果","opinion":"受到广泛认可","evidence":"Smith教授的研究成果受到广泛认可"}}
]

句子4（Aspect 绑定合法 Object）：
"某机构的研究水平较高。"
Entity: 某机构
Relations:
[
  {{"subject":"_paper_author","object":"<某机构的entity_id>","aspect":"研究水平","opinion":"较高","evidence":"某机构的研究水平较高"}}
]

句子5（Aspect 绑定合法 Object）：
"某论文的理论贡献较大。"
Entity: 某论文
Relations:
[
  {{"subject":"_paper_author","object":"<某论文的entity_id>","aspect":"理论贡献","opinion":"较大","evidence":"某论文的理论贡献较大"}}
]

句子6（subject 使用 _cite[N]：评价主体是被引文献）：
"文献[3]认为，李华团队在开放数据平台建设方面成效显著。"
说明：评价由被引文献[3]发出，故 subject=_cite[3]；被评价对象"李华团队"属于组织，是合法客体。
Entity: 李华团队
Relations:
[
  {{"subject":"_cite[3]","object":"<李华团队的entity_id>","aspect":"开放数据平台建设","opinion":"成效显著","evidence":"李华团队在开放数据平台建设方面成效显著"}}
]

subject 约定：
- _paper_author：评价由本文作者发出（默认）。
- _cite[N]：评价由第 N 篇被引文献发出，N 必须是引文编号（阿拉伯数字），如 _cite[3]。
- <人物名字>：当评价主体是具体人物（学者/研究者/作者）时，subject 直接填写该人物名字（如"李华""Smith"），不要写成 _cite[人名]。
- _unknown：无法确定评价主体。

严格约束：_cite[...] 的方括号内只能是引文编号数字，绝对不能出现人名。凡是能确定为具体人物的评价主体，一律直接用人物名字作 subject。

例（评价主体是具体人物，subject 用人名）：
"李华认为张伟团队在开放数据平台建设方面成效显著。" → subject=李华（不是 _cite[李华]），object=<张伟团队的entity_id>。

句子7（研究主题作为合法 Object）：
"我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。"
Entity: 农村图书馆研究（研究主题/复合研究对象，属于合法评价客体）
说明：不要抽取"作者分布""研究水平"为 Entity，它们是 aspect。
Relations:
[
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"作者分布","opinion":"不合理","evidence":"作者分布不合理"}},
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"研究水平","opinion":"偏低","evidence":"研究水平偏低"}}
]

句子8（复合研究对象作为合法 Object）：
"数字图书馆建设在资源整合方面成效显著。"
Entity: 数字图书馆建设（复合研究对象，属于合法评价客体）
Relations:
[
  {{"subject":"_paper_author","object":"<数字图书馆建设的entity_id>","aspect":"资源整合","opinion":"成效显著","evidence":"数字图书馆建设在资源整合方面成效显著"}}
]

句子9（研究主题作为合法 Object、无明确 Aspect，aspect=null；opinion 保留完整片段）：
"农村图书馆研究取得了一定成绩。"
Entity: 农村图书馆研究（研究主题/复合研究对象，属于合法评价客体）
说明：此时没有明确评价维度，aspect=null；opinion 保留完整片段"取得了一定成绩"，不要截断为"取得"。
Relations:
[
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":null,"opinion":"取得了一定成绩","evidence":"农村图书馆研究取得了一定成绩"}}
]

关系输出字段必须使用 subject、object、aspect、opinion、evidence。不要输出 polarity。

## Negative Examples（非评价关系 / 禁止类型）

### 一、事实描述 / 定义说明（无评价）

以下情况不要抽取评价关系，has_evaluation=false，relations=[]。

例1：事实描述（无评价）
句子：
"用户服务平台提供了按资源类型、标题、作者、关键词等多种检索途径查找资源的功能。"
不要抽取：
object=用户服务平台
opinion=提供多种检索途径
原因：
"提供、支持、包含"等在此描述功能或事实，不代表评价。注意"具有/提供"等谓词需按完整语义判断：仅当其后为客观功能/属性（如本例"多种检索途径"）时才不代表评价；若其后引出评价级差（如"具有较强的创新能力"），则仍属评价。

例2：定义说明（无评价）
句子：
"结构化是指将获取的知识内容加以归纳和整理，使之条理化、纲领化。"
不要抽取评价关系。
原因："是指"属于概念定义。

例3：方法流程描述（无评价）
句子：
"本文采用文献计量法对相关研究进行分析。"
不要抽取评价关系。
原因："采用、使用、利用"表示研究方法，不表示评价。

例4：分类枚举（无评价）
句子：
"研究内容主要包括理论研究、应用研究和实践研究。"
不要抽取评价关系。
原因："包括、分为、涉及"属于分类描述。

例5：功能/能力陈述（无评价）
句子：
"该系统能够实现文献检索、数据分析和结果展示。"
不要抽取评价关系。
原因："能够实现"描述功能，不等于"功能好"或"效果显著"。

例6："发展起来、出现、形成、产生"描述变化过程，不代表评价。
句子：
"自动文摘研究逐渐发展起来。"
不要抽取评价关系。

例7："主要集中于"描述研究分布，不表示好坏评价。
句子：
"目前的研究主要集中于文摘生成方法。"
不要抽取评价关系。

例8："实现、完成、构建、提出"描述研究工作，不表示效果评价。
句子：
"该模型实现了自动摘要生成。"
不要抽取评价关系。

例9：动作 / 研究行为 / 方法使用 / 功能实现 / 定义 / 分类 / 统计（一律不抽，has_evaluation=false，relations=[]）
以下句式只是描述"做了什么/发生了什么/是什么"，没有价值、优劣、效果、程度等评价性判断，一律不生成评价关系：
- "……进行了研究。" → 不抽（研究行为）。
- "……进行了分析。" → 不抽（研究行为）。
- "……提出了方法。" → 不抽（研究工作）。
- "……采用了某方法。" → 不抽（方法使用）。
- "……实现了某功能。" → 不抽（功能实现）。
- "……定义为……。" → 不抽（定义）。
- "……逐渐发展起来。" → 不抽（过程描述）。
- "……主要集中于……。" → 不抽（研究分布/统计）。
- "……进行了对比。" → 不抽（研究行为）。

### 二、概念/方法/技术/模型等禁止类型（有评价但客体非法）

以下对象属于禁止类型：句中确有评价表达，但客体非法，因此不生成 relation。
按情形B处理：has_evaluation=true，relations=[]。

例10：概念/技术对象
句子：
"开源大模型性能优异。"
结果：has_evaluation=true, relations=[]
原因："性能优异"是评价，但"开源大模型"属于概念/技术对象，不属于允许的 Evaluation Object。

例11：方法类对象
句子：
"深度学习方法具有较高准确率。"
结果：has_evaluation=true, relations=[]
原因："较高准确率"是评价，但"深度学习方法"属于方法类对象，不作为评价客体。

例12：模型类对象
句子：
"Transformer模型在文本生成任务中表现良好。"
结果：has_evaluation=true, relations=[]
原因："表现良好"是评价，但"Transformer模型"属于模型类对象，不作为评价客体。

例13：方法类对象
句子：
"文献计量法效果较好。"
结果：has_evaluation=true, relations=[]
原因："效果较好"是评价，但"文献计量法"属于方法，不作为评价客体。

例14：模型/技术对象
句子：
"ChatGPT具有较强的文本生成能力。"
结果：has_evaluation=true, relations=[]
原因："较强的文本生成能力"是评价，但"ChatGPT"属于模型/技术对象，不属于允许的 Evaluation Object。

例15：方法类对象
句子：
"深度学习方法取得了较好的效果。"
结果：has_evaluation=true, relations=[]
原因："较好的效果"是评价，但"深度学习方法"属于方法类对象，不作为评价客体。

### 三、过程/工作/属性描述（无评价）

以下情况不要抽取评价关系，has_evaluation=false，relations=[]。

例16：过程描述
句子：
"自动文摘研究逐渐发展起来。"
结果：has_evaluation=false, relations=[]
原因：描述发展过程，不是评价。

例17：工作描述
句子：
"该模型实现了自动摘要生成。"
结果：has_evaluation=false, relations=[]
原因：描述完成某项研究工作，不是评价。

例18：属性陈述
句子：
"文献之间具有重复性和差异性。"
结果：has_evaluation=false, relations=[]
原因：描述客观属性，不代表价值判断。

### 四、避免将 Aspect 误认为 Object（object 合法，正确绑定）

例19：
句子：
"农村图书馆研究水平偏低。"
错误：object=研究水平（把 aspect 当成 object）。
正确：object=农村图书馆研究, aspect=研究水平, opinion=偏低。
说明："农村图书馆研究"是研究主题，属于合法评价客体；"研究水平"是评价方面（aspect），不是被评价对象。应生成关系：
[
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"研究水平","opinion":"偏低","evidence":"农村图书馆研究水平偏低"}}
]
原因：研究水平是评价方面，不是 object；被评价对象是研究主题"农村图书馆研究"，属于合法客体，正常生成 relation（情形C）。

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
      "subject": "_paper_author|_cite[N]|<人物名字>|_unknown",
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
            entities=parsed_entities,
        )