"""
pipeline.entity_extraction_agent — Agent 1: 实体抽取 + 评价关系识别 (V4.4 合并版)
对齐: 实体类型分类体系_opencode版.md — LangGPT 风格提示词
从评价句中高召回抽取全部实体类型: Agent / Artifact / Abstract / Event
同时识别评价关系 (主体-客体-极性)，替代原独立 Agent 4
输出中间结果到 mid_data/ 目录
"""

from __future__ import annotations

from typing import Optional

from .llm import LLMClient

# ===========================================================================
# Agent 1 抽取 Prompt — LangGPT 风格 (Role/Profile/Rules/Workflow/Background/OutputFormat/Examples/Input)
# ===========================================================================

ENTITY_EXTRACTION_PROMPT = (
    "# Role\n"
    "你是一个学术文献实体抽取专家。\n\n"
    "## Profile\n"
    "- 专长: 从学术评价句中高召回抽取具体实体\n"
    "- 能力: 识别四大类别实体 — Agent(行为主体)、Artifact(人工制品)、Abstract(抽象实体)、Event(事件)\n"
    "- 原则: 严格实体性过滤，仅抽取具体可唯一识别的对象\n"
    "- 风格: 精准、忠实原文、不概括、不创造名称\n\n"
    "## Rules\n"
    "### 实体性门槛 — 以下类别词/短语 **一律不抽取**:\n"
    "1. 泛化身份类别词 (不指称具体个人):\n"
    "   科学家、学者、研究者、专家、教授、馆长、图书馆员、工程师、作者、学生、读者、用户\n"
    "2. 泛化机构类别词 (不指称具体组织):\n"
    "   大学图书馆、高校、研究机构、公共图书馆、档案馆、公司、出版社\n"
    "3. 通用连接词/虚义动词: 比较、进行、存在、通过、基于、根据、利用、实现、导致、产生、构成、形成、推动、促进、影响\n"
    "4. 泛指代词/不定指称: 这方面、该问题、上述研究、本文、笔者、相关文献、某论文\n"
    "5. 无学术语义的计量词: 篇数、比例、数量、百分比、平均值、标准差\n"
    "6. 泛化评价用语: 重要、显著、明显、突出、不足、深远、深刻、广泛\n"
    "7. 单纯时间表达: 2005年、20世纪90年代、某年、近年来。除非原文明确表示一个发展阶段，否则不抽取为 stage\n"
    "8. 单纯地名/行政区名: 深圳、青岛、北京。除非指向具体政府机构或组织主体，否则不抽取为 governance\n"
    "   判定标准: 该短语能否指向一个**具体可唯一识别**的对象?\n"
    "   若不能，且属于以上任一类别 → **一律不抽取**。\n\n"
    "### 原文忠实原则:\n"
    "1. mention 必须是**原文中出现的原词/原短语**，严禁概括、总结、改名\n"
    "2. normalized_name 是规范化形式 (去OCR乱码、统一简繁)，无特殊问题时与 mention 一致\n"
    "3. 带缩写/代称/简称/英文名的实体，抽取全名+简称+括号内内容\n"
    "   如: 「巴巴拉·奎恩特（B. Quint）」 → mention 应为完整形式\n\n"
    "### 召回优先原则:\n"
    "1. candidate_l1 允许多个，模糊实体可标多个候选 L1\n"
    "2. evidence 是从原句中截取能证明该实体存在的文本片段\n\n"
    "## Workflow\n"
    "1. 读取待分析评价句，先判断是否存在评价表达或评价判断。\n"
    "2. 若存在评价，先抽取评价关系草图: subject、aspect、opinion、evidence，并标出原文中可能的评价对象文本。\n"
    "3. 对每条评价关系执行 Object Resolution: 根据 aspect/opinion 回溯真正被评价的对象。\n"
    "4. 将每个已解析出的评价对象反推为必须抽取的 Entity，优先保留完整研究对象/主题对象边界。\n"
    "5. 再补充句中与评价关系无直接绑定、但仍符合 Agent/Artifact/Abstract/Event 分类体系的有效实体。\n"
    "6. 对所有 Entity 判定 candidate_l3、candidate_l1，并提取 evidence。\n"
    "7. 用已生成的 entity_id 回填 relation.object；不要在 relation.object 中使用未进入 entities 的新 ID。\n"
    "8. 按 OutputFormat 输出严格 JSON。\n\n"
    "## Background\n"
    "### 一、Agent (行为主体) — 能产生学术行为的主体\n"
    "**Person (个人)** — 刚性类型，身份不随评价语境改变\n"
    "- `scholar`: 以研究/知识生产为主要职能 (教授/研究员/博士生)。vs practitioner: scholar 产出是知识本身(论文/著作)\n"
    "- `practitioner`: 以专业服务为主要职能 (图书馆员/档案馆员/情报分析师)。产出是专业服务\n"
    "- `policy_advocate`: 以推动政策/改革/倡议为主要公开行为的个人\n"
    "- `reviewer`: 承担评审职能 (审稿人/评审专家/项目评委)\n"
    "> 仅抽取**具体可唯一识别的人名**。泛称如「专家」「学者」「教授」不抽取。\n\n"
    "**Organization (组织)** — 刚性类型，type_code 即核心职能\n"
    "- `research`: 知识生产组织 (大学/研究所/智库/研究院/实验室)\n"
    "- `service`: 专业服务组织 (图书馆/档案馆/信息中心/医院)\n"
    "- `professional`: 职业共同体组织 (学会/协会/IFLA/ALA)\n"
    "- `governance`: 行政/治理/资助组织 (教育部/NSF/国家社科基金委/档案局)\n"
    "- `publishing`: 出版传播组织 (Elsevier/商务印书馆/arXiv/出版社)\n"
    "> 仅抽取**具体可唯一识别的机构全称/简称**。泛称如「高校」「研究机构」「图书馆」不抽取。\n"
    "> 区分 Organization vs discipline: 能给出地址/法人/成员名单 → Organization。\n"
    "> 区分 Organization vs school_of_thought: 有行政建制 → Organization。\n\n"
    "### 二、Artifact (人工制品) — 人类有意识生产的制品\n"
    "**Discursive (话语性著作)** — 路径 `Artifact > Info > Discursive`\n"
    "- `journal_article`: 期刊论文(有卷期页) | `conference_paper`: 会议论文\n"
    "- `book`: 专著(有独立ISBN) | `thesis`: 学位论文 | `report`: 研究报告\n"
    "- `preprint`: 预印本(arXiv/SSRN) | `paper`: 论文兜底\n"
    "> 仅抽取具体文献名称。泛称如「某论文」「相关文献」不抽取。\n\n"
    "**Organizational (组织性知识工具)** — 路径 `Artifact > Info > Organizational`\n"
    "- `knowledge_organization_system`: 分类法/叙词表/本体/受控词表(KOS)\n"
    "- `metadata_schema`: 元数据规范(Dublin Core/MARC/EDM)\n"
    "- `reference_tool`: 参考工具书(辞典/百科全书/手册)\n"
    "- `index`: 索引/引文索引(SCI/CSSCI)\n"
    "> vs: KOS 关注概念关系，metadata_schema 关注描述规则，index 关注检索功能。\n\n"
    "**Empirical (经验性制品)** — 路径 `Artifact > Info > Empirical`\n"
    "- `dataset`: 结构化数据集(OpenCitations) | `corpus`: 文本语料库\n"
    "- `database`: 数据资源(CNKI/WoS, 评价数据覆盖质量时)\n"
    "> vs: database 评价数据内容，information_system 评价系统功能。\n\n"
    "**Normative (规范性制品)** — 路径 `Artifact > Info > Normative`\n"
    "- `standard`: 技术标准(ISO/GB/ISBN) | `policy`: 政策文件(HR1858)\n"
    "- `guideline`: 指南/规范(信息素质培养标准)\n"
    "> vs: standard = 技术性，policy = 行政约束力，guideline = 建议性。\n\n"
    "**System (信息系统)** — 路径 `Artifact > Functional > System`\n"
    "- `information_system`: 信息系统/平台(CDWS分词系统/数字图书馆平台)\n"
    "> vs database: system 评价系统功能和界面，database 评价数据内容质量。\n\n"
    "**Tool (软件/工具)** — 路径 `Artifact > Functional > Tool`\n"
    "- `software`: 软件/程序(SPSS/R/SMW/NoteExpress)\n"
    "- `algorithm`: 算法/计算逻辑(PageRank/TF-IDF/KNN/SOM/聚类算法)\n"
    "- `instrument`: 测量工具/量表/问卷/指标体系(调查问卷/心理量表/元素依赖性指数)\n"
    "> vs: algorithm = 明确计算步骤/排序/分类逻辑；software = 算法或功能的软件实现；instrument = 测量、评价、采集数据的工具。\n"
    "> 电子计算机、机器人、VR/AR设备、5G、大数据、人工智能技术、云计算等不得因“技术/设备”泛化标为 algorithm 或 instrument。\n"
    "> 仅抽取具体可识别的软件名。泛称如「统计软件」「分析工具」不抽取。\n\n"
    "### 三、Abstract (抽象实体) — 无物质载体的智识构造物\n"
    "**Conceptual (概念层)**\n"
    "- `concept`: 学术术语/命名单元(Persona/信息素养/知识鸿沟)\n"
    "- `definition`: 对概念边界的特定界定方式(通常含提出者)\n"
    "- `typology`: 分类方案(JCR分类/学科分类体系)\n"
    "> vs: concept = 命名，definition = 界定，typology = 系统分类方案。\n"
    "> `concept` 不是 Abstract 的默认兜底项。只有原句把对象作为术语/概念名本身讨论，且不能归入方法、理论、模型、框架、现象、学科、事件或制品时，才标 concept。\n\n"
    "**KnowledgeClaim (知识主张层)**\n"
    "- `theory`: 有因果主张的系统命题(学习迁移理论/弱连接理论) — 解释「为什么」\n"
    "- `model`: 组件/变量关系描述(研究对象套件模型/TAM) — 描述「如何运作」\n"
    "- `framework`: 分析视角，无因果主张(Argument Zoning II/FRBR) — 提供「分析角度」\n"
    "- `hypothesis`: 单一待检验命题\n"
    "> 判断标准: 有因果解释 → theory；有组件关系描述 → model；仅有分析视角 → framework。\n\n"
    "**Methodological (方法论层)** — 层次 `methodology → method → technique` (高→低)\n"
    "- `methodology`: 认识论立场(实证主义/解释主义/批判理论)\n"
    "- `method`: 完整操作程序(统计回归方法/文献计量法/内容分析法/德尔菲法/排架法)\n"
    "- `technique`: 具体技术手段(共词分析/聚类分析/TF-IDF/LDA/受控技术)\n\n"
    "> `technique` 仅限研究/分析/检索/测量中的具体操作手段。5G、VR/AR、人工智能技术、大数据、云计算是技术体系/技术领域，不是研究 technique。\n\n"
    "**Epistemic (认识论共同体层)** — 共享认识论立场的抽象共同体\n"
    "- `paradigm`: 研究范式(第四范式/实证主义范式)\n"
    "- `approach`: 研究取向(文化下乡/定性取向/用户中心取向)\n"
    "- `discipline`: 学科(图书情报学/社会学/历史学)\n"
    "- `subfield`: 子领域(科学计量学/信息检索/知识管理)\n"
    "- `school_of_thought`: 学派(情报学派/芝加哥学派)\n"
    "> 区分 Organization vs discipline: 有地址/法人/名单 → Organization；否则 → discipline。\n"
    "> 区分 scholar vs school_of_thought: 个人 → scholar；群体传统 → school_of_thought。\n\n"
    "**Phenomenon (现象层)**\n"
    "- `phenomenon`: 可观察的客观/社会现象(数据孤岛/信息茧房/数字鸿沟)\n"
    "  需同时满足: (1)图情领域可观察 (2)有独立学术命名 (3)作为独立研究主题被讨论\n"
    "- `trend`: 带时间方向性的演变(数字化趋势/Web2.0广泛应用)\n"
    "> vs: trend 有时间方向性，phenomenon 是静态现象。\n"
    "> 泛化趋势如「发展趋势」「增长趋势」不抽取。\n\n"
    "### Abstract 内部优先级 — 防止 concept 兜底\n"
    "当一个抽象实体可能被标为 concept 时，必须先排除以下更具体类别:\n"
    "1. 研究操作/程序/分析/测量/检索/聚类/回归 → method 或 technique\n"
    "2. 理论/模型/框架/假设/机制/因果解释/变量关系 → theory/model/framework/hypothesis\n"
    "3. 学科/领域/方向/研究传统/学派 → discipline/subfield/school_of_thought/approach/paradigm\n"
    "4. 可观察社会事实/现实问题/困境/鸿沟/孤岛/茧房 → phenomenon 或 trend\n"
    "5. 运动/倡议/计划/项目/改革/争论/发展阶段 → Event\n"
    "6. 具体系统、数据库、工具、标准、文献、知识组织系统 → Artifact\n"
    "仅当以上均不适用，且原句讨论的是命名单元本身，才使用 concept。\n\n"
    "### 技术类实体临时规则 — 当前本体无 Technology\n"
    "1. 5G/5G技术/第五代移动通信系统: 若强调技术标准/通信制式 → standard；若泛指技术体系或应用语境 → concept，不标 algorithm。\n"
    "2. 人工智能技术/大数据/云计算/VR/AR: 若作为技术领域、技术体系或应用概念 → concept；若作为具体系统/平台/软件 → information_system/software；不标 technique。\n"
    "3. 电子计算机/机器人/智能设备: 若只是设备种类且无具体名称，通常不抽取；若原文作为技术概念讨论 → concept；不得标 instrument，除非它明确用于测量/评价/采集数据。\n"
    "4. 图书馆自动化、档案管理自动化、保存文件遗产等职能/业务理念: 通常标 concept；只有强调历史时段推进才考虑 stage，只有描述可观察社会事实才考虑 phenomenon。\n\n"
    "### 四、Event (事件/过程) — 在时间中展开\n"
    "- `intellectual_turn`: 学术转向(可明确时间节点的范式转变)\n"
    "- `debate`: 学术争论(有明确争论双方和议题)\n"
    "- `movement`: 学术运动(自下而上的改革行动)\n"
    "- `research_program`: 研究计划(CADAL等长期有组织研究)\n"
    "- `policy_initiative`: 政策举措/改革事件(自上而下的制度行动)\n"
    "  > vs policy(Artifact): policy 是政策文件本身(制品)，policy_initiative 是实施过程(事件)\n"
    "- `stage`: 发展阶段(时段性划分，模糊起止时间)\n"
    "- `conference_meeting`: 学术会议(具体召开的会议/年会/论坛)\n"
    "  > vs conference_paper(Artifact): paper 是会议论文制品，meeting 是会议召开本身\n\n"
    "## Evaluation Relation Recognition (评价关系识别)\n"
    "请先识别句子中的评价关系，再根据评价关系所需的 object 反推并抽取实体。\n\n"
    "### 评价判定:\n"
    "评价是指作者、引用文献作者或其他评价主体，对某一对象作出的正向、负向或中性的判断、概括、评价、比较或总结。\n"
    "典型评价用语: 重要、有效、丰富、较高、较低、明显、成熟、完善、不足、较好、优于、落后、提高、降低、具有……价值、存在……问题、有待……\n"
    "如果整个句子只是客观事实描述，没有评价，则 has_evaluation=false, relations=[]。\n\n"
    "### 评价主体 (subject) 判定规则:\n"
    "评价主体是作出评价的人或来源，不是执行动作的行为主体。仅允许以下四种类型:\n"
    "1. **显性实体ID**: 句中明确指出某人/机构进行了评价 → 使用已抽取实体的 entity_id (如 \"1_e3\")\n"
    "2. **_paper_author**: 评价来自当前论文作者，句中无显式评价主体\n"
    "3. **_cite[N]**: 评价来自引用文献 → _cite[12] (单个引用) 或 _cite[3][5] (多个引用)\n"
    "4. **_unknown**: 依据当前句无法确定评价主体\n\n"
    "### 评价客体 (object) 判定规则:\n"
    "评价客体必须是被评价的对象，优先使用已抽取实体的 entity_id。\n"
    "若评价对象不存在于实体列表中 → 使用 \"_missing_entity\"，同时增加 object_text 字段填写原始文本。\n"
    "不要自行创造新的 entity_id。\n\n"
    "### 评价方面 (aspect) 与评价内容 (opinion):\n"
    "aspect 填写评价对象的具体评价维度；若不存在明确维度则为 null。\n"
    "opinion 填写原文中的评价表达或评价内容。\n\n"
    "### 评价依据 (evidence):\n"
    "输出支持该评价的最小文本片段，尽可能短但能完整表达评价。\n\n"
    "### 多评价关系:\n"
    "一个句子可能包含多个评价关系，也可能没有评价关系。\n\n"
    "## OutputFormat\n"
    "严格 JSON，不含 markdown 代码块。entities 数组无实体则为空数组 []，relations 数组无评价则为空数组 []。\n"
    '{{\n'
    '  "entities": [\n'
    '    {{\n'
    '      "mention": "<原文精确短语，严禁概括/改名>",\n'
    '      "normalized_name": "<规范化实体名，无特殊问题时等于mention>",\n'
    '      "candidate_l3": "<type_code, 从 Background 分类体系中选择>",\n'
    '      "candidate_l1": ["Agent|Artifact|Abstract|Event (允许多个，召回优先)"],\n'
    '      "evidence": "<原句中证明该实体存在的文本片段>",\n'
    '      "is_specific_entity": true,\n'
    '      "confidence": 0.95,\n'
    '      "uncertainty": ""\n'
    '    }}\n'
    '  ],\n'
    '  "has_evaluation": true,\n'
    '  "relations": [\n'
    '    {{\n'
    '      "subject": "_paper_author|entity_id|_cite[N]|_unknown",\n'
    '      "object": "entity_id|_missing_entity",\n'
    '      "aspect": "<评价方面或 null>",\n'
    '      "opinion": "<评价表达>",\n'
    '      "evidence": "<评价依据文本片段>",\n'
    '      "object_text": "<仅当 object 为 _missing_entity 时填写原始文本>"\n'
    '    }}\n'
    '  ]\n'
    '}}\n\n'
    "## Examples\n"
    "### 正例1 (scholar — 知识生产者)\n"
    "输入: 建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。他们的研究对谱学理论的普及与发展具有不可磨灭的贡献。\n"
    '输出: {{\"entities\":[{{\"mention\":\"潘光旦\",\"normalized_name\":\"潘光旦\",\"candidate_l3\":\"scholar\",\"candidate_l1\":[\"Agent\"],\"evidence\":\"在谱学研究领域颇有建树的学者有潘光旦\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}},{{\"mention\":\"罗香林\",\"normalized_name\":\"罗香林\",\"candidate_l3\":\"scholar\",\"candidate_l1\":[\"Agent\"],\"evidence\":\"在谱学研究领域颇有建树的学者有潘光旦、罗香林等人\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}}]}}\n\n'
    "### 正例2 (research — 知识生产组织 + phenomenon)\n"
    "输入: 加拿大不列颠哥伦比亚大学的卫生保健管理中心门户就是一个努力帮助用户克服信息过载的网络信息中介的示例。\n"
    '输出: {{\"entities\":[{{\"mention\":\"加拿大不列颠哥伦比亚大学\",\"normalized_name\":\"不列颠哥伦比亚大学\",\"candidate_l3\":\"research\",\"candidate_l1\":[\"Agent\"],\"evidence\":\"加拿大不列颠哥伦比亚大学的卫生保健管理中心门户\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}},{{\"mention\":\"信息过载\",\"normalized_name\":\"信息过载\",\"candidate_l3\":\"phenomenon\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"帮助用户克服信息过载\",\"is_specific_entity\":true,\"confidence\":0.9,\"uncertainty\":\"现实信息问题,非术语本身\"}}]}}\n\n'
    "### 正例3 (book — 专著)\n"
    "输入: 新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果。与旧版相比,其观点更新颖,内容更充实,结构更合理。\n"
    '输出: {{\"entities\":[{{\"mention\":\"新版《图书馆学概论》\",\"normalized_name\":\"《图书馆学概论》\",\"candidate_l3\":\"book\",\"candidate_l1\":[\"Artifact\"],\"evidence\":\"新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}}]}}\n\n'
    "### 正例4 (knowledge_organization_system)\n"
    "输入: 关于类目虚设问题。这点《中图法》比较突出,尤以自然科学类为最,不但加重了分类法的篇幅,也给分类员制造了麻烦。\n"
    '输出: {{\"entities\":[{{\"mention\":\"《中图法》\",\"normalized_name\":\"《中图法》\",\"candidate_l3\":\"knowledge_organization_system\",\"candidate_l1\":[\"Artifact\"],\"evidence\":\"这点《中图法》比较突出\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}}]}}\n'
    "> 「比较突出」中的「比较」为虚义动词，不抽取。\n\n"
    "### 正例5 (theory — 理论)\n"
    "输入: Ausubel基于学习者认知结构的学习迁移理论与这些研究问题非常契合。已有研究并没有深入探讨学习迁移的基础理论。\n"
    '输出: {{\"entities\":[{{\"mention\":\"学习迁移理论\",\"normalized_name\":\"学习迁移理论\",\"candidate_l3\":\"theory\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"Ausubel基于学习者认知结构的学习迁移理论\",\"is_specific_entity\":true,\"confidence\":0.92,\"uncertainty\":\"\"}}]}}\n'
    "> 「基础理论」为泛称，不抽取。只抽取有具体命名的理论。\n\n"
    "### 正例6 (method — 研究方法)\n"
    "输入: 传统的统计回归方法常采用线性或多项式函数;而机器学习方法更倾向于复杂的非线性模型,能得到较高准确率。\n"
    '输出: {{\"entities\":[{{\"mention\":\"统计回归方法\",\"normalized_name\":\"统计回归方法\",\"candidate_l3\":\"method\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"传统的统计回归方法常采用线性或多项式函数\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}},{{\"mention\":\"机器学习方法\",\"normalized_name\":\"机器学习方法\",\"candidate_l3\":\"method\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"机器学习方法更倾向于复杂的非线性模型\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}}]}}\n'
    "> 「预测结果」「高准确率」为评价用语，不抽取。\n\n"
    "### 正例7 (phenomenon)\n"
    "输入: 长期以来各文化机构独自推进的智改数转造就了一座座数据孤岛,底层关联不足进而会引发上层文化服务割裂。\n"
    '输出: {{\"entities\":[{{\"mention\":\"数据孤岛\",\"normalized_name\":\"数据孤岛\",\"candidate_l3\":\"phenomenon\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"造就了一座座数据孤岛\",\"is_specific_entity\":true,\"confidence\":0.95,\"uncertainty\":\"\"}}]}}\n'
    "> 「文化服务割裂」若无独立学术命名则不抽取。\n\n"
    "### 正例7b (subfield vs concept)\n"
    "输入: 信息检索在图书情报学研究中形成了稳定的问题域和方法传统。\n"
    '输出: {{\"entities\":[{{\"mention\":\"信息检索\",\"normalized_name\":\"信息检索\",\"candidate_l3\":\"subfield\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"信息检索在图书情报学研究中形成了稳定的问题域和方法传统\",\"is_specific_entity\":true,\"confidence\":0.9,\"uncertainty\":\"作为研究子领域,非concept兜底\"}}]}}\n\n'
    "### 正例7c (concept 的正向用法)\n"
    "输入: 「信息素养」这一概念强调个体识别、获取和评价信息的能力。\n"
    '输出: {{\"entities\":[{{\"mention\":\"信息素养\",\"normalized_name\":\"信息素养\",\"candidate_l3\":\"concept\",\"candidate_l1\":[\"Abstract\"],\"evidence\":\"「信息素养」这一概念\",\"is_specific_entity\":true,\"confidence\":0.9,\"uncertainty\":\"原句讨论术语/概念名本身\"}}]}}\n\n'
    "### 正例8 (debate + movement)\n"
    "输入: 情报学中对于Information与Intelligence的争论应该是有益的。开放获取意味着文章一旦被创造出来,将通过网络让读者免费获取和利用。\n"
    '输出: {{\"entities\":[{{\"mention\":\"Information与Intelligence的争论\",\"normalized_name\":\"Information与Intelligence的争论\",\"candidate_l3\":\"debate\",\"candidate_l1\":[\"Event\"],\"evidence\":\"情报学中对于Information与Intelligence的争论应该是有益的\",\"is_specific_entity\":true,\"confidence\":0.92,\"uncertainty\":\"\"}},{{\"mention\":\"开放获取\",\"normalized_name\":\"开放获取\",\"candidate_l3\":\"movement\",\"candidate_l1\":[\"Event\"],\"evidence\":\"开放获取意味着文章一旦被创造出来\",\"is_specific_entity\":true,\"confidence\":0.9,\"uncertainty\":\"可兼为concept\"}}]}}\n'
    "> 「开放获取」既可作 concept 也可作 movement。此处描述其作为运动的方式，优先标 movement。\n\n"
    "### 负例1 (泛称身份不抽取)\n"
    "输入: 许多科学家认为开放获取能推动学术交流与合作。\n"
    '输出: {{\"entities\":[]}}\n'
    "> 「科学家」是泛化身份类别词，不指称具体个人 → 不抽取。\n\n"
    "### 负例2 (通用词/评价用语不抽取)\n"
    "输入: 通过比较两种方法的优劣，本文认为该理论具有重要意义。\n"
    '输出: {{\"entities\":[]}}\n'
    "> 「比较」通用动词、「优劣」评价用语、「本文」自指、「重要意义」评价用语 → 均不抽取。\n\n"
    "## Input\n"
    "{statement}"
)

ENTITY_EXTRACTION_PROMPT = ENTITY_EXTRACTION_PROMPT.replace(
    "## Input\n{statement}",
"""## Academic Evaluation Object Rules (增强规则)
本任务采用“先找评价关系，后抽取实体”的关系优先策略：
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
- “中西部地区研究不足” -> Entity: 中西部地区研究；不要抽取“中西部地区”。
- “农村图书馆事业发展良好” -> Entity: 农村图书馆事业发展；不要只抽“农村图书馆”。
- “数字信息资源建设存在不足” -> Entity: 数字信息资源建设；不要只抽“数字信息资源”。

当名词短语后接“研究、建设、发展、问题、实践、应用、水平、能力、体系”，且整体构成被评价的研究对象或主题对象时，优先整体抽取。

以下通常不是 Entity，除非原文把它们作为独立研究对象或术语本身讨论：
作者分布、研究水平、研究质量、理论基础、应用效果、区域分布。
它们在评价关系中通常应放入 aspect 字段。

评价关系不要按“实体 + 评价词”机械抽取，而应识别：
subject = 评价主体
object = 被评价的核心对象 entity_id
aspect = 评价方面；没有则为 null
opinion = 评价表达
evidence = 支持该评价的最小原文片段

评价对象回溯（Object Resolution）：
先识别 aspect 与 opinion，再判断“这个评价是在评价哪个实体”。不要因为 aspect 不是实体，就直接输出 _missing_entity。
1. 优先绑定已有实体：若 aspect 属于某个已抽取实体的属性、组成部分、发展情况、研究维度或评价维度，object 必须绑定该实体。
2. Aspect 不是 Object：aspect 表示评价维度，object 表示真正被评价的对象。例如“作者分布不合理”若句子讨论“农村图书馆研究”，object=农村图书馆研究，aspect=作者分布。
3. 寻找 aspect 所属对象：当 aspect 出现时，优先向左寻找其所属对象。
   - “数字图書館建設的發展速度較快” -> object=数字图書館建設, aspect=發展速度, opinion=較快。
   - “法明頓計畫在協調布局方面堪稱典範” -> object=法明頓計畫, aspect=協調布局, opinion=堪稱典範。
4. 允许跨短语回溯：object 不一定紧邻 aspect。例如“近年来，档案信息化建设取得快速发展，其理论研究仍存在不足”中，“理论研究/不足”应回溯到“档案信息化建设”。
5. 仅当当前句不存在任何可作为评价对象的实体、aspect 无法归属于任何实体、且上下文无法确定评价对象时，才使用 _missing_entity。
6. Entity 优先原则：多个候选实体时，选择最直接被评价、语义距离最近、且能够完整支撑 aspect 的实体。不要选择地名、时间、修饰语。
7. Aspect 属于 object，不是独立 object。例如“研究水平偏低”：object=农村图书馆研究，aspect=研究水平，opinion=偏低；不要 object=研究水平。

例如：
句子：“我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。”
Entity 只抽取“农村图书馆研究”，不要抽取“作者分布”或“研究水平”。
Relations:
[
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"作者分布","opinion":"不合理","evidence":"作者分布不合理"}},
  {{"subject":"_paper_author","object":"<农村图书馆研究的entity_id>","aspect":"研究水平","opinion":"偏低","evidence":"研究水平偏低"}}
]

句子：“中西部地区研究不足。”
Entity: 中西部地区研究
Relation: object=<中西部地区研究的entity_id>, aspect=null, opinion=不足。

关系输出字段必须使用 subject、object、aspect、opinion、evidence。不要输出 polarity。

## Input
{statement}""",
)

# ===========================================================================
# 中间结果数据模型
# ===========================================================================


class ExtractedEntity:
    """Agent 1 抽取的单条实体中间结果"""

    def __init__(
        self,
        entity_id: str = "",
        mention: str = "",
        normalized_name: str = "",
        candidate_l3: str = "",
        candidate_l1: Optional[list[str]] = None,
        evidence: str = "",
        is_specific_entity: bool = True,
        confidence: float = 0.0,
        uncertainty: str = "",
    ):
        self.entity_id = entity_id
        self.mention = mention
        self.normalized_name = normalized_name
        self.candidate_l3 = candidate_l3
        self.candidate_l1 = candidate_l1 or []
        self.evidence = evidence
        self.is_specific_entity = is_specific_entity
        self.confidence = confidence
        self.uncertainty = uncertainty

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "mention": self.mention,
            "normalized_name": self.normalized_name,
            "candidate_l3": self.candidate_l3,
            "candidate_l1": self.candidate_l1,
            "evidence": self.evidence,
            "is_specific_entity": self.is_specific_entity,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExtractedEntity":
        return ExtractedEntity(
            entity_id=data.get("entity_id", ""),
            mention=data.get("mention", ""),
            normalized_name=data.get("normalized_name", ""),
            candidate_l3=data.get("candidate_l3", ""),
            candidate_l1=data.get("candidate_l1", []),
            evidence=data.get("evidence", ""),
            is_specific_entity=data.get("is_specific_entity", True),
            confidence=data.get("confidence", 0.0),
            uncertainty=data.get("uncertainty", ""),
        )


class ExtractedRelation:
    """Agent 1 识别的一条评价关系 (V4.4 新增)"""

    def __init__(
        self,
        subject: str = "",
        object: str = "",
        aspect: Optional[str] = None,
        opinion: str = "",
        evidence: str = "",
        object_text: str = "",
    ):
        self.subject = subject
        self.object = object
        self.aspect = aspect if aspect not in ("", "null") else None
        self.opinion = opinion
        self.evidence = evidence
        self.object_text = object_text

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

    @staticmethod
    def from_dict(data: dict) -> "ExtractedRelation":
        return ExtractedRelation(
            subject=data.get("subject", ""),
            object=data.get("object", ""),
            aspect=data.get("aspect"),
            opinion=data.get("opinion", ""),
            evidence=data.get("evidence", ""),
            object_text=data.get("object_text", ""),
        )


class SentenceExtractionOutput:
    """单句的抽取输出 (V4.4: 增加评价关系)"""

    def __init__(
        self,
        sentence_id: str = "",
        sentence: str = "",
        entities: Optional[list[ExtractedEntity]] = None,
        has_evaluation: bool = False,
        relations: Optional[list[ExtractedRelation]] = None,
    ):
        self.sentence_id = sentence_id
        self.sentence = sentence
        self.entities = entities or []
        self.has_evaluation = has_evaluation
        self.relations = relations or []

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "entities": [e.to_dict() for e in self.entities],
            "has_evaluation": self.has_evaluation,
            "relations": [r.to_dict() for r in self.relations],
        }

    @staticmethod
    def from_dict(data: dict) -> "SentenceExtractionOutput":
        return SentenceExtractionOutput(
            sentence_id=data.get("sentence_id", ""),
            sentence=data.get("sentence", ""),
            entities=[ExtractedEntity.from_dict(e) for e in data.get("entities", [])],
            has_evaluation=data.get("has_evaluation", False),
            relations=[ExtractedRelation.from_dict(r) for r in data.get("relations", [])],
        )


# ===========================================================================
# Agent 1: Entity Extraction Agent
# ===========================================================================


class EntityExtractionAgent:
    """
    Agent 1 — 实体抽取 + 评价关系识别 (V4.4 合并版)
    单次 LLM 调用完成:
      1. 全部 Agent/Artifact/Abstract/Event 实体抽取
      2. 评价关系识别 (主体-客体-极性)
    输出中间结果 (SentenceExtractionOutput) 写入 mid_data/。
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, statement: str, sentence_id: str = "") -> SentenceExtractionOutput:
        prompt = ENTITY_EXTRACTION_PROMPT.format(statement=statement)
        data = self.llm.call_json(prompt, {})
        if not isinstance(data, dict):
            return SentenceExtractionOutput(
                sentence_id=sentence_id, sentence=statement, entities=[], relations=[]
            )

        # ── 实体抽取 (原逻辑) ──
        entities: list[ExtractedEntity] = []
        entity_ids: set[str] = set()
        raw_entities = data.get("entities", []) or []
        for i, item in enumerate(raw_entities):
            if not isinstance(item, dict):
                continue
            mention = item.get("mention") or item.get("entity_name") or ""
            if not mention:
                continue
            eid = f"{sentence_id}_e{i + 1}"
            entity_ids.add(eid)
            entities.append(
                ExtractedEntity(
                    entity_id=eid,
                    mention=mention,
                    normalized_name=item.get("normalized_name") or mention,
                    candidate_l3=item.get("candidate_l3") or "",
                    candidate_l1=item.get("candidate_l1") or [],
                    evidence=item.get("evidence") or "",
                    is_specific_entity=item.get("is_specific_entity", True),
                    confidence=item.get("confidence", 0.0),
                    uncertainty=item.get("uncertainty", ""),
                )
            )

        # ── 评价关系识别 (V4.4 新增) ──
        has_evaluation = bool(data.get("has_evaluation", False))
        raw_relations = data.get("relations", [])
        if not isinstance(raw_relations, list):
            raw_relations = []

        relations: list[ExtractedRelation] = []
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
            # 验证 object 是否为已识别的实体ID
            if obj != "_missing_entity" and obj not in entity_ids:
                object_text = object_text or str(item.get("object", "")).strip()
                obj = "_missing_entity"

            relations.append(
                ExtractedRelation(
                    subject=subject,
                    object=obj,
                    aspect=aspect,
                    opinion=opinion,
                    evidence=evidence,
                    object_text=object_text,
                )
            )

        return SentenceExtractionOutput(
            sentence_id=sentence_id,
            sentence=statement,
            entities=entities,
            has_evaluation=has_evaluation or bool(relations),
            relations=relations,
        )
