"""
pipeline.entity_extraction_agent — Agent 1: 实体抽取 (一体化 L1+L2)
对齐: 实体类型分类体系_opencode版.md — LangGPT 风格提示词
从评价句中高召回抽取全部实体类型: Agent / Artifact / Abstract / Event
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
    "1. 读取待分析评价句\n"
    "2. 扫描句中所有名词短语/专有名词\n"
    "3. 逐一过实体性门槛 → 剔除泛称、通用词、评价用语\n"
    "4. 对通过过滤的短语 → 判定 candidate_l3 (从 Background 分类体系中选)\n"
    "5. 对每个实体判定 candidate_l1 (可多选，召回优先)\n"
    "6. 提取 evidence (原句中的证据片段)\n"
    "7. 按 OutputFormat 输出严格 JSON\n\n"
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
    "- `algorithm`: 算法/模型(SOM/聚类算法)\n"
    "- `instrument`: 测量工具(调查问卷/心理量表/元素依赖性指数)\n"
    "> vs: algorithm = 计算逻辑，software = 算法实现，instrument = 测量工具。\n"
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
    "## OutputFormat\n"
    "严格 JSON，不含 markdown 代码块。entities 数组，无实体则为空数组 []。\n"
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


class SentenceExtractionOutput:
    """单句的抽取输出"""

    def __init__(
        self,
        sentence_id: str = "",
        sentence: str = "",
        entities: Optional[list[ExtractedEntity]] = None,
    ):
        self.sentence_id = sentence_id
        self.sentence = sentence
        self.entities = entities or []

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "entities": [e.to_dict() for e in self.entities],
        }

    @staticmethod
    def from_dict(data: dict) -> "SentenceExtractionOutput":
        return SentenceExtractionOutput(
            sentence_id=data.get("sentence_id", ""),
            sentence=data.get("sentence", ""),
            entities=[ExtractedEntity.from_dict(e) for e in data.get("entities", [])],
        )


# ===========================================================================
# Agent 1: Entity Extraction Agent
# ===========================================================================


class EntityExtractionAgent:
    """
    Agent 1 — 实体抽取
    单次 LLM 调用完成全部 Agent + Artifact + Abstract + Event 实体抽取。
    输出中间结果 (SentenceExtractionOutput) 写入 mid_data/。
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, statement: str, sentence_id: str = "") -> SentenceExtractionOutput:
        prompt = ENTITY_EXTRACTION_PROMPT.format(statement=statement)
        data = self.llm.call_json(prompt, {})
        if not isinstance(data, dict):
            return SentenceExtractionOutput(
                sentence_id=sentence_id, sentence=statement, entities=[]
            )
        entities: list[ExtractedEntity] = []
        raw_entities = data.get("entities", []) or []
        for i, item in enumerate(raw_entities):
            if not isinstance(item, dict):
                continue
            mention = item.get("mention") or item.get("entity_name") or ""
            if not mention:
                continue
            eid = f"{sentence_id}_e{i + 1}"
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
        return SentenceExtractionOutput(
            sentence_id=sentence_id, sentence=statement, entities=entities
        )
