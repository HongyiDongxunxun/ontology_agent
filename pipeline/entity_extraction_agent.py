"""
pipeline.entity_extraction_agent — 实体抽取（Agent 2）

对齐: 实体类型分类体系_opencode版.md — LangGPT 风格提示词
从评价句中高召回抽取全部实体类型: Agent / Artifact / Abstract / Event
接收上游 Agent 1 评价关系的 subject/object 原文短语作为必抽提示，
由本 Agent 统一负责实体的识别与分类，避免与 Agent 1 职责重叠。
输出中间结果到 mid_data/ 目录
"""

from __future__ import annotations

from typing import Optional

from .llm import LLMClient

# ===========================================================================
# 实体抽取 Prompt — LangGPT 风格 (Role/Profile/Rules/Workflow/Background/OutputFormat/Examples/Input)
# 不包含评价关系识别，仅做实体抽取
# ===========================================================================

ENTITY_EXTRACTION_PROMPT = (
    "# Role\n"
    "你是一个学术文献实体抽取专家。\n\n"
    "## Profile\n"
    "- 专长: 从学术评价句中抽取符合实体性双闸门的实体 (含抽象概念与构念)\n"
    "- 能力: 识别四大类别实体 — Agent(行为主体)、Artifact(人工制品)、Abstract(抽象实体)、Event(事件)\n"
    "- 原则: 论元位置＋学术身份双闸门判定; 抽象不等于无效\n"
    "- 风格: 精准、忠实原文、不概括、不创造名称\n\n"
    "## Rules\n"
    "### 实体性判定 — 双闸门 (entity-hood 核心标准)\n"
    "**抽象 ≠ 无效。** 一个短语是实体，当且仅当同时通过两道闸门:\n"
    "**闸门一 · 论元位置**: 它在本句中是**被论及的对象** (被定义/提出/采用/改进/测量/评价/分类，或充当「X 的 Y」中的 Y)，而不是说话者正在执行的动作或给出的判断。\n"
    "**闸门二 · 学术身份**: 该字符串本身是可复用的学术名称，满足任一锚定:\n"
    "  a. 专名锚定: 具体的人/机构/文献/系统/数据集/标准/会议名\n"
    "  b. 术语锚定: 学术界稳定复用的概念/理论/模型/框架/方法/技术/现象/领域名 (能回答「X 是一种___」)\n"
    "  c. 构念锚定: 带限定域、可定义或测量的变量性名词，限定域与构念外壳**整体成词**\n"
    "     如: 专利价值、专利保护强度、专利授权速度、知识生产策略、知识共享障碍、网络媒体态势感知\n"
    "  d. 事件锚定: 有参与方、在时间中展开的命名事件/运动/争论/阶段\n"
    "> 「独立指称」指字符串本身是可复用名称，而非能脱离语境定位唯一实例 — 因此抽象术语与构念同样有效。\n\n"
    "### 三问测试 (每个候选依次自问):\n"
    "1. **论元问**: 它是被论及的对象，还是句中的动作/判断本身? 后者不抽 (但动作所涉及的命名对象要抽，见下方句法模式 C)。\n"
    "2. **脱句问**: 单说「X 是一种___」能否填入学术类别 (方法/现象/理论/构念/系统/领域…)? 填不出 → 不抽。\n"
    "3. **限定问**: 剥去限定后是否只剩零内涵外壳 (能力/因素/作用/水平/问题/方式/方面/内容/情况/特征/关系/结果/效果/意义)?\n"
    "   - 外壳与限定域整体构成可定义构念 → 抽整体 (「专利保护强度」「个体知识共享动机」);\n"
    "   - 带限定后仍只是**类别标签** (研究方法/文献类型/新闻公告/分类问题/技术名词) 或**句内指代** (该方法/这种方法/现有方法/本文/当前研究) → 不抽。\n\n"
    "### 无锚定信号词 (仅在裸现时排除):\n"
    "以下词**单独出现、无任何锚定**时不抽取; 一旦整体构成具体名称或命名术语则照抽。判定看整体身份，不看是否命中词表:\n"
    "1. 泛化身份 (科学家/学者/研究者/专家/教授/馆长/图书馆员/工程师/作者/学生/读者用户): 裸现不抽; 「Schankerman」「朱雪忠」抽。\n"
    "2. 泛化机构 (大学图书馆/高校/研究机构/公共图书馆/档案馆/公司/出版社): 裸现不抽; 「北京大学图书馆」「英国法院」抽。\n"
    "3. 虚义动词 (比较/进行/存在/通过/基于/利用/实现/导致/产生/推动/促进/影响)\n"
    "4. 句内指代 (这方面/该问题/上述研究/本文/笔者/相关文献/某论文/现有文献)\n"
    "5. 计量词 (篇数/比例/数量/百分比/平均值/标准差/准确率/召回率)\n"
    "6. 泛化评价词 (重要/显著/明显/突出/不足/深远/广泛)\n"
    "7. 单纯时间 (2005年/近年来): 除非明确指发展阶段，否则不抽为 stage\n"
    "8. 单纯地名 (深圳/青岛): 除非指具体机构主体，否则不抽为 governance\n\n"
    "### 非实体形式 (按句法模式，不按词项; 一律**先修剪、后判定**):\n"
    "A. 完整小句/命题: 含主谓或动宾、表达完整命题的片段不抽。先剥命题外壳 (「…的认识/的问题/的证据/的结果/的有效性/…之间的关系」)，再对核心走双闸门。\n"
    "   「发明人数量越多，剩余寿命越长，专利价值越高」不抽;「专利年龄与专利价值之间的关系」剥壳后无独立构念不抽;「专利保护强度」抽。\n"
    "B. 评价/状态语: 含「较为/日益/蓬勃/较窄/更高/更低」等评价修饰的整串不抽; 其中被评价的命名对象单独抽。「保护范围较窄的专利申请」不抽，「专利保护范围」抽。\n"
    "C. 动作短语: 谓语框架 (运用/利用/提升/分析/构建/追踪/甄别…＋宾语) 整体不抽，**但框架中的命名宾语照抽**。\n"
    "   「运用机器学习」→ 只抽「机器学习」;「提升健康信息素养」→ 抽「健康信息素养」。\n"
    "   注意: **「以动词开头」不是排除理由** — 动词性技术名处于论元位置时照抽: 信息检索、多模态数据融合、注意力机制、局部序列特征提取。\n"
    "D. 裸泛称: 适用三问第 3 问。「安全」「新闻公告」「能力」「个人因素」「知识共享动机的作用」不抽;「知识共享动机」抽。\n"
    "E. 人名/文献引用: **引用不是非实体**，整体保留，规则见边界修剪第 3 条。\n"
    "F. 「基于…的…方法」长名: 不按整句判死。先剥「基于…的」等修饰，对核心方法名走双闸门:\n"
    "   「基于GAT的会话图学习方法」→ 抽「会话图学习方法」;「基于RNN的会话推荐方法」→ 抽「会话推荐方法」;\n"
    "   剥后核心仍是动宾描述 (「通过数据挖掘获取热点主题的方法」) → 不抽。\n"
    "G. 研究/文献泛称: 「XX研究/XX研究文献/最近的研究/现有文献/有关XX的研究」无具体文献名时不抽; 有具体题名 (《…》、具名报告) 才抽。\n\n"
    "### 原文忠实原则:\n"
    "1. mention 必须是**原文中连续出现的文字**，严禁概括、总结、改名\n"
    "2. normalized_name 是规范化形式 (去OCR乱码、统一简繁)，无特殊问题时与 mention 一致\n"
    "3. 带缩写/中英文名对照的实体，按下方边界修剪第 2 条括号三分支处理\n\n"
    "### 边界修剪规则 — mention 取「最小独立指称短语」:\n"
    "1. **连续子串、允许切分**: mention 必须是原文连续文字; 允许在并列连词、括号、命题外壳边界处切分; 不得改写，不得在句中截断。「湖南…学科导航库的建…」须还原为完整短语「学科导航库」才抽; 无法还原则不抽 (二选一，不得输出残片)。\n"
    "2. **括号三分支**:\n"
    "   (a) 括号内是官方中英文名/缩写对照 → **保留完整形式**作为一个实体 (「卷积神经网络（convolutional neural network,CNN）」「巴巴拉·奎恩特（B. Quint）」); 其中缩写或英文名在文中独立出现时可另抽一条;\n"
    "   (b) 括号外是描述、括号内是专名代号 → 取括号内专名 (「融入引用信息的模型(citation-author-topic,cat)」→「citation-author-topic」);\n"
    "   (c) 括号内是注释/举例/OCR 残缺 (如「Processes)模型」式残片) → 剥离; 剥后不完整则不抽。\n"
    "3. **引用与人名**: 人名/文献引用**整体保留**，不剥离年份、et al.、等 — 「薛明皋（2013）」「Wah et al(2007)」「Paul等」均可作一个 scholar/paper 实体; 并列人名各自单独成实体 (「Davenport和Prusak」→ 两条;「Siemsen, Roth 和」→「Siemsen」「Roth」两条)。仅当引用串 OCR 截断到无法辨认 (「L.W.Anderson 和D.」) 才不抽。\n"
    "4. **剥修饰性定语**: 剥「传统的/新兴的/本文提出的/基于…的」等修饰，保留核心名 (「传统的学术期刊出版模式」→「学术期刊出版模式」)。但**构念限定语不剥**:「专利保护强度」中「专利保护」是构念成分，须整体保留。\n"
    "5. **拆并列**: 「A、B、C」「A和B」各自独立抽取，不输出连词与逗号; 「GRU4REC(Gated Recurrent Unit…)」→「GRU4REC」。\n"
    "6. **剥命题外壳**: 剥「…的认识/问题/证据/关系/结果/有效性/作用」后取核心名词;「Lerner（1994）设计的专利保护范围变量」→「专利保护范围变量」; 剥壳后无独立构念 (「知识共享动机的作用」) → 不抽。\n"
    "7. evidence 从原句截取，保留完整上下文用于核查，不受 mention 修剪影响。\n\n"
    "### 抽取决策顺序 (替代「宁漏勿滥」):\n"
    "1. 先修剪候选 span (边界规则 1–6)\n"
    "2. 对修剪后的核心短语走双闸门三问\n"
    "3. 过闸 → 抽取; 类型不清 → 多标 candidate_l1、降低 confidence、uncertainty 写明疑点，**不要丢弃**\n"
    "4. 不过闸 → 不进 entities; 若该短语来自上游必抽清单，仍输出一条但置 is_specific_entity=false、candidate_l3 留空、uncertainty 注明 (不占用正常实体)\n"
    "5. 不使用「宁漏勿滥」也不降低门槛: 通过双闸门的命名实体 (含抽象术语、可测构念、动词性技术名) 不得因谨慎跳过; 没通过的短语不得因召回压力放行。\n\n"
    "## Workflow\n"
    "1. 读取待分析句子，结合必抽实体列表（来自上游 Agent 1 评价关系的主客体原文短语），识别句中所有符合 Agent/Artifact/Abstract/Event 分类体系的有效实体。\n"
    "2. 必抽实体来自上游评价关系抽取的主客体，是**优先核查候选**而非免检项: 过双闸门者正常抽取; 不过闸者不进 entities，按 Rules「抽取决策顺序」第 4 条置 is_specific_entity=false 输出并注明原因。\n"
    "3. 除必抽实体外，补充抽取句中其他有效实体。\n"
    "4. 对每个实体判定 candidate_l3、candidate_l1，并提取 evidence。\n"
    "5. 按 OutputFormat 输出严格 JSON。\n\n"
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
    "> 电子计算机、机器人、VR/AR设备、5G、大数据、人工智能技术、云计算等不得因「技术/设备」泛化标为 algorithm 或 instrument。\n"
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
    "## OutputFormat\n"
    "严格 JSON，不含 markdown 代码块。entities 数组无实体则为空数组 []。\n"
    '{{\n'
    '  "entities": [\n'
    '    {{\n'
    '      "mention": "<原文精确短语，严禁概括/改名>",\n'
    '      "normalized_name": "<规范化实体名，无特殊问题时等于mention>",\n'
    '      "candidate_l3": "<type_code, 从 Background 分类体系中选择>",\n'
    '      "candidate_l1": ["Agent|Artifact|Abstract|Event (类型不清时允许多个)"],\n'
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
    '输出: {{"entities":[{{"mention":"潘光旦","normalized_name":"潘光旦","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"在谱学研究领域颇有建树的学者有潘光旦","is_specific_entity":true,"confidence":0.95,"uncertainty":""}},{{"mention":"罗香林","normalized_name":"罗香林","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"在谱学研究领域颇有建树的学者有潘光旦、罗香林等人","is_specific_entity":true,"confidence":0.95,"uncertainty":""}}]}}\n\n'
    "### 正例2 (research — 知识生产组织 + phenomenon)\n"
    "输入: 加拿大不列颠哥伦比亚大学的卫生保健管理中心门户就是一个努力帮助用户克服信息过载的网络信息中介的示例。\n"
    '输出: {{"entities":[{{"mention":"加拿大不列颠哥伦比亚大学","normalized_name":"不列颠哥伦比亚大学","candidate_l3":"research","candidate_l1":["Agent"],"evidence":"加拿大不列颠哥伦比亚大学的卫生保健管理中心门户","is_specific_entity":true,"confidence":0.95,"uncertainty":""}},{{"mention":"信息过载","normalized_name":"信息过载","candidate_l3":"phenomenon","candidate_l1":["Abstract"],"evidence":"帮助用户克服信息过载","is_specific_entity":true,"confidence":0.9,"uncertainty":"现实信息问题,非术语本身"}}]}}\n\n'
    "### 正例3 (book — 专著)\n"
    "输入: 新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果。与旧版相比,其观点更新颖,内容更充实,结构更合理。\n"
    '输出: {{"entities":[{{"mention":"新版《图书馆学概论》","normalized_name":"《图书馆学概论》","candidate_l3":"book","candidate_l1":["Artifact"],"evidence":"新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果","is_specific_entity":true,"confidence":0.95,"uncertainty":""}}]}}\n\n'
    "### 正例4 (knowledge_organization_system)\n"
    "输入: 关于类目虚设问题。这点《中图法》比较突出,尤以自然科学类为最,不但加重了分类法的篇幅,也给分类员制造了麻烦。\n"
    '输出: {{"entities":[{{"mention":"《中图法》","normalized_name":"《中图法》","candidate_l3":"knowledge_organization_system","candidate_l1":["Artifact"],"evidence":"这点《中图法》比较突出","is_specific_entity":true,"confidence":0.95,"uncertainty":""}}]}}\n'
    "> 「比较突出」中的「比较」为虚义动词，不抽取。\n\n"
    "### 正例5 (theory — 理论)\n"
    "输入: Ausubel基于学习者认知结构的学习迁移理论与这些研究问题非常契合。已有研究并没有深入探讨学习迁移的基础理论。\n"
    '输出: {{"entities":[{{"mention":"学习迁移理论","normalized_name":"学习迁移理论","candidate_l3":"theory","candidate_l1":["Abstract"],"evidence":"Ausubel基于学习者认知结构的学习迁移理论","is_specific_entity":true,"confidence":0.92,"uncertainty":""}}]}}\n'
    "> 「基础理论」为泛称，不抽取。只抽取有具体命名的理论。\n\n"
    "### 正例6 (method — 研究方法)\n"
    "输入: 传统的统计回归方法常采用线性或多项式函数;而机器学习方法更倾向于复杂的非线性模型,能得到较高准确率。\n"
    '输出: {{"entities":[{{"mention":"统计回归方法","normalized_name":"统计回归方法","candidate_l3":"method","candidate_l1":["Abstract"],"evidence":"传统的统计回归方法常采用线性或多项式函数","is_specific_entity":true,"confidence":0.95,"uncertainty":""}},{{"mention":"机器学习方法","normalized_name":"机器学习方法","candidate_l3":"method","candidate_l1":["Abstract"],"evidence":"机器学习方法更倾向于复杂的非线性模型","is_specific_entity":true,"confidence":0.95,"uncertainty":""}}]}}\n'
    "> 「预测结果」「高准确率」为评价用语，不抽取。\n\n"
    "### 正例7 (phenomenon)\n"
    "输入: 长期以来各文化机构独自推进的智改数转造就了一座座数据孤岛,底层关联不足进而会引发上层文化服务割裂。\n"
    '输出: {{"entities":[{{"mention":"数据孤岛","normalized_name":"数据孤岛","candidate_l3":"phenomenon","candidate_l1":["Abstract"],"evidence":"造就了一座座数据孤岛","is_specific_entity":true,"confidence":0.95,"uncertainty":""}}]}}\n'
    "> 「文化服务割裂」若无独立学术命名则不抽取。\n\n"
    "### 正例7b (subfield vs concept)\n"
    "输入: 信息检索在图书情报学研究中形成了稳定的问题域和方法传统。\n"
    '输出: {{"entities":[{{"mention":"信息检索","normalized_name":"信息检索","candidate_l3":"subfield","candidate_l1":["Abstract"],"evidence":"信息检索在图书情报学研究中形成了稳定的问题域和方法传统","is_specific_entity":true,"confidence":0.9,"uncertainty":"作为研究子领域,非concept兜底"}}]}}\n\n'
    "### 正例7c (concept 的正向用法)\n"
    "输入: 「信息素养」这一概念强调个体识别、获取和评价信息的能力。\n"
    '输出: {{"entities":[{{"mention":"信息素养","normalized_name":"信息素养","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"「信息素养」这一概念","is_specific_entity":true,"confidence":0.9,"uncertainty":"原句讨论术语/概念名本身"}}]}}\n\n'
    "### 正例8 (debate + movement)\n"
    "输入: 情报学中对于Information与Intelligence的争论应该是有益的。开放获取意味着文章一旦被创造出来,将通过网络让读者免费获取和利用。\n"
    '输出: {{"entities":[{{"mention":"Information与Intelligence的争论","normalized_name":"Information与Intelligence的争论","candidate_l3":"debate","candidate_l1":["Event"],"evidence":"情报学中对于Information与Intelligence的争论应该是有益的","is_specific_entity":true,"confidence":0.92,"uncertainty":""}},{{"mention":"开放获取","normalized_name":"开放获取","candidate_l3":"movement","candidate_l1":["Event"],"evidence":"开放获取意味着文章一旦被创造出来","is_specific_entity":true,"confidence":0.9,"uncertainty":"可兼为concept"}}]}}\n'
    "> 「开放获取」既可作 concept 也可作 movement。此处描述其作为运动的方式，优先标 movement。\n\n"
    "### 负例1 (泛称身份不抽取)\n"
    "输入: 许多科学家认为开放获取能推动学术交流与合作。\n"
    '输出: {{"entities":[]}}\n'
    "> 「科学家」是泛化身份类别词，不指称具体个人 → 不抽取。\n\n"
    "### 负例2 (通用词/评价用语不抽取)\n"
    "输入: 通过比较两种方法的优劣，本文认为该理论具有重要意义。\n"
    '输出: {{"entities":[]}}\n'
    "> 「比较」通用动词、「优劣」评价用语、「本文」自指、「重要意义」评价用语 → 均不抽取。\n\n"
    "### 边界修剪正例1 (缩写粘连 → 取专名)\n"
    "输入: 该方法采用GRU4REC(Gated Recurrent Unit)模型进行会话建模。\n"
    '输出: {{"entities":[{{"mention":"GRU4REC","normalized_name":"GRU4REC","candidate_l3":"model","candidate_l1":["Abstract"],"evidence":"GRU4REC(Gated Recurrent Unit)模型","is_specific_entity":true,"confidence":0.9,"uncertainty":""}}]}}\n'
    "> 错误示范: mention 取「GRU4REC(Gated Recurrent Unit)」(夹带括号全称) → 应修剪为「GRU4REC」。\n\n"
    "### 边界修剪正例2 (描述+括号专名 → 取括号内专名)\n"
    "输入: 融入引用信息的模型(citation-author-topic,cat)在链接预测上表现优异。\n"
    '输出: {{"entities":[{{"mention":"citation-author-topic","normalized_name":"citation-author-topic","candidate_l3":"model","candidate_l1":["Abstract"],"evidence":"融入引用信息的模型(citation-author-topic,cat)","is_specific_entity":true,"confidence":0.85,"uncertainty":"括号外为描述,取括号内专名"}}]}}\n'
    "> 错误示范: mention 取「融入引用信息的模型」(描述语) 或整段夹带括号 → 应取专名「citation-author-topic」。\n\n"
    "### 边界修剪正例3 (并列人名 → 拆分)\n"
    "输入: Davenport和Prusak提出了知识管理理论。\n"
    '输出: {{"entities":[{{"mention":"Davenport","normalized_name":"Davenport","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"Davenport和Prusak","is_specific_entity":true,"confidence":0.9,"uncertainty":""}},{{"mention":"Prusak","normalized_name":"Prusak","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"Davenport和Prusak","is_specific_entity":true,"confidence":0.9,"uncertainty":""}}]}}\n'
    "> 错误示范: mention 取「Davenport和Prusak」(并列粘连) → 应拆为两个人名分别抽取。\n\n"
    "### 边界修剪正例4 (构念锚定＋命题外壳)\n"
    "输入: 实证研究表明，专利授权速度与专利保护范围之间存在 U 型关系。\n"
    '输出: {{"entities":[{{"mention":"专利授权速度","normalized_name":"专利授权速度","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"专利授权速度与专利保护范围之间存在U型关系","is_specific_entity":true,"confidence":0.9,"uncertainty":"限定域+可测构念,整体成词"}},{{"mention":"专利保护范围","normalized_name":"专利保护范围","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"专利授权速度与专利保护范围之间存在U型关系","is_specific_entity":true,"confidence":0.9,"uncertainty":""}},{{"mention":"U型关系","normalized_name":"U型关系","candidate_l3":"phenomenon","candidate_l1":["Abstract"],"evidence":"存在U型关系","is_specific_entity":true,"confidence":0.85,"uncertainty":""}}]}}\n'
    "> 「…之间存在…关系」是命题框架不抽; 两端构念 (专利授权速度/专利保护范围) 虽抽象但可定义可测，必须抽。\n\n"
    "### 边界修剪正例5 (引用整体保留)\n"
    "输入: 薛明皋（2013）及唐恒等(2014)针对专利质押贷款这一情境讨论专利价值的分析指标体系。\n"
    '输出: {{"entities":[{{"mention":"薛明皋（2013）","normalized_name":"薛明皋","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"薛明皋（2013）及唐恒等(2014)","is_specific_entity":true,"confidence":0.9,"uncertainty":"引用整体保留,normalized去年份"}},{{"mention":"唐恒等(2014)","normalized_name":"唐恒","candidate_l3":"scholar","candidate_l1":["Agent"],"evidence":"薛明皋（2013）及唐恒等(2014)","is_specific_entity":true,"confidence":0.9,"uncertainty":""}},{{"mention":"专利质押贷款","normalized_name":"专利质押贷款","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"针对专利质押贷款这一情境","is_specific_entity":true,"confidence":0.85,"uncertainty":""}},{{"mention":"专利价值","normalized_name":"专利价值","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"讨论专利价值的分析指标体系","is_specific_entity":true,"confidence":0.9,"uncertainty":""}}]}}\n'
    "> 年份与「等」不剥离，引用串整体是实体;「分析指标体系」在本句是类别性宾语，不单抽。\n\n"
    "### 边界修剪正例6 (同词不同命 — 论元位置决定)\n"
    "输入: 网络媒体态势感知研究常使用大数据、机器学习算法，涵盖自然语言处理、计算机视觉等技术。\n"
    '输出: {{"entities":[{{"mention":"网络媒体态势感知研究","normalized_name":"网络媒体态势感知研究","candidate_l3":"subfield","candidate_l1":["Abstract"],"evidence":"网络媒体态势感知研究常使用","is_specific_entity":true,"confidence":0.85,"uncertainty":"作为研究领域被论及"}},{{"mention":"大数据","normalized_name":"大数据","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"常使用大数据","is_specific_entity":true,"confidence":0.8,"uncertainty":"论元位置上的命名技术概念"}},{{"mention":"机器学习算法","normalized_name":"机器学习算法","candidate_l3":"algorithm","candidate_l1":["Artifact"],"evidence":"大数据、机器学习算法","is_specific_entity":true,"confidence":0.85,"uncertainty":""}},{{"mention":"自然语言处理","normalized_name":"自然语言处理","candidate_l3":"subfield","candidate_l1":["Abstract"],"evidence":"涵盖自然语言处理、计算机视觉等技术","is_specific_entity":true,"confidence":0.9,"uncertainty":""}},{{"mention":"计算机视觉","normalized_name":"计算机视觉","candidate_l3":"subfield","candidate_l1":["Abstract"],"evidence":"涵盖自然语言处理、计算机视觉等技术","is_specific_entity":true,"confidence":0.9,"uncertainty":""}}]}}\n'
    "> 同一词 (如「大数据」「深度学习」) 在论元位置作为命名技术/领域被论及时抽; 仅作「本文/现有研究」式泛指或铺垫时不抽。判定看句法语境，不看词本身。\n\n"
    "### 负例3 (构念外壳不抽，核心构念抽)\n"
    "输入: 该文综述了知识共享动机的作用、组成因素与影响因素。\n"
    '输出: {{"entities":[{{"mention":"知识共享动机","normalized_name":"知识共享动机","candidate_l3":"concept","candidate_l1":["Abstract"],"evidence":"知识共享动机的作用、组成因素与影响因素","is_specific_entity":true,"confidence":0.9,"uncertainty":""}}]}}\n'
    "> 「知识共享动机的作用」「组成因素」「影响因素」是零内涵外壳，不抽; 核心构念「知识共享动机」抽。\n\n"
    "## 反例清单（Negative Examples）\n"
    "> 以下为真实误抽样例，标注处理方式。抽取时须主动规避，不得复现同类错误。\n\n"
    "### 问题1: 过度抽取 — 未过双闸门的短语\n"
    "| 原文 span | 形式类型 | 处理 |\n"
    "|---|---|---|\n"
    "| 建模搜索者知识改变过程及策略的研… | 描述性小句 | 不抽取 |\n"
    "| 知识提供者的信念、态度、意图和行… | 描述性小句 | 不抽取 |\n"
    "| 资源浪费较为严重 | 评价/状态语 | 整串不抽取 (核心另判) |\n"
    "| 总体生活环境的发展 | 评价/状态语 | 不抽取 |\n"
    "| 正在出现的,具备蓬勃的发展趋势 | 评价/状态语 | 不抽取 |\n"
    "| 运用机器学习 | 动作短语 | 框架不抽，宾语「机器学习」抽 |\n"
    "| 提升健康信息素养、甄别疫情虚假信… | 动作短语 | 框架不抽，「健康信息素养」抽 |\n"
    "| 安全 | 泛称短词 | 裸现无锚定，不抽取 |\n"
    "| 新闻公告 | 泛称短词 | 不抽取 |\n"
    "| 研究方法 | 泛称类别词 | 不抽取 |\n"
    "| 文献类型 | 泛称类别词 | 不抽取 |\n"
    "| Siemsen, Roth 和 | 并列人名 | 拆分为「Siemsen」「Roth」两条 |\n"
    "| Davenport和Prusak | 并列人名 | 拆分为「Davenport」「Prusak」两条 |\n"
    "| Xie等[73] | 人名引用 | 整体保留为一条 scholar (不剥离[73]) |\n"
    "| L.W.Anderson 和D. | OCR 截断残片 | 无法辨认，不抽取 |\n"
    "| 基于混合依赖序列特征提取的抽取方法 | 方法长名 | 剥修饰后取核心方法名; 非专名则不抽 |\n"
    "| 基于GAT的会话图学习方法 | 方法长名 | 应修剪为「会话图学习方法」 |\n\n"
    "### 问题2: 边界切分不规范 — 修剪为核心名词短语\n"
    "| 原文 span | 处理 |\n"
    "|---|---|\n"
    "| 融入引用信息的模型(citation-author-topic,cat） | 括号内专名代号 → 取「citation-author-topic」 |\n"
    "| 卷积神经网络（convolutional neural network,CNN） | 官方中英对照 → 保留完整形式 |\n"
    "| GRU4REC(Gated Re… | 括号残缺 → 取「GRU4REC」 |\n"
    "| Processes)模型 | OCR 括号残片 → 剥离后不完整，不抽取 |\n"
    "| 湖南高校图书馆学科导航库的建… | 唯一处理: 还原为「学科导航库」; 无法还原则不抽 |\n"
    "| 欧洲专利价值调研问卷数据 | 应修剪为「欧洲专利价值调研问卷」 |\n"
    "| 传统的学术期刊出版模式 | 应修剪为「学术期刊出版模式」 |\n"
    "| 作者过去的推特总数和转发率 | 应修剪为「推特总数和转发率」 |\n\n"
    "## 必抽实体（来自上游 Agent 1 评价关系的主客体原文短语）\n"
    "以下短语在评价关系中被作为 subject/object 引用，是**优先核查候选** (非免检项):\n"
    "- 过双闸门者 → 抽取为正常实体 (is_specific_entity=true)\n"
    "- 不过闸门者 (泛指/指代/动作/命题外壳) → 仍输出一条但 is_specific_entity=false、candidate_l3 留空，在 uncertainty 中说明原因\n"
    "{required_mentions_text}\n\n"
    "## Input\n"
    "{statement}"
)

# ===========================================================================
# 中间结果数据模型
# ===========================================================================


class ExtractedEntity:
    """抽取的单条实体中间结果"""

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
    """单句的抽取输出（仅实体，不含评价关系）"""

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
# 实体抽取 Agent
# ===========================================================================


class EntityExtractionAgent:
    """
    实体抽取 Agent（Agent 2）
    从评价句中高召回抽取全部 Agent/Artifact/Abstract/Event 实体。
    接收上游 Agent 1 评价关系抽取的主客体原文短语作为必抽提示，
    由本 Agent 统一负责实体的识别、ID 分配与分类。
    输出中间结果 (SentenceExtractionOutput) 写入 mid_data/。
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(
        self,
        statement: str,
        sentence_id: str = "",
        required_mentions: Optional[list[str]] = None,
    ) -> SentenceExtractionOutput:
        """抽取实体。

        Args:
            statement: 待分析句子原文。
            sentence_id: 句子 ID，用于生成实体 ID (格式 {sentence_id}_eN)。
            required_mentions: 上游 Agent 1 评价关系的主客体原文短语，
                这些短语会被注入 prompt 作为必抽提示。占位符
                (_paper_author / _cite[N] / _unknown / _missing_entity)
                由调用方过滤，不应出现在本列表中。
        """
        mentions_text = self._format_required_mentions(required_mentions or [])
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            statement=statement, required_mentions_text=mentions_text
        )
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
            sentence_id=sentence_id,
            sentence=statement,
            entities=entities,
        )

    @staticmethod
    def _format_required_mentions(mentions: list[str]) -> str:
        """将上游评价关系的主客体原文短语格式化为提示词文本。

        - 输入是字符串列表 (已是原文短语，不含占位符)。
        - 空列表返回 "无"。
        """
        if not mentions:
            return "无"
        # 去重保序
        seen: set[str] = set()
        lines: list[str] = []
        for m in mentions:
            m = (m or "").strip()
            if not m or m in seen:
                continue
            seen.add(m)
            lines.append(f"- {m}")
        return "\n".join(lines) if lines else "无"