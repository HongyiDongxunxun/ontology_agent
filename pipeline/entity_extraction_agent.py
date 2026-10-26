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
from .compact_output import derive_evidence, parse_flags
from .taxonomy import get_l1l2_from_l3

# ===========================================================================
# 实体抽取 Prompt — LangGPT 风格 (Role/Profile/Rules/Workflow/Background/OutputFormat/Examples/Input)
# 不包含评价关系识别，仅做实体抽取
# ===========================================================================

ENTITY_EXTRACTION_PROMPT = (
    "# Role\n"
    "你是一个学术文献实体抽取专家。\n\n"
    "## Profile\n"
    "- 从学术评价句中抽取实体，覆盖 Agent / Artifact / Abstract / Event 四大类\n"
    "- 判定原则: 论元位置＋学术身份「双闸门」; 抽象不等于无效\n"
    "- 风格: 忠实原文、不概括、不创造名称\n\n"
    "## Rules\n"
    "### 闸门 3 · 可入库性 (新增)\n"
    "`is_specific_entity` 与下游 `valid_entity` 同义。**false = 默认不抽取、不进入实体列表**。严禁\"先抽进来再让下游判断\"——`is_specific_entity=false` 的短语默认不输出。仅当上游必抽清单强制要求时，才允许输出 `is_specific_entity=false` 的记录，且必须在 uncertainty 字段注明\"上游强制保留\"。\n"
    "### 输出前质量自检\n"
    "输出前逐条检查 mention:\n"
    "- 长度: 15 字触发外壳剥离检查，30 字为硬上限；超过 30 字须剥除非核心定语；\n"
    "- 不含 LaTeX 符号（$、\\mathbb{{}}）、OCR 连字符（⁃、‐）、连续空格；含则清洗或拒绝；\n"
    "- 单字 mention 默认不抽，除非为公认缩写且有明确 evidence；\n"
    "- 同句去重: 按规则四.3 执行，同一实体的全称/缩写/中英文只保留信息最全的一条。\n\n"
    "### 一、实体性双闸门 (同时通过才抽取)\n"
    "**闸门 1 · 论元位置**: 短语在本句中是**被论及的对象** (被定义/提出/采用/改进/测量/评价/分类，或「X 的 Y」中的 Y)，而不是说话者的动作或判断本身。\n"
    "**闸门 2 · 学术身份**: 字符串本身是可复用的学术名称，满足任一锚定:\n"
    "- a 专名: 具体的人/机构/文献/系统/数据集/标准/会议名\n"
    "- b 术语: 学术界稳定复用的概念/理论/模型/框架/方法/技术/现象/领域名，能回答「X 是一种___」\n"
    "- c 构念: 带领域限定、可定义或测量的变量性名词，限定域与构念外壳**整体成词** (专利价值、专利保护强度、知识共享动机、网络媒体态势感知、信息素养)\n"
    "- d 事件: 有参与方、在时间中展开的命名事件/运动/争论/阶段\n\n"
    "**三问快速判定**: ①它是对象还是动作/判断? ②单说「X 是一种___」能否填入学术类别? ③剥掉限定后是否只剩零内涵外壳 (能力/因素/作用/水平/问题/方式/方面/内容/情况/特征/关系/结果/意义)? 外壳与领域整体构成可定义构念则抽整体; 带限定后仍是类别标签 (研究方法/分析方法/文献类型/分类问题) 或句内指代 (该方法/现有方法/本文) 则不抽。\n"
    "**两个操作测试 (灰区短语必做)**:\n"
    "- **替换测试**: 把该短语换成同后缀的另一个具体名称——若句子讨论的客体随之改变 (「采用文献计量法」换成「内容分析法」后说的不再是一回事)，说明它有独立身份，抽; 若换成任何同类词句子都照样成立 (「采用研究方法/该方法」)，它只是占位泛称，不抽。\n"
    "- **删除测试**: 删掉该短语后若句子失去明确客体 (不知道采用/测量/评价的是什么)，抽; 若句意基本不变、只少了铺垫或泛指，不抽。\n"
    "> 核心追问: 它在**当前语境**中是否作为独立研究对象被定义、命名、测量、对比或改进? 仅被顺带提及、充当泛指类别或填充成分的，不抽。\n\n"
    "### 二、高频非实体模式 (按条件判别，不按词封禁)\n"
    "1. **指代表达**: 带 该/此/这/本/其/上述/现有/目前 等限定且无独立命名 → 不抽; 剥掉指代后是命名实体则抽核心 (「本文提出的TAM模型」→抽「TAM模型」，「本文提出的理论框架」→不抽)。\n"
    "2. **动作短语**: 谓语框架 (运用/利用/提升/分析/构建/追踪/甄别…＋宾语) 整体不抽，**其中的命名宾语照抽** (「运用机器学习」只抽「机器学习」)。「以动词开头」本身不是排除理由——动词性技术名处于论元位置照抽: 信息检索、多模态数据融合、注意力机制、局部序列特征提取。\n"
    "3. **评价/状态语**: 含 较为/日益/不断/蓬勃/较窄/更高/偏低 等评价趋势成分的整串不抽; 被评价的命名对象单独抽 (「保护范围较窄的专利申请」不抽，「专利保护范围」抽)。\n"
    "4. **命题/关系外壳**: 「…的认识/问题/证据/结果/有效性/作用」「X 与 Y 之间的关系」整串不抽; 剥壳后对核心名词再走双闸门 (「专利年龄与专利价值之间的关系」不抽，「专利年龄」「专利价值」抽)。**关系形态描述** (U型关系/倒U型/正相关/线性关系) 是对关系的刻画而非可独立指称的实体，不抽。\n"
    "   **性质/状态/作用/结果/行为**等描述性成分本身不抽 (…的现状、…的作用、…的影响、…的效果、…的结果、对…的行为)，只抽被描述的实体; 仅当该成分已被学界命名、并在本句被定义或测量为变量时才按 concept 抽 (「知识共享行为」作为被研究的命名构念时抽)。**两个实体之间形成的关系默认不抽** (A 对 B 的影响、A 与 B 的关系/相关性)，抽 A、B 两端; 例外: 关系已被打包成有专名的理论/模型 (TAM、技术接受模型) 时按 theoretical_construct 抽整体。\n"
    "5. **研究/文献泛称**: 「XX研究/XX研究文献/最近的研究/有关XX的研究」无具体题名时不抽; 有《题名》或具名报告才抽。\n"
    "6. **枚举举例项**: 「如/包括 A、B、C 等」中的普通事物成员不抽 (音乐节、体育比赛、道路、桥梁、医院); 其中具学术身份者才抽 (智慧城市建设、地理标记的推文)。\n"
    "7. **通用部件/计量/属性词**: 无独立学术命名的技术部件 (卷积核、池化操作、邻居节点、隐藏状态)、计量指标 (准确率、召回率、比例、搜索次数)、普通属性 (位置、内容属性) 不抽; 具名方法/机制/量表 (注意力机制、Tie-or-Break规则、元素依赖性指数) 抽。\n"
    "8. **裸泛称/描述性名词**: 安全、新闻公告、能力、个人因素、情报工作、处理措施、研究成果、知识内容、竞争优势(未被当作被定义/测量的研究构念时)、数据的有效利用(动作性描述)、大数据(仅作铺垫泛指时) 等无锚定裸现不抽; 同词在论元位置作为命名概念被专门论及时照抽 (如「常使用大数据、机器学习算法」中二者均抽)。判定看句法语境，不看词本身。\n"
    "9. **后缀不决定实体性** (方法/模型/研究/技术/策略/内容/信息)，一律以替换/删除测试判定:\n"
    "   - 方法: 具名稳定方法抽 (文献计量法、共词分析、LDA);「研究方法/分析方法/该方法/两种方法」不抽\n"
    "   - 模型: 具名抽 (TAM模型、GRU4REC、PageRank);「该模型/现有模型/理论模型/裸「模型」」不抽\n"
    "   - 研究: 有具体题名或命名项目才抽;「XX的研究/现有研究/相关研究/实证研究」不抽\n"
    "   - 技术: 命名技术/领域抽 (自然语言处理、注意力机制);「该技术/相关技术/新兴技术」不抽\n"
    "   - 策略: 被定义的命名策略构念抽 (知识生产策略);「应对策略/优化策略/相关策略」不抽\n"
    "   - 内容/信息: 普通填充性名词不抽 (知识内容、研究内容、网络信息、相关信息、大量信息); 构成领域名/构念时抽 (健康信息素养、信息素养)\n"
    "10. **期刊本体不抽**: 期刊本身（如《经济研究》《中国工业经济》）是连续出版物，不是 journal_article（判定边界见 Background · Discursive）；期刊整体作被评价对象时无更贴切类型则不抽。\n\n"
    "### 三、最小充分语义单元 (mention 边界)\n"
    "mention = 剥离指称/评价/命题外壳等非命名成分后，仍能独立指称该对象的**最短名词短语**。「充分」优先于「最小」:\n"
    "1. **连续、完整、不截断**: 必须是原文连续文字，允许在并列连词/括号/外壳边界切分; 残片须还原为完整短语 (「…学科导航库的建…」→「学科导航库」)，无法还原则不抽。\n"
    "2. **领域限定不剥**: 领域限定是构念成分 (专利保护强度≠强度、知识共享动机≠动机、专利授权速度)，整体保留; 指称/评价/数量/所有者/用途-情境定语不是构念成分 (该、本文、传统的、新兴的、大量、高价值、用于…的、在…场景下的)，剥除 (「传统的学术期刊出版模式」→「学术期刊出版模式」;「用于专利分类的CNN模型」→「CNN模型」)。若限定把通名收窄为另一个被命名的子构念则保留 (「网络媒体态势感知」)。\n"
    "3. **括号三分支**: (a) 官方中英/缩写对照→保留完整形式 (「卷积神经网络（convolutional neural network,CNN)」「巴巴拉·奎恩特（B. Quint）」); (b) 括号外是描述、括号内是专名代号→取括号内 (「…模型(citation-author-topic,cat)」→「citation-author-topic」); (c) 注释/举例/OCR残片→剥离，剥后不完整则不抽。\n"
    "4. **引用与人名**: 人名/文献引用**整体保留**，不剥年份、[N]、et al.、等 (「薛明皋（2013）」「Wah et al(2007)」「Paul等」各为一条); 并列人名各自单独成条 (「Davenport和Prusak」→两条); OCR 截断到不可辨认 (「L.W.Anderson 和D.」) 不抽。\n"
    "5. **并列拆取**: 「A、B、C」「A和B」各自独立抽取，不带连词与逗号。\n\n"
    "### 四、长度、嵌套与去重\n"
    "1. **过长检查**: span ≥15 字或含 ≥2 个「的」时，先剥命题/指称外壳取核心再判定; 仍是完整命题/小句 (含主谓、动宾) 的不抽; 30 字为硬上限。\n"
    "2. **嵌套不重复**: 中心词与扩展式在本句同指一个对象时，只保留最小充分语义单元一条; 扩展式因领域限定构成**不同的可定义/可测构念**时才另抽。区分: 加的是指称/评价/数量/所有者定语 → 同指不另抽 (「企业专利价值」仅指「专利价值」时不另抽); 加的是领域限定形成新构念 → 不同对象 (「专利保护」与「专利保护强度」各一条)。**并列结构中若两项构成不同可定义构念，按规则三.5 并列拆取; 若为同指 (如北京大学/北大)，按本规则去重。**\n"
    "3. **同句去重**: 同一对象在句中多次出现、或全称与缩写、或同义改写，只输出一条，evidence 取信息最完整的一次; normalized_name 统一为规范名。\n\n"
    "### 五、决策与灰度\n"
    "1. 顺序: 修剪 span (规则三) → 双闸门三问 (规则一) → 排除高频非实体模式 (规则二) → 类型判定。\n"
    "2. 过闸但类型不清 → 正常抽取，candidate_l1 多标、降低 confidence、uncertainty 写明疑点，**不得丢弃**。\n"
    "3. 不过闸 → 不进 entities; 来自上游必抽清单的处理见闸门 3。\n"
    "4. 不靠少抽换精确率，也不降低门槛: 过闸的命名实体 (含抽象术语、可测构念、动词性技术名) 不得因谨慎跳过; 不过闸的短语不得因召回压力放行。\n\n"
    "## Workflow\n"
    "1. 结合必抽清单，逐句识别候选短语; 必抽清单是**优先核查候选**，不是免检项。\n"
    "2. 对每个候选先做边界修剪 (规则三)，再过双闸门三问 (规则一)，排除高频非实体模式 (规则二)。\n"
    "3. 处理嵌套与同句重复 (规则四)，为保留实体判定 candidate_l3 / candidate_l1，截取 evidence。\n"
    "4. 按 OutputFormat 输出严格 JSON。\n\n"
    "## Background\n"
    "### 一、Agent — 能产生学术行为的主体\n"
    "**Person (刚性类型)**\n"
    "- `scholar`: 以知识生产为职能 (教授/研究员/博士生) | `practitioner`: 以专业服务为职能 (图书馆员/情报分析师)\n"
    "- 其他个人 (政策推动者/审稿人/科普作家等) 暂归 `other` 并在候选说明中注明职能。\n"
    "> 仅抽具体人名; 泛称 (专家/学者/研究者) 不抽。\n\n"
    "**Organization**\n"
    "- `research`: 大学/研究所/智库/实验室 | `service`: 图书馆/档案馆/信息中心/医院\n"
    "- `professional`: 学会/协会 (IFLA/ALA) | `governance`: 行政/治理/资助组织 (教育部/NSF/档案局)\n"
    "- `publishing`: 出版传播组织 (Elsevier/商务印书馆/arXiv)\n"
    "> 仅抽具体机构名; 有地址/法人/名单→Organization，否则可能是 research_field。\n\n"
    "### 二、Artifact — 人类有意识生产的制品\n"
    "**Discursive** — `journal_article`(单篇期刊论文，有卷期页)/`conference_paper`/`book`(独立ISBN)/`thesis`/`report`/`preprint`(arXiv,SSRN)/`paper`(仅限无法归入前者的具体单篇文献兜底)。期刊整体不抽；仅抽具体文献名。\n"
    "**Organizational** — `knowledge_organization_system`(分类法/叙词表/本体/受控词表)、`metadata_schema`(Dublin Core/MARC/EDM)、`information_retrieval_resource`(辞典/百科/手册/文摘/引文索引: SCI/CSSCI)。\n"
    "**Empirical** — `dataset`(结构化数据集与语料库)、`database`(数据资源); database 评数据内容，software_system 评系统功能。\n"
    "**Normative** — `standard`(ISO/GB)、`policy`(政策文件)、`guideline`(建议性指南)。\n"
    "**Functional** — `software_system`(可运行的软件/平台/系统/数字图书馆，如 SPSS/R/CNKI 平台/ChatGPT)、`algorithm`(可执行计算逻辑: PageRank/TF-IDF/KNN/各类机器学习模型)、`instrument`(测量工具/量表/问卷/指标体系)。\n"
    "> 仅抽具体可识别名称;「统计软件」「分析工具」不抽。5G/大数据/云计算/VR/AR 等技术体系不标 algorithm/method (见技术临时规则)。\n\n"
    "### 三、Abstract — 无物质载体的智识构造物\n"
    "**Conceptual** — `concept`(命名术语: Persona/信息素养/知识鸿沟)、`definition`(特定界定)、`typology`(分类方案)。concept 不是兜底: 只有原句讨论术语名本身、且不能归入下列各类时才用。\n"
    "**KnowledgeClaim** — `theoretical_construct`(由概念/命题/变量及其关系组成的结构化解释或分析构造，含理论、概念模型、分析框架，如学习迁移理论/TAM/FRBR)、`hypothesis`(单一待检验命题)。\n"
    "**Methodological** (methodology→method) — `methodology`(实证主义/解释主义)、`method`(研究各环节的可操作程序与具体手段: 文献计量法/内容分析法/德尔菲法/共词分析/聚类分析/LDA)。**排除**: 管理制度、评价程序、考核流程、编辑流程（如\"综合考核\"\"分类管理\"\"盲审专家的确定\"）不是研究方法，不抽或按语境归入 policy / policy_initiative / concept。\n"
    "**Epistemic** — `research_orientation`(共享的范式/立场/视角/取向，如第四范式)、`research_field`(建制化学科、学科分支与跨学科领域: 图书情报学/科学计量学/信息检索/知识管理)、`school_of_thought`(有名称的学术流派，如芝加哥学派/学衡派)。**排除**: research_field 仅限学术问题域，机构类型、院校层级、行业领域（如\"985院校\"\"新建本科院校\"\"高校学报\"）不抽；school_of_thought 仅限有名称的学派，\"学术共同体\"\"研究传统\"\"学术圈\"等泛指群体不抽或归入 professional。\n"
    "**Phenomenon** — `phenomenon`(可观察的状态、行为模式或变化表现，含带时间方向的增长/衰退/扩散/转变: 数据孤岛/信息茧房); 泛化的「发展趋势」不抽。\n"
    "> Abstract 内部优先级: 研究操作/手段→method; 结构化解释构造→theoretical_construct; 待检验命题→hypothesis; 学科领域→research_field; 立场范式→research_orientation; 学派→school_of_thought; 可观察现实及其变化→phenomenon; 运动倡议→Event; 具体系统/工具/文献→Artifact; 均不符合才用 concept。\n\n"
    "**技术类临时规则** (本体无 Technology): 5G/人工智能/大数据/云计算/VR/AR 作技术体系或应用概念→concept，作具体可运行系统/软件→software_system; 设备种类无具体名称不抽; 职能/业务理念 (图书馆自动化)→concept。\n\n"
    "### 四、Event — 在时间中展开\n"
    "- `intellectual_turn`(范式转向) | `debate`(有双方与议题的学术争论) | `movement`(自下而上的运动，如开放获取)\n"
    "- `research_program`(长期有组织研究，如 CADAL) | `policy_initiative`(自上而下的制度行动; 区别于作为文件的 policy)\n"
    "- `stage`(模糊起止的发展阶段) | `conference_meeting`(具体会议; 区别于 conference_paper 制品)\n\n"
    "## OutputFormat\n"
    "严格 JSON，不含 markdown 代码块。无实体时只输出 {{\"e\":[]}}。\n"
    "根对象: {{\"e\": [ ... ], \"n\": {{ ... }} }}\n"
    "- e: 每条实体为**位置固定的紧凑数组** [mention, candidate_l3, flags?]，不要写字段名:\n"
    "  1. mention: 原文连续短语，最小充分语义单元，严禁概括/改名\n"
    "  2. candidate_l3: 从 Background 选择 type_code\n"
    "  3. flags: 可选标志串。? = 类型/边界不确定；! = 上游强制保留但非具体实体；!? = 两者\n"
    "- n: 可选。仅当 normalized_name 与 mention 不同时，以实体下标为键输出规范名，"
    "如 {{\"0\":\"卷积神经网络/CNN\"}}\n"
    "- 不要输出 candidate_l1 / evidence / confidence / normalized_name / is_specific_entity 字段，"
    "这些由下游推导\n"
    "- e 数组长度必须等于本句抽出的实体数，顺序即实体 ID 顺序\n\n"
    "## Examples\n"
    "### 例1 (具体人名; 泛称不抽)\n"
    "输入: 谱学研究领域颇有建树的学者有潘光旦、罗香林等人。\n"
    '输出: {{"e":[["潘光旦","scholar"],["罗香林","scholar"]]}}\n'
    "> 「学者」泛称、「等人」引用词不另抽。\n\n"
    "### 例2 (文献专名; 剥版本修饰)\n"
    "输入: 新版《图书馆学概论》反映了网络时代图书馆学研究的最新成果。\n"
    '输出: {{"e":[["《图书馆学概论》","book"]]}}\n\n'
    "### 例3 (理论; 只抽具名稳定方法，类别词不抽)\n"
    "输入: Ausubel 基于学习迁移理论; 该文使用文献计量法、内容分析法与德尔菲法。\n"
    '输出: {{"e":[["学习迁移理论","theoretical_construct"],["文献计量法","method"],["内容分析法","method"],["德尔菲法","method"]]}}\n'
    "> 方法只抽具名、稳定的方法 (文献计量法/内容分析法/德尔菲法);「研究方法」「分析方法」是类别词不抽;「该文」指代不抽。\n\n"
    "### 例4 (抽象构念有效; 命题框架与关系形态不抽)\n"
    "输入: 研究表明专利授权速度与专利保护范围密切相关。\n"
    '输出: {{"e":[["专利授权速度","concept"],["专利保护范围","concept"]]}}\n'
    "> 「…与…密切相关」是命题框架不抽;「U型关系/正相关」等关系形态描述也不抽。两个构念虽抽象但可定义可测，必须抽。\n\n"
    "### 例5 (引用整体保留; 构念与类别词)\n"
    "输入: 薛明皋（2013）及唐恒等(2014)讨论专利价值的分析指标体系。\n"
    '输出: {{"e":[["薛明皋（2013）","scholar"],["唐恒等(2014)","scholar"],["专利价值","concept"]],"n":{{"0":"薛明皋","1":"唐恒"}}}}\n'
    "> 年份/「等」不剥; 并列作者拆条; 「分析指标体系」是类别性宾语不抽。\n\n"
    "### 例6 (括号中英对照保留; 全称与缩写合并一条)\n"
    "输入: 模型采用卷积神经网络（convolutional neural network,CNN)，CNN 在分类任务上表现稳定。\n"
    '输出: {{"e":[["卷积神经网络（convolutional neural network,CNN)","algorithm"]],"n":{{"0":"卷积神经网络/CNN"}}}}\n\n'
    "### 例7 (抽象概念与研究领域)\n"
    "输入: 「信息素养」这一概念强调个体识别与评价信息的能力; 信息检索在图书情报学中有稳定的方法传统。\n"
    '输出: {{"e":[["信息素养","concept"],["信息检索","research_field"],["图书情报学","research_field"]]}}\n'
    "> 「能力」是零内涵外壳不抽。\n\n"
    "### 例8 (事件; 动词性运动名)\n"
    "输入: 情报学中对 Information 与 Intelligence 的争论是有益的; 开放获取让读者免费获取论文。\n"
    '输出: {{"e":[["Information与Intelligence的争论","debate"],["开放获取","movement","?"]]}}\n'
    "> 「开放获取」兼为 concept, 此处取运动义; 标注 ? 表示类型有歧义。\n\n"
    "### 例9 (枚举普通成员不抽; 只抽具名数据集)\n"
    "输入: 本地化事件如音乐节、体育比赛是智慧城市建设中态势感知监测的重点，研究使用 Sentiment140 数据集训练事件过滤器。\n"
    '输出: {{"e":[["智慧城市建设","policy_initiative"],["Sentiment140 数据集","dataset"]],"n":{{"1":"Sentiment140"}}}}\n'
    "> 「音乐节」「体育比赛」是枚举中的普通事件成员不抽;「本地化事件」裸类别不抽; 只有具名数据集 (Sentiment140) 才抽，「地理标记的推文」这类数据类别描述不抽。\n\n"
    "### 例10 (负例: 指代/泛称/外壳/动作)\n"
    "输入: 本文通过比较两种方法的优劣，综述了知识共享动机的作用，现有研究认为该方法能运用机器学习提升信息素养。\n"
    '输出: {{"e":[["知识共享动机","concept"],["机器学习","method"],["信息素养","concept"]]}}\n'
    "> 「本文」「该方法」「现有研究」指代，「比较/优劣」虚义，「…的作用」「两种方法」类别外壳，「运用/提升」谓语框架——均不单独抽取。\n\n"
    "### 例11 (形似学术实体、实为描述 — 全部不抽)\n"
    "输入: 本文探讨数据的有效利用与处理措施，梳理知识内容对竞争优势的作用，总结研究方法、分析方法与研究成果，展望发展趋势，指出二者呈 U 型关系。\n"
    '输出: {{"e":[]}}\n'
    "> 逐条说明:「本文」指代;「数据的有效利用」动作性描述;「处理措施」「知识内容」「研究成果」裸泛称;「竞争优势」在本句只是普通名词 (若另句作为被定义/测量的研究构念，如变量「企业竞争优势」，则抽);「…的作用」命题外壳;「研究方法」「分析方法」类别词;「发展趋势」泛化趋势;「U型关系」关系形态描述。\n\n"
    "### 例12 (看似抽象、实为真正学术实体 — 全部要抽)\n"
    "输入: 专利价值与专利保护强度是常用的专利计量变量;知识共享动机、信息素养是被广泛讨论的概念;文献计量学与信息检索是图情领域的成熟方向。\n"
    '输出: {{"e":[["专利价值","concept"],["专利保护强度","concept"],["知识共享动机","concept"],["信息素养","concept"],["文献计量学","research_field"],["信息检索","research_field"]]}}\n'
    "> 与例11 对照: 可测构念 (专利价值/专利保护强度)、命名概念 (知识共享动机/信息素养)、研究领域 (文献计量学/信息检索) 在论元位置被论及时照抽——抽象不等于无效。\n\n"
    "### 例13 (灰区后缀: 具名对象抽，占位泛称不抽)\n"
    "输入: 现有研究多采用这类方法; 本文使用文献计量法与 LDA 主题模型，该模型在相关技术中表现稳定。\n"
    '输出: {{"e":[["文献计量法","method"],["LDA 主题模型","algorithm"]],"n":{{"1":"LDA主题模型"}}}}\n'
    "> 替换测试不通过者均不抽:「现有研究」「这类方法」「该模型」「相关技术」换成任何同类词句子都成立;「文献计量法」「LDA主题模型」换掉则客体改变。\n\n"
    "### 例14 (关系与描述成分不抽，只抽两端实体)\n"
    "输入: 专利保护强度正向影响企业专利价值，二者呈倒 U 型关系; 专利质量决定专利保护的效果。\n"
    '输出: {{"e":[["专利保护强度","concept"],["专利价值","concept"],["专利质量","concept"],["专利保护","concept"]]}}\n'
    "> 「正向影响」「倒 U 型关系」是关系不抽;「…的效果」是结果外壳不抽;「企业」是所有者定语，mention 取「专利价值」。\n\n"
    "### 例15 (负例: 期刊本体)\n"
    "输入: 《经济研究》的影响因子位居前列。\n"
    '输出: {{"e":[]}}\n'
    "> 《经济研究》是期刊本体，不是某篇论文，journal_article 不成立；无更贴切类型则不抽。\n\n"
    "### 例16 (负例: 管理实践 ≠ 研究方法)\n"
    "输入: 每3年进行一次综合考核，实行动态的评价方式。\n"
    '输出: {{"e":[]}}\n'
    "> \"综合考核\"\"动态的评价方式\"是管理/评价实践，不是学术研究方法。\n\n"
    "### 例17 (负例: 指代与泛称)\n"
    "输入: 该期刊的影响因子较高；评价体系需要完善。\n"
    '输出: {{"e":[]}}\n'
    "> \"该期刊\"指代表达不抽；\"评价体系\"是泛称类别词，替换测试不通过（换成\"评价方法\"句意不变）。\n\n"
    "## 必抽实体（上游 Agent 1 评价关系主客体原文短语）\n"
    "以下为**优先核查候选** (非免检项): 过双闸门 → 正常抽取 (is_specific_entity=true); 不过闸 → 按闸门 3 处理。\n"
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
    接收上游 Agent 1 评价关系的主客体原文短语作为必抽提示，
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
        return self._parse_output(data, statement, sentence_id)

    @staticmethod
    def _parse_output(
        data: object,
        statement: str,
        sentence_id: str = "",
    ) -> SentenceExtractionOutput:
        """解析 Agent 2 输出。

        兼容两种格式：
        - 紧凑格式: {"e": [[mention, l3, flags?], ...], "n": {"0": "规范名"}}
        - 旧格式:   {"entities": [{mention, normalized_name, candidate_l3, ...}]}
        """
        name_map: dict = {}
        if isinstance(data, list):
            raw_entities = data
        elif isinstance(data, dict):
            raw_entities = data.get("e")
            if raw_entities is None:
                raw_entities = data.get("entities", [])
            name_map = data.get("n") or {}
        else:
            raw_entities = []
        if not isinstance(raw_entities, list):
            raw_entities = []
        if not isinstance(name_map, dict):
            name_map = {}

        entities: list[ExtractedEntity] = []
        for i, item in enumerate(raw_entities):
            if isinstance(item, dict):
                # 兼容旧格式
                mention = item.get("mention") or item.get("entity_name") or ""
                if not mention:
                    continue
                normalized_name = item.get("normalized_name") or mention
                candidate_l3 = item.get("candidate_l3") or ""
                candidate_l1 = item.get("candidate_l1") or []
                evidence = item.get("evidence") or ""
                is_specific = item.get("is_specific_entity", True)
                confidence = item.get("confidence", 0.0)
                uncertainty = item.get("uncertainty", "")
            elif isinstance(item, (list, tuple)):
                # 紧凑格式: [mention, candidate_l3, flags?]
                vals = list(item)
                mention = str(vals[0]).strip() if len(vals) > 0 and vals[0] else ""
                if not mention:
                    continue
                candidate_l3 = str(vals[1]).strip() if len(vals) > 1 and vals[1] else ""
                uncertain, forced = parse_flags(vals[2] if len(vals) > 2 else "")
                normalized_name = str(name_map.get(str(i)) or mention).strip()
                candidate_l1 = []
                evidence = ""
                is_specific = not forced
                confidence = 0.6 if uncertain else 0.9
                if forced:
                    uncertainty = "上游强制保留"
                elif uncertain:
                    uncertainty = "类型/边界不确定"
                else:
                    uncertainty = ""
            else:
                continue

            # 本地派生 candidate_l1
            if not candidate_l1 and candidate_l3:
                l1_code, _ = get_l1l2_from_l3(candidate_l3)
                if l1_code:
                    candidate_l1 = [l1_code]
            # 本地回填 evidence
            if not evidence:
                evidence = derive_evidence(statement, mention) or mention

            entities.append(
                ExtractedEntity(
                    entity_id=f"{sentence_id}_e{i + 1}",
                    mention=mention,
                    normalized_name=normalized_name,
                    candidate_l3=candidate_l3,
                    candidate_l1=candidate_l1,
                    evidence=evidence,
                    is_specific_entity=is_specific,
                    confidence=confidence,
                    uncertainty=uncertainty,
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
