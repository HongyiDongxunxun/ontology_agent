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
    "- 操作程序/分析/测量/检索/聚类/回归/实验/方法 → method 或 technique\n"
    "- 理论/模型/框架/假设/机制/因果解释/变量关系 → theory/model/framework/hypothesis\n"
    "- 学科/领域/方向/问题域/研究传统/学派 → discipline/subfield/school_of_thought/approach/paradigm\n"
    "- 可观察社会事实/现实问题/困境/鸿沟/孤岛/茧房 → phenomenon 或 trend\n"
    "- 运动/倡议/改革/计划/项目/争论/发展阶段 → Event\n"
    "- 具体系统、数据库、工具、标准、文献、知识组织系统 → Artifact\n"
    "边界提示: 信息检索/文献计量学若作为研究领域 → subfield；引文分析法/内容分析法 → method；共词分析/聚类分析 → technique；信息过载/数据孤岛若作为现实问题 → phenomenon；开放获取若作为推广运动 → movement。\n\n"
    "### `technique` / `instrument` / `phenomenon` 严格边界:\n"
    "- `technique` 仅限研究、分析、检索、测量中的具体操作手段，如共词分析、聚类分析、TF-IDF、LDA。技术领域/技术体系不得标 technique。\n"
    "- `instrument` 仅限测量工具、问卷、量表、评价指标或采集数据的工具。计算机、机器人、VR/AR设备等普通设备不得标 instrument。\n"
    "- `algorithm` 仅限明确计算步骤/排序/分类/预测逻辑，如 PageRank、KNN、TF-IDF。5G、VR/AR、大数据、人工智能技术、云计算不得标 algorithm。\n"
    "- `phenomenon` 仅限可观察的社会/信息现实问题，如数据孤岛、信息茧房。业务职能、服务模式、技术概念不得因难分类而标 phenomenon。\n\n"
    "### 技术类实体临时规则 — 当前本体无 Technology:\n"
    "- 5G/5G技术/第五代移动通信系统: 若强调技术标准/通信制式 → Artifact.Normative.standard；若泛指技术体系或应用概念 → Abstract.Conceptual.concept。\n"
    "- 人工智能技术/大数据/云计算/VR/AR: 若作为技术领域、技术体系或应用概念 → Abstract.Conceptual.concept；若是具体系统/平台/软件 → Artifact.System.information_system 或 Artifact.Tool.software。\n"
    "- 电子计算机/机器人/智能设备: 若只是设备种类且无具体名称，优先判 invalid；若原句作为技术概念讨论 → Abstract.Conceptual.concept；只有用于测量/采集数据才可标 instrument。\n"
    "- 图书馆自动化、档案管理自动化、保存文件遗产等业务职能/理念 → Abstract.Conceptual.concept；只有强调历史时段推进才考虑 stage。\n\n"
    "## Workflow\n"
    "1. 读取每个候选实体及其原句\n"
    "2. 有效性验证 → 无效则填 invalid_reason, 跳过后续步骤\n"
    "3. 判定 L1 (Agent|Artifact|Abstract|Event)\n"
    "4. 在 L1 下选定 L2\n"
    "5. 在 L2 下选定 L3 (type_code + label)\n"
    "6. 按 OutputFormat 输出 JSON 数组\n\n"
    "## Background\n"
    "### 分类体系 (L1 → L2 → L3)\n"
    "{taxonomy_text}\n\n"
    "## OutputFormat\n"
    "严格 JSON 数组，不含 markdown 代码块。每个元素对应一个实体。\n"
    "无效实体: 仅填 entity_id, valid_entity, invalid_reason，其余字段留空。\n"
    '[\n'
    '  {{\n'
    '    "entity_id": "<原始entity_id>",\n'
    '    "valid_entity": true,\n'
    '    "invalid_reason": "",\n'
    '    "l1": "Agent",\n'
    '    "l2": "Person",\n'
    '    "l2_label": "个人",\n'
    '    "l3_type_code": "scholar",\n'
    '    "l3_label": "知识生产者",\n'
    '    "evidence": "分类证据原文片段",\n'
    '    "reason": "简短分类理由 (<=30字)",\n'
    '    "other_suggestion": ""\n'
    '  }}\n'
    ']\n\n'
    "## Examples\n"
    "### 例1 — scholar vs practitioner 区分\n"
    "实体: 罗香林 | 原句: 建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。\n"
    "分类: l1=Agent, l2=Person, l3_type_code=scholar, reason=以研究/知识生产为主要职能\n"
    "规则: scholar 产出是知识本身(论文/著作)；practitioner 产出是专业服务(咨询/管理)。同一人物以研究身份被评价→scholar。\n\n"
    "### 例2 — book vs journal_article 区分\n"
    "实体: 新版《图书馆学概论》 | 原句: 新版《图书馆学概论》反映了网络时代国内外图书馆学研究的最新成果。\n"
    "分类: l1=Artifact, l2=Discursive, l3_type_code=book, reason=独立出版物发行,有独立书名\n"
    "规则: 专著有独立 ISBN 和书名，整本出版；期刊论文有卷期页码属于某期刊。\n\n"
    "### 例3 — policy(Artifact) vs policy_initiative(Event) 区分\n"
    "实体A: HR1858 | 原句: HR1858很好地平衡了反盗与利用信息制作新数据库产品需求之间的关系。\n"
    "分类: l1=Artifact, l2=Normative, l3_type_code=policy, reason=政策文件本身,是制品\n"
    "实体B: 文献检索与利用课 | 原句: 教育部下发了在高校开设「文献检索与利用课」的红头文件,大大推动了其发展。\n"
    "分类: l1=Event, l2=Event, l3_type_code=policy_initiative, reason=政策实施过程中的制度性行动事件\n"
    "规则: policy 是政策文件本身(制品, Artifact)；policy_initiative 是政策实施事件(事件, Event)。判断语境评价的是文件内容还是实施行动。\n\n"
    "### 例4 — concept vs definition 区分\n"
    "实体: 知识流动 | 原句: Szulanski认为知识流动是在特定环境下知识从来源方到接收方的传递过程。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=definition, reason=对概念边界的特定界定方式,含有提出者\n"
    "规则: concept 是命名单元本身；definition 是对命名单元边界的特定界定(通常含提出者)。同一概念可有多个 definition。\n\n"
    "### 例5 — concept vs subfield/method/phenomenon 区分\n"
    "实体A: 信息检索 | 原句: 信息检索在图书情报学研究中形成了稳定的问题域和方法传统。\n"
    "分类: l1=Abstract, l2=Epistemic, l3_type_code=subfield, reason=作为研究子领域被评价\n"
    "实体B: 引文分析法 | 原句: 本文采用引文分析法揭示知识流动路径。\n"
    "分类: l1=Abstract, l2=Methodological, l3_type_code=method, reason=完整研究方法\n"
    "实体C: 信息过载 | 原句: 网络环境中的信息过载加重了用户筛选负担。\n"
    "分类: l1=Abstract, l2=Phenomenon, l3_type_code=phenomenon, reason=现实信息问题,非术语本身\n"
    "实体D: 信息素养 | 原句: 「信息素养」这一概念强调个体识别、获取和评价信息的能力。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=concept, reason=原句讨论概念名本身\n"
    "规则: concept 只收术语/概念名本身；若原句评价研究领域、方法或现实现象，必须归入更具体类别。\n\n"
    "### 例6 — 技术类实体边界 (无 Technology 时)\n"
    "实体A: 5G技术 | 原句: 5G技术为智慧图书馆提供高速低时延连接。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=concept, reason=技术体系概念,非算法\n"
    "实体B: 第五代移动通信系统 | 原句: 第五代移动通信系统作为通信制式标准支撑移动互联。\n"
    "分类: l1=Artifact, l2=Normative, l3_type_code=standard, reason=通信制式/技术标准\n"
    "实体C: VR/AR | 原句: VR/AR被用于构建沉浸式阅读场景。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=concept, reason=应用技术概念,非研究技术手段\n"
    "实体D: 电子计算机 | 原句: 电子计算机推动了图书馆自动化的发展。\n"
    "分类: l1=Abstract, l2=Conceptual, l3_type_code=concept, reason=技术概念,非测量工具\n"
    "规则: 技术领域/技术体系不标 technique；普通设备不标 instrument；5G 不标 algorithm。\n\n"
    "### 例7 — 时间/地名过滤\n"
    "实体A: 2005年 | 原句: 2005年档案信息化建设进一步推进。\n"
    "分类: valid_entity=false, invalid_reason=单纯时间表达,非发展阶段实体\n"
    "实体B: 深圳 | 原句: 深圳和青岛的数字图书馆建设进展较快。\n"
    "分类: valid_entity=false, invalid_reason=单纯地名,非具体治理机构\n"
    "规则: 年份不是 stage；城市名不是 governance，除非原句明确指城市政府/行政主体。\n\n"
    "### 例8 — Organization 子类型选择 (research vs service vs governance)\n"
    "实体A: 哥伦比亚大学 | 分类: l3_type_code=research, reason=以知识生产为主要职能的大学\n"
    "实体B: 沈阳师范大学图书馆 | 分类: l3_type_code=service, reason=以专业服务为主要职能的图书馆\n"
    "实体C: 广州市档案局 | 分类: l3_type_code=governance, reason=有行政/治理职能的政府机构\n"
    "规则: 以核心职能判断。大学/研究院→research; 图书馆/档案馆→service; 学会/协会→professional; 政府部门→governance; 出版社→publishing。\n\n"
    "### 例9 — conference_paper(Artifact) vs conference_meeting(Event) 区分\n"
    "实体: 第二届全国灰色文献年会 | 原句: 本次会议在2018年第一次全国灰色文献年会的基础上进行了深入探讨。\n"
    "分类: l1=Event, l2=Event, l3_type_code=conference_meeting, reason=具体召开的学术会议事件\n"
    "规则: conference_paper 是会议论文(制品, Artifact)；conference_meeting 是会议召开本身(事件, Event)。\n\n"
    "## Few-Shot 参考 (来自动态术语库的高置信度实体类型)\n"
    "以下是从已确认术语库中检索到的与当前实体相似的参考分类，仅供参考:\n"
    "{few_shot_hints}\n\n"
    "## Input\n"
    "{entities_text}\n\n"
    "# 原始评价句\n"
    "{sentence}"
)

TAXONOMY_HIERARCHY: dict[str, dict[str, list[str]]] = {
    "Agent": {
        "Person":       ["scholar", "practitioner", "policy_advocate", "reviewer"],
        "Organization": ["research", "service", "professional", "governance", "publishing"],
    },
    "Artifact": {
        "Discursive":    ["journal_article", "conference_paper", "book",
                          "thesis", "report", "preprint", "paper"],
        "Organizational":["knowledge_organization_system", "metadata_schema",
                          "reference_tool", "index"],
        "Empirical":     ["dataset", "corpus", "database"],
        "Normative":     ["standard", "policy", "guideline"],
        "System":        ["information_system"],
        "Tool":          ["software", "algorithm", "instrument"],
    },
    "Abstract": {
        "Conceptual":     ["concept", "definition", "typology"],
        "KnowledgeClaim": ["theory", "model", "framework", "hypothesis"],
        "Methodological": ["methodology", "method", "technique"],
        "Epistemic":      ["paradigm", "approach", "discipline", "subfield", "school_of_thought"],
        "Phenomenon":     ["phenomenon", "trend"],
    },
    "Event": {
        "Event": ["intellectual_turn", "debate", "movement",
                   "research_program", "policy_initiative",
                   "stage", "conference_meeting"],
    },
}

# ===========================================================================
# L2 中文标签
# ===========================================================================

L2_LABELS: dict[str, str] = {
    "Person":              "个人",
    "Organization":        "组织机构",
    "Discursive":          "话语性著作",
    "Organizational":      "组织性知识工具",
    "Empirical":           "经验性制品",
    "Normative":           "规范性制品",
    "System":              "信息系统",
    "Tool":                "软件/工具",
    "Conceptual":          "概念层",
    "KnowledgeClaim":      "知识主张",
    "Methodological":      "方法论",
    "Epistemic":           "认识论共同体",
    "Phenomenon":          "现象层",
    "Event":               "事件/过程",
}

# ===========================================================================
# L3 中文标签
# ===========================================================================

L3_LABELS: dict[str, str] = {
    # ── Agent > Person ──
    "scholar":                       "知识生产者",
    "practitioner":                  "专业从业者",
    "policy_advocate":               "政策倡导者",
    "reviewer":                      "评审者",
    # ── Agent > Organization ──
    "research":                      "知识生产组织",
    "service":                       "专业服务组织",
    "professional":                  "职业共同体组织",
    "governance":                    "行政/治理/资助组织",
    "publishing":                    "出版传播组织",
    # ── Artifact > Discursive ──
    "journal_article":               "期刊论文",
    "conference_paper":              "会议论文",
    "book":                          "专著",
    "thesis":                        "学位论文",
    "report":                        "研究报告",
    "preprint":                      "预印本",
    "paper":                         "论文(兜底)",
    # ── Artifact > Organizational ──
    "knowledge_organization_system":  "知识组织系统",
    "metadata_schema":               "元数据规范",
    "reference_tool":                "参考工具书",
    "index":                         "索引/引文索引",
    # ── Artifact > Empirical ──
    "dataset":                       "数据集",
    "corpus":                        "语料库",
    "database":                      "数据库",
    # ── Artifact > Normative ──
    "standard":                      "技术标准",
    "policy":                        "政策文件",
    "guideline":                     "操作指南",
    # ── Artifact > System ──
    "information_system":            "信息系统",
    # ── Artifact > Tool ──
    "software":                      "软件/程序",
    "algorithm":                     "算法/模型",
    "instrument":                    "测量工具/量表",
    # ── Abstract > Conceptual ──
    "concept":                       "学术概念/术语",
    "definition":                    "概念定义",
    "typology":                      "分类方案",
    # ── Abstract > KnowledgeClaim ──
    "theory":                        "理论",
    "model":                         "模型",
    "framework":                     "分析框架",
    "hypothesis":                    "研究假设",
    # ── Abstract > Methodological ──
    "methodology":                   "方法论立场",
    "method":                        "研究方法",
    "technique":                     "具体技术",
    # ── Abstract > Epistemic ──
    "paradigm":                      "研究范式",
    "approach":                      "研究取向",
    "discipline":                    "学科",
    "subfield":                      "子领域",
    "school_of_thought":             "学派",
    # ── Abstract > Phenomenon ──
    "phenomenon":                    "学术现象",
    "trend":                         "演变趋势",
    # ── Event ──
    "intellectual_turn":             "学术转向",
    "debate":                        "学术争论",
    "movement":                      "学术运动",
    "research_program":              "研究计划",
    "policy_initiative":             "政策举措/改革事件",
    "stage":                         "发展阶段",
    "conference_meeting":            "学术会议",
    # ── 兜底 ──
    "other":                         "其他",
}