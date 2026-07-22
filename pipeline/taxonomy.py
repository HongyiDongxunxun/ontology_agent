"""
pipeline.taxonomy — 评论索引实体分类体系 (opencode版 L1/L2/L3 层级 + 中文标签)
对齐: 实体类型分类体系_opencode版.md
"""

from __future__ import annotations

# ===========================================================================
# L1 → L2 → L3 层级映射表
# ===========================================================================

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

# ===========================================================================
# 反向索引: L3 type_code → (L1, L2)
# ===========================================================================

_L3_TO_L1L2: dict[str, tuple[str, str]] = {}
for _l1, _l2_map in TAXONOMY_HIERARCHY.items():
    for _l2, _l3_list in _l2_map.items():
        for _l3 in _l3_list:
            _L3_TO_L1L2[_l3] = (_l1, _l2)


def get_l1_from_l3(l3_code: str) -> str:
    return _L3_TO_L1L2.get(l3_code, ("", ""))[0]


def get_l2_from_l3(l3_code: str) -> str:
    return _L3_TO_L1L2.get(l3_code, ("", ""))[1]


def get_l1l2_from_l3(l3_code: str) -> tuple[str, str]:
    return _L3_TO_L1L2.get(l3_code, ("", ""))


def get_l1_options() -> list[str]:
    return list(TAXONOMY_HIERARCHY.keys())


def get_l2_options(l1: str) -> list[str]:
    return list(TAXONOMY_HIERARCHY.get(l1, {}).keys())


def get_l3_options(l1: str, l2: str) -> list[str]:
    return TAXONOMY_HIERARCHY.get(l1, {}).get(l2, [])


def get_l2_label(l2_code: str) -> str:
    return L2_LABELS.get(l2_code, l2_code)


def get_l3_label(l3_code: str) -> str:
    return L3_LABELS.get(l3_code, l3_code)


def build_taxonomy_text() -> str:
    """构建分类体系文本描述，供 Agent 2/3 Prompt 使用"""
    lines: list[str] = []
    for l1, l2_map in TAXONOMY_HIERARCHY.items():
        lines.append(f"**L1 = {l1}**")
        for l2, l3_list in l2_map.items():
            l2_label = L2_LABELS.get(l2, l2)
            l3_details = ", ".join(
                f"{code}({L3_LABELS.get(code, code)})" for code in l3_list
            )
            lines.append(f"  L2 = {l2} ({l2_label}) → L3: {l3_details}")
    return "\n".join(lines)
