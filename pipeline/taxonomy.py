"""
pipeline.taxonomy — 评论索引实体分类体系 (V3 简化版 L1/L2/L3 层级 + 中文标签)
对齐: 实体类型分类体系（简化版）.docx
旧版归并关系见 pipeline.taxonomy_migration
"""

from __future__ import annotations

# ===========================================================================
# L1 → L2 → L3 层级映射表
# 各 L2 分组下无法归入具体 L3 时，统一使用全局兜底码 "other" 并填 other_suggestion
# ===========================================================================

TAXONOMY_HIERARCHY: dict[str, dict[str, list[str]]] = {
    "Agent": {
        "Person":       ["scholar", "practitioner"],
        "Organization": ["research", "service", "professional", "governance", "publishing"],
    },
    "Artifact": {
        "Discursive":    ["journal_article", "conference_paper", "book",
                          "thesis", "report", "preprint", "paper"],
        "Organizational":["knowledge_organization_system", "metadata_schema",
                          "information_retrieval_resource"],
        "Empirical":     ["dataset", "database"],
        "Normative":     ["standard", "policy", "guideline"],
        "System":        ["software_system"],
        "Tool":          ["algorithm", "instrument"],
    },
    "Abstract": {
        "Conceptual":     ["concept", "definition", "typology"],
        "KnowledgeClaim": ["theoretical_construct", "hypothesis"],
        "Methodological": ["methodology", "method"],
        "Epistemic":      ["research_orientation", "research_field", "school_of_thought"],
        "Phenomenon":     ["phenomenon"],
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
    "System":              "系统",
    "Tool":                "工具",
    "Conceptual":          "概念层",
    "KnowledgeClaim":      "知识主张层",
    "Methodological":      "方法论层",
    "Epistemic":           "认识论共同体层",
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
    "report":                        "报告",
    "preprint":                      "预印本",
    "paper":                         "其他文献论文(兜底)",
    # ── Artifact > Organizational ──
    "knowledge_organization_system":  "知识组织系统",
    "metadata_schema":               "元数据规范",
    "information_retrieval_resource":"信息检索资源",
    # ── Artifact > Empirical ──
    "dataset":                       "数据集",
    "database":                      "数据库(数据资源)",
    # ── Artifact > Normative ──
    "standard":                      "标准",
    "policy":                        "政策文件",
    "guideline":                     "指南/规范",
    # ── Artifact > System / Tool ──
    "software_system":               "软件系统",
    "algorithm":                     "算法",
    "instrument":                    "测量工具",
    # ── Abstract > Conceptual ──
    "concept":                       "概念/术语",
    "definition":                    "定义",
    "typology":                      "分类方案",
    # ── Abstract > KnowledgeClaim ──
    "theoretical_construct":         "理论构造",
    "hypothesis":                    "假设",
    # ── Abstract > Methodological ──
    "methodology":                   "方法论",
    "method":                        "研究方法",
    # ── Abstract > Epistemic ──
    "research_orientation":          "研究取向",
    "research_field":                "研究领域",
    "school_of_thought":             "学派",
    # ── Abstract > Phenomenon ──
    "phenomenon":                    "现象",
    # ── Event ──
    "intellectual_turn":             "学术转向",
    "debate":                        "学术争论",
    "movement":                      "学术运动",
    "research_program":              "研究计划",
    "policy_initiative":             "政策举措/改革事件",
    "stage":                         "发展阶段",
    "conference_meeting":            "学术会议",
    # ── 全局兜底 (任一分组无法归入时使用，须同时填 other_suggestion) ──
    "other":                         "其他(兜底)",
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
