"""
pipeline.taxonomy_migration — V3 简化版实体类型归并表与历史数据迁移工具

对齐: doc/实体类型分类体系（简化版）.docx
用途:
  1. 记录旧版(opencode版) → V3 简化版的类型归并关系；
  2. 为历史结果 (mid_data/、output/、dynamic_terms.json) 提供旧码迁移函数。

归并概览:
  - 保持不变 38 个 type_code
  - 合并消失 15 个旧码 → 7 个新码 (theory/model/framework→theoretical_construct 等)
  - 新增 5 个 type_code: information_retrieval_resource / software_system /
    theoretical_construct / research_orientation / research_field
  - policy_advocate / reviewer → other (Person 分组 Other，须填 other_suggestion)
"""

from __future__ import annotations

# ===========================================================================
# 归并明细表
# scope: merged(多码并入) | renamed(等价更名) | other(兜底) | split(按语境拆分)
# ===========================================================================

MERGE_TABLE: list[dict] = [
    # ── Agent > Person ──
    {"old": "policy_advocate", "old_label": "政策倡导者",
     "new": "other", "new_label": "其他(兜底)",
     "scope": "other",
     "note": "V3 Person 仅保留 scholar/practitioner；政策倡导者等走 Person 组 Other，须在 other_suggestion 注明职能"},
    {"old": "reviewer", "old_label": "评审者",
     "new": "other", "new_label": "其他(兜底)",
     "scope": "other",
     "note": "同上；审稿人作为其他个人，须在 other_suggestion 注明"},
    # ── Artifact > Organizational ──
    {"old": "reference_tool", "old_label": "参考工具书",
     "new": "information_retrieval_resource", "new_label": "信息检索资源",
     "scope": "merged",
     "note": "辞典/百科/手册/文摘等统一为「信息检索资源」"},
    {"old": "index", "old_label": "索引/引文索引",
     "new": "information_retrieval_resource", "new_label": "信息检索资源",
     "scope": "merged",
     "note": "SCI/SSCI/CSSCI 等引文索引并入信息检索资源"},
    # ── Artifact > Empirical ──
    {"old": "corpus", "old_label": "语料库",
     "new": "dataset", "new_label": "数据集",
     "scope": "merged",
     "note": "V3 dataset 定义明确包含原 corpus 所指语料库，不以表格结构为区分条件"},
    # ── Artifact > Functional ──
    {"old": "information_system", "old_label": "信息系统",
     "new": "software_system", "new_label": "软件系统",
     "scope": "merged",
     "note": "平台/系统/数字图书馆等可运行应用统一为软件系统；评价数据内容时仍用 database"},
    {"old": "software", "old_label": "软件/程序",
     "new": "software_system", "new_label": "软件系统",
     "scope": "merged",
     "note": "独立软件(SPSS/R/ChatGPT)与平台合并为软件系统"},
    # ── Abstract > KnowledgeClaim ──
    {"old": "theory", "old_label": "理论",
     "new": "theoretical_construct", "new_label": "理论构造",
     "scope": "merged",
     "note": "理论/概念模型/分析框架合并为「理论构造」(含命题、要素与关系的结构化构造)"},
    {"old": "framework", "old_label": "分析框架",
     "new": "theoretical_construct", "new_label": "理论构造",
     "scope": "merged",
     "note": "同上；不要求具有因果主张或图示"},
    {"old": "model", "old_label": "模型",
     "new": "theoretical_construct / algorithm", "new_label": "理论构造 / 算法",
     "scope": "split",
     "note": "按语境拆分: 理论/概念模型(TAM、态势感知模型、大五人格模型)→theoretical_construct；"
             "可执行的机器学习模型(GRU4REC、PageRank、马尔可夫模型)→algorithm。默认见 TERM_SPLIT_HINTS"},
    # ── Abstract > Methodological ──
    {"old": "technique", "old_label": "具体技术",
     "new": "method", "new_label": "研究方法",
     "scope": "merged",
     "note": "V3 method 定义覆盖「整体程序+某一环节的操作方式」，共词分析/聚类分析/LDA 等并入 method；"
             "计算机可执行算法仍归 algorithm；技术体系(5G/大数据)归 concept"},
    # ── Abstract > Epistemic ──
    {"old": "paradigm", "old_label": "研究范式",
     "new": "research_orientation", "new_label": "研究取向",
     "scope": "merged",
     "note": "研究取向涵盖共同体共享的研究范式与较具体的研究取向(含「第四范式」)"},
    {"old": "approach", "old_label": "研究取向",
     "new": "research_orientation", "new_label": "研究取向",
     "scope": "renamed",
     "note": "等价更名"},
    {"old": "discipline", "old_label": "学科",
     "new": "research_field", "new_label": "研究领域",
     "scope": "merged",
     "note": "建制化学科与学科内部分支统一为「研究领域」"},
    {"old": "subfield", "old_label": "子领域",
     "new": "research_field", "new_label": "研究领域",
     "scope": "merged",
     "note": "同上(文献计量学/信息检索/会话推荐研究等)"},
    # ── Abstract > Phenomenon ──
    {"old": "trend", "old_label": "演变趋势",
     "new": "phenomenon", "new_label": "现象",
     "scope": "merged",
     "note": "V3 phenomenon 定义明确包含带时间演变方向的增长/衰退/扩散/转变，不再单列 trend；"
             "泛化的「发展趋势」本就不抽"},
]

# ===========================================================================
# 旧码 → 默认新码 (model 默认按理论构造迁移，可执行模型见 TERM_SPLIT_HINTS)
# ===========================================================================

OLD_TO_NEW: dict[str, str] = {row["old"]: row["new"].split(" / ")[0] for row in MERGE_TABLE}

# ===========================================================================
# 拆分型旧码的词条级提示 (子串匹配，命中即覆盖默认迁移)
# 仅对 scope=split 的旧码有意义
# ===========================================================================

TERM_SPLIT_HINTS: dict[str, list[tuple[tuple[str, ...], str]]] = {
    "model": [
        # 可执行的机器学习/计算模型 → algorithm
        (("GNN", "GRU", "TAGNN", "SR-", "GCE", "GNR", "A-PGNN", "PGNN", "SGNN", "DAT-MDI",
          "GEL", "GC-SAN", "Beyond Clicks", "Graph Neural", "马尔可夫", "Word2Vec", "LDA",
          "PageRank", "KNN", "SOM", "TF-IDF", "BERT", "CNN", "RNN", "LSTM"), "algorithm"),
        # 理论/概念模型 → theoretical_construct
        (("人格", "态势感知", "TAM", "技术接受", "结构方程", "研究对象套件"), "theoretical_construct"),
    ],
    "technique": [
        # 计算机可执行的具体算法 → algorithm
        (("Word2Vec", "TF-IDF", "PageRank", "KNN", "LDA", "聚类算法"), "algorithm"),
        # 技术领域/技术体系 → concept
        (("文本挖掘技术", "数据挖掘技术", "大数据技术"), "concept"),
        # 其余研究操作手段 → method (默认)
    ],
}


def migrate_l3(old_code: str, term: str | None = None) -> str:
    """将旧 L3 type_code 迁移为 V3 码。

    Args:
        old_code: 旧类型码。
        term: 可选的实体原文/术语名，用于 model/technique 等拆分型旧码消歧。
    Returns:
        新类型码；本就是 V3 码或无法识别时原样返回。
    """
    if not old_code:
        return old_code
    # 已是新码: 直接返回
    try:
        from .taxonomy import L3_LABELS
    except ImportError:  # 直接以脚本方式运行
        from taxonomy import L3_LABELS
    if old_code in L3_LABELS:
        return old_code

    # 拆分型: 先查词条提示
    if term and old_code in TERM_SPLIT_HINTS:
        for keywords, target in TERM_SPLIT_HINTS[old_code]:
            if any(k.lower() in term.lower() for k in keywords):
                return target

    return OLD_TO_NEW.get(old_code, old_code)


def print_merge_table() -> None:
    """命令行打印归并表。"""
    scope_cn = {"merged": "合并", "renamed": "更名", "other": "归兜底", "split": "拆分"}
    print(f"{'旧码':<22}{'旧标签':<14}{'范围':<6}{'新码':<30}说明")
    print("-" * 110)
    for r in MERGE_TABLE:
        print(f"{r['old']:<22}{r['old_label']:<14}{scope_cn[r['scope']]:<6}"
              f"{r['new']:<30}{r['note']}")
    removed = sorted(OLD_TO_NEW)
    added = ["information_retrieval_resource", "software_system",
             "theoretical_construct", "research_orientation", "research_field"]
    print("-" * 110)
    print(f"消失旧码 {len(removed)} 个: {', '.join(removed)}")
    print(f"新增新码 {len(added)} 个: {', '.join(added)}")


if __name__ == "__main__":
    print_merge_table()
