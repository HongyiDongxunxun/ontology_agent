"""
eval — Ontology_Agent 评估包 (V1.1: + 评价关系评估)

评估体系:
- gold_standard.py : 标注数据加载、验证、导出 (含评价关系标注)
- metrics.py      : F1/Precision/Recall 计算 + 混淆矩阵 + 关系级指标
- reporter.py     : Markdown 评测报告生成
- annotation_tool.py : 命令行交互式标注工具
"""

__version__ = "1.1.0"

from .gold_standard import GoldStandard, GoldEntity, GoldSentence, GoldRelation
from .metrics import (
    compute_metrics,
    compute_extraction_metrics,
    compute_classification_metrics,
    compute_strict_metrics,
    compute_per_type_metrics,
    build_confusion_matrix,
    normalize_pipeline_output,
    normalize_relation_output,
    compute_relation_metrics,
    compute_has_eval_accuracy,
    EvalResult,
)
from .reporter import generate_report, print_summary

__all__ = [
    "GoldStandard", "GoldEntity", "GoldSentence", "GoldRelation",
    "compute_metrics", "compute_extraction_metrics", "compute_classification_metrics",
    "compute_strict_metrics", "compute_per_type_metrics", "build_confusion_matrix",
    "normalize_pipeline_output", "normalize_relation_output",
    "compute_relation_metrics", "compute_has_eval_accuracy",
    "EvalResult",
    "generate_report", "print_summary",
]
