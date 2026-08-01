"""
eval — Ontology_Agent 评估包

评估体系:
- gold_standard.py : 标注数据加载、验证、导出
- metrics.py      : F1/Precision/Recall 计算 + 混淆矩阵
- reporter.py     : Markdown 评测报告生成
- annotation_tool.py : 命令行交互式标注工具
"""

__version__ = "1.0.0"

from .gold_standard import GoldStandard, GoldEntity, GoldSentence
from .metrics import (
    compute_metrics,
    compute_extraction_metrics,
    compute_classification_metrics,
    compute_strict_metrics,
    compute_per_type_metrics,
    build_confusion_matrix,
    normalize_pipeline_output,
    EvalResult,
)
from .reporter import generate_report, print_summary

__all__ = [
    "GoldStandard", "GoldEntity", "GoldSentence",
    "compute_metrics", "compute_extraction_metrics", "compute_classification_metrics",
    "compute_strict_metrics", "compute_per_type_metrics", "build_confusion_matrix",
    "normalize_pipeline_output",
    "EvalResult",
    "generate_report", "print_summary",
]
