"""
eval.reporter — Markdown 评测报告生成

生成易读的评估报告，包含:
- 总体指标摘要
- 各层级分类准确率
- Per-type F1 详情 (排序, Top-10 最优/最劣)
- L3 混淆矩阵 (Top 混淆对)
- 改进建议
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .gold_standard import GoldStandard
from .metrics import EvalResult


def generate_report(
    eval_result: EvalResult,
    gold_standard: Optional[GoldStandard] = None,
    title: str = "Ontology_Agent 评估报告",
    output_path: Optional[str] = None,
) -> str:
    """
    生成 Markdown 格式的评测报告。

    参数:
    - eval_result: compute_metrics() 返回的评估结果
    - gold_standard: 标注数据集 (用于统计标注分布)
    - title: 报告标题
    - output_path: 输出文件路径 (可选, 不传则只返回字符串)

    返回: Markdown 报告字符串
    """
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 头部 ──
    lines.append(f"# {title}")
    lines.append(f"")
    lines.append(f"> 生成时间: {now}  |  评估句数: {eval_result.total_sentences}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # ── 1. 总体指标摘要 ──
    lines.append(f"## 1. 总体指标")
    lines.append(f"")
    lines.append(f"| 维度 | Precision | Recall | F1 | TP | FP | FN |")
    lines.append(f"|------|-----------|--------|----|----|----|----|")
    lines.append(
        f"| **实体抽取级** | {_pct(eval_result.extraction_precision)} | "
        f"{_pct(eval_result.extraction_recall)} | "
        f"**{_pct(eval_result.extraction_f1)}** | "
        f"{eval_result.extraction_tp} | {eval_result.extraction_fp} | {eval_result.extraction_fn} |"
    )
    lines.append(
        f"| **端到端严格** | {_pct(eval_result.strict_precision)} | "
        f"{_pct(eval_result.strict_recall)} | "
        f"**{_pct(eval_result.strict_f1)}** | "
        f"{eval_result.strict_tp} | {eval_result.strict_fp} | {eval_result.strict_fn} |"
    )
    lines.append(f"")

    # ── 2. 分类准确率 ──
    lines.append(f"## 2. 分类准确率 (已匹配实体)")
    lines.append(f"")
    lines.append(f"| 层级 | Accuracy | 正确数 | 总数 |")
    lines.append(f"|------|----------|--------|------|")
    lines.append(
        f"| L1 大类 | {_pct(eval_result.l1_accuracy)} | "
        f"{eval_result.l1_correct} | {eval_result.l1_total} |"
    )
    lines.append(
        f"| L2 子类 | {_pct(eval_result.l2_accuracy)} | "
        f"{eval_result.l2_correct} | {eval_result.l2_total} |"
    )
    lines.append(
        f"| L3 细类 | {_pct(eval_result.l3_accuracy)} | "
        f"{eval_result.l3_correct} | {eval_result.l3_total} |"
    )
    lines.append(
        f"| 有效/无效判定 | {_pct(eval_result.validity_accuracy)} | "
        f"{eval_result.validity_correct} | {eval_result.validity_total} |"
    )
    lines.append(f"")

    # ── 3. Per-type F1 ──
    if eval_result.per_type:
        lines.append(f"## 3. Per-type F1 详情 (抽取级)")
        lines.append(f"")
        lines.append(
            f"| L3 类型 | Precision | Recall | F1 | TP | FP | FN | 严格F1 |"
        )
        lines.append(
            f"|---------|-----------|--------|----|----|----|----|--------|"
        )

        sorted_types = sorted(
            eval_result.per_type.items(),
            key=lambda x: x[1].f1,
            reverse=True,
        )
        for type_code, tm in sorted_types:
            lines.append(
                f"| `{type_code}` | {_pct(tm.precision)} | {_pct(tm.recall)} | "
                f"**{_pct(tm.f1)}** | {tm.tp} | {tm.fp} | {tm.fn} | "
                f"{_pct(tm.strict_f1)} |"
            )
        lines.append(f"")

        # Top-10 最优 / 最劣
        valid_types = [(tc, tm) for tc, tm in sorted_types if tm.f1 > 0]
        if len(valid_types) >= 5:
            lines.append(f"### Top-5 F1 最高类型")
            lines.append(f"")
            for tc, tm in valid_types[-5:]:
                lines.append(f"- ✅ `{tc}`: F1={_pct(tm.f1)}")
            lines.append(f"")

        low_types = [(tc, tm) for tc, tm in sorted_types if tm.f1 < 0.5 and tm.tp + tm.fn > 0]
        if low_types:
            lines.append(f"### Top-{min(len(low_types), 10)} F1 最低类型 (需重点优化)")
            lines.append(f"")
            for tc, tm in low_types[:10]:
                lines.append(
                    f"- ❌ `{tc}`: F1={_pct(tm.f1)}  "
                    f"(P={_pct(tm.precision)}, R={_pct(tm.recall)}, "
                    f"TP={tm.tp}, FP={tm.fp}, FN={tm.fn})"
                )
            lines.append(f"")

        # 未出现的类型
        if gold_standard:
            gs_l3 = set(gold_standard.l3_distribution().keys())
            pred_l3 = set(eval_result.per_type.keys())
            missing = gs_l3 - pred_l3
            if missing:
                lines.append(f"### 标注中存在但未被预测的类型 ({len(missing)} 个)")
                lines.append(f"")
                for t in sorted(missing):
                    count = gold_standard.l3_distribution().get(t, 0)
                    lines.append(f"- ⚠️ `{t}`: 标注 {count} 个, 预测 0 个 (完全漏召)")
                lines.append(f"")

    # ── 4. L3 混淆矩阵 ──
    if eval_result.l3_confusion:
        lines.append(f"## 4. L3 混淆矩阵 (Top 混淆对)")
        lines.append(f"")

        # 找 top 非对角线混淆
        confusions: list[tuple[str, str, int]] = []
        all_labels = set(eval_result.l3_confusion.keys())
        for gold_label, preds in eval_result.l3_confusion.items():
            for pred_label, count in preds.items():
                if gold_label != pred_label and count > 0:
                    confusions.append((gold_label, pred_label, count))
                    all_labels.add(pred_label)

        confusions.sort(key=lambda x: -x[2])

        if confusions:
            lines.append(f"| 正确类型 | 误判为 | 次数 |")
            lines.append(f"|----------|--------|------|")
            for gold, pred, count in confusions[:20]:
                lines.append(f"| `{gold}` | `{pred}` | {count} |")
            lines.append(f"")

        if len(confusions) > 20:
            lines.append(f"> 仅显示 Top-20 混淆对, 共 {len(confusions)} 对")
            lines.append(f"")

    # ── 5. 改进建议 ──
    lines.append(f"## 5. 改进建议")
    lines.append(f"")

    suggestions = _generate_suggestions(eval_result)
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"*报告由 eval/reporter.py 自动生成*")

    report = "\n".join(lines)

    # 写入文件
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Reporter] 报告已生成 -> {path.resolve()}")

    return report


def print_summary(eval_result: EvalResult) -> None:
    """Print brief evaluation summary to terminal."""
    try:
        _print_summary_impl(eval_result)
    except UnicodeEncodeError:
        # Fallback: strip emoji for Windows GBK consoles
        _print_summary_impl(eval_result, safe=True)


def _print_summary_impl(eval_result: EvalResult, safe: bool = False) -> None:
    emoji_warn = "[WARN]" if safe else "⚠️"
    sep = "-" * 50 if safe else "─" * 50
    equals = "=" * 60

    print(f"\n{equals}")
    print(f"  Evaluation Summary")
    print(f"{equals}")
    print(f"  Sentences: {eval_result.total_sentences}  |  "
          f"Predicted: {eval_result.total_predicted}  |  Gold: {eval_result.total_gold}")
    print(f"{equals}")
    print(f"  Metric              Precision   Recall      F1")
    print(f"  {sep}")
    print(f"  Extraction           {_pct(eval_result.extraction_precision):>9}  "
          f"{_pct(eval_result.extraction_recall):>9}  "
          f"{_pct(eval_result.extraction_f1):>9}")
    print(f"  Strict (end-to-end)  {_pct(eval_result.strict_precision):>9}  "
          f"{_pct(eval_result.strict_recall):>9}  "
          f"{_pct(eval_result.strict_f1):>9}")
    print(f"  {sep}")
    print(f"  L1 Accuracy: {_pct(eval_result.l1_accuracy):>9}")
    print(f"  L2 Accuracy: {_pct(eval_result.l2_accuracy):>9}")
    print(f"  L3 Accuracy: {_pct(eval_result.l3_accuracy):>9}")
    print(f"  Validity:    {_pct(eval_result.validity_accuracy):>9}")

    if eval_result.per_type:
        low_f1 = [
            (tc, tm) for tc, tm in eval_result.per_type.items()
            if tm.f1 < 0.5 and tm.tp + tm.fn > 0
        ]
        if low_f1:
            low_f1.sort(key=lambda x: x[1].f1)
            print(f"\n  {emoji_warn}  Types with F1 < 0.5:")
            for tc, tm in low_f1[:10]:
                print(f"    {tc:30s}  F1={_pct(tm.f1)}  P={_pct(tm.precision)}  R={_pct(tm.recall)}")
    print(f"{equals}\n")


# ===========================================================================
# 内部工具
# ===========================================================================


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _generate_suggestions(eval_result: EvalResult) -> list[str]:
    """根据评估结果自动生成改进建议"""
    suggestions: list[str] = []

    # 召回率低
    if eval_result.extraction_recall < 0.70:
        suggestions.append(
            "**召回率偏低**: Agent 1 抽取遗漏较多实体。建议增加 Prompt 中的多策略发现引导 "
            "(书名号、引号、英文缩写、机构全称), 并添加'常见漏抽模式'反例。"
        )

    # 精确率低
    if eval_result.extraction_precision < 0.70:
        suggestions.append(
            "**精确率偏低**: Agent 1 抽取了过多无效实体。建议强化实体性门槛规则, "
            "并增加更多负例示范(泛称身份、通用词)。"
        )

    # L3 分类错误
    if eval_result.l3_accuracy < 0.75:
        suggestions.append(
            "**L3 分类准确率不足**: 建议引入 RAG 增强 (Phase 3.1), "
            "为容易混淆的类型对增加区分性示例, 以及考虑自一致性投票机制。"
        )

    # 有效/无效判定
    if eval_result.validity_accuracy < 0.80:
        suggestions.append(
            "**有效/无效判定误差较大**: 检查 Agent 2 的有效性验证 Prompt, "
            "特别是图情领域专业术语是否被误判为无效。"
        )

    # 个别类型
    if eval_result.per_type:
        low_types = [
            (tc, tm) for tc, tm in eval_result.per_type.items()
            if tm.f1 < 0.5 and tm.tp + tm.fn > 0
        ]
        if low_types:
            type_list = ", ".join(f"`{tc}`" for tc, _ in low_types[:5])
            suggestions.append(
                f"**关注低 F1 类型**: {type_list} 的表现显著低于平均水平。"
                f"建议在 Prompt 中为这些类型增加专门的区分特征和更多示例。"
            )

    # L1 分类
    if eval_result.l1_accuracy < 0.85:
        suggestions.append(
            "**L1 大类分类有较大改善空间**: 检查 Agent 2 对 Agent/Artifact/Abstract/Event 的判定逻辑, "
            "特别是 Event vs Artifact 的边界区分。"
        )

    if not suggestions:
        suggestions.append("所有指标均处于良好水平, 继续按计划优化即可。")

    return suggestions
