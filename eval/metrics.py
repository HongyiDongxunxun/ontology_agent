"""
eval.metrics — F1/Precision/Recall 计算 + 混淆矩阵

支持四个评估维度:
1. 实体抽取级: 基于 mention 匹配的 P/R/F1
2. 分类级: L1/L2/L3 逐层 Accuracy (仅对正确抽取的实体)
3. 端到端严格匹配: mention+L1+L2+L3 全对的 P/R/F1
4. 评价关系抽取级 (V5.0): subject/object/opinion 匹配的 P/R/F1 + 句级 has_evaluation 准确性

以及 per-type 细分和混淆矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .gold_standard import GoldStandard, GoldEntity


def _normalize_relation_name(name: str) -> str:
    """规范化实体名用于关系匹配 (与 pipeline 侧保持一致)"""
    name = (name or "").strip()
    for a, b in (("（", "("), ("）", ")"), ("《", ""), ("》", "")):
        name = name.replace(a, b)
    return name


# ===========================================================================
# 数据模型
# ===========================================================================


@dataclass
class EvalResult:
    """单次评估的完整结果"""

    # ── 实体抽取级 ──
    extraction_precision: float = 0.0
    extraction_recall: float = 0.0
    extraction_f1: float = 0.0
    extraction_tp: int = 0
    extraction_fp: int = 0
    extraction_fn: int = 0

    # ── L1 分类 ──
    l1_accuracy: float = 0.0
    l1_correct: int = 0
    l1_total: int = 0

    # ── L2 分类 ──
    l2_accuracy: float = 0.0
    l2_correct: int = 0
    l2_total: int = 0

    # ── L3 分类 ──
    l3_accuracy: float = 0.0
    l3_correct: int = 0
    l3_total: int = 0

    # ── 端到端严格匹配 ──
    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0
    strict_tp: int = 0
    strict_fp: int = 0
    strict_fn: int = 0

    # ── Per-type 细分 ──
    per_type: dict[str, "TypeMetrics"] = field(default_factory=dict)

    # ── 混淆矩阵 (L3) ──
    l3_confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    # ── 有效/无效判定 ──
    validity_accuracy: float = 0.0
    validity_correct: int = 0
    validity_total: int = 0

    # ── 评价关系抽取级 (V5.0) ──
    relation_precision: float = 0.0
    relation_recall: float = 0.0
    relation_f1: float = 0.0
    relation_tp: int = 0
    relation_fp: int = 0
    relation_fn: int = 0
    relation_total_gold: int = 0

    # ── 句级 has_evaluation 准确性 (V5.0) ──
    has_eval_accuracy: float = 0.0
    has_eval_correct: int = 0
    has_eval_total: int = 0

    # ── 总体统计 ──
    total_predicted: int = 0    # 管道输出的实体总数
    total_gold: int = 0         # 标注的实体总数
    total_sentences: int = 0    # 评估的句子数


@dataclass
class TypeMetrics:
    """单个 L3 类型的指标"""
    type_code: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    # 严格匹配版本
    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0
    strict_tp: int = 0


# ===========================================================================
# Pipeline 输出标准化
# ===========================================================================


def normalize_pipeline_output(results: list[dict]) -> dict[str, list[dict]]:
    """
    将管道输出的 FinalEntityResult[] 按 sentence_id 分组。

    输入: [{"sentence_id": ..., "entity": ..., "l1": ..., ...}, ...]
    输出: {"1": [entity_dict, ...], "2": [...]}
    """
    grouped: dict[str, list[dict]] = {}
    for r in results:
        sid = r.get("sentence_id", "")
        if sid:
            grouped.setdefault(sid, []).append(r)
    return grouped


# ===========================================================================
# 实体匹配逻辑
# ===========================================================================


def _match_entity(
    pred: dict, gold: GoldEntity, match_mode: str = "exact"
) -> bool:
    """
    判断预测实体与标注实体是否匹配。

    match_mode:
    - "exact": mention 精确匹配 (大小写不敏感)
    - "normalized": normalized_name 匹配
    - "fuzzy": 模糊子串匹配 (高召回, 用于抽取评估)
    """
    pred_mention = (pred.get("entity") or pred.get("mention") or "").lower().strip()
    gold_mention = gold.mention.lower().strip()

    if match_mode == "exact":
        return pred_mention == gold_mention

    elif match_mode == "normalized":
        pred_norm = (pred.get("normalized_name") or pred_mention).lower().strip()
        gold_norm = (gold.normalized_name or gold.mention).lower().strip()
        return pred_norm == gold_norm

    elif match_mode == "fuzzy":
        # 任意方向子串包含 ≥50%
        if pred_mention == gold_mention:
            return True
        if pred_mention in gold_mention or gold_mention in pred_mention:
            return True
        # 字符级别 Jaccard 快速近似
        p_set = set(pred_mention)
        g_set = set(gold_mention)
        if not p_set or not g_set:
            return False
        intersection = p_set & g_set
        union = p_set | g_set
        return len(intersection) / len(union) >= 0.5

    return False


# ===========================================================================
# 核心指标计算
# ===========================================================================


def compute_extraction_metrics(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    match_mode: str = "fuzzy",
) -> tuple[float, float, float, int, int, int]:
    """
    实体抽取级 P/R/F1 (仅关心 实体是否被抽出, 不关心分类)。

    返回: (precision, recall, f1, tp, fp, fn)
    """
    tp, fp, fn = 0, 0, 0

    for sid, pred_entities in predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            # 无标注的句子 —— 所有预测实体都算 FP
            # (保守起见, 跳过未标注句子的评估)
            continue

        gold_valid = [e for e in gold_sentence.gold_entities if e.valid_entity]
        matched_gold: set[int] = set()  # 已匹配的标注实体索引

        for pred in pred_entities:
            if not pred.get("valid_entity", True):
                continue  # 管道判定为无效的不参与抽取评估

            # 查找匹配的 gold entity
            found = False
            for gi, gold_entity in enumerate(gold_valid):
                if gi in matched_gold:
                    continue
                if _match_entity(pred, gold_entity, match_mode=match_mode):
                    matched_gold.add(gi)
                    found = True
                    break

            if found:
                tp += 1
            else:
                fp += 1

        fn += len(gold_valid) - len(matched_gold)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return precision, recall, f1, tp, fp, fn


def compute_classification_metrics(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    match_mode: str = "exact",
) -> dict:
    """
    分类级指标: L1/L2/L3 逐层 Accuracy。
    仅对 抽取正确 (匹配到 gold entity) 的实体计算分类准确性。

    返回: {l1_acc, l2_acc, l3_acc, l1_correct, l2_correct, l3_correct, total_matched}
    """
    l1_correct = l2_correct = l3_correct = 0
    total_matched = 0

    for sid, pred_entities in predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue

        gold_valid = [e for e in gold_sentence.gold_entities if e.valid_entity]

        for pred in pred_entities:
            if not pred.get("valid_entity", True):
                continue

            # 找匹配的 gold entity
            matched_gold = None
            for gold_entity in gold_valid:
                if _match_entity(pred, gold_entity, match_mode=match_mode):
                    matched_gold = gold_entity
                    break

            if matched_gold is None:
                continue  # 抽取错误的实体不参与分类评估

            total_matched += 1

            if pred.get("l1", "") == matched_gold.l1:
                l1_correct += 1
            if pred.get("l2", "") == matched_gold.l2:
                l2_correct += 1
            if pred.get("l3_type_code", "") == matched_gold.l3_type_code:
                l3_correct += 1

    return {
        "l1_accuracy": l1_correct / max(total_matched, 1),
        "l2_accuracy": l2_correct / max(total_matched, 1),
        "l3_accuracy": l3_correct / max(total_matched, 1),
        "l1_correct": l1_correct,
        "l2_correct": l2_correct,
        "l3_correct": l3_correct,
        "total_matched": total_matched,
    }


def compute_strict_metrics(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    match_mode: str = "exact",
) -> tuple[float, float, float, int, int, int]:
    """
    端到端严格匹配 P/R/F1。
    条件: mention 匹配 + L1 正确 + L2 正确 + L3 正确 + valid_entity 一致。

    返回: (precision, recall, f1, tp, fp, fn)
    """
    tp, fp, fn = 0, 0, 0

    for sid, pred_entities in predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue

        gold_all = gold_sentence.gold_entities  # 包括 valid 和 invalid
        matched_gold: set[int] = set()

        for pred in pred_entities:
            found = False
            for gi, gold_entity in enumerate(gold_all):
                if gi in matched_gold:
                    continue
                if (
                    _match_entity(pred, gold_entity, match_mode=match_mode)
                    and pred.get("l1", "") == gold_entity.l1
                    and pred.get("l2", "") == gold_entity.l2
                    and pred.get("l3_type_code", "") == gold_entity.l3_type_code
                    and pred.get("valid_entity", True) == gold_entity.valid_entity
                ):
                    matched_gold.add(gi)
                    found = True
                    break

            if found:
                tp += 1
            else:
                fp += 1

        fn += len(gold_all) - len(matched_gold)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return precision, recall, f1, tp, fp, fn


def normalize_relation_output(
    relations: list[dict],
) -> dict[str, dict]:
    """
    将管道的评价关系输出按 sentence_id 分组 (V5.0)。

    输入: [{"sentence_id": ..., "subject": ..., "object": ..., ...}, ...]
    输出: {"1": {"has_evaluation": true, "relations": [rel_dict, ...]}, ...}

    has_evaluation 由该句是否存在关系推断 (管道层面关系为空的句子不进入输出,
    因此这里只能覆盖有关系的句子; 无关系句子的 has_evaluation 由调用方补充)。
    """
    grouped: dict[str, dict] = {}
    for rel in relations:
        sid = str(rel.get("sentence_id", ""))
        if not sid:
            continue
        entry = grouped.setdefault(sid, {"has_evaluation": False, "relations": []})
        entry["relations"].append(rel)
        entry["has_evaluation"] = True
    return grouped


def compute_relation_metrics(
    relation_predictions: dict[str, dict],
    gold_standard: GoldStandard,
    entity_predictions: dict[str, list[dict]],
) -> tuple[float, float, float, int, int, int]:
    """
    评价关系抽取级 P/R/F1 (V5.0)。

    匹配条件 (1:1 贪心匹配, 每句内):
    - subject 完全一致 (_paper_author/_cite[N]/人名/_unknown)
    - object 一致: 预测的 entity_id 先解析回实体名 (或 _missing_entity 的 object_text),
      再与标注的 object mention 做规范化匹配
    - opinion 规范化后完全一致

    参数:
    - relation_predictions: normalize_relation_output() 的分组结果
    - gold_standard: 含 gold_relations 的标注数据集
    - entity_predictions: normalize_pipeline_output() 的实体分组 (用于 ID→名字解析)

    返回: (precision, recall, f1, tp, fp, fn)
    """
    tp, fp, fn = 0, 0, 0

    # 预构建每句的 entity_id → 名字映射
    id_to_name: dict[str, dict[str, str]] = {}
    for sid, entities in entity_predictions.items():
        mapping: dict[str, str] = {}
        for e in entities:
            eid = str(e.get("entity_id", ""))
            if not eid:
                continue
            name = e.get("entity") or e.get("mention") or ""
            if name:
                mapping[eid] = name
            norm = e.get("normalized_name", "")
            if norm:
                mapping.setdefault(f"{eid}::norm", norm)
        id_to_name[sid] = mapping

    gold_relation_total = 0
    for sid, pred_entry in relation_predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue

        gold_relations = gold_sentence.gold_relations
        gold_relation_total += len(gold_relations)
        matched_gold: set[int] = set()

        for pred_rel in pred_entry.get("relations", []):
            pred_subject = str(pred_rel.get("subject", "")).strip()
            pred_obj_id = str(pred_rel.get("object", "")).strip()
            pred_opinion = _normalize_relation_name(str(pred_rel.get("opinion", "")))

            # 解析预测 object → 名字
            pred_obj_name = ""
            mapping = id_to_name.get(sid, {})
            if pred_obj_id == "_missing_entity":
                pred_obj_name = pred_rel.get("object_text", "")
            else:
                pred_obj_name = (
                    mapping.get(pred_obj_id, "")
                    or mapping.get(f"{pred_obj_id}::norm", "")
                )
            pred_obj_name = _normalize_relation_name(pred_obj_name)

            if not pred_subject or not pred_obj_name or not pred_opinion:
                fp += 1
                continue

            # 1:1 贪心匹配
            found = False
            for gi, gold_rel in enumerate(gold_relations):
                if gi in matched_gold:
                    continue
                if (
                    gold_rel.subject == pred_subject
                    and _normalize_relation_name(gold_rel.object) == pred_obj_name
                    and _normalize_relation_name(gold_rel.opinion) == pred_opinion
                ):
                    matched_gold.add(gi)
                    found = True
                    break

            if found:
                tp += 1
            else:
                fp += 1

        fn += len(gold_relations) - len(matched_gold)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return precision, recall, f1, tp, fp, fn


def compute_has_eval_accuracy(
    has_evaluation_predictions: dict[str, bool],
    gold_standard: GoldStandard,
) -> tuple[float, int, int]:
    """
    句级 has_evaluation 二元准确性 (V5.0)。

    - has_evaluation_predictions: {sentence_id: bool}
    - 标注侧: GoldSentence.get_has_evaluation()

    返回: (accuracy, correct, total)
    """
    correct = total = 0
    for sid, pred_has_eval in has_evaluation_predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue
        total += 1
        if pred_has_eval == gold_sentence.get_has_evaluation():
            correct += 1
    accuracy = correct / max(total, 1)
    return accuracy, correct, total


def compute_per_type_metrics(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    match_mode: str = "exact",
) -> dict[str, TypeMetrics]:
    """
    按 L3 类型分别计算抽取级 + 严格匹配级 P/R/F1。
    """
    # 收集所有 L3 类型
    all_types: set[str] = set()
    for sid, pred_entities in predictions.items():
        for pred in pred_entities:
            if pred.get("valid_entity", True):
                t = pred.get("l3_type_code", "")
                if t:
                    all_types.add(t)
    for gs in gold_standard._sentences.values():
        for ge in gs.gold_entities:
            if ge.valid_entity:
                all_types.add(ge.l3_type_code)

    per_type: dict[str, TypeMetrics] = {}

    for type_code in sorted(all_types):
        # 筛选该类型的预测和标注
        type_predictions: dict[str, list[dict]] = {}
        for sid, pred_entities in predictions.items():
            filtered = [
                p for p in pred_entities
                if p.get("valid_entity", True) and p.get("l3_type_code", "") == type_code
            ]
            if filtered:
                type_predictions[sid] = filtered

        # 构建临时 GoldStandard (仅含该类型的实体)
        temp_gold = GoldStandard()
        for gs in gold_standard._sentences.values():
            filtered_entities = [
                e for e in gs.gold_entities
                if e.valid_entity and e.l3_type_code == type_code
            ]
            if filtered_entities:
                # 创建一个只含此类型 gold entities 的 GoldSentence
                from .gold_standard import GoldSentence
                temp_gold.add_sentence(
                    GoldSentence(
                        sentence_id=gs.sentence_id,
                        sentence=gs.sentence,
                        gold_entities=filtered_entities,
                    )
                )

        # 计算
        ext_p, ext_r, ext_f1, _tp, _fp, _fn = compute_extraction_metrics(
            type_predictions, temp_gold, match_mode
        )
        str_p, str_r, str_f1, stp, _, _ = compute_strict_metrics(
            type_predictions, temp_gold, match_mode
        )

        per_type[type_code] = TypeMetrics(
            type_code=type_code,
            precision=ext_p,
            recall=ext_r,
            f1=ext_f1,
            tp=_tp,
            fp=_fp,
            fn=_fn,
            strict_precision=str_p,
            strict_recall=str_r,
            strict_f1=str_f1,
            strict_tp=stp,
        )

    return per_type


def build_confusion_matrix(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    level: str = "l3",
) -> dict[str, dict[str, int]]:
    """
    构建分类混淆矩阵。

    level: "l1" | "l2" | "l3" (默认 L3)

    返回: {gold_label: {pred_label: count, ...}, ...}
    """
    confusion: dict[str, dict[str, int]] = {}

    for sid, pred_entities in predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue

        gold_valid = [e for e in gold_sentence.gold_entities if e.valid_entity]

        for pred in pred_entities:
            if not pred.get("valid_entity", True):
                continue

            # 找匹配的 gold entity
            matched_gold = None
            for gold_entity in gold_valid:
                if _match_entity(pred, gold_entity, match_mode="exact"):
                    matched_gold = gold_entity
                    break

            if matched_gold is None:
                continue

            # 获取正确标签和预测标签
            if level == "l1":
                gold_label = matched_gold.l1
                pred_label = pred.get("l1", "")
            elif level == "l2":
                gold_label = matched_gold.l2
                pred_label = pred.get("l2", "")
            else:  # l3
                gold_label = matched_gold.l3_type_code
                pred_label = pred.get("l3_type_code", "")

            if gold_label not in confusion:
                confusion[gold_label] = {}
            confusion[gold_label][pred_label] = (
                confusion[gold_label].get(pred_label, 0) + 1
            )

    return confusion


# ===========================================================================
# 一站式评估 (推荐入口)
# ===========================================================================


def compute_metrics(
    predictions: dict[str, list[dict]],
    gold_standard: GoldStandard,
    match_mode: str = "exact",
    detailed: bool = True,
    relation_predictions: Optional[dict[str, dict]] = None,
    has_evaluation_predictions: Optional[dict[str, bool]] = None,
) -> EvalResult:
    """
    一站式评估: 计算所有指标并返回 EvalResult。

    参数:
    - predictions: {sentence_id: [entity_dict, ...]}
    - gold_standard: 标注数据集
    - match_mode: "exact" | "fuzzy" | "normalized"
    - detailed: 是否计算 per-type 和混淆矩阵 (耗时稍多)
    - relation_predictions (V5.0): normalize_relation_output() 的分组关系结果;
      传入时额外计算评价关系抽取级 P/R/F1
    - has_evaluation_predictions (V5.0): {sentence_id: bool}; 传入时额外计算
      句级 has_evaluation 准确性

    返回: EvalResult
    """
    result = EvalResult()

    # 统计
    result.total_predicted = sum(
        1 for entities in predictions.values()
        for e in entities if e.get("valid_entity", True)
    )
    result.total_gold = sum(
        1 for gs in gold_standard._sentences.values()
        for e in gs.gold_entities if e.valid_entity
    )
    result.total_sentences = len(predictions)

    # 1. 抽取级
    ep, er, ef1, etp, efp, efn = compute_extraction_metrics(
        predictions, gold_standard, match_mode
    )
    result.extraction_precision = ep
    result.extraction_recall = er
    result.extraction_f1 = ef1
    result.extraction_tp = etp
    result.extraction_fp = efp
    result.extraction_fn = efn

    # 2. 分类级
    clf = compute_classification_metrics(predictions, gold_standard, match_mode)
    result.l1_accuracy = clf["l1_accuracy"]
    result.l2_accuracy = clf["l2_accuracy"]
    result.l3_accuracy = clf["l3_accuracy"]
    result.l1_correct = clf["l1_correct"]
    result.l2_correct = clf["l2_correct"]
    result.l3_correct = clf["l3_correct"]
    result.l1_total = clf["total_matched"]
    result.l2_total = clf["total_matched"]
    result.l3_total = clf["total_matched"]

    # 3. 严格匹配
    sp, sr, sf1, stp, sfp, sfn = compute_strict_metrics(
        predictions, gold_standard, match_mode
    )
    result.strict_precision = sp
    result.strict_recall = sr
    result.strict_f1 = sf1
    result.strict_tp = stp
    result.strict_fp = sfp
    result.strict_fn = sfn

    # 4. 有效/无效判定准确性
    validity_correct = 0
    validity_total = 0
    for sid, pred_entities in predictions.items():
        gold_sentence = gold_standard.get_sentence(sid)
        if gold_sentence is None:
            continue
        gold_all = gold_sentence.gold_entities
        for pred in pred_entities:
            for gold_entity in gold_all:
                if _match_entity(pred, gold_entity, match_mode=match_mode):
                    validity_total += 1
                    if pred.get("valid_entity", True) == gold_entity.valid_entity:
                        validity_correct += 1
                    break
    result.validity_accuracy = validity_correct / max(validity_total, 1)
    result.validity_correct = validity_correct
    result.validity_total = validity_total

    if detailed:
        # 5. Per-type
        result.per_type = compute_per_type_metrics(
            predictions, gold_standard, match_mode
        )
        # 6. 混淆矩阵
        result.l3_confusion = build_confusion_matrix(
            predictions, gold_standard, "l3"
        )

    # 7. 评价关系抽取级 (V5.0, 可选)
    if relation_predictions is not None:
        rp, rr, rf1, rtp, rfp, rfn = compute_relation_metrics(
            relation_predictions, gold_standard, predictions
        )
        result.relation_precision = rp
        result.relation_recall = rr
        result.relation_f1 = rf1
        result.relation_tp = rtp
        result.relation_fp = rfp
        result.relation_fn = rfn
        result.relation_total_gold = gold_standard.relation_count()

    # 8. 句级 has_evaluation 准确性 (V5.0, 可选)
    if has_evaluation_predictions is not None:
        acc, correct, total = compute_has_eval_accuracy(
            has_evaluation_predictions, gold_standard
        )
        result.has_eval_accuracy = acc
        result.has_eval_correct = correct
        result.has_eval_total = total

    return result
