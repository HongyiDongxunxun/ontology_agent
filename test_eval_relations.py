"""V5.0 关系评估单元测试 — eval 模块的关系级指标"""

from eval.gold_standard import (
    GoldStandard, GoldEntity, GoldSentence, GoldRelation,
)
from eval.metrics import (
    normalize_relation_output,
    compute_relation_metrics,
    compute_has_eval_accuracy,
    compute_metrics,
    normalize_pipeline_output,
)


def _build_gold() -> GoldStandard:
    """构建含关系标注的 GoldStandard (2句)"""
    gs = GoldStandard()
    gs.add_sentence(GoldSentence(
        sentence_id="1",
        sentence="农村图书馆研究取得了一定成绩，但研究水平偏低。",
        gold_entities=[
            GoldEntity(mention="农村图书馆研究", l1="Abstract", l2="Epistemic",
                       l3_type_code="subfield", valid_entity=True),
        ],
        gold_relations=[
            GoldRelation(subject="_paper_author", object="农村图书馆研究",
                         opinion="取得了一定成绩", aspect=None,
                         evidence="取得了一定成绩"),
            GoldRelation(subject="_paper_author", object="农村图书馆研究",
                         opinion="偏低", aspect="研究水平",
                         evidence="研究水平偏低"),
        ],
    ))
    gs.add_sentence(GoldSentence(
        sentence_id="2",
        sentence="本文采用了文献计量法。",
        gold_entities=[],
        gold_relations=[],
        has_evaluation=False,
    ))
    return gs


def test_gold_relation_roundtrip():
    r = GoldRelation(subject="_paper_author", object="农村图书馆研究",
                     opinion="偏低", aspect="研究水平", evidence="研究水平偏低")
    d = r.to_dict()
    assert d["aspect"] == "研究水平"
    r2 = GoldRelation.from_dict(d)
    assert r2 == r
    # aspect=None 时序列化与恢复
    r3 = GoldRelation.from_dict({"subject": "_paper_author", "object": "X",
                                 "opinion": "好", "aspect": None})
    assert r3.aspect is None


def test_gold_sentence_relation_roundtrip_and_inference():
    s = GoldSentence(sentence_id="1", sentence="t",
                     gold_relations=[GoldRelation(
                         subject="_paper_author", object="X", opinion="好")])
    d = s.to_dict()
    assert "gold_relations" in d
    s2 = GoldSentence.from_dict(d)
    assert len(s2.gold_relations) == 1
    assert s2.get_has_evaluation() is True  # 由关系非空推断

    # 显式 has_evaluation=False + 无关系 → False
    s3 = GoldSentence(sentence_id="2", sentence="t", has_evaluation=False)
    assert s3.get_has_evaluation() is False


def test_gold_standard_relation_stats():
    gs = _build_gold()
    assert gs.relation_count() == 2
    assert gs.evaluated_sentence_count() == 1  # 句1有评价, 句2无
    stats = gs.get_stats()
    assert stats["relations"] == 2
    assert stats["evaluated_sentences"] == 1


def test_normalize_relation_output_groups_by_sentence():
    relations = [
        {"sentence_id": "1", "subject": "_paper_author", "object": "1_e1",
         "opinion": "好", "evidence": "好"},
        {"sentence_id": "1", "subject": "_paper_author", "object": "1_e1",
         "opinion": "不足", "evidence": "不足"},
        {"sentence_id": "2", "subject": "_paper_author", "object": "2_e1",
         "opinion": "显著", "evidence": "显著"},
    ]
    grouped = normalize_relation_output(relations)
    assert set(grouped.keys()) == {"1", "2"}
    assert grouped["1"]["has_evaluation"] is True
    assert len(grouped["1"]["relations"]) == 2
    assert len(grouped["2"]["relations"]) == 1


def test_compute_relation_metrics_with_id_resolution():
    """ID→名字解析 + 1:1 匹配: 全对 → F1=1"""
    gs = _build_gold()

    entity_predictions = {
        "1": [
            {"entity_id": "1_e1", "entity": "农村图书馆研究",
             "normalized_name": "农村图书馆研究", "valid_entity": True},
        ],
    }
    relation_predictions = {
        "1": {"has_evaluation": True, "relations": [
            {"subject": "_paper_author", "object": "1_e1",
             "opinion": "取得了一定成绩", "evidence": "取得了一定成绩"},
            {"subject": "_paper_author", "object": "1_e1",
             "opinion": "偏低", "evidence": "研究水平偏低"},
        ]},
    }

    p, r, f1, tp, fp, fn = compute_relation_metrics(
        relation_predictions, gs, entity_predictions)
    assert (tp, fp, fn) == (2, 0, 0)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_compute_relation_metrics_detects_fp_fn():
    """错误 opinion → FP + FN"""
    gs = _build_gold()
    entity_predictions = {
        "1": [{"entity_id": "1_e1", "entity": "农村图书馆研究",
               "normalized_name": "农村图书馆研究", "valid_entity": True}],
    }
    # 预测: 1条正确, 1条错误opinion → TP=1, FP=1, FN=1
    relation_predictions = {
        "1": {"has_evaluation": True, "relations": [
            {"subject": "_paper_author", "object": "1_e1",
             "opinion": "取得了一定成绩", "evidence": "x"},
            {"subject": "_paper_author", "object": "1_e1",
             "opinion": "完全错误", "evidence": "y"},
        ]},
    }
    p, r, f1, tp, fp, fn = compute_relation_metrics(
        relation_predictions, gs, entity_predictions)
    assert (tp, fp, fn) == (1, 1, 1)
    assert abs(p - 0.5) < 1e-9 and abs(r - 0.5) < 1e-9


def test_compute_relation_metrics_missing_entity_text():
    """_missing_entity 用 object_text 参与匹配"""
    gs = _build_gold()
    entity_predictions = {"1": []}  # 无实体可解析
    relation_predictions = {
        "1": {"has_evaluation": True, "relations": [
            {"subject": "_paper_author", "object": "_missing_entity",
             "object_text": "农村图书馆研究",
             "opinion": "取得了一定成绩", "evidence": "x"},
        ]},
    }
    p, r, f1, tp, fp, fn = compute_relation_metrics(
        relation_predictions, gs, entity_predictions)
    assert tp == 1  # object_text 匹配成功
    assert fn == 1  # 另一条 gold 关系未预测


def test_compute_has_eval_accuracy():
    gs = _build_gold()
    preds = {"1": True, "2": True}  # 句2应为False → 1/2正确
    acc, correct, total = compute_has_eval_accuracy(preds, gs)
    assert total == 2 and correct == 1
    assert abs(acc - 0.5) < 1e-9

    preds2 = {"1": True, "2": False}
    acc2, correct2, _ = compute_has_eval_accuracy(preds2, gs)
    assert correct2 == 2 and acc2 == 1.0


def test_compute_metrics_integration_with_relations():
    """一站式评估集成: relation + has_eval 字段被填充"""
    gs = _build_gold()
    entity_predictions = normalize_pipeline_output([
        {"sentence_id": "1", "entity_id": "1_e1", "entity": "农村图书馆研究",
         "normalized_name": "农村图书馆研究", "valid_entity": True,
         "l1": "Abstract", "l2": "Epistemic", "l3_type_code": "subfield"},
    ])
    relation_predictions = normalize_relation_output([
        {"sentence_id": "1", "subject": "_paper_author", "object": "1_e1",
         "opinion": "取得了一定成绩", "evidence": "x"},
        {"sentence_id": "1", "subject": "_paper_author", "object": "1_e1",
         "opinion": "偏低", "evidence": "y"},
    ])
    has_eval = {"1": True, "2": False}

    result = compute_metrics(
        entity_predictions, gs, match_mode="exact", detailed=True,
        relation_predictions=relation_predictions,
        has_evaluation_predictions=has_eval,
    )
    assert result.relation_total_gold == 2
    assert result.relation_tp == 2
    assert abs(result.relation_f1 - 1.0) < 1e-9
    assert result.has_eval_total == 2
    assert result.has_eval_correct == 2
    assert abs(result.has_eval_accuracy - 1.0) < 1e-9

    # 不传关系参数时不计算关系指标 (保持向后兼容)
    result2 = compute_metrics(entity_predictions, gs, detailed=False)
    assert result2.relation_total_gold == 0
    assert result2.has_eval_total == 0
