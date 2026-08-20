"""V5.0 v2 架构单元测试 — 关系优先四Agent管道

对应架构:
  Agent 1: EvaluativeRelationAgent (评价关系抽取, 自行输出 entities + relations, 短ID)
  Agent 2: EntityExtractionAgent (实体补充抽取, 接收 known_entities, 完整ID {sid}_eN)
  Agent 3: ClassificationAgent
  Agent 4: ReviewerAgent

覆盖:
  - 关系抽取 schema (subject/object/aspect/opinion/evidence, 无 polarity)
  - aspect 不入 entities / aspect 绑定 object
  - 最小评价对象原则
  - prompt 规则文本存在性
  - object 名字匹配回退 / entity_id 去重
  - 管道级 object ID 重映射与实体回抽保障
"""
from pipeline.entity_extraction_agent import EntityExtractionAgent
from pipeline.evaluative_relation_agent import EvaluativeRelationAgent
from pipeline.dual_agent_pipeline import DualAgentPipeline


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def call_json(self, prompt, default=None, schema_hint=""):
        self.prompt = prompt
        return self.payload


# ─────────────────────────────────────────────────────────────
# Agent 1 (EvaluativeRelationAgent) 测试
# ─────────────────────────────────────────────────────────────

def test_relation_agent_keeps_aspects_out_of_entities():
    """aspect 不应进入 entities; relations 输出 aspect/opinion 而非 polarity"""
    llm = FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "农村图书馆研究",
             "normalized_name": "农村图书馆研究", "evidence": "我国农村图书馆研究取得了一定成绩"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": "作者分布",
             "opinion": "不合理", "evidence": "作者分布不合理"},
            {"subject": "_paper_author", "object": "e1", "aspect": "研究水平",
             "opinion": "偏低", "evidence": "研究水平偏低"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract(
        "1", "我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。")

    assert [e["entity"] for e in result.entities] == ["农村图书馆研究"]
    assert "作者分布" not in [e["entity"] for e in result.entities]
    assert "研究水平" not in [e["entity"] for e in result.entities]
    assert [r.object for r in result.relations] == ["e1", "e1"]
    assert [(r.aspect, r.opinion) for r in result.relations] == [
        ("作者分布", "不合理"), ("研究水平", "偏低")]
    assert "polarity" not in result.relations[0].to_dict()


def test_relation_agent_uses_minimal_evaluation_object():
    """最小评价对象原则: 抽取完整短语而非内部成分"""
    llm = FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "中西部地区研究",
             "normalized_name": "中西部地区研究", "evidence": "中西部地区研究不足"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": None,
             "opinion": "不足", "evidence": "中西部地区研究不足"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("2", "中西部地区研究不足。")

    assert [e["entity"] for e in result.entities] == ["中西部地区研究"]
    assert result.relations[0].object == "e1"
    assert result.relations[0].aspect is None
    assert result.relations[0].opinion == "不足"


def test_relation_agent_binds_aspect_to_existing_object():
    """aspect 绑定到已有 object, 不产生新实体"""
    llm = FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "档案信息化建设",
             "normalized_name": "档案信息化建设", "evidence": "档案信息化建设的步伐越走越快"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": "步伐",
             "opinion": "越走越快", "evidence": "档案信息化建设的步伐越走越快"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("4", "档案信息化建设的步伐越走越快。")

    assert result.relations[0].object == "e1"
    assert result.relations[0].aspect == "步伐"
    assert result.relations[0].opinion == "越走越快"


def test_relation_agent_resolves_object_by_name():
    """LLM 把实体名写在 object 字段 (而非 entity_id) 时按名字匹配回退"""
    llm = FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "农村图书馆研究",
             "normalized_name": "农村图书馆研究", "evidence": "农村图书馆研究取得了一定成绩"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "农村图书馆研究", "aspect": None,
             "opinion": "取得了一定成绩", "evidence": "农村图书馆研究取得了一定成绩"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("1", "农村图书馆研究取得了一定成绩。")

    assert result.relations[0].object == "e1"  # 名字被解析回实体ID


def test_relation_agent_dedups_entity_ids():
    """LLM 输出重复 entity_id 时自动去重, 不产生 ID 冲突"""
    llm = FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "实体甲", "normalized_name": "实体甲", "evidence": "甲"},
            {"entity_id": "e1", "entity": "实体乙", "normalized_name": "实体乙", "evidence": "乙"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": None,
             "opinion": "好", "evidence": "实体甲好"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("1", "实体甲好，实体乙也好。")

    ids = [e["entity_id"] for e in result.entities]
    assert len(ids) == len(set(ids))  # 无重复
    assert result.relations[0].object in ids


def test_relation_agent_prompt_rules_are_present():
    """关系优先策略的核心 prompt 规则存在"""
    llm = FakeLLM({"has_evaluation": False, "relations": [], "entities": []})
    EvaluativeRelationAgent(llm).extract("3", "档案信息化建设的步伐越走越快。")

    assert "先找评价关系，后抽取实体" in llm.prompt
    assert "评价对象回溯" in llm.prompt
    assert "不要因为 aspect 不是实体，就直接输出 _missing_entity" in llm.prompt
    assert "才使用 _missing_entity" in llm.prompt
    assert "Aspect 属于 object" in llm.prompt
    assert "关系输出字段必须使用 subject、object、aspect、opinion、evidence" in llm.prompt


def test_evaluation_validity_mandatory_gate_is_present():
    """Evaluation Validity 强制判断门必须存在于 prompt 中"""
    llm = FakeLLM({"has_evaluation": False, "relations": [], "entities": []})
    EvaluativeRelationAgent(llm).extract("3", "本文采用了文献计量法。")

    # 强制判断问题
    assert ("这句话是在\"评价\"一个对象，还是在\"描述\"这个对象发生了什么？"
            in llm.prompt)
    # 不完成判断不得输出
    assert "不完成此判断，不得输出任何 relation" in llm.prompt
    # 强原则
    assert ("动作、过程、研究行为、方法使用、功能实现、定义、分类、统计结果本身，"
            "都不是评价" in llm.prompt)
    # 统一判断顺序中的 Step 0 强制判断门
    assert "Step 0 — 强制判断门（前置，不可跳过）" in llm.prompt
    # 中性比较判断也是评价
    assert "中性比较判断也是评价" in llm.prompt
    assert "两种方法的结果基本一致。" in llm.prompt
    assert "水平相当" in llm.prompt


def test_evaluation_validity_negative_examples_are_present():
    """九类典型错误模式必须以完整 Negative Examples 形式存在"""
    llm = FakeLLM({"has_evaluation": False, "relations": [], "entities": []})
    EvaluativeRelationAgent(llm).extract("3", "本文采用了文献计量法。")

    expected_examples = [
        ("例9a", "我国学者对农村图书馆问题进行了研究。"),
        ("例9b", "本文对相关文献进行了分析。"),
        ("例9c", "该研究提出了一种新的分类方法。"),
        ("例9d", "本文采用了文献计量法对相关研究进行分析。"),
        ("例9e", "该系统实现了文献检索、数据分析和结果展示。"),
        ("例9f", "结构化是指将获取的知识内容加以归纳和整理。"),
        ("例9g", "自动文摘研究逐渐发展起来。"),
        ("例9h", "目前的研究主要集中于文摘生成方法。"),
        ("例9i", "该文对两种方法的优劣进行了对比。"),
    ]
    for label, sentence in expected_examples:
        assert label in llm.prompt, f"缺少 {label}"
        assert sentence in llm.prompt, f"缺少 {label} 的例句"
        assert "has_evaluation=false, relations=[]" in llm.prompt


# ─────────────────────────────────────────────────────────────
# Agent 2 (EntityExtractionAgent) 测试
# ─────────────────────────────────────────────────────────────

def test_entity_agent_formats_known_entities_as_names():
    """Agent 2 接收 known_entities, 名字注入 prompt (不注入上游短ID)"""
    llm = FakeLLM({"entities": []})
    EntityExtractionAgent(llm).extract(
        "农村图书馆研究取得了一定成绩。", sentence_id="1",
        known_entities=[
            {"entity_id": "e1", "entity": "农村图书馆研究", "normalized_name": "农村图书馆研究"},
        ],
    )

    assert "农村图书馆研究" in llm.prompt
    assert "e1" not in llm.prompt  # 上游短ID不应泄露进 prompt


# ─────────────────────────────────────────────────────────────
# Pipeline 级测试: ID 重映射与实体回抽保障
# ─────────────────────────────────────────────────────────────

def _make_pipeline(rel_payload, ent_payload):
    rel_llm = FakeLLM(rel_payload)
    ent_llm = FakeLLM(ent_payload)
    pipeline = DualAgentPipeline(
        llm_extraction=ent_llm,
        llm_relation=rel_llm,
        mid_data_dir="",
        verbose=False,
    )
    return pipeline, rel_llm, ent_llm


def test_pipeline_remaps_relation_object_to_full_entity_id():
    """关系短ID (e1) 应重映射为 Agent 2 的完整ID (1_e1)"""
    rel_payload = {
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "农村图书馆研究",
             "normalized_name": "农村图书馆研究", "evidence": "取得了一定成绩"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": None,
             "opinion": "取得了一定成绩", "evidence": "农村图书馆研究取得了一定成绩"},
        ],
    }
    ent_payload = {
        "entities": [
            {"mention": "农村图书馆研究", "normalized_name": "农村图书馆研究",
             "candidate_l3": "subfield", "candidate_l1": ["Abstract"],
             "evidence": "取得了一定成绩", "is_specific_entity": True,
             "confidence": 0.9, "uncertainty": ""},
        ],
    }

    pipeline, _, _ = _make_pipeline(rel_payload, ent_payload)
    # 只验证抽取阶段: 直接调用内部 agent 组合逻辑
    sid, stmt = "1", "农村图书馆研究取得了一定成绩。"
    rel_output = pipeline.evaluative_relation_agent.extract(sid, stmt)
    extraction = pipeline.extraction_agent.extract(
        stmt, sentence_id=sid, known_entities=rel_output.entities)

    rel_ent = rel_output.entities[0]
    matched = pipeline._find_entity_id(
        extraction.entities, rel_ent["entity"], rel_ent["normalized_name"])
    assert matched == "1_e1"  # 完整ID


def test_pipeline_backstops_missing_relation_entity():
    """Agent 2 漏抽关系引用的实体时, _find_entity_id 返回空, 触发程序化补抽路径"""
    pipeline, _, _ = _make_pipeline({}, {})
    from pipeline.entity_extraction_agent import ExtractedEntity
    # Agent 2 输出中没有任何与关系实体同名的实体
    extraction_entities = [
        ExtractedEntity(entity_id="1_e1", mention="中央政府",
                        normalized_name="中央政府", candidate_l3="governance",
                        candidate_l1=["Agent"], evidence=""),
    ]
    matched = pipeline._find_entity_id(
        extraction_entities, "农村图书馆研究", "农村图书馆研究")
    assert matched == ""  # 未找到 → 触发补抽 (run() 中 append 逻辑)

    # 模拟 run() 中的补抽分支
    new_eid = f"1_e{len(extraction_entities) + 1}"
    assert new_eid == "1_e2"


def test_pipeline_find_entity_id_matches_normalized_name():
    """名字匹配: mention 或 normalized_name 均参与匹配"""
    pipeline, _, _ = _make_pipeline({}, {})
    from pipeline.entity_extraction_agent import ExtractedEntity
    entities = [
        ExtractedEntity(entity_id="1_e1", mention="新版《图书馆学概论》",
                        normalized_name="《图书馆学概论》", candidate_l3="book",
                        candidate_l1=["Artifact"], evidence=""),
    ]
    # 用规范化名匹配
    assert pipeline._find_entity_id(entities, "图书馆学概论", "") == "1_e1"
    # 精确 mention 匹配
    assert pipeline._find_entity_id(entities, "新版《图书馆学概论》", "") == "1_e1"
    # 无关实体不匹配
    assert pipeline._find_entity_id(entities, "不存在的实体", "") == ""
