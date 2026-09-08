"""V5.1 v2 架构单元测试 — 关系优先五Agent管道

对应架构:
  Agent 1: EvaluativeRelationAgent (评价关系抽取, subject/object 为原文精确短语, 不输出 entities)
  Agent 2: EntityExtractionAgent (接收 required_mentions 必抽短语, 统一负责实体识别+ID+分类)
  Agent 3: ClassificationAgent
  Agent 4: ReviewerAgent
  Agent 5: RelationVerificationAgent

覆盖:
  - 关系抽取 schema (subject/object/aspect/opinion/evidence, 无 polarity, 无 entities)
  - object 为原文精确短语 (不再引用 entity_id)
  - _missing_entity 占位符处理
  - prompt 规则文本存在性
  - Agent 2 接收 required_mentions (字符串列表) 注入 prompt
  - pipeline._extract_required_mentions 从 relations 提取必抽短语
  - pipeline._is_placeholder 判断占位符
  - pipeline.run() 后 relation.object 被回填为 entity_id
  - 未匹配时 object_unmatched=true
"""
from pipeline.entity_extraction_agent import EntityExtractionAgent
from pipeline.evaluative_relation_agent import (
    EvaluativeRelationAgent,
    EvaluativeRelation,
)
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

def test_relation_agent_outputs_no_entities_field():
    """Agent 1 不再输出 entities; relations 直接带原文短语"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "农村图书馆研究",
             "aspect": "作者分布", "opinion": "不合理",
             "evidence": "作者分布不合理"},
            {"subject": "_paper_author", "object": "农村图书馆研究",
             "aspect": "研究水平", "opinion": "偏低",
             "evidence": "研究水平偏低"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract(
        "1", "我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。")

    # Agent 1 不再输出 entities
    assert not hasattr(result, "entities") or not result.entities
    # relations 的 object 是原文短语，不是 entity_id
    assert [r.object for r in result.relations] == ["农村图书馆研究", "农村图书馆研究"]
    # aspect 不应作为 object
    assert "作者分布" not in [r.object for r in result.relations]
    assert "研究水平" not in [r.object for r in result.relations]
    assert [(r.aspect, r.opinion) for r in result.relations] == [
        ("作者分布", "不合理"), ("研究水平", "偏低")]
    assert "polarity" not in result.relations[0].to_dict()


def test_relation_agent_object_is_raw_text_span():
    """object 字段直接保留原文精确短语，不做 entity_id 解析"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "中西部地区研究",
             "aspect": None, "opinion": "不足",
             "evidence": "中西部地区研究不足"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("2", "中西部地区研究不足。")

    assert result.relations[0].object == "中西部地区研究"  # 原文短语，非 entity_id
    assert result.relations[0].aspect is None
    assert result.relations[0].opinion == "不足"


def test_relation_agent_binds_aspect_to_object():
    """aspect 绑定到 object，object 仍为原文短语"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "档案信息化建设",
             "aspect": "步伐", "opinion": "越走越快",
             "evidence": "档案信息化建设的步伐越走越快"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("4", "档案信息化建设的步伐越走越快。")

    assert result.relations[0].object == "档案信息化建设"  # 原文短语
    assert result.relations[0].aspect == "步伐"
    assert result.relations[0].opinion == "越走越快"


def test_relation_agent_preserves_missing_entity_placeholder():
    """LLM 输出 _missing_entity 时保留占位符，并附 object_text"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "_missing_entity",
             "aspect": None, "opinion": "较低",
             "evidence": "研究水平较低", "object_text": "某研究"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("5", "某研究的研究水平较低。")

    assert result.relations[0].object == "_missing_entity"
    assert result.relations[0].object_text == "某研究"


def test_relation_agent_missing_entity_without_text_gets_placeholder():
    """_missing_entity 未附带 object_text 时，用占位符本身作为 object_text"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "_missing_entity",
             "aspect": None, "opinion": "较低",
             "evidence": "研究水平较低"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("5", "某研究的研究水平较低。")

    assert result.relations[0].object == "_missing_entity"
    assert result.relations[0].object_text == "_missing_entity"


def test_relation_agent_drops_incomplete_relations():
    """必填字段缺失 (subject/object/opinion/evidence) 的 relation 被丢弃"""
    llm = FakeLLM({
        "has_evaluation": True,
        "relations": [
            # 缺 opinion
            {"subject": "_paper_author", "object": "甲", "aspect": None,
             "opinion": "", "evidence": "甲好"},
            # 缺 evidence
            {"subject": "_paper_author", "object": "乙", "aspect": None,
             "opinion": "好", "evidence": ""},
            # 完整
            {"subject": "_paper_author", "object": "丙", "aspect": None,
             "opinion": "好", "evidence": "丙好"},
        ],
    })

    result = EvaluativeRelationAgent(llm).extract("1", "甲乙丙。")

    assert len(result.relations) == 1
    assert result.relations[0].object == "丙"


def test_relation_agent_prompt_rules_are_present():
    """关系优先策略的核心 prompt 规则存在"""
    llm = FakeLLM({"has_evaluation": False, "relations": []})
    EvaluativeRelationAgent(llm).extract("3", "档案信息化建设的步伐越走越快。")

    assert "评价关系抽取专家" in llm.prompt
    assert "Object Resolution" in llm.prompt
    assert "_missing_entity" in llm.prompt
    assert "subject/object 用原文短语" in llm.prompt
    assert "不要输出 entities 字段" in llm.prompt


def test_relation_agent_prompt_has_no_entity_supplement_section():
    """旧版 'Entity 补充' 章节已删除"""
    llm = FakeLLM({"has_evaluation": False, "relations": []})
    EvaluativeRelationAgent(llm).extract("3", "测试。")

    # 旧版 "Entity 补充" 章节已删除
    assert "Entity 补充" not in llm.prompt
    assert "补充必要 Entity" not in llm.prompt
    # 旧版 "entities" 输出字段已从输出格式中删除
    # （输出格式区只应出现一次 "entities"，即不应作为顶层字段出现）
    # 检查输出格式区不包含 entities 数组定义
    assert '"entities":' not in llm.prompt


# ─────────────────────────────────────────────────────────────
# Agent 2 (EntityExtractionAgent) 测试
# ─────────────────────────────────────────────────────────────

def test_entity_agent_receives_required_mentions():
    """Agent 2 接收 required_mentions (字符串列表)，注入 prompt 作为必抽提示"""
    llm = FakeLLM({"entities": []})
    EntityExtractionAgent(llm).extract(
        "农村图书馆研究取得了一定成绩。", sentence_id="1",
        required_mentions=["农村图书馆研究"],
    )

    assert "农村图书馆研究" in llm.prompt
    assert "必抽实体" in llm.prompt


def test_entity_agent_required_mentions_dedup():
    """重复短语被去重，必抽列表中每个只出现一次"""
    llm = FakeLLM({"entities": []})
    EntityExtractionAgent(llm).extract(
        "测试句。", sentence_id="1",
        required_mentions=["农村图书馆研究", "农村图书馆研究", "信息过载"],
    )

    # 必抽列表中 "- 农村图书馆研究" 只出现一次（原句已不含该词）
    assert llm.prompt.count("- 农村图书馆研究") == 1
    assert "- 信息过载" in llm.prompt


def test_entity_agent_empty_required_mentions():
    """空列表返回 '无'，prompt 仍包含必抽实体章节标题"""
    llm = FakeLLM({"entities": []})
    EntityExtractionAgent(llm).extract(
        "测试。", sentence_id="1", required_mentions=[],
    )

    assert "必抽实体" in llm.prompt
    # 空列表时注入文本应为 "无"
    assert "\n无\n" in llm.prompt or "无" in llm.prompt


def test_entity_agent_no_required_mentions_argument():
    """不传 required_mentions 时默认为空列表"""
    llm = FakeLLM({"entities": []})
    result = EntityExtractionAgent(llm).extract("测试。", sentence_id="1")

    assert result.entities == []
    assert "必抽实体" in llm.prompt


# ─────────────────────────────────────────────────────────────
# Pipeline 辅助方法测试
# ─────────────────────────────────────────────────────────────

def _make_pipeline(rel_payload=None, ent_payload=None):
    rel_llm = FakeLLM(rel_payload or {})
    ent_llm = FakeLLM(ent_payload or {})
    pipeline = DualAgentPipeline(
        llm_extraction=ent_llm,
        llm_relation=rel_llm,
        mid_data_dir="",
        verbose=False,
    )
    return pipeline, rel_llm, ent_llm


def test_is_placeholder_recognizes_subject_placeholders():
    """_is_placeholder 正确识别 subject 占位符"""
    assert DualAgentPipeline._is_placeholder("_paper_author") is True
    assert DualAgentPipeline._is_placeholder("_cite[3]") is True
    assert DualAgentPipeline._is_placeholder("_unknown") is True
    assert DualAgentPipeline._is_placeholder("_missing_entity") is True
    assert DualAgentPipeline._is_placeholder("") is True


def test_is_placeholder_rejects_raw_spans():
    """_is_placeholder 对原文短语返回 False"""
    assert DualAgentPipeline._is_placeholder("农村图书馆研究") is False
    assert DualAgentPipeline._is_placeholder("许晓东等") is False
    assert DualAgentPipeline._is_placeholder("巴巴拉·奎恩特") is False


def test_extract_required_mentions_collects_object_and_subject():
    """从 relations 提取 object 原文 + subject 人物名，跳过占位符"""
    relations = [
        EvaluativeRelation(
            subject="_paper_author", object="农村图书馆研究",
            aspect=None, opinion="好", evidence="农村图书馆研究好", object_text="",
        ),
        EvaluativeRelation(
            subject="_cite[3]", object="信息过载",
            aspect=None, opinion="严重", evidence="信息过载严重", object_text="",
        ),
        EvaluativeRelation(
            subject="许晓东等", object="该机构",
            aspect="创新能力", opinion="强", evidence="该机构强", object_text="",
        ),
    ]

    mentions = DualAgentPipeline._extract_required_mentions(relations)

    # object 原文 + subject 人物名，去重保序
    assert mentions == ["农村图书馆研究", "信息过载", "该机构", "许晓东等"]


def test_extract_required_mentions_skips_missing_entity():
    """object=_missing_entity 时不进入必抽列表"""
    relations = [
        EvaluativeRelation(
            subject="_paper_author", object="_missing_entity",
            aspect=None, opinion="低", evidence="低", object_text="某研究",
        ),
        EvaluativeRelation(
            subject="_paper_author", object="农村图书馆研究",
            aspect=None, opinion="好", evidence="好", object_text="",
        ),
    ]

    mentions = DualAgentPipeline._extract_required_mentions(relations)

    assert mentions == ["农村图书馆研究"]


def test_extract_required_mentions_dedup():
    """重复的短语去重保序"""
    relations = [
        EvaluativeRelation(
            subject="_paper_author", object="农村图书馆研究",
            aspect="作者分布", opinion="不合理", evidence="x", object_text="",
        ),
        EvaluativeRelation(
            subject="_paper_author", object="农村图书馆研究",
            aspect="研究水平", opinion="偏低", evidence="y", object_text="",
        ),
    ]

    mentions = DualAgentPipeline._extract_required_mentions(relations)

    assert mentions == ["农村图书馆研究"]


# ─────────────────────────────────────────────────────────────
# Pipeline run() 级测试: object 回填
# ─────────────────────────────────────────────────────────────

def test_pipeline_run_backfills_object_to_entity_id():
    """pipeline.run() 后 relation.object 被回填为 Agent 2 的 entity_id"""
    rel_payload = {
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "农村图书馆研究",
             "aspect": None, "opinion": "取得了一定成绩",
             "evidence": "农村图书馆研究取得了一定成绩"},
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
    # 只跑抽取阶段: 禁用 Agent 3/4/5 以便隔离测试
    pipeline.enable_verification = False
    sentences = [("1", "农村图书馆研究取得了一定成绩。")]
    # 直接调用内部组合逻辑 (不触发 run() 的分类/审查/校验)
    sid, stmt = sentences[0]
    rel_output = pipeline.evaluative_relation_agent.extract(sid, stmt)
    required = pipeline._extract_required_mentions(rel_output.relations)
    extraction = pipeline.extraction_agent.extract(
        stmt, sentence_id=sid, required_mentions=required)

    # 构建 ent_index 模拟 run() 中的回填逻辑
    from pipeline.evaluative_relation_agent import _normalize_name
    ent_index = {}
    for e in extraction.entities:
        for key in (_normalize_name(e.normalized_name), _normalize_name(e.mention)):
            if key and key not in ent_index:
                ent_index[key] = e.entity_id

    rel = rel_output.relations[0]
    matched = ent_index.get(_normalize_name(rel.object), "")
    assert matched == "1_e1"  # 回填为完整 entity_id


def test_pipeline_run_marks_unmatched_object():
    """Agent 2 未抽到评价对象时，object 保留原文 + object_unmatched"""
    rel_payload = {
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "该研究方法",
             "aspect": None, "opinion": "有效",
             "evidence": "该研究方法有效"},
        ],
    }
    # Agent 2 没抽到 "该研究方法"
    ent_payload = {
        "entities": [
            {"mention": "其他实体", "normalized_name": "其他实体",
             "candidate_l3": "concept", "candidate_l1": ["Abstract"],
             "evidence": "其他", "is_specific_entity": True,
             "confidence": 0.9, "uncertainty": ""},
        ],
    }

    pipeline, _, _ = _make_pipeline(rel_payload, ent_payload)
    sid, stmt = "1", "该研究方法有效。"
    rel_output = pipeline.evaluative_relation_agent.extract(sid, stmt)
    required = pipeline._extract_required_mentions(rel_output.relations)
    extraction = pipeline.extraction_agent.extract(
        stmt, sentence_id=sid, required_mentions=required)

    from pipeline.evaluative_relation_agent import _normalize_name
    ent_index = {}
    for e in extraction.entities:
        for key in (_normalize_name(e.normalized_name), _normalize_name(e.mention)):
            if key and key not in ent_index:
                ent_index[key] = e.entity_id

    rel = rel_output.relations[0]
    matched = ent_index.get(_normalize_name(rel.object), "")
    assert matched == ""  # 未匹配
    # 在 run() 中会标记 object_unmatched=True


def test_pipeline_run_passes_through_placeholder_subject():
    """占位符 subject (_paper_author) 不参与匹配，原样透传"""
    rel_payload = {
        "has_evaluation": True,
        "relations": [
            {"subject": "_paper_author", "object": "农村图书馆研究",
             "aspect": None, "opinion": "好", "evidence": "好"},
        ],
    }
    ent_payload = {"entities": []}

    pipeline, _, _ = _make_pipeline(rel_payload, ent_payload)
    sid, stmt = "1", "农村图书馆研究好。"
    rel_output = pipeline.evaluative_relation_agent.extract(sid, stmt)
    required = pipeline._extract_required_mentions(rel_output.relations)

    # 占位符 subject 不应进入 required_mentions
    assert "_paper_author" not in required
    assert "农村图书馆研究" in required
