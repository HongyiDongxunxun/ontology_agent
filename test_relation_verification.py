"""评价关系校验 Agent 单元测试"""

from pipeline.relation_verification_agent import (
    RelationVerificationAgent,
    RelationVerification,
    SentenceVerificationOutput,
)
from pipeline.evaluative_relation_agent import EvaluativeRelationAgent


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def call_json(self, prompt, default=None, schema_hint=""):
        self.prompt = prompt
        return self.payload


def _sample_relations():
    return [
        {"subject": "_paper_author", "object": "1_e1", "aspect": "研究水平",
         "opinion": "偏低", "evidence": "研究水平偏低"},
        {"subject": "_paper_author", "object": "1_e1", "aspect": None,
         "opinion": "进行了研究", "evidence": "对农村图书馆问题进行了研究"},
    ]


def test_verify_marks_facts_and_passes_evaluations():
    """评价放行(is_evaluation=true), 事实标记(false + fact_type)"""
    llm = FakeLLM([
        {"relation_id": "r1", "is_evaluation": True,
         "verdict_reason": "opinion带评价级差'偏低'"},
        {"relation_id": "r2", "is_evaluation": False,
         "verdict_reason": "研究行为描述", "fact_type": "研究行为"},
    ])
    agent = RelationVerificationAgent(llm)
    out = agent.verify("1", "农村图书馆研究水平偏低，进行了研究。", _sample_relations())

    assert out.total_relations == 2
    assert out.evaluation_count == 1
    assert out.fact_count == 1
    v1, v2 = out.verifications
    assert v1.is_evaluation is True and v1.fact_type == ""
    assert v2.is_evaluation is False and v2.fact_type == "研究行为"
    # 原始关系透传
    assert v2.original["opinion"] == "进行了研究"
    assert "is_evaluation" in v2.to_dict()
    assert "fact_type" in v2.to_dict()


def test_verify_prompt_contains_mandatory_question():
    """校验 prompt 包含强制判断问题与九类事实规则"""
    llm = FakeLLM([])
    agent = RelationVerificationAgent(llm)
    agent.verify("1", "测试句。", _sample_relations())
    assert "在\"评价\"一个对象，还是在\"描述\"这个对象发生了什么" in llm.prompt
    assert "进行了研究/分析/对比" in llm.prompt
    assert "基本一致" in llm.prompt
    # 输入关系带编号
    assert '"relation_id": "r1"' in llm.prompt
    assert '"relation_id": "r2"' in llm.prompt


def test_verify_prompt_has_scholar_persona():
    """校验 prompt 具备图书情报学者身份与语言学/逻辑学依据"""
    llm = FakeLLM([])
    agent = RelationVerificationAgent(llm)
    agent.verify("1", "测试句。", _sample_relations())
    # 学者身份
    assert "图书情报学学者" in llm.prompt
    assert "同行评议" in llm.prompt
    # 语言学: 评价理论三系统 + 立场标记
    assert "评价理论" in llm.prompt
    assert "级差" in llm.prompt
    assert "立场标记" in llm.prompt
    assert "名物化" in llm.prompt
    # 逻辑学: 休谟问题 事实命题/价值命题
    assert "休谟问题" in llm.prompt
    assert "事实命题" in llm.prompt
    assert "价值命题" in llm.prompt
    assert "从\"是\"推不出\"应该\"" in llm.prompt
    # 严格原则
    assert "存疑从严" in llm.prompt
    assert "独立裁决" in llm.prompt


def test_verify_handles_abnormal_llm_output():
    """LLM 返回异常时保守放行 (默认 is_evaluation=true)"""
    llm = FakeLLM({"not": "a list"})
    agent = RelationVerificationAgent(llm)
    out = agent.verify("1", "测试句。", _sample_relations())
    assert out.total_relations == 2
    assert out.fact_count == 0  # 全部保守放行
    assert all(v.is_evaluation for v in out.verifications)


def test_verify_handles_missing_verdicts():
    """LLM 漏掉部分关系的判定时, 缺失项默认放行"""
    llm = FakeLLM([
        {"relation_id": "r1", "is_evaluation": False,
         "verdict_reason": "x", "fact_type": "定义"},
        # r2 缺失
    ])
    agent = RelationVerificationAgent(llm)
    out = agent.verify("1", "测试句。", _sample_relations())
    assert out.verifications[0].is_evaluation is False
    assert out.verifications[1].is_evaluation is True  # 缺失默认放行
    assert out.evaluation_count == 1 and out.fact_count == 1


def test_verify_empty_relations():
    """无关系输入直接返回空输出"""
    agent = RelationVerificationAgent(FakeLLM([]))
    out = agent.verify("1", "测试句。", [])
    assert out.total_relations == 0
    assert out.verifications == []


def test_filter_evaluations_returns_only_passed():
    """filter_evaluations 只保留放行的关系"""
    out = SentenceVerificationOutput(
        sentence_id="1", sentence="t", total_relations=2,
        evaluation_count=1, fact_count=1,
        verifications=[
            RelationVerification(relation_id="r1", is_evaluation=True,
                                 original={"subject": "s1"}),
            RelationVerification(relation_id="r2", is_evaluation=False,
                                 original={"subject": "s2"}),
        ],
    )
    passed = RelationVerificationAgent.filter_evaluations(out)
    assert len(passed) == 1
    assert passed[0]["subject"] == "s1"


def test_verification_agent_works_with_relation_agent_output():
    """端到端(离线): Agent 1 抽取的关系可直接送入校验 Agent"""
    rel_agent = EvaluativeRelationAgent(FakeLLM({
        "has_evaluation": True,
        "entities": [
            {"entity_id": "e1", "entity": "农村图书馆研究",
             "normalized_name": "农村图书馆研究", "evidence": "x"},
        ],
        "relations": [
            {"subject": "_paper_author", "object": "e1", "aspect": None,
             "opinion": "进行了研究", "evidence": "对农村图书馆问题进行了研究"},
        ],
    }))
    rel_out = rel_agent.extract("1", "对农村图书馆问题进行了研究。")
    rel_dicts = [r.to_dict() for r in rel_out.relations]

    verify_agent = RelationVerificationAgent(FakeLLM([
        {"relation_id": "r1", "is_evaluation": False,
         "verdict_reason": "研究行为", "fact_type": "研究行为"},
    ]))
    vout = verify_agent.verify("1", "对农村图书馆问题进行了研究。", rel_dicts)
    assert vout.fact_count == 1
    assert vout.verifications[0].fact_type == "研究行为"
