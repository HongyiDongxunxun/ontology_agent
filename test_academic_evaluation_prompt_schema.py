from pipeline.entity_extraction_agent import EntityExtractionAgent
from pipeline.evaluative_relation_agent import EvaluativeRelationAgent


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def call_json(self, prompt, default=None, schema_hint=""):
        self.prompt = prompt
        return self.payload


def test_joint_extraction_keeps_aspects_out_of_entities():
    llm = FakeLLM(
        {
            "entities": [
                {
                    "mention": "农村图书馆研究",
                    "normalized_name": "农村图书馆研究",
                    "candidate_l3": "subfield",
                    "candidate_l1": ["Abstract"],
                    "evidence": "我国农村图书馆研究取得了一定成绩",
                    "is_specific_entity": True,
                    "confidence": 0.95,
                    "uncertainty": "",
                }
            ],
            "has_evaluation": True,
            "relations": [
                {
                    "subject": "_paper_author",
                    "object": "1_e1",
                    "aspect": "作者分布",
                    "opinion": "不合理",
                    "evidence": "作者分布不合理",
                },
                {
                    "subject": "_paper_author",
                    "object": "1_e1",
                    "aspect": "研究水平",
                    "opinion": "偏低",
                    "evidence": "研究水平偏低",
                },
            ],
        }
    )

    result = EntityExtractionAgent(llm).extract(
        "我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。",
        sentence_id="1",
    )

    assert [entity.mention for entity in result.entities] == ["农村图书馆研究"]
    assert "作者分布" not in [entity.mention for entity in result.entities]
    assert "研究水平" not in [entity.mention for entity in result.entities]
    assert [rel.object for rel in result.relations] == ["1_e1", "1_e1"]
    assert [(rel.aspect, rel.opinion) for rel in result.relations] == [
        ("作者分布", "不合理"),
        ("研究水平", "偏低"),
    ]
    assert "polarity" not in result.relations[0].to_dict()


def test_joint_extraction_uses_minimal_evaluation_object():
    llm = FakeLLM(
        {
            "entities": [
                {
                    "mention": "中西部地区研究",
                    "normalized_name": "中西部地区研究",
                    "candidate_l3": "subfield",
                    "candidate_l1": ["Abstract"],
                    "evidence": "中西部地区研究不足",
                    "is_specific_entity": True,
                    "confidence": 0.95,
                    "uncertainty": "",
                }
            ],
            "has_evaluation": True,
            "relations": [
                {
                    "subject": "_paper_author",
                    "object": "2_e1",
                    "aspect": None,
                    "opinion": "不足",
                    "evidence": "中西部地区研究不足",
                }
            ],
        }
    )

    result = EntityExtractionAgent(llm).extract("中西部地区研究不足。", sentence_id="2")

    assert [entity.mention for entity in result.entities] == ["中西部地区研究"]
    assert "中西部地区" not in [entity.mention for entity in result.entities]
    assert result.relations[0].object == "2_e1"
    assert result.relations[0].aspect is None
    assert result.relations[0].opinion == "不足"


def test_relation_agent_outputs_aspect_and_opinion():
    llm = FakeLLM(
        {
            "has_evaluation": True,
            "relations": [
                {
                    "subject": "_paper_author",
                    "object": "1_e1",
                    "aspect": "作者分布",
                    "opinion": "不合理",
                    "evidence": "作者分布不合理",
                }
            ],
        }
    )

    result = EvaluativeRelationAgent(llm).extract(
        "1",
        "我国农村图书馆研究取得了一定成绩，但作者分布不合理、研究水平偏低。",
        [{"entity_id": "1_e1", "entity": "农村图书馆研究", "valid_entity": True}],
    )

    assert result.relations[0].to_dict() == {
        "subject": "_paper_author",
        "object": "1_e1",
        "aspect": "作者分布",
        "opinion": "不合理",
        "evidence": "作者分布不合理",
    }


def test_object_resolution_prompt_rules_are_present():
    joint_llm = FakeLLM({"entities": [], "has_evaluation": False, "relations": []})
    EntityExtractionAgent(joint_llm).extract("档案信息化建设的步伐越走越快。", sentence_id="3")

    relation_llm = FakeLLM({"has_evaluation": False, "relations": []})
    EvaluativeRelationAgent(relation_llm).extract(
        "3",
        "档案信息化建设的步伐越走越快。",
        [{"entity_id": "3_e1", "entity": "档案信息化建设", "valid_entity": True}],
    )

    assert "评价对象回溯" in joint_llm.prompt
    assert "不要因为 aspect 不是实体，就直接输出 _missing_entity" in joint_llm.prompt
    assert "仅在真正不存在对象时使用 _missing_entity" in relation_llm.prompt
    assert "Aspect 属于 Object，而不是独立 Object" in relation_llm.prompt


def test_joint_prompt_uses_relation_first_strategy():
    llm = FakeLLM({"entities": [], "has_evaluation": False, "relations": []})
    EntityExtractionAgent(llm).extract("中西部地区研究不足。", sentence_id="5")

    assert "先找评价关系，后抽取实体" in llm.prompt
    assert "先识别句子中的评价关系，再根据评价关系所需的 object 反推并抽取实体" in llm.prompt
    assert "entities 必须覆盖所有可解析的 relation.object" in llm.prompt


def test_relation_agent_binds_aspect_to_existing_object():
    llm = FakeLLM(
        {
            "has_evaluation": True,
            "relations": [
                {
                    "subject": "_paper_author",
                    "object": "4_e1",
                    "aspect": "步伐",
                    "opinion": "越走越快",
                    "evidence": "档案信息化建设的步伐越走越快",
                }
            ],
        }
    )

    result = EvaluativeRelationAgent(llm).extract(
        "4",
        "档案信息化建设的步伐越走越快。",
        [{"entity_id": "4_e1", "entity": "档案信息化建设", "valid_entity": True}],
    )

    assert result.relations[0].object == "4_e1"
    assert result.relations[0].aspect == "步伐"
    assert result.relations[0].opinion == "越走越快"
