"""
pipeline — 四Agent端到端文献知识挖掘系统核心包
Agent 1: 评价关系抽取 → Agent 2: 实体抽取补充 → Agent 3: 分类 → Agent 4: 审查
"""

from .llm import LLMClient, build_llm_client
from .taxonomy import (
    TAXONOMY_HIERARCHY, L2_LABELS, L3_LABELS,
    get_l1_from_l3, get_l2_from_l3, get_l1l2_from_l3,
    get_l1_options, get_l2_options, get_l3_options,
    get_l2_label, get_l3_label,
)
from .entity_extraction_agent import (
    EntityExtractionAgent,
    ExtractedEntity,
    SentenceExtractionOutput,
)
from .classification_agent import (
    ClassificationAgent,
    FinalEntityResult,
)
from .reviewer_agent import (
    ReviewerAgent,
    ReviewResult,
)
from .evaluative_relation_agent import (
    EvaluativeRelationAgent,
    EvaluativeRelation,
    SentenceRelationOutput,
)
from .relation_verification_agent import (
    RelationVerificationAgent,
    RelationVerification,
    SentenceVerificationOutput,
)
from .dynamic_term_db import DynamicTermDB
from .dual_agent_pipeline import (
    DualAgentPipeline,
    export_jsonl,
    export_summary_json,
    export_relation_jsonl,
)

__all__ = [
    "LLMClient", "build_llm_client",
    "TAXONOMY_HIERARCHY", "L2_LABELS", "L3_LABELS",
    "get_l1_from_l3", "get_l2_from_l3", "get_l1l2_from_l3",
    "get_l1_options", "get_l2_options", "get_l3_options",
    "get_l2_label", "get_l3_label",
    "EntityExtractionAgent", "ExtractedEntity", "SentenceExtractionOutput",
    "ClassificationAgent", "FinalEntityResult",
    "ReviewerAgent", "ReviewResult",
    "EvaluativeRelationAgent", "EvaluativeRelation", "SentenceRelationOutput",
    "RelationVerificationAgent", "RelationVerification", "SentenceVerificationOutput",
    "DynamicTermDB",
    "DualAgentPipeline", "export_jsonl", "export_summary_json", "export_relation_jsonl",
]

__version__ = "5.1.0"
