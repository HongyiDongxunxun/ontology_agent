"""
pipeline — 三Agent端到端文献知识挖掘系统核心包
"""

from .llm import LLMClient, build_llm_client
from .taxonomy import (
    TAXONOMY_HIERARCHY, L2_LABELS, L3_LABELS,
    build_taxonomy_text,
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
from .dynamic_term_db import DynamicTermDB
from .dual_agent_pipeline import (
    AgentPipeline,
    export_jsonl,
    export_summary_json,
)
from .rag import (
    RAGExampleDB,
    get_rag_db,
    build_rag_index,
)
from .voting import (
    ExtractionVoter,
)

__all__ = [
    "LLMClient", "build_llm_client",
    "TAXONOMY_HIERARCHY", "L2_LABELS", "L3_LABELS",
    "get_l1_from_l3", "get_l2_from_l3", "get_l1l2_from_l3",
    "get_l1_options", "get_l2_options", "get_l3_options",
    "get_l2_label", "get_l3_label",
    "build_taxonomy_text",
    "EntityExtractionAgent", "ExtractedEntity", "SentenceExtractionOutput",
    "ClassificationAgent", "FinalEntityResult",
    "ReviewerAgent", "ReviewResult",
    "DynamicTermDB",
    "AgentPipeline", "export_jsonl", "export_summary_json",
    "RAGExampleDB", "get_rag_db", "build_rag_index",
    "ExtractionVoter",
]

__version__ = "4.3.0"
