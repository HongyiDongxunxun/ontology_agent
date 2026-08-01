"""
===========================================================================
三Agent文献知识挖掘系统 — 统一配置
V4.3: + thinking mode + voting + RAG + dynamic terms path
===========================================================================
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path, PosixPath

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MID_DATA_DIR = PROJECT_ROOT / "mid_data"

@dataclass
class LLMConfig:
    model: str = os.environ.get("LLM_MODEL", os.environ.get("B2_LLM_MODEL", "deepseek-v4-flash"))
    base_url: str = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("B2_LLM_BASE_URL",
        os.environ.get("LOCAL_MODEL_ENDPOINT", "https://api.deepseek.com")))
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60
    enable_thinking_extraction: bool = False
    enable_thinking_classification: bool = False
    enable_thinking_reviewer: bool = False
    reasoning_effort: str = "max"
    api_key_extraction: str = os.environ.get("DEEPSEEK_API_KEY_EXTRACTION",
        os.environ.get("DEEPSEEK_API_KEY_L1", os.environ.get("B2_LLM_API_KEY", "")))
    api_key_classification: str = os.environ.get("DEEPSEEK_API_KEY_CLASSIFICATION",
        os.environ.get("DEEPSEEK_API_KEY_L2", os.environ.get("B2_LLM_API_KEY", "")))
    api_key_reviewer: str = os.environ.get("DEEPSEEK_API_KEY_REVIEWER",
        os.environ.get("DEEPSEEK_API_KEY_L3", os.environ.get("B2_LLM_API_KEY", "")))
    api_key: str = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("B2_LLM_API_KEY", ""))

@dataclass
class PipelineConfig:
    batch_size: int = 12
    max_l1_per_entity: int = 2
    enable_voting: bool = False
    voting_rounds: int = 3
    voting_temperature: float = 0.3
    enable_rag: bool = False
    rag_k_examples: int = 5

@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    live_mode: bool = False
    verbose: bool = True
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    mid_data_dir: Path = MID_DATA_DIR
    dynamic_terms_path: str = ""

config = Config()

def build_config(live_mode: bool = False, verbose: bool = True) -> Config:
    return Config(live_mode=live_mode, verbose=verbose)
