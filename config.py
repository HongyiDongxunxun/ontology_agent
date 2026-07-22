"""
===========================================================================
三Agent文献知识挖掘系统 — 统一配置
===========================================================================
Agent 1 (抽取) + Agent 2 (分类) + Agent 3 (审查) 三Agent模式，使用 3 个独立 LLM API Key。
使用方式:
    from config import config
    model_name = config.llm.model
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ===========================================================================
# 项目根路径
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MID_DATA_DIR = PROJECT_ROOT / "mid_data"

# ===========================================================================
# LLM 配置
# ===========================================================================


@dataclass
class LLMConfig:
    """大语言模型调用参数 — 支持 3 层独立 API Key (Agent 1 抽取 + Agent 2 分类 + Agent 3 审查)"""

    model: str = os.environ.get("LLM_MODEL", os.environ.get("B2_LLM_MODEL", "deepseek-chat"))
    base_url: str = os.environ.get(
        "DEEPSEEK_BASE_URL",
        os.environ.get("B2_LLM_BASE_URL", os.environ.get("LOCAL_MODEL_ENDPOINT", "https://api.deepseek.com")),
    )
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: int = 180
    # Per-agent thinking mode: ALL fast for max throughput
    enable_thinking_extraction: bool = False     # Agent 1: fast extraction
    enable_thinking_classification: bool = False  # Agent 2: fast classification
    enable_thinking_reviewer: bool = False       # Agent 3: fast review
    reasoning_effort: str = "max"     # Reasoning depth when thinking enabled (min/low/medium/high/max)

    # 3 层独立 API Key: Agent1=抽取, Agent2=分类, Agent3=审查
    # 若未分层配置，回退到统一 DEEPSEEK_API_KEY / B2_LLM_API_KEY
    api_key_extraction: str = os.environ.get(
        "DEEPSEEK_API_KEY_EXTRACTION",
        os.environ.get("DEEPSEEK_API_KEY_L1", os.environ.get("B2_LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))),
    )
    api_key_classification: str = os.environ.get(
        "DEEPSEEK_API_KEY_CLASSIFICATION",
        os.environ.get("DEEPSEEK_API_KEY_L2", os.environ.get("B2_LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))),
    )
    api_key_reviewer: str = os.environ.get(
        "DEEPSEEK_API_KEY_REVIEWER",
        os.environ.get("DEEPSEEK_API_KEY_L3", os.environ.get("B2_LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))),
    )


# ===========================================================================
# 三Agent管道配置
# ===========================================================================


@dataclass
class PipelineConfig:
    """三Agent管道参数"""

    batch_size: int = 12          # Agent 2 分类时每批处理实体数
    max_l1_per_entity: int = 2    # 每个实体最多进入几个候选 L1

    # ── RAG Few-shot 配置 ──
    enable_rag: bool = False          # 是否启用动态 RAG few-shot
    rag_k_examples: int = 5           # 每个句子检索的示例数
    rag_similarity_threshold: float = 0.3  # 最低相似度阈值
    rag_model: str = "paraphrase-multilingual-MiniLM-L12-v2"  # embedding 模型

    # ── Voting 投票配置 (可选) ──
    enable_voting: bool = False       # 是否启用 Self-Consistency 投票
    voting_rounds: int = 3            # 投票轮数
    voting_threshold: int = 2         # 最少同意轮数 (>=2/3)
    voting_temperature: float = 0.3   # 投票时使用的 temperature
    voting_auto: bool = True          # 自动判断是否需要投票 (复杂句子)


# ===========================================================================
# L4 MicroMapping 术语底库 — 已改为动态 (由 DynamicTermDB 运行时积累)
# 参见: pipeline/dynamic_term_db.py
# 项目启动时为空库，运行过程中 Likert 5 分实体自动积累为术语底库。
# ===========================================================================


# ===========================================================================
# 总配置容器
# ===========================================================================


@dataclass
class Config:
    """系统总配置，聚合所有子配置"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # 运行时模式
    live_mode: bool = False
    verbose: bool = True

    # 路径
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    mid_data_dir: Path = MID_DATA_DIR

    # 动态术语库路径 (用于预加载 few-shot)
    dynamic_terms_path: str = ""


# 全局单例 — 可直接 import 使用
config = Config()


def build_config(live_mode: bool = False, verbose: bool = True) -> Config:
    """工厂函数：按模式更新全局配置并返回"""
    config.live_mode = live_mode
    config.verbose = verbose
    return config
