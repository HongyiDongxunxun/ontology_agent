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
from typing import Optional

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
        os.environ.get("B2_LLM_BASE_URL", os.environ.get("LOCAL_MODEL_ENDPOINT", "https://api.deepseek.com"))
    )
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60

    # 3 层独立 API Key: Agent1=抽取, Agent2=分类, Agent3=审查
    api_key_extraction: str = os.environ.get(
        "DEEPSEEK_API_KEY_EXTRACTION",
        os.environ.get("DEEPSEEK_API_KEY_L1", os.environ.get("B2_LLM_API_KEY", "")),
    )
    api_key_classification: str = os.environ.get(
        "DEEPSEEK_API_KEY_CLASSIFICATION",
        os.environ.get("DEEPSEEK_API_KEY_L2", os.environ.get("B2_LLM_API_KEY", "")),
    )
    api_key_reviewer: str = os.environ.get(
        "DEEPSEEK_API_KEY_REVIEWER",
        os.environ.get("DEEPSEEK_API_KEY_L3", os.environ.get("B2_LLM_API_KEY", "")),
    )

    # 回退: 若未分层配置，统一使用此 key
    api_key: str = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("B2_LLM_API_KEY", ""))


# ===========================================================================
# 三Agent管道配置
# ===========================================================================


@dataclass
class PipelineConfig:
    """三Agent管道参数"""
    batch_size: int = 12          # Agent 2 分类时每批处理实体数
    max_l1_per_entity: int = 2    # 每个实体最多进入几个候选 L1


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


# 全局单例 — 可直接 import 使用
config = Config()


def build_config(live_mode: bool = False, verbose: bool = True) -> Config:
    """工厂函数：按模式构建配置"""
    return Config(
        live_mode=live_mode,
        verbose=verbose,
    )
