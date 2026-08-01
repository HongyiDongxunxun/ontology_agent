"""
src — 后向兼容重导出层 (已迁移至 pipeline/)
直接导入 pipeline.* 是推荐方式。
"""

from pipeline import *  # noqa: F401, F403

__version__ = "2.0.0"
