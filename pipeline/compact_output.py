"""
pipeline.compact_output — 紧凑输出格式的公共解析辅助。

五个 Agent 的 LLM 输出由「长字段名 + 全字段」改为「位置元组 + 稀疏字段」。
本模块只放与模型无关的本地派生逻辑：证据片段回填、标志位解析。
"""

from __future__ import annotations

from typing import Any


def derive_evidence(sentence: str, anchor: str, extra: str = "", window: int = 12) -> str:
    """从原句截取最小证据片段，替代 LLM 重复输出 evidence。

    - anchor 找不到时返回空串，由调用方回退。
    - extra 给出时，取覆盖 anchor 与 extra 的最小区间。
    - 否则取 anchor ± window 字的窗口。
    """
    if not sentence or not anchor:
        return ""
    start = sentence.find(anchor)
    if start < 0:
        return ""
    end = start + len(anchor)
    if extra:
        pos = sentence.find(extra, max(0, start - 60))
        if pos >= 0:
            lo, hi = min(start, pos), max(end, pos + len(extra))
            if hi - lo <= len(anchor) + len(extra) + 40:
                return sentence[lo:hi]
    lo = max(0, start - window)
    hi = min(len(sentence), end + window)
    return sentence[lo:hi]


def parse_flags(value: Any) -> tuple[bool, bool]:
    """解析实体标志位串：'?' 不确定，'!' 非具体实体（上游强制保留）。"""
    flags = value if isinstance(value, str) else ""
    return ("?" in flags, "!" in flags)


def as_tuple(item: Any) -> list:
    """把 LLM 返回的单条记录统一成列表；非列表返回空列表。"""
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return list(item)
    return []
