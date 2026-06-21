"""
pipeline.dynamic_term_db — 线程安全的动态 L4 MicroMapping 术语底库
在项目运行过程中不断积累 Likert 5 分（最高置信度）实体作为术语底库。
Agent 2 每次构建 L4 匹配 prompt 时从该库实时读取最新快照。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class DynamicTermDB:
    """线程安全的动态 L4 术语底库，去重，支持并发读写。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._term_set: set[str] = set()
        self._terms: list[tuple[str, str]] = []
        self._added_count = 0
        self._skipped_count = 0

    def add(self, term: str, l3_type_code: str) -> bool:
        """添加术语，返回 True=新增, False=重复跳过"""
        if not term or not l3_type_code:
            return False
        key = f"{term}::{l3_type_code}"
        with self._lock:
            if key in self._term_set:
                self._skipped_count += 1
                return False
            self._term_set.add(key)
            self._terms.append((term, l3_type_code))
            self._added_count += 1
            return True

    def get_micro_terms(self) -> list[tuple[str, str]]:
        """获取当前术语快照，返回 (term, type_code) 列表 — 兼容 Agent 2 原格式"""
        with self._lock:
            return list(self._terms)

    def size(self) -> int:
        with self._lock:
            return len(self._terms)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_terms": len(self._terms),
                "added": self._added_count,
                "skipped_duplicates": self._skipped_count,
            }

    def export(self, filepath: str) -> None:
        """持久化当前术语库到 JSON 文件"""
        with self._lock:
            data = {
                "total_terms": len(self._terms),
                "added": self._added_count,
                "skipped_duplicates": self._skipped_count,
                "terms": [
                    {"term": t, "type_code": tc} for t, tc in self._terms
                ],
            }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
