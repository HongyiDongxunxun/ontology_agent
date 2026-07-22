"""
pipeline.dynamic_term_db — 线程安全的动态 L4 MicroMapping 术语底库
在项目运行过程中不断积累 Likert 5 分（最高置信度）实体作为术语底库。
Agent 2 每次构建 L4 匹配 prompt 时从该库实时读取最新快照。

V4.3: 增加 JSON 预加载 + 字符 trigram 相似度检索 (用于 RAG few-shot)
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path


class DynamicTermDB:
    """线程安全的动态 L4 术语底库，去重，支持并发读写 + 相似度检索。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._term_set: set[str] = set()
        self._terms: list[tuple[str, str]] = []
        self._added_count = 0
        self._skipped_count = 0

        # ── 字符 trigram 倒排索引 (用于快速相似度检索) ──
        self._trigram_index: dict[str, set[int]] = defaultdict(set)
        self._index_built = False

    # ═══════════════════════════════════════════════════════════
    # 基本操作
    # ═══════════════════════════════════════════════════════════

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
            idx = len(self._terms)
            self._terms.append((term, l3_type_code))
            self._added_count += 1
            # 增量更新 trigram index
            for trigram in self._get_trigrams(term):
                self._trigram_index[trigram].add(idx)
            self._index_built = True
            return True

    def get_micro_terms(self) -> list[tuple[str, str]]:
        """获取当前术语快照，返回 (term, type_code) 列表"""
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

    # ═══════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════
    # 从 JSON 预加载 (用于启动时注入已有术语库)
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def from_json(cls, filepath: str, max_terms: int = 0) -> "DynamicTermDB":
        """
        从 JSON 文件批量加载术语。
        文件格式: {"terms": [{"term": "...", "type_code": "..."}, ...]}
        max_terms: 最大加载条数 (0=全部), 用于限制内存。

        返回: DynamicTermDB 实例
        """
        db = cls()
        path = Path(filepath)
        if not path.exists():
            print(f"[DynamicTermDB] 文件不存在: {filepath}")
            return db

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        terms_list = data.get("terms", [])
        if not terms_list:
            print(f"[DynamicTermDB] 文件中无 terms 数据: {filepath}")
            return db

        limit = min(max_terms, len(terms_list)) if max_terms > 0 else len(terms_list)
        print(f"[DynamicTermDB] 正在加载 {limit:,} / {len(terms_list):,} 条术语...")

        # 批量添加 (不加锁, 构造阶段单线程)
        for item in terms_list[:limit]:
            term = item.get("term", "")
            type_code = item.get("type_code", "")
            if not term or not type_code:
                continue
            key = f"{term}::{type_code}"
            if key in db._term_set:
                db._skipped_count += 1
                continue
            db._term_set.add(key)
            db._terms.append((term, type_code))
            db._added_count += 1

        # 构建 trigram 索引
        db._build_trigram_index()
        print(f"[DynamicTermDB] 加载完成: {db._added_count:,} 条 (跳过 {db._skipped_count:,} 条重复)")
        return db

    # ═══════════════════════════════════════════════════════════
    # 字符 trigram 相似度检索 (用于 RAG few-shot)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _get_trigrams(text: str) -> set[str]:
        """提取字符 trigram (用于中文等 CJK 语言, 逐字符切分)"""
        text = text.lower().strip()
        if len(text) < 3:
            return {text}
        trigrams: set[str] = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i:i+3])
        return trigrams

    def _build_trigram_index(self) -> None:
        """(重新)构建 trigram 倒排索引"""
        self._trigram_index = defaultdict(set)
        for idx, (term, _) in enumerate(self._terms):
            for trigram in self._get_trigrams(term):
                self._trigram_index[trigram].add(idx)
        self._index_built = True

    def search_similar(self, query: str, k: int = 5, min_overlap: int = 1) -> list[tuple[str, str, float]]:
        """
        使用字符 trigram 重叠检索与 query 最相似的术语。

        参数:
        - query: 查询字符串 (实体 mention)
        - k: 返回 Top-K 个结果
        - min_overlap: 最少共同 trigram 数

        返回: [(term, type_code, jaccard_similarity), ...]  按相似度降序
        """
        with self._lock:
            if not self._index_built or not self._terms:
                return []

            query_trigrams = self._get_trigrams(query)
            if not query_trigrams:
                return []

            # 收集候选项 (有至少 min_overlap 个共同 trigram)
            candidate_scores: dict[int, int] = defaultdict(int)  # idx -> overlap count
            for trigram in query_trigrams:
                for idx in self._trigram_index.get(trigram, set()):
                    candidate_scores[idx] += 1

            if not candidate_scores:
                return []

            # 精确匹配优先
            query_lower = query.lower().strip()
            exact_match = None
            for idx, (term, type_code) in enumerate(self._terms):
                if term.lower().strip() == query_lower:
                    exact_match = (term, type_code, 1.0)
                    break

            # 计算 Jaccard 相似度
            scored: list[tuple[float, str, str]] = []
            for idx, overlap in candidate_scores.items():
                if overlap < min_overlap:
                    continue
                term, type_code = self._terms[idx]
                term_trigrams = self._get_trigrams(term)
                union = len(query_trigrams | term_trigrams)
                jaccard = overlap / max(union, 1)
                scored.append((jaccard, term, type_code))

            # 按 Jaccard 降序
            scored.sort(key=lambda x: -x[0])
            results = [(term, tc, round(score, 3)) for score, term, tc in scored[:k]]

            # 如果精确匹配存在且不在结果中, 插入到首位
            if exact_match and exact_match not in [(r[0], r[1], r[2]) for r in results]:
                results.insert(0, exact_match)

            return results[:k]

    def get_few_shot_hints(self, entity_mention: str, k: int = 3) -> str:
        """
        为实体 mention 获取 few-shot 提示文本。

        返回格式:
          「潘光旦」→ scholar (知识生产者)
          「阮冈纳赞」→ scholar (知识生产者)
          「中国图书馆学会」→ professional (职业共同体组织)
        """
        similar = self.search_similar(entity_mention, k=k, min_overlap=1)
        if not similar:
            return ""

        # 从 pipeline.taxonomy 获取标签 (延迟导入避免循环依赖)
        try:
            from .taxonomy import L3_LABELS
        except ImportError:
            L3_LABELS = {}

        lines: list[str] = []
        for term, type_code, score in similar:
            label = L3_LABELS.get(type_code, type_code)
            lines.append(f"  「{term}」→ {type_code} ({label}) [相似度:{score:.0%}]")

        return "\n".join(lines)
