"""
pipeline.dynamic_term_db — 线程安全的动态 L4 MicroMapping 术语底库
V4.3: 增加 JSON 预加载 + 字符 trigram 相似度检索 (用于 RAG few-shot)
"""
from __future__ import annotations
import json, threading
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
        self._trigram_index: dict[str, set[int]] = defaultdict(set)
        self._index_built = False

    def add(self, term: str, l3_type_code: str) -> bool:
        if not term or not l3_type_code: return False
        key = f"{term}::{l3_type_code}"
        with self._lock:
            if key in self._term_set:
                self._skipped_count += 1; return False
            self._term_set.add(key)
            idx = len(self._terms); self._terms.append((term, l3_type_code))
            self._added_count += 1
            for tg in self._get_trigrams(term): self._trigram_index[tg].add(idx)
            self._index_built = True
            return True

    def get_micro_terms(self) -> list[tuple[str, str]]:
        with self._lock: return list(self._terms)

    def size(self) -> int:
        with self._lock: return len(self._terms)

    def get_stats(self) -> dict:
        with self._lock: return {"total_terms": len(self._terms), "added": self._added_count, "skipped_duplicates": self._skipped_count}

    def export(self, filepath: str) -> None:
        with self._lock:
            data = {"total_terms": len(self._terms), "added": self._added_count, "skipped_duplicates": self._skipped_count, "terms": [{"term": t, "type_code": tc} for t, tc in self._terms]}
        path = Path(filepath); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, filepath: str, max_terms: int = 0) -> "DynamicTermDB":
        db = cls(); path = Path(filepath)
        if not path.exists(): print(f"[DynamicTermDB] file not found: {filepath}"); return db
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        terms_list = data.get("terms", [])
        if not terms_list: return db
        limit = min(max_terms, len(terms_list)) if max_terms > 0 else len(terms_list)
        print(f"[DynamicTermDB] loading {limit:,} / {len(terms_list):,} terms...")
        for item in terms_list[:limit]:
            t, tc = item.get("term", ""), item.get("type_code", "")
            if not t or not tc: continue
            key = f"{t}::{tc}"
            if key in db._term_set: db._skipped_count += 1; continue
            db._term_set.add(key); db._terms.append((t, tc)); db._added_count += 1
        db._build_trigram_index()
        print(f"[DynamicTermDB] loaded: {db._added_count:,} terms (skipped {db._skipped_count:,})")
        return db

    @staticmethod
    def _get_trigrams(text: str) -> set[str]:
        text = text.lower().strip()
        if len(text) < 3: return {text}
        return {text[i:i+3] for i in range(len(text) - 2)}

    def _build_trigram_index(self) -> None:
        self._trigram_index = defaultdict(set)
        for idx, (term, _) in enumerate(self._terms):
            for tg in self._get_trigrams(term): self._trigram_index[tg].add(idx)
        self._index_built = True

    def search_similar(self, query: str, k: int = 5, min_overlap: int = 1) -> list[tuple[str, str, float]]:
        with self._lock:
            if not self._index_built or not self._terms: return []
            qts = self._get_trigrams(query)
            if not qts: return []
            scores: dict[int, int] = defaultdict(int)
            for tg in qts:
                for idx in self._trigram_index.get(tg, set()): scores[idx] += 1
            if not scores: return []
            scored = []
            for idx, overlap in scores.items():
                if overlap < min_overlap: continue
                term, tc = self._terms[idx]
                union = len(qts | self._get_trigrams(term))
                scored.append((overlap / max(union, 1), term, tc))
            scored.sort(key=lambda x: -x[0])
            return [(term, tc, round(s, 3)) for s, term, tc in scored[:k]]

    def get_few_shot_hints(self, entity_mention: str, k: int = 3) -> str:
        similar = self.search_similar(entity_mention, k=k, min_overlap=1)
        if not similar: return ""
        try:
            from .taxonomy import L3_LABELS
        except ImportError:
            L3_LABELS = {}
        lines = []
        for term, type_code, score in similar:
            label = L3_LABELS.get(type_code, type_code)
            lines.append(f"  [{term}] -> {type_code} ({label}) [sim:{score:.0%}]")
        return "\n".join(lines)
