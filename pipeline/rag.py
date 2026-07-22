"""
pipeline.rag — Dynamic RAG + Few-shot 动态词表增强模块
基于原始 Ontology_Agent 的 DynamicTermDB 机制扩展：
1. 从 gold_flat 标注数据构建语义检索索引
2. 对每个输入句子检索最相似的标注示例
3. 动态注入 Agent prompt 作为 few-shot demonstrations

支持两种检索模式:
- TF-IDF (默认, 无需下载模型, 对中文友好)
- Sentence-Transformers (需下载模型, 语义匹配更好)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class RAGExampleDB:
    """
    RAG 示例库 — 从 gold_flat 标注数据构建语义检索索引。

    默认使用 TF-IDF (字符级 n-gram) 进行检索，
    无需下载外部模型。也可切换到 sentence-transformers。
    """

    def __init__(
        self,
        gold_flat_dir: str = "",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str = "",
        use_tfidf: bool = True,
    ):
        self.gold_flat_dir = gold_flat_dir
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_tfidf = use_tfidf
        self._model = None
        self._vectorizer = None
        self._embeddings: Optional[np.ndarray] = None
        self._examples: list[dict] = []
        self._sentence_texts: list[str] = []
        self._built = False

    # ── Build Index ──────────────────────────────────────────

    def build(self, force: bool = False) -> int:
        if self._built and not force:
            return len(self._examples)

        cache_file = ""
        if self.cache_dir:
            suffix = "_tfidf" if self.use_tfidf else "_sbert"
            cache_file = os.path.join(self.cache_dir, f"rag_index{suffix}.npz")

        if cache_file and os.path.exists(cache_file) and not force:
            return self._load_cache(cache_file)

        self._build_from_gold_flat()

        if cache_file:
            self._save_cache(cache_file)

        self._built = True
        return len(self._examples)

    def _build_from_gold_flat(self) -> None:
        if not self.gold_flat_dir or not os.path.isdir(self.gold_flat_dir):
            print("[RAG] gold_flat dir not found, RAG disabled")
            return

        gold_path = Path(self.gold_flat_dir)
        raw_examples: list[dict] = []
        sentences: list[str] = []

        for fpath in sorted(gold_path.glob("*.jsonl")):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sentence = entry.get("sentence", "")
                    entity = entry.get("entity", "")
                    etype = entry.get("type", "")
                    if not sentence or not entity or not etype:
                        continue
                    raw_examples.append(entry)
                    sentences.append(sentence)

        if not sentences:
            print("[RAG] No examples found in gold_flat")
            return

        print(f"[RAG] Building {'TF-IDF' if self.use_tfidf else 'SBERT'} index from {len(sentences)} examples...")

        if self.use_tfidf:
            self._embeddings = self._build_tfidf(sentences)
        else:
            self._embeddings = self._build_sbert(sentences)

        self._examples = raw_examples
        self._sentence_texts = sentences
        print(f"[RAG] Index built: {len(self._examples)} examples, dim={self._embeddings.shape[1]}")

    def _build_tfidf(self, sentences: list[str]) -> np.ndarray:
        """构建字符级 TF-IDF 矩阵 (n-gram 1-3)"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            max_features=10000,
            lowercase=True,
        )
        matrix = self._vectorizer.fit_transform(sentences)
        # Normalize to unit vectors for cosine similarity
        from sklearn.preprocessing import normalize
        return normalize(matrix, norm="l2").toarray().astype(np.float32)

    def _build_sbert(self, sentences: list[str]) -> np.ndarray:
        """构建 SBERT 向量索引"""
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(
            sentences,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self._built or self._embeddings is None or len(self._embeddings) == 0:
            return []

        if self.use_tfidf and self._vectorizer is not None:
            from sklearn.preprocessing import normalize
            q_vec = normalize(self._vectorizer.transform([query]), norm="l2").toarray().astype(np.float32)
        else:
            from sentence_transformers import SentenceTransformer
            if self._model is None:
                self._model = SentenceTransformer(self.model_name)
            q_vec = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)

        scores = np.dot(self._embeddings, q_vec[0])
        top_k_idx = np.argsort(scores)[-k:][::-1]

        results: list[dict] = []
        for idx in top_k_idx:
            if scores[idx] < 0.15:  # Minimum similarity
                continue
            ex = self._examples[idx]
            results.append({
                "sentence": ex.get("sentence", ""),
                "entity": ex.get("entity", ""),
                "type": ex.get("type", ""),
                "score": float(scores[idx]),
            })
        return results

    def build_few_shot_text(
        self,
        query: str,
        k: int = 5,
        max_examples: int = 5,
        exclude_types: Optional[set] = None,
    ) -> str:
        """
        构建 few-shot 文本片段，可直接注入 prompt。

        Args:
            query: 查询句子
            k: 检索候选数
            max_examples: 最终返回的示例数上限
            exclude_types: 排除的实体类型（如 "other"）

        Returns:
            格式化的 few-shot 文本，如：
            "示例1: 句子: ... 实体: ... 类型: ..."
            若无结果返回空字符串
        """
        results = self.search(query, k=k)
        if not results:
            return ""

        # Filter and deduplicate by entity
        seen_entities: set[str] = set()
        filtered: list[dict] = []
        for r in results:
            entity = r["entity"].lower().strip()
            etype = r["type"]
            if entity in seen_entities:
                continue
            if exclude_types and etype in exclude_types:
                continue
            seen_entities.add(entity)
            filtered.append(r)
            if len(filtered) >= max_examples:
                break

        if not filtered:
            return ""

        lines = ["\n## Few-shot 示例 (从相似标注句中检索):"]
        for i, ex in enumerate(filtered, 1):
            lines.append(
                f"{i}. 句子: \"{ex['sentence'][:150]}\"\n"
                f"   实体: \"{ex['entity']}\" → 类型: `{ex['type']}`"
                f"  (相似度: {ex['score']:.2f})"
            )

        return "\n".join(lines)

    # ── Cache ─────────────────────────────────────────────────

    def _save_cache(self, cache_file: str) -> None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(
            cache_file,
            embeddings=self._embeddings,
            examples=np.array(self._examples, dtype=object),
            sentences=np.array(self._sentence_texts, dtype=object),
        )

    def _load_cache(self, cache_file: str) -> int:
        data = np.load(cache_file, allow_pickle=True)
        self._embeddings = data["embeddings"]
        self._examples = list(data["examples"])
        self._sentence_texts = list(data["sentences"])
        self._built = True
        print(f"[RAG] Loaded {len(self._examples)} examples from cache")
        return len(self._examples)

    # ── Stats ─────────────────────────────────────────────────

    def size(self) -> int:
        return len(self._examples)

    def is_ready(self) -> bool:
        return self._built and len(self._examples) > 0


# ── Global singleton ──────────────────────────────────────────

_rag_db: Optional[RAGExampleDB] = None


def get_rag_db(
    gold_flat_dir: str = "",
    cache_dir: str = "",
    force_rebuild: bool = False,
) -> RAGExampleDB:
    """获取全局 RAG 示例库单例"""
    global _rag_db
    if _rag_db is None or force_rebuild:
        _rag_db = RAGExampleDB(
            gold_flat_dir=gold_flat_dir or "",
            cache_dir=cache_dir or "",
        )
    return _rag_db


def build_rag_index(
    gold_flat_dir: str,
    cache_dir: str = "",
) -> RAGExampleDB:
    """构建并返回 RAG 索引"""
    db = get_rag_db(gold_flat_dir, cache_dir)
    db.build()
    return db
