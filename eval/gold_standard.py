"""
eval.gold_standard — 标注数据管理

GoldStandard 类管理标注数据 (ground truth) 的加载、验证、导出。
标注文件为 JSONL 格式，每行一个句子及其标注实体。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ===========================================================================
# 标注数据模型
# ===========================================================================


@dataclass
class GoldEntity:
    """单条标注实体 — 预期正确的抽取+分类结果"""

    mention: str                    # 原文实体提及 (原文精确短语)
    l1: str                        # 正确 L1: Agent|Artifact|Abstract|Event
    l2: str                        # 正确 L2 代码
    l3_type_code: str              # 正确 L3 type_code
    valid_entity: bool = True      # 是否有效实体
    normalized_name: str = ""      # 规范化名称 (默认同 mention)

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self.mention

    def to_dict(self) -> dict:
        return {
            "mention": self.mention,
            "normalized_name": self.normalized_name,
            "l1": self.l1,
            "l2": self.l2,
            "l3_type_code": self.l3_type_code,
            "valid_entity": self.valid_entity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldEntity":
        return cls(
            mention=data.get("mention", ""),
            normalized_name=data.get("normalized_name", ""),
            l1=data.get("l1", ""),
            l2=data.get("l2", ""),
            l3_type_code=data.get("l3_type_code", ""),
            valid_entity=data.get("valid_entity", True),
        )


@dataclass
class GoldSentence:
    """单句标注 — 一句原始评价句 + 其中所有正确实体"""

    sentence_id: str                              # 句编号 (如 "reviewed_full_1::3")
    sentence: str                                 # 完整原文 (含上下文拼接)
    gold_entities: list[GoldEntity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "gold_entities": [e.to_dict() for e in self.gold_entities],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldSentence":
        entities = [GoldEntity.from_dict(e) for e in data.get("gold_entities", [])]
        return cls(
            sentence_id=data.get("sentence_id", ""),
            sentence=data.get("sentence", ""),
            gold_entities=entities,
        )


# ===========================================================================
# GoldStandard — 标注数据集管理
# ===========================================================================


class GoldStandard:
    """标注数据集 (ground truth)，用于评估管道输出"""

    def __init__(self, name: str = "default"):
        self.name = name
        self._sentences: dict[str, GoldSentence] = {}
        self._entity_index: dict[str, GoldEntity] = {}

    # ── 加载/导出 ──────────────────────────────────────────────

    @classmethod
    def from_jsonl(cls, filepath: str | Path) -> "GoldStandard":
        """从 JSONL 加载标注数据。每行格式: {sentence_id, sentence, gold_entities: [...]}"""
        gs = cls(name=Path(filepath).stem)
        path = Path(filepath)
        if not path.exists():
            print(f"[GoldStandard] 文件不存在: {path}")
            return gs

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sentence = GoldSentence.from_dict(data)
                    gs.add_sentence(sentence)
                except json.JSONDecodeError as e:
                    print(f"[GoldStandard] JSON 解析错误 (行 {line_no}): {e}")
        return gs

    def to_jsonl(self, filepath: str | Path) -> None:
        """导出标注数据到 JSONL"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sentence in self._sentences.values():
                f.write(json.dumps(sentence.to_dict(), ensure_ascii=False) + "\n")
        print(f"[GoldStandard] 已导出 {len(self._sentences)} 句 -> {path.resolve()}")

    # ── 增删查改 ──────────────────────────────────────────────

    def add_sentence(self, sentence: GoldSentence) -> None:
        self._sentences[sentence.sentence_id] = sentence
        for entity in sentence.gold_entities:
            key = self._entity_key(sentence.sentence_id, entity.mention)
            self._entity_index[key] = entity

    def get_sentence(self, sentence_id: str) -> Optional[GoldSentence]:
        return self._sentences.get(sentence_id)

    def remove_sentence(self, sentence_id: str) -> None:
        if sentence_id in self._sentences:
            sentence = self._sentences.pop(sentence_id)
            for entity in sentence.gold_entities:
                key = self._entity_key(sentence_id, entity.mention)
                self._entity_index.pop(key, None)

    # ── 查询 ───────────────────────────────────────────────────

    def get_entity(self, sentence_id: str, mention: str) -> Optional[GoldEntity]:
        """根据句ID和mention精确查找标注实体"""
        key = self._entity_key(sentence_id, mention)
        return self._entity_index.get(key)

    def find_entity_by_mention(
        self, sentence_id: str, mention: str
    ) -> Optional[GoldEntity]:
        """宽松查找：先精确，再忽略大小写，再归一化"""
        # 精确匹配
        key = self._entity_key(sentence_id, mention)
        if key in self._entity_index:
            return self._entity_index[key]

        # 忽略大小写
        mention_lower = mention.lower().strip()
        for stored_key, entity in self._entity_index.items():
            if stored_key.lower() == f"{sentence_id}::{mention_lower}":
                return entity

        # 子串包含 (高召回)
        for stored_key, entity in self._entity_index.items():
            _sid, stored_mention = stored_key.split("::", 1)
            if _sid == sentence_id:
                sl = stored_mention.lower().strip()
                if sl in mention_lower or mention_lower in sl:
                    return entity
        return None

    # ── 统计 ───────────────────────────────────────────────────

    def sentence_count(self) -> int:
        return len(self._sentences)

    def entity_count(self) -> int:
        return len(self._entity_index)

    def valid_entity_count(self) -> int:
        return sum(1 for e in self._entity_index.values() if e.valid_entity)

    def l1_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for e in self._entity_index.values():
            if e.valid_entity and e.l1:
                dist[e.l1] = dist.get(e.l1, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def l3_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for e in self._entity_index.values():
            if e.valid_entity and e.l3_type_code:
                dist[e.l3_type_code] = dist.get(e.l3_type_code, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "sentences": self.sentence_count(),
            "entities": self.entity_count(),
            "valid_entities": self.valid_entity_count(),
            "invalid_entities": self.entity_count() - self.valid_entity_count(),
            "l1_distribution": self.l1_distribution(),
            "l3_distribution": self.l3_distribution(),
        }

    # ── 内部方法 ───────────────────────────────────────────────

    @staticmethod
    def _entity_key(sentence_id: str, mention: str) -> str:
        return f"{sentence_id}::{mention}"
