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
class GoldRelation:
    """单条标注评价关系 — 预期正确的评价关系抽取结果 (V5.0)

    object 字段存被评价对象的实体 mention (标注时用名字而非ID,
    评估时由 metrics 解析预测 entity_id 到名字后匹配)。
    """

    subject: str                              # _paper_author | _cite[N] | <人名> | _unknown
    object: str                               # 被评价对象的实体 mention (原文精确短语)
    opinion: str                              # 评价表达 (完整片段)
    aspect: Optional[str] = None              # 评价方面; 无则 None
    evidence: str = ""                        # 评价依据文本片段

    def to_dict(self) -> dict:
        data = {
            "subject": self.subject,
            "object": self.object,
            "aspect": self.aspect,
            "opinion": self.opinion,
            "evidence": self.evidence,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GoldRelation":
        aspect = data.get("aspect")
        return cls(
            subject=data.get("subject", ""),
            object=data.get("object", ""),
            opinion=data.get("opinion", ""),
            aspect=None if aspect is None else str(aspect),
            evidence=data.get("evidence", ""),
        )


@dataclass
class GoldSentence:
    """单句标注 — 一句原始评价句 + 其中所有正确实体 + 评价关系 (V5.0)"""

    sentence_id: str                              # 句编号 (如 "reviewed_full_1::3")
    sentence: str                                 # 完整原文 (含上下文拼接)
    gold_entities: list[GoldEntity] = field(default_factory=list)
    gold_relations: list[GoldRelation] = field(default_factory=list)
    has_evaluation: Optional[bool] = None         # None = 由 gold_relations 是否为空推断

    def to_dict(self) -> dict:
        data = {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "gold_entities": [e.to_dict() for e in self.gold_entities],
            "gold_relations": [r.to_dict() for r in self.gold_relations],
        }
        if self.has_evaluation is not None:
            data["has_evaluation"] = self.has_evaluation
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GoldSentence":
        entities = [GoldEntity.from_dict(e) for e in data.get("gold_entities", [])]
        relations = [GoldRelation.from_dict(r) for r in data.get("gold_relations", [])]
        has_eval = data.get("has_evaluation")
        return cls(
            sentence_id=data.get("sentence_id", ""),
            sentence=data.get("sentence", ""),
            gold_entities=entities,
            gold_relations=relations,
            has_evaluation=None if has_eval is None else bool(has_eval),
        )

    def get_has_evaluation(self) -> bool:
        """句中是否存在评价: 显式标注优先, 否则由关系是否为空推断"""
        if self.has_evaluation is not None:
            return self.has_evaluation
        return bool(self.gold_relations)


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

    def relation_count(self) -> int:
        return sum(len(s.gold_relations) for s in self._sentences.values())

    def evaluated_sentence_count(self) -> int:
        """标注为含评价的句子数 (含情形B: 有评价但无合法关系)"""
        return sum(1 for s in self._sentences.values() if s.get_has_evaluation())

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "sentences": self.sentence_count(),
            "entities": self.entity_count(),
            "valid_entities": self.valid_entity_count(),
            "invalid_entities": self.entity_count() - self.valid_entity_count(),
            "l1_distribution": self.l1_distribution(),
            "l3_distribution": self.l3_distribution(),
            "relations": self.relation_count(),
            "evaluated_sentences": self.evaluated_sentence_count(),
        }

    # ── 内部方法 ───────────────────────────────────────────────

    @staticmethod
    def _entity_key(sentence_id: str, mention: str) -> str:
        return f"{sentence_id}::{mention}"
