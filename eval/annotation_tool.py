"""
eval.annotation_tool — 命令行交互式标注工具

加载 pipeline 的输出 (JSONL) 或原始句子 (JSON),
逐条展示并允许人工标注正确的 L1/L2/L3/valid_entity。

使用方式:
    # 从 pipeline 输出标注
    python -m eval.annotation_tool --input output/reviewed_full_1_result.jsonl

    # 从原始输入标注 (需要先展示句子, 手动输入实体)
    python -m eval.annotation_tool --input input/reviewed_full_1.json --mode raw

    # 指定已有标注文件 (续标)
    python -m eval.annotation_tool --input output/ --gold eval/gold_data.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# 添加项目根路径到 sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.taxonomy import (
    TAXONOMY_HIERARCHY,
    L2_LABELS,
    L3_LABELS,
    get_l1_options,
    get_l2_options,
    get_l3_options,
)

from .gold_standard import GoldStandard, GoldSentence, GoldEntity


# ===========================================================================
# 标注器
# ===========================================================================


class AnnotationTool:
    """命令行交互式标注工具"""

    def __init__(self, gold_path: str = ""):
        self.gold = GoldStandard(name="annotation")
        self.gold_path = gold_path
        self._session_count = 0

        # 预加载已有标注
        if gold_path and Path(gold_path).exists():
            self.gold = GoldStandard.from_jsonl(gold_path)
            print(f"[标注工具] 已加载 {self.gold.sentence_count()} 句标注数据")

    def run_from_pipeline_output(self, jsonl_files: list[str]) -> None:
        """
        从 pipeline 输出 JSONL 加载实体，逐条确认/修正分类。

        工作流:
        1. 读取 JSONL, 按句分组展示
        2. 对每句的每个实体, 显示 pipeline 分类结果
        3. 用户确认正确 (y) 或输入修正 (n 进入修正模式)
        4. 最终输出 gold JSONL
        """
        # 按句分组
        sentences: dict[str, dict] = {}  # {sid: {sentence: str, entities: [...]}}
        for filepath in jsonl_files:
            self._load_pipeline_jsonl(filepath, sentences)

        print(f"\n[标注工具] 共加载 {len(sentences)} 句, "
              f"{sum(len(s['entities']) for s in sentences.values())} 个实体\n")
        print("标注说明:")
        print("  y / Enter  = 确认 pipeline 分类正确")
        print("  n          = 进入修正模式")
        print("  s          = 跳过此实体")
        print("  q          = 保存并退出")
        print()

        processed = 0
        for sid in sorted(sentences.keys(), key=lambda k: int(k) if k.isdigit() else k):
            sent_data = sentences[sid]
            sentence_text = sent_data["sentence"]
            entities = sent_data["entities"]

            if not entities:
                continue

            gold_entities: list[GoldEntity] = []

            print(f"\n{'─'*70}")
            # 截断过长句子
            display_text = sentence_text[:200] + "..." if len(sentence_text) > 200 else sentence_text
            print(f"  📄 句 [{sid}]: {display_text}")
            print(f"{'─'*70}")

            for i, entity in enumerate(entities):
                processed += 1
                mention = entity.get("entity") or entity.get("mention") or "?"
                l1 = entity.get("l1", "")
                l2 = entity.get("l2", "")
                l3 = entity.get("l3_type_code", "")
                valid = entity.get("valid_entity", True)

                print(f"\n  [{i+1}/{len(entities)}] 实体: 「{mention}」")
                print(f"       Pipeline 分类: L1={l1}  L2={l2}  L3={l3}  "
                      f"valid={'✓' if valid else '✗'}")

                while True:
                    choice = input("       确认? [y(确认)/n(修正)/s(跳过)/q(退出)]: ").strip().lower()

                    if choice in ("y", ""):
                        gold_entities.append(GoldEntity(
                            mention=mention,
                            normalized_name=entity.get("normalized_name", mention),
                            l1=l1,
                            l2=l2,
                            l3_type_code=l3,
                            valid_entity=valid,
                        ))
                        break

                    elif choice == "n":
                        # 修正模式
                        corrected = self._interactive_correction(mention, l1, l2, l3, valid)
                        if corrected:
                            gold_entities.append(corrected)
                        break

                    elif choice == "s":
                        print("       ⏭ 已跳过")
                        break

                    elif choice == "q":
                        self._save_and_exit(sid, gold_entities)
                        return

                    else:
                        print("       ❓ 无效输入, 请重新选择")

            # 保存当前句的标注
            if gold_entities:
                self.gold.add_sentence(GoldSentence(
                    sentence_id=sid,
                    sentence=sentence_text,
                    gold_entities=gold_entities,
                ))

            # 每 10 句自动保存
            if processed % 50 == 0:
                self._auto_save()

        self._auto_save()
        print(f"\n✅ 标注完成! 共标注 {self.gold.sentence_count()} 句, "
              f"{self.gold.entity_count()} 个实体")

    def _load_pipeline_jsonl(self, filepath: str, sentences: dict) -> None:
        """加载 pipeline 输出的 JSONL 文件"""
        path = Path(filepath)
        if not path.exists():
            print(f"[警告] 文件不存在: {filepath}")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    sid = item.get("sentence_id", "")
                    if sid not in sentences:
                        sentences[sid] = {
                            "sentence": item.get("sentence", ""),
                            "entities": [],
                        }
                    sentences[sid]["entities"].append(item)
                except json.JSONDecodeError:
                    continue

    def _interactive_correction(
        self, mention: str, cur_l1: str, cur_l2: str, cur_l3: str, cur_valid: bool
    ) -> Optional[GoldEntity]:
        """交互式修正一个实体的分类"""
        print(f"\n       === 修正模式 ===")
        print(f"       当前: L1={cur_l1}, L2={cur_l2}, L3={cur_l3}, valid={cur_valid}")

        # 选 L1
        l1_options = get_l1_options()
        print(f"       L1 选项: {', '.join(l1_options)}")
        l1 = input(f"       输入 L1 [默认={cur_l1}]: ").strip()
        if not l1:
            l1 = cur_l1
        if l1 not in l1_options:
            print(f"       ❌ 无效 L1: {l1}, 使用默认值")
            l1 = cur_l1

        # 选 L2
        l2_options = get_l2_options(l1)
        print(f"       L2 选项: {', '.join(l2_options)}")
        l2 = input(f"       输入 L2 [默认={cur_l2}]: ").strip()
        if not l2:
            l2 = cur_l2
        if l2 not in l2_options:
            print(f"       ❌ 无效 L2: {l2} (对 L1={l1})")
            l2 = l2_options[0] if l2_options else cur_l2

        # 选 L3
        l3_options = get_l3_options(l1, l2)
        l3_display = ", ".join(f"{c}({L3_LABELS.get(c, c)})" for c in l3_options)
        print(f"       L3 选项: {l3_display}")
        l3 = input(f"       输入 L3 type_code [默认={cur_l3}]: ").strip()
        if not l3:
            l3 = cur_l3
        if l3 not in l3_options:
            print(f"       ❌ 无效 L3: {l3} (对 L1={l1}, L2={l2})")
            l3 = l3_options[0] if l3_options else cur_l3

        # valid_entity
        valid_str = input(f"       valid_entity? [y/n, 默认={'y' if cur_valid else 'n'}]: ").strip().lower()
        if valid_str == "n":
            valid = False
        elif valid_str == "y":
            valid = True
        else:
            valid = cur_valid

        print(f"       ✅ 修正为: L1={l1}, L2={l2}, L3={l3}, valid={valid}")
        return GoldEntity(
            mention=mention,
            l1=l1,
            l2=l2,
            l3_type_code=l3,
            valid_entity=valid,
        )

    def _auto_save(self) -> None:
        """自动保存到 gold_path"""
        if self.gold_path:
            self.gold.to_jsonl(self.gold_path)
            print(f"       💾 已自动保存 ({self.gold.sentence_count()} 句, "
                  f"{self.gold.entity_count()} 实体)")

    def _save_and_exit(self, current_sid: str, pending: list[GoldEntity]) -> None:
        """保存并退出"""
        if pending:
            self.gold.add_sentence(GoldSentence(
                sentence_id=current_sid,
                sentence="",
                gold_entities=pending,
            ))
        self._auto_save()
        print(f"\n🛑 已退出。标注进度已保存。")


# ===========================================================================
# 入口
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ontology_Agent 实体标注工具 (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Pipeline 输出 JSONL 文件或目录 (多个文件用逗号分隔)",
    )
    parser.add_argument(
        "--gold", "-g", type=str, default="eval/gold_data.jsonl",
        help="标注数据输出路径 (默认: eval/gold_data.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从已有标注文件续标 (跳过已标注的句子)",
    )
    args = parser.parse_args()

    # 解析输入文件
    input_paths: list[str] = []
    for part in args.input.split(","):
        part = part.strip()
        p = Path(part)
        if p.is_dir():
            input_paths.extend(str(f) for f in p.glob("*_result.jsonl"))
        elif p.exists():
            input_paths.append(str(p))
        else:
            print(f"[警告] 路径不存在: {part}")

    if not input_paths:
        print("[错误] 无有效的输入文件")
        return 1

    print(f"[标注工具] 输入文件: {len(input_paths)} 个")
    for p in input_paths:
        print(f"  - {p}")
    print(f"[标注工具] 标注输出: {args.gold}")

    # 确保输出目录存在
    Path(args.gold).parent.mkdir(parents=True, exist_ok=True)

    # 运行标注
    tool = AnnotationTool(gold_path=args.gold)
    tool.run_from_pipeline_output(input_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
