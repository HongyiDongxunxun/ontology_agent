# -*- coding: utf-8 -*-
"""
按 Silver 金标准过滤 Test 句子组。

流程：
1. 读取 gold_files.txt 每一行，提取其中的数字记作 num。
2. 读取 Test:  review_sentence/reviewed_full_{num}.json
3. 读取 Silver: gold_flat/reviewed_full_{num}_result_gold_flat.jsonl
   收集 Silver 中的 sentence_id 集合，Test 的键即为 sentence_id，
   删除 Silver 中不存在的 sentence_id 对应的句子组。
4. 保留原 JSON 格式输出到 input 文件夹。
"""
import json
import os
import re

GOLD_FILES = r"D:\codes\AE-ablation\gold_files.txt"
TEST_DIR = r"D:\codes\AcademicEvaluation\data\output\review_sentence"
SILVER_DIR = r"D:\codes\AE-goldline\output\gold_flat"
OUT_DIR = r"d:\codes\ontology_agent\input"


def read_text(path):
    """自动识别 UTF-8/UTF-16(BOM) 编码读取文本。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_num(line):
    m = re.search(r"(\d+)", line)
    return m.group(1) if m else None


def load_silver_ids(path):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("sentence_id")
            if sid is not None:
                ids.add(str(sid))
    return ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [l.strip() for l in read_text(GOLD_FILES).splitlines() if l.strip()]

    total_files = 0
    total_kept = 0
    total_removed = 0
    missing = []

    for line in lines:
        num = extract_num(line)
        if not num:
            continue
        test_path = os.path.join(TEST_DIR, f"reviewed_full_{num}.json")
        silver_path = os.path.join(SILVER_DIR, f"reviewed_full_{num}_result_gold_flat.jsonl")
        if not os.path.exists(test_path):
            missing.append(f"[TEST ] {test_path}")
            continue
        if not os.path.exists(silver_path):
            missing.append(f"[SILVER] {silver_path}")
            continue

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        silver_ids = load_silver_ids(silver_path)

        filtered = {}
        kept = 0
        removed = 0
        for key, group in test_data.items():
            if str(key) in silver_ids:
                filtered[key] = group
                kept += 1
            else:
                removed += 1

        out_path = os.path.join(OUT_DIR, f"reviewed_full_{num}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=4)

        total_files += 1
        total_kept += kept
        total_removed += removed
        if removed:
            print(f"num={num}: kept={kept}, removed={removed}")

    print("\n=== 汇总 ===")
    print(f"处理文件数: {total_files}")
    print(f"保留句子组: {total_kept}")
    print(f"删除句子组: {total_removed}")
    if missing:
        print("\n缺失文件:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
