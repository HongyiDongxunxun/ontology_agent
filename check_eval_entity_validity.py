import json
import os
from collections import Counter, defaultdict


# ============================================================
# 0. 手动配置
# ============================================================

# 例如：
# FILE_NO = 9
#
# 修改这里即可切换不同批次：
# FILE_NO = 1
# FILE_NO = 2
# FILE_NO = 9
# ...

FILE_NO = 44


# ============================================================
# 1. 文件路径
# ============================================================

ENTITY_FILE = (
    f"output/entities/reviewed_full_{FILE_NO}_result.jsonl"
)

RELATION_FILE = (
    f"output/evaluative_relation/relation_full_{FILE_NO}.jsonl"
)

VERIFICATION_FILE = (
    f"output/relation_verification/"
    f"reviewed_full_{FILE_NO}_verification.jsonl"
)


# ============================================================
# 2. 输出文件
# ============================================================

# ------------------------------------------------------------
# 最终结果：
# verification=true 的关系
# + 主客体实体有效性检查
# ------------------------------------------------------------

OUTPUT_FILE = (
    f"output/evaluative_relation/"
    f"relation_validity_{FILE_NO}.jsonl"
)


# ------------------------------------------------------------
# Verification 中：
# is_evaluation=false 的关系
# ------------------------------------------------------------

REJECTED_FILE = (
    f"output/evaluative_relation/"
    f"relation_verification_rejected_{FILE_NO}.jsonl"
)


# ------------------------------------------------------------
# relation_full 中：
# 找不到对应 Verification 的关系
# ------------------------------------------------------------

UNMATCHED_FILE = (
    f"output/evaluative_relation/"
    f"relation_verification_unmatched_{FILE_NO}.jsonl"
)


# ============================================================
# 3. 特殊主体
# ============================================================

# `_paper_author` 是系统预定义的论文作者主体。
# 它不要求在实体抽取结果中出现。

SPECIAL_VALID_SUBJECTS = {
    "_paper_author"
}


# ============================================================
# 4. JSONL 读取
# ============================================================

def read_jsonl(file_path):
    """
    读取 JSONL 文件。

    Parameters
    ----------
    file_path : str
        JSONL 文件路径

    Returns
    -------
    list
        JSON 对象列表
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\n文件不存在：{file_path}"
        )

    data = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(
            f,
            1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                data.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                print(
                    f"[Warning] "
                    f"{file_path} "
                    f"第 {line_no} 行 JSON 解析失败："
                    f"{e}"
                )

    return data


# ============================================================
# 5. 构造关系匹配键
# ============================================================

def relation_key(item):
    """
    用于匹配 relation_full 与 verification 中的同一关系。

    不依赖 relation_id，
    而是使用：

        sentence_id
        subject
        object
        aspect
        opinion
        evidence
    """

    return (
        str(item.get("sentence_id")),
        item.get("subject"),
        item.get("object"),
        item.get("aspect"),
        item.get("opinion"),
        item.get("evidence"),
    )


# ============================================================
# 6. 建立 Verification 索引
# ============================================================

def build_verification_index(
    verification_data
):
    """
    将 verification 文件中的所有关系建立索引。

    结构：

        relation_key
            ↓
        [verification1, verification2, ...]
    """

    verification_index = defaultdict(list)

    for document in verification_data:

        verifications = document.get(
            "verifications",
            []
        )

        for verification in verifications:

            key = relation_key(
                verification
            )

            verification_index[key].append(
                verification
            )

    return verification_index


# ============================================================
# 7. 建立实体索引
# ============================================================

def build_entity_index(
    entity_data
):
    """
    构建：

        entity_id
            ↓
        entity record
    """

    entity_index = {}

    duplicate_count = 0

    for item in entity_data:

        entity_id = item.get(
            "entity_id"
        )

        if not entity_id:
            continue

        if entity_id in entity_index:

            duplicate_count += 1

        entity_index[
            entity_id
        ] = item

    if duplicate_count > 0:

        print(
            f"[Warning] "
            f"发现重复 entity_id："
            f"{duplicate_count} 条"
        )

    return entity_index


# ============================================================
# 8. 检查主体实体
# ============================================================

def check_subject(
    subject,
    entity_index
):
    """
    返回：

        valid
        reason
    """

    # --------------------------------------------------------
    # 特殊主体
    # --------------------------------------------------------

    if subject in SPECIAL_VALID_SUBJECTS:

        return (
            True,
            "special_subject"
        )

    # --------------------------------------------------------
    # 普通实体
    # --------------------------------------------------------

    if subject in entity_index:

        entity = entity_index[
            subject
        ]

        if entity.get(
            "valid_entity"
        ) is True:

            return (
                True,
                "valid_entity"
            )

        return (
            False,
            "entity_exists_but_invalid"
        )

    # --------------------------------------------------------
    # 找不到实体
    # --------------------------------------------------------

    return (
        False,
        "entity_not_found"
    )


# ============================================================
# 9. 检查客体实体
# ============================================================

def check_object(
    obj,
    entity_index
):
    """
    返回：

        valid
        reason
    """

    # --------------------------------------------------------
    # 空客体
    # --------------------------------------------------------

    if obj is None or obj == "":

        return (
            False,
            "empty_object"
        )

    # --------------------------------------------------------
    # 普通实体
    # --------------------------------------------------------

    if obj in entity_index:

        entity = entity_index[
            obj
        ]

        if entity.get(
            "valid_entity"
        ) is True:

            return (
                True,
                "valid_entity"
            )

        return (
            False,
            "entity_exists_but_invalid"
        )

    # --------------------------------------------------------
    # 客体找不到
    # --------------------------------------------------------

    return (
        False,
        "entity_not_found"
    )


# ============================================================
# 10. 统计 Verification 本身
# ============================================================

def analyze_verification(
    verification_data
):
    """
    独立统计 verification 文件：

        总关系数
        is_evaluation=true
        is_evaluation=false
        其他值
    """

    all_verifications = []

    for document in verification_data:

        verifications = document.get(
            "verifications",
            []
        )

        all_verifications.extend(
            verifications
        )

    true_relations = [
        x
        for x in all_verifications
        if x.get("is_evaluation") is True
    ]

    false_relations = [
        x
        for x in all_verifications
        if x.get("is_evaluation") is False
    ]

    other_relations = [
        x
        for x in all_verifications
        if (
            x.get("is_evaluation") is not True
            and x.get("is_evaluation") is not False
        )
    ]

    return {
        "total": len(all_verifications),
        "true": len(true_relations),
        "false": len(false_relations),
        "other": len(other_relations),
        "true_relations": true_relations,
        "false_relations": false_relations,
        "other_relations": other_relations,
    }


# ============================================================
# 11. relation_full 与 Verification 匹配
# ============================================================

def match_relations_to_verification(
    relation_data,
    verification_data
):
    """
    将 relation_full 中的每一条关系
    与 verification 建立对应关系。

    返回：

        matched_true
        matched_false
        matched_other
        unmatched

    注意：

    verification 中存在 false，
    并不意味着 relation_full 一定包含这些 false 关系。

    因此这里专门统计：
        relation_full 实际命中了哪些 true / false。
    """

    verification_index = (
        build_verification_index(
            verification_data
        )
    )

    # --------------------------------------------------------
    # 防止重复 key 导致错误匹配
    # --------------------------------------------------------

    used_index = defaultdict(int)

    matched_true = []

    matched_false = []

    matched_other = []

    unmatched = []

    # --------------------------------------------------------
    # 遍历 relation_full
    # --------------------------------------------------------

    for relation in relation_data:

        key = relation_key(
            relation
        )

        candidates = (
            verification_index.get(
                key,
                []
            )
        )

        index = used_index[key]

        # ----------------------------------------------------
        # 找不到 Verification
        # ----------------------------------------------------

        if index >= len(candidates):

            item = relation.copy()

            item.update({

                "verification_matched": False,

                "verification_is_evaluation":
                    None,

                "verification_match_reason":
                    "verification_not_found"
            })

            unmatched.append(
                item
            )

            continue

        # ----------------------------------------------------
        # 成功匹配
        # ----------------------------------------------------

        verification = candidates[
            index
        ]

        used_index[key] += 1

        is_evaluation = verification.get(
            "is_evaluation"
        )

        item = relation.copy()

        item.update({

            "verification_matched": True,

            "verification_is_evaluation":
                is_evaluation,

            "verification_relation_id":
                verification.get(
                    "relation_id"
                ),

            "verification_reason":
                verification.get(
                    "verdict_reason"
                ),

            "verification_fact_type":
                verification.get(
                    "fact_type"
                )
        })

        # ----------------------------------------------------
        # true
        # ----------------------------------------------------

        if is_evaluation is True:

            matched_true.append(
                item
            )

        # ----------------------------------------------------
        # false
        # ----------------------------------------------------

        elif is_evaluation is False:

            item[
                "exclusion_reason"
            ] = "is_evaluation_false"

            matched_false.append(
                item
            )

        # ----------------------------------------------------
        # 其他情况
        # ----------------------------------------------------

        else:

            item[
                "exclusion_reason"
            ] = "invalid_is_evaluation_value"

            matched_other.append(
                item
            )

    return {
        "true": matched_true,
        "false": matched_false,
        "other": matched_other,
        "unmatched": unmatched,
    }


# ============================================================
# 12. 对合法评价关系进行实体有效性检查
# ============================================================

def evaluate_entity_validity(
    legal_relations,
    entity_index
):
    """
    只处理：

        verification.is_evaluation == true

    的关系。

    检查：

        subject
        object

    是否对应 valid_entity=true。
    """

    results = []

    for relation in legal_relations:

        result = relation.copy()

        subject = relation.get(
            "subject"
        )

        obj = relation.get(
            "object"
        )

        # ----------------------------------------------------
        # Subject
        # ----------------------------------------------------

        (
            subject_valid,
            subject_reason
        ) = check_subject(
            subject,
            entity_index
        )

        # ----------------------------------------------------
        # Object
        # ----------------------------------------------------

        (
            object_valid,
            object_reason
        ) = check_object(
            obj,
            entity_index
        )

        # ----------------------------------------------------
        # 关系整体有效性
        # ----------------------------------------------------

        relation_valid = (
            subject_valid
            and object_valid
        )

        # ----------------------------------------------------
        # 无效原因
        # ----------------------------------------------------

        reasons = []

        if not subject_valid:

            reasons.append(
                f"subject:{subject_reason}"
            )

        if not object_valid:

            reasons.append(
                f"object:{object_reason}"
            )

        relation_invalid_reason = (
            "; ".join(reasons)
            if reasons
            else ""
        )

        # ----------------------------------------------------
        # 写入
        # ----------------------------------------------------

        result.update({

            "subject_valid":
                subject_valid,

            "subject_valid_reason":
                subject_reason,

            "object_valid":
                object_valid,

            "object_valid_reason":
                object_reason,

            "relation_valid":
                relation_valid,

            "relation_invalid_reason":
                relation_invalid_reason
        })

        results.append(
            result
        )

    return results


# ============================================================
# 13. 保存 JSONL
# ============================================================

def write_jsonl(
    data,
    file_path
):
    """
    保存 JSONL。
    """

    directory = os.path.dirname(
        file_path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# 14. 打印 Verification 总体统计
# ============================================================

def print_verification_statistics(
    statistics
):

    total = statistics[
        "total"
    ]

    true_count = statistics[
        "true"
    ]

    false_count = statistics[
        "false"
    ]

    other_count = statistics[
        "other"
    ]

    print("\n")
    print("=" * 75)
    print("Verification 文件总体统计")
    print("=" * 75)

    print(
        f"Verification 总关系数："
        f"{total}"
    )

    print(
        f"is_evaluation = true："
        f"{true_count}"
    )

    print(
        f"is_evaluation = false："
        f"{false_count}"
    )

    print(
        f"其他 is_evaluation 值："
        f"{other_count}"
    )

    if total > 0:

        print(
            f"\ntrue 占比："
            f"{true_count / total:.2%}"
        )

        print(
            f"false 占比："
            f"{false_count / total:.2%}"
        )

    print("=" * 75)


# ============================================================
# 15. 打印 relation_full 匹配统计
# ============================================================

def print_relation_match_statistics(
    relation_data,
    match_results
):

    total = len(
        relation_data
    )

    matched_true = len(
        match_results["true"]
    )

    matched_false = len(
        match_results["false"]
    )

    matched_other = len(
        match_results["other"]
    )

    unmatched = len(
        match_results["unmatched"]
    )

    matched_total = (
        matched_true
        + matched_false
        + matched_other
    )

    print("\n")
    print("=" * 75)
    print(
        "relation_full 与 Verification 匹配统计"
    )
    print("=" * 75)

    print(
        f"relation_full 总关系数："
        f"{total}"
    )

    print(
        f"匹配到 Verification："
        f"{matched_total}"
    )

    print(
        f"  ├─ 对应 is_evaluation = true："
        f"{matched_true}"
    )

    print(
        f"  ├─ 对应 is_evaluation = false："
        f"{matched_false}"
    )

    print(
        f"  ├─ 对应其他值："
        f"{matched_other}"
    )

    print(
        f"  └─ Verification 未匹配："
        f"{unmatched}"
    )

    if total > 0:

        print(
            f"\nrelation_full → true："
            f"{matched_true / total:.2%}"
        )

        print(
            f"relation_full → false："
            f"{matched_false / total:.2%}"
        )

        print(
            f"relation_full → 未匹配："
            f"{unmatched / total:.2%}"
        )

    print("=" * 75)


# ============================================================
# 16. 打印实体有效性统计
# ============================================================

def print_entity_statistics(
    final_results
):

    total = len(
        final_results
    )

    subject_valid = sum(
        item["subject_valid"]
        for item in final_results
    )

    object_valid = sum(
        item["object_valid"]
        for item in final_results
    )

    relation_valid = sum(
        item["relation_valid"]
        for item in final_results
    )

    relation_invalid = (
        total
        - relation_valid
    )

    print("\n")
    print("=" * 75)
    print(
        "合法评价关系的实体有效性统计"
    )
    print("=" * 75)

    print(
        f"进入实体有效性检查："
        f"{total}"
    )

    if total > 0:

        print(
            f"主体有效："
            f"{subject_valid}"
            f" ({subject_valid / total:.2%})"
        )

        print(
            f"客体有效："
            f"{object_valid}"
            f" ({object_valid / total:.2%})"
        )

        print(
            f"主客体均有效："
            f"{relation_valid}"
            f" ({relation_valid / total:.2%})"
        )

        print(
            f"存在无效实体："
            f"{relation_invalid}"
            f" ({relation_invalid / total:.2%})"
        )

    # --------------------------------------------------------
    # Subject 原因
    # --------------------------------------------------------

    subject_reasons = Counter(
        item["subject_valid_reason"]
        for item in final_results
        if not item["subject_valid"]
    )

    print("\n主体无效原因：")

    if subject_reasons:

        for reason, count in (
            subject_reasons.most_common()
        ):

            print(
                f"  {reason}: {count}"
            )

    else:

        print("  无")

    # --------------------------------------------------------
    # Object 原因
    # --------------------------------------------------------

    object_reasons = Counter(
        item["object_valid_reason"]
        for item in final_results
        if not item["object_valid"]
    )

    print("\n客体无效原因：")

    if object_reasons:

        for reason, count in (
            object_reasons.most_common()
        ):

            print(
                f"  {reason}: {count}"
            )

    else:

        print("  无")

    print("=" * 75)


# ============================================================
# 17. 打印 Verification 中被判 false 的关系
# ============================================================

def print_verification_false_relations(
    false_relations
):

    print("\n")
    print("=" * 85)
    print(
        "Verification 中 is_evaluation = false 的关系"
    )
    print("=" * 85)

    if not false_relations:

        print("无")

        print("=" * 85)

        return

    for i, item in enumerate(
        false_relations,
        1
    ):

        print(
            f"\n[{i}] "
            f"sentence_id = "
            f"{item.get('sentence_id')}"
        )

        print(
            f"subject = "
            f"{item.get('subject')}"
        )

        print(
            f"object = "
            f"{item.get('object')}"
        )

        print(
            f"aspect = "
            f"{item.get('aspect')}"
        )

        print(
            f"opinion = "
            f"{item.get('opinion')}"
        )

        print(
            f"relation_id = "
            f"{item.get('relation_id')}"
        )

        print(
            f"reason = "
            f"{item.get('verdict_reason')}"
        )

        print(
            f"fact_type = "
            f"{item.get('fact_type')}"
        )

    print("\n" + "=" * 85)


# ============================================================
# 18. 打印 relation_full 中命中 false 的关系
# ============================================================

def print_matched_false_relations(
    matched_false
):

    print("\n")
    print("=" * 85)
    print(
        "relation_full 中对应到 Verification=false 的关系"
    )
    print("=" * 85)

    if not matched_false:

        print("无")

        print("=" * 85)

        return

    for i, item in enumerate(
        matched_false,
        1
    ):

        print(
            f"\n[{i}] "
            f"sentence_id = "
            f"{item.get('sentence_id')}"
        )

        print(
            f"subject = "
            f"{item.get('subject')}"
        )

        print(
            f"object = "
            f"{item.get('object')}"
        )

        print(
            f"aspect = "
            f"{item.get('aspect')}"
        )

        print(
            f"opinion = "
            f"{item.get('opinion')}"
        )

        print(
            f"reason = "
            f"{item.get('verification_reason')}"
        )

    print("\n" + "=" * 85)


# ============================================================
# 19. 打印 Verification 未匹配关系
# ============================================================

def print_unmatched_relations(
    unmatched
):

    print("\n")
    print("=" * 85)
    print(
        "relation_full 中无法匹配 Verification 的关系"
    )
    print("=" * 85)

    if not unmatched:

        print("无")

        print("=" * 85)

        return

    for i, item in enumerate(
        unmatched,
        1
    ):

        print(
            f"\n[{i}] "
            f"sentence_id = "
            f"{item.get('sentence_id')}"
        )

        print(
            f"subject = "
            f"{item.get('subject')}"
        )

        print(
            f"object = "
            f"{item.get('object')}"
        )

        print(
            f"aspect = "
            f"{item.get('aspect')}"
        )

        print(
            f"opinion = "
            f"{item.get('opinion')}"
        )

        print(
            f"evidence = "
            f"{item.get('evidence')}"
        )

    print("\n" + "=" * 85)


# ============================================================
# 20. 打印实体无效关系
# ============================================================

def print_invalid_entity_relations(
    final_results
):

    invalid_results = [
        item
        for item in final_results
        if not item["relation_valid"]
    ]

    print("\n")
    print("=" * 85)
    print(
        "合法评价关系中，"
        "存在无效主体/客体实体的关系"
    )
    print("=" * 85)

    if not invalid_results:

        print("无")

        print("=" * 85)

        return

    for i, item in enumerate(
        invalid_results,
        1
    ):

        print(
            f"\n[{i}] "
            f"sentence_id = "
            f"{item.get('sentence_id')}"
        )

        print(
            f"subject = "
            f"{item.get('subject')} "
            f"| valid = "
            f"{item.get('subject_valid')}"
        )

        print(
            f"object = "
            f"{item.get('object')} "
            f"| valid = "
            f"{item.get('object_valid')}"
        )

        print(
            f"reason = "
            f"{item.get('relation_invalid_reason')}"
        )

        print(
            f"opinion = "
            f"{item.get('opinion')}"
        )

        print(
            f"evidence = "
            f"{item.get('evidence')}"
        )

    print("\n" + "=" * 85)


# ============================================================
# 21. 主程序
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print(
        "评价关系 Verification + 实体有效性检查"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # 文件
    # --------------------------------------------------------

    print(
        f"\nENTITY_FILE：\n"
        f"{ENTITY_FILE}"
    )

    print(
        f"\nRELATION_FILE：\n"
        f"{RELATION_FILE}"
    )

    print(
        f"\nVERIFICATION_FILE：\n"
        f"{VERIFICATION_FILE}"
    )

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    print(
        "\n正在读取文件……"
    )

    entity_data = read_jsonl(
        ENTITY_FILE
    )

    relation_data = read_jsonl(
        RELATION_FILE
    )

    verification_data = read_jsonl(
        VERIFICATION_FILE
    )

    print("\n读取完成：")

    print(
        f"  实体记录："
        f"{len(entity_data)}"
    )

    print(
        f"  relation_full："
        f"{len(relation_data)}"
    )

    print(
        f"  verification 文档："
        f"{len(verification_data)}"
    )

    # ========================================================
    # 建立实体索引
    # ========================================================

    entity_index = (
        build_entity_index(
            entity_data
        )
    )

    print(
        f"  可索引实体："
        f"{len(entity_index)}"
    )

    # ========================================================
    # 第一部分
    #
    # Verification 本身统计
    # ========================================================

    verification_statistics = (
        analyze_verification(
            verification_data
        )
    )

    print_verification_statistics(
        verification_statistics
    )

    # ========================================================
    # 第二部分
    #
    # relation_full 与 Verification 匹配
    # ========================================================

    match_results = (
        match_relations_to_verification(
            relation_data,
            verification_data
        )
    )

    # ========================================================
    # 第三部分
    #
    # 只有 Verification=true 才进入实体检查
    # ========================================================

    legal_relations = (
        match_results["true"]
    )

    final_results = (
        evaluate_entity_validity(
            legal_relations,
            entity_index
        )
    )

    print_entity_statistics(
        final_results
    )

    # ========================================================
    # 第四部分
    #
    # 保存最终结果
    # ========================================================

    write_jsonl(
        final_results,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # 保存 Verification=false
    # --------------------------------------------------------

    write_jsonl(
        verification_statistics[
            "false_relations"
        ],
        REJECTED_FILE
    )

    # --------------------------------------------------------
    # 保存 relation_full 中未匹配的关系
    # --------------------------------------------------------

    write_jsonl(
        match_results[
            "unmatched"
        ],
        UNMATCHED_FILE
    )

    # ========================================================
    # 第五部分
    #
    # 详细输出
    # ========================================================

    print_verification_false_relations(
        verification_statistics[
            "false_relations"
        ]
    )

    print_matched_false_relations(
        match_results[
            "false"
        ]
    )

    print_unmatched_relations(
        match_results[
            "unmatched"
        ]
    )

    print_invalid_entity_relations(
        final_results
    )

    # ========================================================
    # 最终
    # ========================================================

    print("\n")
    print("=" * 75)
    print("处理完成")
    print("=" * 75)

    print(
        f"\n最终结果：\n"
        f"  {OUTPUT_FILE}"
    )

    print(
        f"\nVerification=false：\n"
        f"  {REJECTED_FILE}"
    )

    print(
        f"\nVerification 未匹配：\n"
        f"  {UNMATCHED_FILE}"
    )

    print("\n")


# ============================================================
# 22. 程序入口
# ============================================================

if __name__ == "__main__":
    main()