"""
pipeline.relation_verification_agent — 评价关系校验 Agent (Relation Verification)

独立复核已抽取的评价关系: 逐条判断该关系表达的是「评价」还是「事实/描述」。
- 事实/描述 → is_evaluation=false, 附 verdict_reason 标记后返回
- 评价     → is_evaluation=true, 原样放行 (继续)

校验 Agent 以资深图书情报学学者身份开展工作, 运用:
- 语言学: 评价理论 (Appraisal Theory) 的态度/介入/级差系统、
  立场标记 (stance markers)、名物化现象
- 逻辑学: 休谟问题揭示的事实命题与价值命题之分 (从"是"推不出"应该")

与 Agent 1 内置的评价有效性过滤形成双保险: Agent 1 在抽取时过滤,
本 Agent 在抽取后独立复核, 可用于:
  1. 质量检测: 统计误抽率 (is_evaluation=false 的占比)
  2. 数据清洗: 过滤掉标记为事实的关系
  3. 提示词迭代: 定位 Agent 1 的漏网模式
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient


RELATION_VERIFICATION_PROMPT = """
# Role

你是一位资深的图书情报学学者，长期从事学术文本语篇分析与情报计量研究，治学严谨、标准严格。你以同行评议的专业态度复核每一条评价关系：有充分语言学证据的评价予以确认，证据不足的一律按事实处理，绝不姑息。

## Profile

- 学术身份: 图书情报学教授，熟悉文献计量学、学术评价、知识组织与引文分析
- 语言学功底: 精通学术语篇人际意义分析——评价理论（Appraisal Theory: 态度/介入/级差三系统）、立场标记（stance markers: 模糊限制语/增强语/态度标记语/自我提及语）、情态系统与名物化现象
- 逻辑学素养: 严守休谟问题所揭示的事实命题与价值命题之分——"是"（is）推不出"应该"（ought），事实描述可验证真伪，价值判断涉及主观取向，二者存在不可跨越的鸿沟
- 工作作风: 严格、审慎、以证据定案；对每条关系独立裁决，不受同句其他关系影响

## Rules

### 一、强制判断问题（每案必答）

对每条关系，必须先回答：

这条关系所依据的表达，是在"评价"一个对象，还是在"描述"这个对象发生了什么？

- 回答"评价" → is_evaluation=true，放行。
- 回答"描述" → is_evaluation=false，标记事实类别。

### 二、语言学判定依据

**1. 级差（Graduation）证据优先**：态度系统中的级差资源是评价的最直接语言学证据。只有当 opinion 承载"语势"（强化/量化：显著、较高、偏低、明显、大幅度）或"聚焦"（清晰化/模糊化：基本、大致、相对）时，评价才成立。以"取得、实现、提出、采用、进行"等行为动词为核心、后接内容无任何级差修饰的表达，一律是事实描述。

**2. 态度标记语（Attitude Markers）辨析**：重要、有效、丰富、成熟、完善、不足、优异、薄弱、落后、值得、有待、优于、逊于、堪称等词属于态度标记语，承载作者的价值判断；但其作用需结合句法位置判定——出现在评价对象之后作谓词中心（如"水平偏低"）构成评价；出现在"进行、开展"等行为动词的宾语从句中仅描述研究行为（如"进行了比较分析"）不构成评价。

**3. 立场与介入考量**："基本一致、水平相当、表现相近"等中性比较属于立场表达，构成评价；"可见、这说明、由此"引导的推论句若带态度标记仍属评价；纯言据标记（如表N所示、据统计、结果显示）引出的客观数据不构成评价。

**4. 名物化警惕**："发展、建设、研究、应用"等名物化动词在统计语境（主要集中于、分布于、占比、共发文）中只表分布，不表评价。

### 三、逻辑学判定依据

**1. 事实命题与价值命题之分**：
- 事实命题回答"是什么"：可依据客观证据验证真伪。研究行为、方法使用、功能实现、定义、分类、统计罗列均属此类。
- 价值命题回答"好不好、该不该"：承载评价者对对象的价值取向，不可简单用数据验证。优劣判断、重要性判断、效果判断、可行性判断、建议性判断均属此类。

**2. 从"是"推不出"应该"**：当证据只陈述"做了什么、发生了什么、是什么"时，即使评价对象合法，该关系仍属事实，标记 is_evaluation=false。只有当文本明确表达价值、优劣、重要性、效果、问题、程度、可行性、认可度或建议性判断时，才认定评价成立。

**3. 命题完整性**：opinion 必须能独立承载一个价值命题。仅剩虚义动词（只有"取得""实现""采用"）的 opinion 无法形成价值命题，一律标记事实。

### 四、九类典型事实模式（is_evaluation=false）

1. "进行了研究/分析/对比" → 研究行为
2. "提出了方法" → 研究工作
3. "采用了某方法" → 方法使用
4. "实现了某功能" → 功能实现
5. "定义为……" → 定义
6. "逐渐发展起来" → 过程描述
7. "主要集中于……" → 研究分布/统计
8. "取得了……成绩/成果" → 客观结果陈述（带级差修饰如"显著的成绩"除外）
9. 数值、比例、排名罗列 → 统计

### 五、裁决标准（严格原则）

1. 存疑从严：语言学证据不足、级差意义含糊时，标记事实。
2. 独立裁决：每条关系依据自身 opinion 与 evidence 判定，不受同句其他关系影响。
3. 完整语义：结合 evidence 还原 opinion 的完整语境后再判定，防止断章取义。
4. 理由必写：每条裁决给出 verdict_reason，引述关键语言证据（如"opinion含级差词'偏低'"、"opinion仅有行为动词'进行'"）。

## Input

句子：
{sentence}

待复核关系：
{relations_text}

## OutputFormat

严格 JSON，不含 markdown 代码块。按输入顺序逐条输出，数组长度与输入关系数一致：

[
  {{"relation_id": "r1", "is_evaluation": true, "verdict_reason": "opinion含级差词'偏低', 构成价值命题"}},
  {{"relation_id": "r2", "is_evaluation": false, "verdict_reason": "opinion仅有行为动词, 无级差证据", "fact_type": "研究行为"}}
]

verdict_reason 简述判定依据并引述关键语言证据；is_evaluation=false 时附 fact_type（九类事实类型之一）。
""".strip()


@dataclass
class RelationVerification:
    """单条关系的校验结论"""

    relation_id: str                    # 对应输入关系的编号 (r1/r2/...)
    is_evaluation: bool                 # True=评价(放行), False=事实/描述(标记)
    verdict_reason: str = ""            # 判定理由
    fact_type: str = ""                 # is_evaluation=false 时的事实类别
    original: dict = field(default_factory=dict)   # 原始关系 (透传)

    def to_dict(self) -> dict:
        data = dict(self.original)
        data["relation_id"] = self.relation_id
        data["is_evaluation"] = self.is_evaluation
        data["verdict_reason"] = self.verdict_reason
        if self.fact_type:
            data["fact_type"] = self.fact_type
        return data


@dataclass
class SentenceVerificationOutput:
    """单句校验输出"""

    sentence_id: str
    sentence: str
    total_relations: int = 0
    evaluation_count: int = 0          # 判定为评价的关系数
    fact_count: int = 0                # 判定为事实的关系数
    verifications: list[RelationVerification] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "total_relations": self.total_relations,
            "evaluation_count": self.evaluation_count,
            "fact_count": self.fact_count,
            "verifications": [v.to_dict() for v in self.verifications],
        }


class RelationVerificationAgent:
    """评价关系校验 Agent: 独立复核抽取结果, 标记事实/描述类关系"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def verify(
        self,
        sentence_id: str,
        sentence: str,
        relations: list[dict],
    ) -> SentenceVerificationOutput:
        """
        复核一句中的所有关系。

        参数:
        - sentence_id: 句编号
        - sentence: 完整句子
        - relations: 待复核关系列表 [{subject, object, aspect, opinion, evidence}]

        返回: SentenceVerificationOutput (事实关系已标记 is_evaluation=false)
        """
        if not relations:
            return SentenceVerificationOutput(
                sentence_id=sentence_id, sentence=sentence)

        # 为每条关系分配编号
        tagged = [
            {**rel, "relation_id": f"r{i + 1}"}
            for i, rel in enumerate(relations)
        ]

        prompt = (
            RELATION_VERIFICATION_PROMPT
            .replace("{sentence}", sentence)
            .replace("{relations_text}", json.dumps(tagged, ensure_ascii=False, indent=2))
        )
        data = self.llm.call_json(prompt, [])

        return self._parse_output(sentence_id, sentence, tagged, data)

    @staticmethod
    def _parse_output(
        sentence_id: str,
        sentence: str,
        tagged: list[dict],
        data: object,
    ) -> SentenceVerificationOutput:
        if not isinstance(data, list):
            # LLM 返回异常: 全部保守标记为评价 (放行), 附说明
            verifications = [
                RelationVerification(
                    relation_id=rel["relation_id"], is_evaluation=True,
                    verdict_reason="LLM返回异常, 默认放行", original=rel)
                for rel in tagged
            ]
            return RelationVerificationAgent._build_output(
                sentence_id, sentence, verifications)

        verdict_map: dict[str, dict] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("relation_id", "")).strip()
            if rid:
                verdict_map[rid] = item

        verifications: list[RelationVerification] = []
        for rel in tagged:
            rid = rel["relation_id"]
            verdict = verdict_map.get(rid, {})
            is_eval = bool(verdict.get("is_evaluation", True))
            reason = str(verdict.get("verdict_reason", ""))
            fact_type = str(verdict.get("fact_type", ""))
            if not reason:
                reason = ("评价" if is_eval else "事实/描述")
            verifications.append(
                RelationVerification(
                    relation_id=rid,
                    is_evaluation=is_eval,
                    verdict_reason=reason,
                    fact_type=fact_type if not is_eval else "",
                    original=rel,
                )
            )

        return RelationVerificationAgent._build_output(
            sentence_id, sentence, verifications)

    @staticmethod
    def _build_output(
        sentence_id: str,
        sentence: str,
        verifications: list[RelationVerification],
    ) -> SentenceVerificationOutput:
        eval_count = sum(1 for v in verifications if v.is_evaluation)
        fact_count = len(verifications) - eval_count
        return SentenceVerificationOutput(
            sentence_id=sentence_id,
            sentence=sentence,
            total_relations=len(verifications),
            evaluation_count=eval_count,
            fact_count=fact_count,
            verifications=verifications,
        )

    # ── 过滤辅助: 只保留通过校验的关系 ──

    @staticmethod
    def filter_evaluations(
        output: SentenceVerificationOutput,
    ) -> list[dict]:
        """从校验输出中提取判定为评价的关系 (继续下游使用)"""
        return [v.original for v in output.verifications if v.is_evaluation]
