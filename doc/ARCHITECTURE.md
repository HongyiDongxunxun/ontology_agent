# final_Agent V4.2 — 三Agent端到端文献知识挖掘系统 技术文档

## 一、项目概览

**定位**: 面向图书情报学领域的学术评价句全自动知识实体挖掘系统。从含OCR噪声的学术文献中，经三级Agent流水线（抽取→分类→审查），将评价句中的学术实体自动标注为L1/L2/L3四级分类体系，并附带L4规则匹配与Likert 5点量表置信度。

**核心指标**:
- **版本**: V4.2 (V4.1 → L4改为规则匹配)
- **语言**: Python 3.x
- **LLM**: DeepSeek API (OpenAI兼容协议)
- **处理能力**: ThreadPoolExecutor并发，默认40并发，支持约10,000+文件批量处理
- **分类体系**: 4 L1大类 × 14 L2子类 × 59 L3叶子类型

---

## 二、总体架构

### 2.1 三Agent流水线

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        input/reviewed_full_*.json                        │
│            (JSON: 逐句 previous + evaluative + next 上下文)              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────────────────────┐
        │                   Agent 1: Entity Extraction                 │
        │  角色: 学术文献实体抽取专家 (高召回)                           │
        │  输入: 单句评价文本                                           │
        │  输出: candidates (mention, candidate_l1, candidate_l3,       │
        │        evidence, confidence)                                  │
        │  写入: mid_data/{name}_extracted.json                         │
        └──────────────────────┬──────────────────────────────────────┘
                               │ SentenceExtractionOutput[]
        ┌──────────────────────▼──────────────────────────────────────┐
        │                   Agent 2: Classification                    │
        │  角色: 学术文献实体精分类与验证专家                             │
        │  输入: Agent 1 的候选实体 + 原文句子                           │
        │  功能:                                                        │
        │    ① 有效性验证 (valid_entity + invalid_reason)               │
        │    ② L1→L2→L3 逐级精标注                                      │
        │    ③ L4 规则匹配 (与 DynamicTermDB 做字符串匹配，不依赖 LLM)        │
        │  输出: FinalEntityResult[]                                    │
        └──────────────────────┬──────────────────────────────────────┘
                               │ FinalEntityResult[]
        ┌──────────────────────▼──────────────────────────────────────┐
        │             Agent 3: Library Science Reviewer                │
        │  角色: 图书馆学专业学长审查员                                   │
        │  输入: Agent 2 的分类结果 + 原文句子                           │
        │  输出: Likert 5点量表评分 (1-5) + reviewer_comment            │
        │        + suggested_correction (≤2分时)                        │
        └──────────────────────┬──────────────────────────────────────┘
                               │ FinalEntityResult[] (已填充 likert)
        ┌──────────────────────▼──────────────────────────────────────┐
        │                   Dynamic Feedback Loop                      │
        │  收集 likert_confidence == 5 的实体 → 写入 DynamicTermDB      │
        │  后续文件 Agent 2 L4 匹配时读取更新后的 DB                     │
        └──────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────────────────────┐
        │                Export Layer                                  │
        │  output/{name}_result.jsonl           (逐实体JSONL)          │
        │  output/{name}_summary.json           (单文件统计)            │
        │  output/dynamic_terms.json            (动态术语库导出)        │
        └─────────────────────────────────────────────────────────────┘
```

### 2.2 项目目录结构

```
final_Agent/
├── run.py                          # 入口: 批量并行启动
├── config.py                       # 配置: LLMConfig + PipelineConfig + 路径
├── requirements.txt                # openai
├── pipeline/                       # 核心包
│   ├── __init__.py                 # 导出所有公共接口, 版本 4.1.0
│   ├── llm.py                      # LLMClient: OpenAI兼容调用层
│   ├── taxonomy.py                 # 分类体系: TAXONOMY_HIERARCHY 字典 + 反向索引
│   ├── entity_extraction_agent.py  # Agent 1: 实体抽取 (LangGPT prompt)
│   ├── classification_agent.py     # Agent 2: L1/L2/L3精分类 + L4 (LangGPT prompt)
│   ├── reviewer_agent.py           # Agent 3: Likert 5点量表审查 (LangGPT prompt)
│   ├── dynamic_term_db.py          # 线程安全动态术语底库
│   └── dual_agent_pipeline.py      # 管道编排 + export_jsonl/export_summary_json
├── input/                          # 输入: ~10,000+ reviewed_full_*.json
├── output/                         # 输出: *_result.jsonl + *_summary.json
├── mid_data/                       # 中间: *_extracted.json
└── 实体类型分类体系_opencode版.md    # 分类体系参考文档 (889行)
```

### 2.3 并发模型

- 每个输入文件 (`reviewed_full_*.json`) 是一个独立处理单元
- `run.py` 使用 `ThreadPoolExecutor` (默认40并发) 并行调度
- 每个线程内: 创建3个独立 `LLMClient` (各自用不同API Key) → 实例化 `DualAgentPipeline` → 调用 `pipeline.run()`
- 线程间共享: `DynamicTermDB`（有 `threading.Lock` 保护）

---

## 三、实体分类体系

### 3.1 四级层级结构 (L1→L2→L3→L4)

分类体系定义于 `pipeline/taxonomy.py`，是一个 **L1→L2→L3** 的字典映射，带中英文标签：

| L1 | L2 | L3 叶子类型 | 典型示例 |
|----|----|-----------|---------|
| **Agent** | Person | `scholar`, `practitioner`, `policy_advocate`, `reviewer` | 潘光旦, 巴巴拉·奎恩特 |
| | Organization | `research`, `service`, `professional`, `governance`, `publishing` | 哥伦比亚大学, 中国图书馆学会 |
| **Artifact** | Discursive | `journal_article`, `conference_paper`, `book`, `thesis`, `report`, `preprint`, `paper` | 《图书馆学概论》 |
| | Organizational | `knowledge_organization_system`, `metadata_schema`, `reference_tool`, `index` | 《中图法》, Dublin Core |
| | Empirical | `dataset`, `corpus`, `database` | CNKI, OpenCitations |
| | Normative | `standard`, `policy`, `guideline` | HR1858, ISBN |
| | System | `information_system` | CDWS分词系统 |
| | Tool | `software`, `algorithm`, `instrument` | SPSS, SOM |
| **Abstract** | Conceptual | `concept`, `definition`, `typology` | 信息素养, 知识鸿沟 |
| | KnowledgeClaim | `theory`, `model`, `framework`, `hypothesis` | 学习迁移理论, TAM |
| | Methodological | `methodology`, `method`, `technique` | 引文分析法, 共词分析 |
| | Epistemic | `paradigm`, `approach`, `discipline`, `subfield`, `school_of_thought` | 图书情报学, 情报学派 |
| | Phenomenon | `phenomenon`, `trend` | 数据孤岛, 数字化趋势 |
| **Event** | Event | `intellectual_turn`, `debate`, `movement`, `research_program`, `policy_initiative`, `stage`, `conference_meeting` | 开放获取运动, 第二届全国灰色文献年会 |

**L4 规则匹配**: 由 `DynamicTermDB` 提供运行时可积累的术语库，Agent 2 在 LLM 完成 L1/L2/L3 分类后，通过 `_match_l4_rules()` 方法对实体名称与术语库做精确匹配和包含匹配，匹配成功则标注 `l4_matched_term` 和 `l4_term_type`。不依赖 LLM。项目启动时为空库，随着处理进程逐步积累 Likert 5 分的高置信实体。

### 3.2 反向索引 (L3→L1+L2)

`taxonomy.py` 在导入时自动构建 `_L3_TO_L1L2` 反向索引，支持从 L3 type_code 快速查询其 (L1, L2) 对。Agent 2 的 LLM 调用失败时，使用此索引进行兜底分类。

---

## 四、LLM 调用基础设施 (`pipeline/llm.py`)

### 4.1 LLMClient 类

| 属性/方法 | 说明 |
|-----------|------|
| `__init__(model, api_key, base_url, temperature, max_tokens, timeout)` | 构造函数，默认 `deepseek-chat` |
| `_call(prompt)` | 单轮对话调用，通过 `openai.OpenAI` 发 `chat.completions.create`，返回 `content` 字符串 |
| `call_json(prompt, default)` | 调用 LLM 并进行 JSON 解析，含多级容错 |
| `_safe_json_parse(raw, default)` | **三级容错**: ① 提取 markdown 代码块 → ② 去除尾逗号 → ③ 正则提取 JSON 片段 |

**关键参数**: `temperature=0.0`（确保分类一致性）、`max_tokens=4096`、`timeout=60s`。

### 4.2 三层 API Key 隔离

三个Agent各用独立API Key（`config.py` 的 `LLMConfig`）：

| 环境变量 | 对应Agent |
|----------|-----------|
| `DEEPSEEK_API_KEY_EXTRACTION` | Agent 1 (抽取) |
| `DEEPSEEK_API_KEY_CLASSIFICATION` | Agent 2 (分类) |
| `DEEPSEEK_API_KEY_REVIEWER` | Agent 3 (审查) |

未配置时回退到 `DEEPSEEK_API_KEY` 统一Key。

---

## 五、Agent 1 — 实体抽取 (`pipeline/entity_extraction_agent.py`)

### 5.1 Prompt 设计 (LangGPT 风格)

采用 LangGPT 八段式结构：

| 段落 | 内容 |
|------|-----|
| `# Role` | "学术文献实体抽取专家" |
| `## Profile` | 专长、能力、原则、风格 |
| `## Rules` | ① 实体性门槛（6类泛称/通用词一律不抽取）② 原文忠实原则 ③ 召回优先原则 |
| `## Workflow` | 7步：读句→扫描名词→过滤→选candidate_l3→选candidate_l1→提取evidence→输出JSON |
| `## Background` | 完整四大类实体定义 + 邻接类型区分规则（如 scholar vs practitioner, concept vs definition） |
| `## OutputFormat` | 严格JSON: `{entities: [{mention, normalized_name, candidate_l3, candidate_l1, evidence, is_specific_entity, confidence, uncertainty}]}` |
| `## Examples` | 8个正例 + 2个负例，覆盖各类实体及边界判断 |
| `## Input` | `{statement}` 占位符 |

### 5.2 核心过滤规则 (实体性门槛)

Agent 1 明确排除以下六类短语：

1. **泛化身份类别词**: 科学家、学者、教授、图书馆员 (不指称具体个人)
2. **泛化机构类别词**: 大学图书馆、高校、研究机构 (不指称具体组织)
3. **通用连接词/虚义动词**: 比较、进行、通过、基于
4. **泛指代词/自指**: 本文、笔者、相关文献
5. **无学术语义计量词**: 篇数、比例、百分比
6. **泛化评价用语**: 重要、显著、不足

### 5.3 数据模型

**ExtractedEntity** — 单条抽取结果:

```
entity_id: str          # "1_e1" (句号_实体序号)
mention: str            # 原文精确短语
normalized_name: str    # 规范化名称
candidate_l3: str       # 候选L3 type_code
candidate_l1: list[str] # 候选L1 (可多选, 召回优先)
evidence: str           # 证据片段
is_specific_entity: bool
confidence: float       # Agent 1 自评置信度 0~1
uncertainty: str        # 不确定性描述
```

**SentenceExtractionOutput** — 单句抽取结果:

```
sentence_id: str
sentence: str
entities: list[ExtractedEntity]
```

### 5.4 实现细节

- `EntityExtractionAgent.extract(statement, sentence_id)`: 调用LLM，解析JSON，为每个实体生成 `{sentence_id}_e{i+1}` 格式的 entity_id
- 若LLM返回非dict → 返回空 `SentenceExtractionOutput`
- 容错：支持 `mention` 或 `entity_name` 两种字段名

---

## 六、Agent 2 — 精分类 (`pipeline/classification_agent.py`)

### 6.1 Prompt 设计

| 段落 | 内容 |
|------|-----|
| `# Role` | "学术文献实体精分类与验证专家" |
| `## Rules` | ① 有效性验证（先验证后分类）② L1→L2→L3逐级分类 |
| `## Workflow` | 6步：读实体→验证有效性→判L1→选L2→定L3→输出JSON |
| `## Background` | 动态注入 `{taxonomy_text}`（分类体系），L4 术语不进入 Prompt |
| `## Examples` | 6个详细区分案例（scholar/practitioner, book/journal_article, policy/policy_initiative, concept/definition, Organization子类, conference_paper/conference_meeting） |

### 6.2 双重任务

**任务一：有效性验证 (先验后类)**

- 有效实体 (valid_entity=true): 具体可唯一识别的人名/机构名/文献名/系统名/事件名
- 无效实体 (valid_entity=false): 泛称身份、泛称机构、泛称文献、通用词、评价用语
- 无效实体仅填写 `entity_id`, `valid_entity=false`, `invalid_reason`，其余字段留空

**任务二：L1→L2→L3 逐级精标注**

- L1: 基于对象功能/语境四选一
- L2: L1下选择二级类
- L3: L2下选择叶子type_code + label
- L3若为other必须填写 `other_suggestion`

### 6.3 L4 规则匹配

Agent 2 不将术语库送入 LLM Prompt，而是在 LLM 返回 L1/L2/L3 分类结果后，通过 `_match_l4_rules()` 方法进行纯字符串规则匹配：

```python
def _match_l4_rules(self, entity_name, normalized_name):
    """与 DynamicTermDB 做精确匹配 + 包含匹配"""
    1. 收集 entity_name 和 normalized_name 作为候选
    2. 对每个候选 (lowercase, strip)，跳过 < 2 字符的短词
    3. 先精确匹配: 候选与术语完全相同 → 返回
    4. 再包含匹配: 术语含于候选 或 (术语≥3字且候选含于术语) → 返回
    5. 无匹配 → 返回 ("", None)
```

**匹配优先级**: 精确匹配 > 包含匹配。精确匹配优先保证不会误匹配同义异形词。

**优势**: 消除 Agent 2 Prompt 中的 `{micro_terms}` 占位符，减少 Prompt 长度，避免 LLM 幻觉匹配，提升性能。

### 6.4 数据模型 FinalEntityResult

```
# 元数据
sentence_id: str       # 来源句编号
entity_id: str         # 实体编号 (继承自Agent 1)
sentence: str          # 完整上下文句
entity: str            # 实体原文表述
normalized_name: str   # 规范化实体名

# 有效性验证
valid_entity: bool     # true=有效具体实体, false=无效
invalid_reason: str    # 无效原因

# L1/L2/L3 层级分类
l1: str                # 顶层类别: Agent/Artifact/Abstract/Event
l2: str                # 第二层子类代码
l2_label: str          # L2中文标签
l3_type_code: str      # 第三层叶子类型代码
l3_label: str          # L3中文标签

# L4 MicroMapping
l4_matched_term: str   # 匹配到的术语 (未匹配为空)
l4_term_type: str|null # 匹配到的术语类型代码 (未匹配为null)

# 分类依据
evidence: str          # 原句证据片段
reason: str            # 简短分类理由说明
other_suggestion: str  # L3为Other时的建议类型

# Agent 3 Likert评分（审查后填入）
likert_confidence: int    # 1-5
reviewer_comment: str
suggested_correction: str
```

### 6.5 批处理与容错

- 每批最多 `batch_size=12` 个实体进入一次LLM调用
- LLM返回非list → 调用 `_fallback_results()`，利用 `taxonomy._L3_TO_L1L2` 反向索引从 `candidate_l3` 推断 L1/L2
- LLM返回的元素数少于输入 → 剩余实体用 `candidate_l3` 兜底，`l1=""` `l2=""`

---

## 七、Agent 3 — 图书馆学专业学长审查 (`pipeline/reviewer_agent.py`)

### 7.1 Prompt 设计

| 段落 | 内容 |
|------|-----|
| `# Role` | "资深的图书馆学专业学长" |
| `## Profile` | 图情学科核心专长: 图书馆学、情报学、文献分类学、知识组织、信息检索、文献计量学 |
| `## Rules` | ① Likert 5点量表详细标准 ② 7条图情学科特殊审查规则 |
| `## Background` | 动态分类体系 + 图情学科常见实体对照表 (13组典型正确分类) |
| `## OutputFormat` | JSON数组: `[{entity_id, likert_score, reviewer_comment, suggested_correction}]` |
| `## Examples` | 6个案例 (5分/4分/3分/2分/1分/无效实体审查) |

### 7.2 Likert 5点量表

| 分值 | 语义 | 判定标准 |
|------|------|---------|
| 5 | 完全认同 | L1/L2/L3全精准，L4也匹配正确，有效性判定合理 |
| 4 | 认同 | 分类基本正确，小瑕疵不影响整体准确性 |
| 3 | 不确定 | 信息不足或处于边界，无法准确判断 |
| 2 | 不认同 | L1正确但L2/L3错误、边界模糊处理不当、有效误判无效 |
| 1 | 完全不认同 | L1大类错误、L3完全错配、无效判定不合理 |

### 7.3 图情学科7条特殊审查规则

1. **KOS匹配**: 叙词表/分类法/本体 必须归 `knowledge_organization_system`
2. **文献类型**: 期刊论文/会议论文/学位论文/专著/报告 必须准确区分
3. **学者vs实践者**: 以研究产出为主→scholar, 以管理服务为主→practitioner
4. **概念vs定义**: 有公认界定方式→definition, 仅为命名单元→concept
5. **标准/政策**: 文件本体→Artifact.Normative, 制定事件→Event
6. **机构类型**: 大学图书馆→service, 大学→research
7. **无效判定审查**: 防止图情专业术语被误判为无效

### 7.4 数据模型 ReviewResult

```python
@dataclass
class ReviewResult:
    entity_id: str
    likert_score: int         # 1-5
    reviewer_comment: str     # 图书馆学专业角度的评分理由 (≤50字)
    suggested_correction: str # likert_score ≤2 时必填
```

### 7.5 实现细节

- 评分 ≤2 自动触发 `suggested_correction`
- LLM调用失败 → 兜底输出 `likert_score=3, reviewer_comment="LLM调用失败,默认评3分"`
- 输入实体信息按 `[idx] entity_id=..., entity=..., valid_entity=..., l1=..., l2=..., l3=..., l4=..., reason=..., evidence=...` 格式构造

---

## 八、动态术语底库 (`pipeline/dynamic_term_db.py`)

### 8.1 设计动机

项目启动时术语库为空，通过运行时 Likert 5 分实体逐步积累。L4 匹配采用纯字符串规则，不依赖 LLM，保证确定性与可复现性。

### 8.2 核心实现

```python
class DynamicTermDB:
    _lock: threading.Lock         # 线程安全锁
    _term_set: set[str]           # 去重键: "term::type_code"
    _terms: list[tuple[str,str]]  # 有序存储 (term, l3_type_code)
    _added_count: int             # 累计新增数
    _skipped_count: int           # 累计去重跳过数
```

| 方法 | 说明 |
|------|------|
| `add(term, l3_type_code) -> bool` | 添加术语，True=新增，False=重复跳过 |
| `get_micro_terms() -> list[(str,str)]` | 获取当前术语快照 |
| `size() -> int` | 当前术语总数 |
| `get_stats() -> dict` | 统计信息: total_terms, added, skipped_duplicates |
| `export(filepath)` | 持久化到JSON文件 |

### 8.3 积累触发机制

在 `DualAgentPipeline.run()` 的 Agent 3 结束后触发：

```python
if self.dynamic_term_db:
    for r in all_results:
        if r.likert_confidence == 5 and r.valid_entity:
            self.dynamic_term_db.add(r.entity, r.l3_type_code)
```

### 8.4 跨文件反馈效应

```
文件1: TermDB空 → Agent2无L4匹配 → Agent3评出3个Likert5 → TermDB:+3
文件2: TermDB含3条 → Agent2规则匹配L4 → Agent3评出2个Likert5 → TermDB:+2 (去重后)
...
→ 术语库随处理进程逐步丰富，后续文件受益于前序文件的积累
```

### 8.5 导出格式

```json
{
    "total_terms": 156,
    "added": 200,
    "skipped_duplicates": 44,
    "terms": [
        {"term": "中国图书馆分类法", "type_code": "knowledge_organization_system"},
        {"term": "引文分析法", "type_code": "method"},
        ...
    ]
}
```

---

## 九、三Agent管道编排 (`pipeline/dual_agent_pipeline.py`)

### 9.1 DualAgentPipeline 类

| 方法 | 说明 |
|------|------|
| `__init__(llm_extraction, llm_classification, llm_reviewer, dynamic_term_db, batch_size, mid_data_dir, verbose)` | 创建三个Agent实例 |
| `run(sentences, base_name) -> list[FinalEntityResult]` | 执行完整三阶段流水线 |
| `_log(msg)` | verbose 控制日志输出 |

### 9.2 run() 执行流程

```
1. [Agent 1] 逐句实体抽取 → SentenceExtractionOutput[]
2. 写中间结果 → mid_data/{base_name}_extracted.json
3. [Agent 2] 逐句L1/L2/L3精分类 + L4规则匹配 → FinalEntityResult[]
4. [Agent 3] 逐句Likert 5点量表审查 → 填充 likert_confidence + reviewer_comment
5. [TermDB] 收集 likert_confidence==5 且 valid_entity==true 的实体 → dynamic_term_db.add()
6. 返回 all_results
```

### 9.3 导出工具

**export_jsonl(results, filepath)**: 逐实体写入JSONL，每行一个完整 `FinalEntityResult.to_dict()`

**export_summary_json(results, base_name, filepath)**: 写入汇总JSON：

```json
{
    "base_name": "reviewed_full_1",
    "total_entities": 144,
    "valid_entities": 138,
    "invalid_entities": 6,
    "l1_distribution": {"Abstract": 63, "Agent": 54, ...},
    "l3_type_distribution": {"discipline": 28, "research": 22, ...},
    "likert_distribution": {
        "1_完全不认同": 2,
        "2_不认同": 5,
        "3_不确定": 12,
        "4_认同": 48,
        "5_完全认同": 71
    },
    "likert_average": 4.31,
    "dynamic_term_db_size": 144
}
```

---

## 十、入口程序 (`run.py`)

### 10.1 运行模式

| 模式 | 触发方式 | 说明 |
|------|---------|------|
| 测试模式 | `--mode test` (默认) | 随机抽取10个文件 |
| 全量模式 | `--mode full` | 处理 input/ 下所有文件 |

### 10.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--live` | False | 必须开启才真正调用LLM API |
| `--concurrency N` | 40 | 并行处理文件数 |
| `--limit N` | 0 | 限制处理文件数 (0=全部) |
| `--no-skip` | False | 不跳过已有输出，强制重跑 |
| `--mode` | test | 运行模式: test / full |
| `--dynamic-out` | output/dynamic_terms.json | 动态术语库导出路径 |

### 10.3 核心函数 process_one_file

```python
def process_one_file(fpath, g, output_dir, mid_data_dir, idx, total, dynamic_db):
    # 1. 创建3个 LLMClient (各自独立 API Key)
    # 2. 创建 DualAgentPipeline (传入 dynamic_db)
    # 3. load_reviewed_json → 拼接前后句为完整上下文字符串
    # 4. pipeline.run(sentences, base_name)
    # 5. export_jsonl + export_summary_json
    # 6. 收集统计数据返回
```

### 10.4 main() 汇总报告

执行完毕后依次输出：
- 总文件数/成功/失败/耗时
- 总评价句数、总实体数 (有效/无效)
- Likert 5点分布及百分比
- L3实体类型分布 (Top 20)
- DynamicTermDB 统计 (累计术语/新增/去重跳过)
- 导出 `dynamic_terms.json`

### 10.5 使用方法

```bash
# 测试模式
python run.py --live

# 全量模式
python run.py --live --mode full

# 控制并发
python run.py --live --mode full --concurrency 20

# 限制文件数
python run.py --live --limit 50

# 强制重新处理
python run.py --live --mode full --no-skip
```

---

## 十一、配置体系 (`config.py`)

### 11.1 LLMConfig

```python
@dataclass
class LLMConfig:
    model: str = "deepseek-chat"               # LLM模型名称
    base_url: str = "https://api.deepseek.com" # API地址
    temperature: float = 0.0                    # 确保确定性分类
    max_tokens: int = 4096                      # 最大输出token数
    timeout: int = 60                           # 请求超时秒数
    api_key_extraction: str                     # Agent 1 专用Key (env: DEEPSEEK_API_KEY_EXTRACTION)
    api_key_classification: str                 # Agent 2 专用Key (env: DEEPSEEK_API_KEY_CLASSIFICATION)
    api_key_reviewer: str                       # Agent 3 专用Key (env: DEEPSEEK_API_KEY_REVIEWER)
    api_key: str                                # 统一回退Key (env: DEEPSEEK_API_KEY)
```

### 11.2 PipelineConfig

```python
@dataclass
class PipelineConfig:
    batch_size: int = 12         # Agent 2/3 每批处理实体数
    max_l1_per_entity: int = 2   # 每个实体最多进入几个候选 L1
```

### 11.3 Config 全局单例

```python
@dataclass
class Config:
    llm: LLMConfig                           # LLM调用参数
    pipeline: PipelineConfig                  # 管道参数
    live_mode: bool = False                   # 是否真实调用API
    verbose: bool = True                      # 是否详细日志
    input_dir: Path = PROJECT_ROOT / "input"   # 输入目录
    output_dir: Path = PROJECT_ROOT / "output" # 输出目录
    mid_data_dir: Path = PROJECT_ROOT / "mid_data" # 中间数据目录
```

### 11.4 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY_EXTRACTION` | Agent 1 抽取 Key | — |
| `DEEPSEEK_API_KEY_CLASSIFICATION` | Agent 2 分类 Key | — |
| `DEEPSEEK_API_KEY_REVIEWER` | Agent 3 审查 Key | — |
| `DEEPSEEK_API_KEY` | 回退统一 Key (未分层时) | — |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |

---

## 十二、完整数据流图

```
┌─ input/reviewed_full_1.json ──────────────────────────────┐
│ {"1":{"evaluative_sentence":"期刊排架是...","previous":"",  │
│  "next":"如果说我们把江乃武同志..."}}                        │
└────────────────────────────────────────────────────────────┘
                          │ load_reviewed_json (拼接上下文)
                          ▼
       "期刊排架是期刊管理工作中十分重要的一环...如果说我们把江乃武同志..."

┌─ Agent 1 ─────────────────────────────────────────────────┐
│ LLM input: ENTITY_EXTRACTION_PROMPT.format(statement)     │
│ LLM output: {"entities":[{mention:"期刊排架",...},...]}   │
│ Parse to: ExtractedEntity[]                              │
│ → SentenceExtractionOutput(sentence_id="1", entities=[])  │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Agent 2 ─────────────────────────────────────────────────┐
│ LLM input: CLASSIFICATION_PROMPT.format(                  │
│   taxonomy_text=...,                                          │
│   entities_text="[0] entity_id=1_e1...",                      │
│   sentence="..."                                              │
│ )                                                             │
│ LLM output: [{entity_id:"1_e1", valid_entity:true,            │
│   l1:"Abstract", l2:"Methodological",                         │
│   l3_type_code:"method", l3_label:"研究方法", ...}, ...]       │
│ → 规则匹配 L4: _match_l4_rules(entity, normalized)            │
│ Parse to: FinalEntityResult[]                                 │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Agent 3 ─────────────────────────────────────────────────┐
│ LLM input: REVIEWER_PROMPT.format(                        │
│   taxonomy_text=..., entities_text="[0] entity_id=1_e1,   │
│   entity=期刊排架, valid_entity=true, l1=Abstract,        │
│   l2=Methodological, l3=method(研究方法), reason=...",     │
│   sentence="..."                                          │
│ )                                                         │
│ LLM output: [{entity_id:"1_e1", likert_score:5,           │
│   reviewer_comment:"排架方法是图情领域基础研究方法...",     │
│   suggested_correction:""}, ...]                          │
│ Merge back → FinalEntityResult.likert_confidence = 5      │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ DynamicTermDB ───────────────────────────────────────────┐
│ if likert_confidence == 5 and valid_entity:               │
│     dynamic_term_db.add(entity, l3_type_code)             │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Export ──────────────────────────────────────────────────┐
│ output/reviewed_full_1_result.jsonl                       │
│ output/reviewed_full_1_summary.json                       │
│ output/dynamic_terms.json                                 │
└────────────────────────────────────────────────────────────┘
```

---

## 十三、LLM Prompt 策略设计原则

所有三个Agent的Prompt均遵循统一的 **LangGPT 八段式** 结构，确保指令清晰、输出可控：

| 段落 | 功能 | Agent 1 | Agent 2 | Agent 3 |
|------|------|---------|---------|---------|
| `# Role` | 角色声明 | 实体抽取专家 | 精分类验证专家 | 图书馆学专业学长 |
| `## Profile` | 专业边界 | 高召回抽取 | 精准分类 | 图情学科核心专长 |
| `## Rules` | 硬性约束 | 6类实体性门槛 | 有效性验证+逐级分类 | 5点量表+7条学科规则 |
| `## Workflow` | 操作指令 | 7步 | 7步 | 7步 |
| `## Background` | 领域知识 | 完整四类定义 | 分类体系 | 分类体系+学科对照表 |
| `## OutputFormat` | JSON约束 | 对象含entities数组 | 数组每元素一个实体 | 数组每元素一个评分 |
| `## Examples` | 正负例 | 8正+2负 | 6组区分案例 | 6组评分案例 |
| `## Input` | 输入占位 | `{statement}` | `{entities_text}` + `{sentence}` | `{entities_text}` + `{sentence}` |

关键技术权衡：
- Agent 1 **高召回** (允许 candidate_l1 多选) vs Agent 2 **高精度** (严格单分类)
- Agent 2 **独立验证** (不受Agent 1 candidate 约束) + 兜底回退 (利用反向索引)
- Agent 3 **领域知识注入** (7条图情规则 + 13组典型对照) → 确保 Likert 评分有学科依据
- **Temperature=0.0** 全程保证分类一致性

---

## 十四、输入输出格式总览

### 14.1 输入格式 (`input/reviewed_full_*.json`)

```json
{
    "1": {
        "previous_sentence": "前一句上下文",
        "evaluative_sentence": "评价句主体内容",
        "next_sentence": "后一句上下文"
    },
    "2": { ... }
}
```

程序自动将三句拼接为完整上下文：`"{previous} {evaluative} {next}"`

### 14.2 中间数据 (`mid_data/{name}_extracted.json`)

```json
{
    "base_name": "reviewed_full_1",
    "stage": "agent1_extraction",
    "extractions": [
        {
            "sentence_id": "1",
            "sentence": "...完整上下文...",
            "entities": [
                {
                    "entity_id": "1_e1",
                    "mention": "期刊排架",
                    "normalized_name": "期刊排架",
                    "candidate_l3": "method",
                    "candidate_l1": ["Abstract"],
                    "evidence": "期刊排架是期刊管理工作中十分重要的一环",
                    "is_specific_entity": true,
                    "confidence": 0.95,
                    "uncertainty": ""
                }
            ]
        }
    ]
}
```

### 14.3 最终 JSONL (`output/{name}_result.jsonl`)

每行一个完整的分类结果JSON对象：

```json
{
    "sentence_id": "1",
    "entity_id": "1_e1",
    "sentence": "...完整上下文...",
    "entity": "期刊排架",
    "normalized_name": "期刊排架",
    "valid_entity": true,
    "invalid_reason": "",
    "l1": "Abstract",
    "l2": "Methodological",
    "l2_label": "方法论",
    "l3_type_code": "method",
    "l3_label": "研究方法",
    "l4_matched_term": "",
    "l4_term_type": null,
    "evidence": "期刊排架是期刊管理工作中十分重要的一环",
    "reason": "期刊排架方法是图情领域的具体研究方法",
    "other_suggestion": "",
    "likert_confidence": 5,
    "reviewer_comment": "排架法是图情领域基础研究方法，分类正确",
    "suggested_correction": ""
}
```

### 14.4 无效实体示例

```json
{
    "entity_id": "5_e4",
    "valid_entity": false,
    "invalid_reason": "泛称身份类别，非具体可唯一识别的实体",
    "l1": "", "l2": "", "l2_label": "",
    "l3_type_code": "", "l3_label": "",
    "l4_matched_term": "", "l4_term_type": null,
    "likert_confidence": 5,
    "reviewer_comment": "确为泛称身份类别，无效判定正确",
    "suggested_correction": ""
}
```

---

## 十五、关键设计决策总结

| 决策 | 理由 |
|------|------|
| 三Agent而非单Agent | 职责分离：抽取(高召回) + 分类(高精度) + 审查(领域验证)，避免单一prompt过长导致的注意力分散 |
| LangGPT八段式 Prompt | 结构化Prompt工程原语，经V3→V4迭代验证有效，确保指令清晰可复现 |
| 线程安全 DynamicTermDB | 支持并发处理下的术语积累，set去重，L4 规则匹配的确定性数据源 |
| L3→L1+L2 反向索引 | LLM调用失败时的确定性兜底，保证管道不中断 |
| 每批12实体 | 平衡prompt长度与LLM处理精度，保证JSON输出格式稳定 |
| 前句+评价句+后句 拼接 | 提供完整语境，避免孤立判断导致的分类错误 |
| 独立API Key | 成本隔离、并发管理、权限分离 |
| Temperature=0.0 | 全程保证分类确定性，避免随机性导致同类实体被分入不同类别 |
| L4 规则匹配 | 消除 LLM 幻觉匹配，改为纯字符串精确+包含匹配，保证确定性与可复现性 |
| 领域知识注入 (Agent 3) | 7条图情学科审查规则确保LIS视角下的分类验证，区别于通用NLP分类任务 |
