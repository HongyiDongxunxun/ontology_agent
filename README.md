# Ontology Agent V5 — 四Agent端到端文献知识挖掘系统

> **多智能体协同 · DeepSeek API · 并行批处理 · 关系优先抽取 · L1/L2/L3 精分类 + L4 规则匹配 · Likert 审查 · 评估体系**

面向情报学/图书馆学领域学术文献，从评价句到结构化知识实体的全自动端到端挖掘管道。采用**关系优先策略**：先抽取评价关系及其评价对象，再补充抽取其余实体，然后分类、审查打分，完整链路为 **评价关系抽取 → 实体抽取补充 → 精分类 → Likert 审查**。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent 1: Evaluative Relation Extraction (评价关系抽取 — 关系优先) │
│  评价句 → 评价关系(subject/object/aspect/opinion/evidence)         │
│         + 评价对象实体 (作为下游已知实体)                          │
│  输出: output/evaluative_relation/relation_full_{num}.jsonl            │
├──────────────────────────────────────────────────────────────────┤
│  Agent 2: Entity Extraction (实体抽取补充 — 高召回)               │
│  接收 Agent 1 的评价对象实体作为"已知实体"，补抽关系之外的实体     │
│  输出: mid_data/{name}_extracted.json                            │
├──────────────────────────────────────────────────────────────────┤
│  Agent 3: Classification (精分类 — L1/L2/L3 + L4)                 │
│  Step 1: 验证实体有效性 (过滤泛称/类别词)                          │
│  Step 2: L1/L2/L3 层级精标注 (Agent/Artifact/Abstract/Event)       │
│  Step 3: L4 规则匹配 (与 DynamicTermDB 做字符串匹配)               │
├──────────────────────────────────────────────────────────────────┤
│  Agent 4: Review (审查 — Likert 5点量表)                          │
│  从图书馆学专业视角审查分类结果，输出 1-5 分置信度                 │
│  输出: output/entities/{name}_result.jsonl + {name}_summary.json  │
└──────────────────────────────────────────────────────────────────┘
```

**关系优先策略**：Agent 1 先识别评价关系并解析出被评价的对象（object）实体，这些实体作为「已知实体」传给 Agent 2；Agent 2 在此基础上补抽评价关系之外的其余实体，避免遗漏，同时保证被评价对象一定进入实体列表。

**ID 重映射**：Agent 1 输出使用句内短 ID（e1/e2），Agent 2 输出使用全局 ID（`{句号}_eN`）。管道编排器按实体名将关系引用的短 ID 重映射为最终实体 ID；若 Agent 2 漏抽了关系引用的实体，编排器会将该实体补入实体列表，保证每条关系都能解析到实体。无法解析时降级为 `_missing_entity` 并保留原文。

**评价有效性过滤（Evaluation Validity）**：Agent 1 在输出任何关系之前必须回答一个问题——这句话在"评价"一个对象，还是在"描述"这个对象发生了什么？研究行为（进行研究、进行分析、进行对比）、方法使用、功能实现、定义、分类、统计罗列均属于描述，一律不生成关系；中性比较判断（基本一致、水平相当、表现相近）属于评价。该机制把九类高频误判模式以完整示例写入了提示词。

### 特性一览

| 特性 | 说明 |
|------|------|
| **关系优先抽取** | Agent 1 先抽评价关系与评价对象，实体列表覆盖所有可解析的 relation.object |
| **实体补充** | Agent 2 接收上游已知实体，补抽关系外实体，提升召回 |
| **ID 重映射** | 关系短 ID 自动重映射为最终实体 ID，漏抽实体自动补入 |
| **评价有效性过滤** | 输出前强制区分"评价"与"描述"，过滤九类典型误判 |
| **Thinking Mode** | 支持 DeepSeek 推理模式，每个 Agent 可独立开关 |
| **Voting** | Agent 3 分类多轮投票机制，减少分类不确定性 |
| **RAG Few-Shot** | 基于动态术语库的相似度检索，为分类提供示例 |
| **动态术语库** | Likert 5 分高置信实体自动积累，跨文件复用 |
| **评估体系** | 实体评估 + 评价关系评估：标注工具、指标计算、报告生成 |

---

## 实体分类体系

### 顶层 L1 (4 大类)

| L1 | 说明 | L2 子类 |
|----|------|---------|
| `Agent` | 行为主体 | Person / Organization |
| `Artifact` | 人工制品 | Discursive / Organizational / Empirical / Normative / System / Tool |
| `Abstract` | 抽象实体 | Conceptual / KnowledgeClaim / Methodological / Epistemic / Phenomenon |
| `Event` | 事件 | Event |

### L4 MicroMapping (术语底库)

L4 层采用**规则匹配**：运行过程中 Likert 5 分的高置信实体自动积累入 DynamicTermDB，后续文件受益于前序积累。支持精确匹配和包含匹配，初始可从 `dynamic_terms.json` 预加载。

---

## 目录结构与文件说明

```
ontology_agent/
│
├── run.py                              # 【主入口】批量并行处理 四Agent管道
├── run_eval.py                         # 【评估入口】一键评估脚本 (自动加载api.txt)
├── run_relation.py                     # 【关系补抽】从 mid_data/实体结果补充抽取评价关系 (兼容/辅助)
├── run_relation_agent.py               # 【独立关系抽取】单独运行 Agent 1 抽取评价关系
├── run_relation_verify.py              # 【关系校验】复核抽取结果, 标记事实/描述类关系
├── run_sample_review.py                # 【抽样审阅】随机采样N句运行管道并输出人工审阅报告
├── config.py                           # 统一配置系统 (LLM / Pipeline / 路径)
├── api.txt                             # API Key 存储文件 (可选)
├── requirements.txt                    # Python 依赖
├── dynamic_terms.json                  # 预加载的动态术语库 (可选)
│
├── test_academic_evaluation_prompt_schema.py   # 单元测试: 关系schema/ID重映射/评价有效性规则
├── test_eval_relations.py              # 单元测试: 评价关系评估指标
├── test_relation_verification.py       # 单元测试: 评价关系校验 Agent
│
├── pipeline/                           # 核心管道包 (v5.0.0)
│   ├── __init__.py                     # 包定义 + 公开API导出
│   ├── llm.py                          # LLM客户端: OpenAI兼容, Thinking/Voting/JSON重试
│   ├── taxonomy.py                     # L1/L2/L3 分类体系层级映射 + 中文标签
│   ├── evaluative_relation_agent.py    # Agent 1: 评价关系抽取 + 评价对象实体
│   ├── relation_verification_agent.py  # 校验Agent: 复核关系是"评价"还是"事实/描述"
│   ├── entity_extraction_agent.py      # Agent 2: 实体抽取补充 (接收上游已知实体)
│   ├── classification_agent.py         # Agent 3: 精分类 + Voting + RAG + L4匹配
│   ├── reviewer_agent.py               # Agent 4: Likert 5点量表审查 Prompt
│   ├── dynamic_term_db.py              # 线程安全动态术语底库 (trigram检索)
│   └── dual_agent_pipeline.py          # 四Agent管道编排 + ID重映射 + JSONL/Summary/Relation导出
│
├── eval/                               # 评估模块
│   ├── __init__.py                     # 评估包定义
│   ├── gold_standard.py                # 标注数据模型 (实体 + 评价关系标注)
│   ├── metrics.py                      # 实体指标 + 关系指标 + has_evaluation准确性
│   ├── reporter.py                     # Markdown 评测报告生成
│   └── annotation_tool.py             # 命令行交互式标注工具
│
├── src/                                # 向后兼容层
│   └── __init__.py                     # 重新导出 pipeline.*
│
├── doc/                                # 文档
│   ├── ARCHITECTURE.md                 # 架构详细文档
│   ├── RFREADME.md                     # 评价关系抽取工具文档
│   └── 实体类型分类体系_opencode版.md    # 完整分类体系规格 (L1×4 L2×14 L3×59)
│
├── input/                              # 输入: reviewed_full_*.json
├── output/                             # 输出目录
│   ├── entities/                       # 最终实体结果 (JSONL + Summary)
│   ├── evaluative_relation/            # Agent 1 评价关系结果
│   └── eval/                           # 评估报告输出
└── mid_data/                           # 中间数据: Agent 2 实体抽取结果 + 关系
```

### 核心文件详解

#### 入口脚本

| 文件 | 作用 | 启动方式 |
|------|------|----------|
| [run.py](run.py) | 四Agent管道主入口，批量并行处理所有输入文件（评价关系由 Agent 1 在管道内直接产出并导出，无需独立步骤） | `python run.py --live` |
| [run_eval.py](run_eval.py) | 一键评估：自动加载 `api.txt`，运行 Pipeline 并计算指标 | `python run_eval.py` |
| [run_relation.py](run_relation.py) | 关系补抽/兼容脚本：从 `mid_data/` 或实体结果中单独抽取/读取评价关系 | `python run_relation.py --live` |
| [run_relation_agent.py](run_relation_agent.py) | 独立运行 Agent 1：单独抽取评价关系（快速试跑/批量生产关系数据） | `python run_relation_agent.py --live --sample 12` |
| [run_relation_verify.py](run_relation_verify.py) | 关系校验：对抽取结果逐条复核，事实/描述类关系标记 `is_evaluation=false`，评价类放行 | `python run_relation_verify.py --live --input-relations <jsonl>` |

#### 管道模块 (`pipeline/`)

| 文件 | 职责 |
|------|------|
| [llm.py](pipeline/llm.py) | **LLM 抽象层**：封装 DeepSeek/OpenAI API，支持 Thinking 推理模式、JSON 输出自动重试与修复、多轮 Voting 投票机制、指数退避重连 |
| [taxonomy.py](pipeline/taxonomy.py) | **分类体系**：定义 L1→L2→L3 完整层级映射表，提供 `get_l1_options()`、`get_l2_label()` 等查询工具函数 |
| [evaluative_relation_agent.py](pipeline/evaluative_relation_agent.py) | **Agent 1**：关系优先策略，先识别评价关系（subject/object/aspect/opinion/evidence），解析被评价对象并输出评价对象实体，作为下游已知实体 |
| [relation_verification_agent.py](pipeline/relation_verification_agent.py) | **校验 Agent**：对已抽取关系逐条复核，强制判断该关系是「评价」还是「事实/描述」。事实类标记 `is_evaluation=false` 并附 `fact_type`（研究行为/方法使用/定义/过程描述等九类），评价类放行。与 Agent 1 的内置过滤形成双保险，用于质量检测与数据清洗 |
| [entity_extraction_agent.py](pipeline/entity_extraction_agent.py) | **Agent 2**：接收 Agent 1 的已知实体，从评价句中补充抽取评价关系之外的其余实体，输出 mention、normalized_name、候选 L1/L3、evidence、confidence，严格过滤泛称/类别词 |
| [classification_agent.py](pipeline/classification_agent.py) | **Agent 3**：验证实体有效性 → 标注 L1/L2/L3 → L4 规则匹配。支持 Voting（多轮投票）、RAG（从 TermDB 检索相似示例）、schema 约束输出 |
| [reviewer_agent.py](pipeline/reviewer_agent.py) | **Agent 4**：以"图书馆学专业学长"视角审查 Agent 3 的分类结果，输出 Likert 1-5 置信度 + 修正建议 |
| [dynamic_term_db.py](pipeline/dynamic_term_db.py) | **动态术语底库**：线程安全的术语积累和检索系统，支持 trigram 相似度搜索（用于 RAG）、JSON 导入导出、自动去重 |
| [dual_agent_pipeline.py](pipeline/dual_agent_pipeline.py) | **管道编排器**：串联 Agent 1→2→3→4 的执行流程，管理中间数据写入、Likert 5 分实体入 TermDB、调用 JSONL/Summary/Relation 导出工具 |

#### 评估模块 (`eval/`)

| 文件 | 职责 |
|------|------|
| [gold_standard.py](eval/gold_standard.py) | 标注数据加载与验证：从 JSONL 读取人工标注，支持实体标注（mention/L1/L2/L3/valid_entity）和评价关系标注（subject/object/opinion/aspect/evidence） |
| [metrics.py](eval/metrics.py) | 指标计算：实体抽取/分类/严格匹配的 F1/Precision/Recall、逐类型指标、混淆矩阵；评价关系级 P/R/F1；句级 has_evaluation 准确性 |
| [reporter.py](eval/reporter.py) | 报告生成：输出 Markdown 格式的量化评估报告（含关系抽取与评价句判定章节） |
| [annotation_tool.py](eval/annotation_tool.py) | 交互式命令行标注工具，用于人工标注 Gold Standard 数据 |

#### 评估体系说明

评估覆盖两个层面：

1. **实体层面**（原有）：抽取级 P/R/F1、L1/L2/L3 逐层准确率、端到端严格匹配、有效/无效判定准确性、逐类型 F1、L3 混淆矩阵。
2. **评价关系层面**（V5.0 新增）：关系级 P/R/F1（subject + object + opinion 三元一致才算匹配，预测的 entity_id 自动解析回实体名参与匹配）；句级 has_evaluation 二元准确性（覆盖"有评价但无合法关系"的情形）。

运行方式：

```bash
python run_eval.py              # 一键评估: 加载api.txt + 术语库 + 标注数据
python run.py --eval --gold output/eval/gold_standard.jsonl
```

标注数据格式（JSONL，每行一句）：

```json
{
  "sentence_id": "reviewed_full_9::20",
  "sentence": "……完整上下文……",
  "gold_entities": [{"mention": "农村图书馆研究", "normalized_name": "农村图书馆研究",
                     "l1": "Abstract", "l2": "Epistemic", "l3_type_code": "subfield",
                     "valid_entity": true}],
  "gold_relations": [{"subject": "_paper_author", "object": "农村图书馆研究",
                      "aspect": "研究水平", "opinion": "有待提高", "evidence": "研究水平有待提高"}]
}
```

旧版标注文件（仅含 `gold_entities`）可直接加载，关系指标自动跳过。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

系统为 4 个 Agent 使用独立的 API Key（也可全部回退到同一把 Key），完全通过环境变量读取，配置在进程启动时即固定，**请先设置环境变量再运行 `run.py`**。

**方式 A — 环境变量 (推荐)：**

Linux / macOS：

```bash
export DEEPSEEK_API_KEY_RELATION="sk-xxx"        # Agent 1 评价关系用
export DEEPSEEK_API_KEY_EXTRACTION="sk-xxx"      # Agent 2 实体抽取用
export DEEPSEEK_API_KEY_CLASSIFICATION="sk-xxx"  # Agent 3 分类用
export DEEPSEEK_API_KEY_REVIEWER="sk-xxx"        # Agent 4 审查用
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # 可选
export LLM_MODEL="deepseek-chat"                     # 可选
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY_RELATION="sk-xxx"
$env:DEEPSEEK_API_KEY_EXTRACTION="sk-xxx"
$env:DEEPSEEK_API_KEY_CLASSIFICATION="sk-xxx"
$env:DEEPSEEK_API_KEY_REVIEWER="sk-xxx"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:LLM_MODEL="deepseek-chat"
```

> **单把 Key 全共用**：只设 `B2_LLM_API_KEY` 即可（它是所有 Agent 的最终兜底 Key）。
> 各 Agent 的 Key 回退链：`DEEPSEEK_API_KEY_*` → 各自次选 → `B2_LLM_API_KEY`。

**方式 B — api.txt 文件（供 `run_eval.py` 使用）：**

在项目根目录创建 `api.txt`，每行一个 Key（抽取/分类/审查）：
```
sk-xxx-extraction
sk-xxx-classification
sk-xxx-reviewer
```

### 3. 准备输入数据

将 `reviewed_full_*.json` 文件放入 `input/` 目录，格式如下：

```json
{
    "1": {
        "previous_sentence": "前一句文本",
        "evaluative_sentence": "评价句主体文本",
        "next_sentence": "后一句文本"
    }
}
```

> 程序自动将前后句与评价句拼接为完整上下文送入模型。

### 4. 运行

```bash
# 全量模式 (默认处理所有输入文件)
python run.py --live

# 显式指定全量 / 测试(随机抽取 10 个文件)
python run.py --live --mode full
python run.py --live --mode test

# 单文件冒烟测试
python run.py --live --limit 1 --no-skip --concurrency 1

# 高级: 启用 Voting + RAG + Thinking
python run.py --live --mode full --voting --rag --thinking

# 控制并发
python run.py --live --mode full --concurrency 20

# 强制重新处理 (不跳过已有输出)
python run.py --live --mode full --no-skip

# 评估模式 (需标注数据)
python run.py --eval --gold eval/gold_data.jsonl

# 一键评估 (自动加载 api.txt)
python run_eval.py

# 关系补抽 (兼容/辅助)
python run_relation.py --live

# 随机采样50句运行, 输出人工审阅报告 (默认seed=42, 修改脚本内SAMPLE_SIZE调整数量)
python run_sample_review.py
```

> 运行 `run.py` 时，评价关系由 Agent 1 在主管道内直接产出，并自动导出到 `output/evaluative_relation/`，无需再单独运行关系抽取脚本。`run_relation.py` 用于对已有中间/实体结果做补抽或兼容旧流程。`run_sample_review.py` 生成的审阅报告（`output/sample_review_50_report.md`）逐句展示原文、实体表与评价关系，适合人工检查效果。

---

## 命令行参数

### run.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--live` | 启用真实 LLM API (需设置 4 把 API Key) | `False` |
| `--mode` | 运行模式: `test` / `full` | `full` |
| `--limit N` | 限制处理文件数, 0=不限制 | `0` |
| `--concurrency N` | 并行处理文件数 | `40` |
| `--no-skip` | 不跳过已有输出, 强制重跑 | `False` |
| `--dynamic-out` | 动态术语库导出路径 | `output/dynamic_terms.json` |
| `--eval` | 启用评估模式 | `False` |
| `--gold` | 标注数据路径 | `eval/gold_data.jsonl` |
| `--eval-output` | 评估报告输出路径 | `eval/report.md` |
| `--thinking` | 启用 DeepSeek 推理模式 | `False` |
| `--no-thinking` | 禁用推理模式 | — |
| `--voting` | 启用 Agent 3 分类多轮投票 | `False` |
| `--no-voting` | 禁用多轮投票 | — |
| `--rag` | 启用 RAG Few-Shot 示例 | `False` |
| `--dynamic-terms` | 预加载术语库路径 | `dynamic_terms.json` |
| `--max-terms` | 术语库最大加载量, 0=不限制 | `0` |

---

## 输出格式

### 最终结果 (`output/entities/{name}_result.jsonl`)

每行一个实体分类结果，完整字段：

```json
{
    "sentence_id": "1",
    "entity_id": "1_e1",
    "sentence": "完整上下文句",
    "entity": "中国图书馆学会",
    "normalized_name": "中国图书馆学会",
    "valid_entity": true,
    "invalid_reason": "",
    "l1": "Agent",
    "l2": "Organization",
    "l2_label": "组织机构",
    "l3_type_code": "professional",
    "l3_label": "职业共同体组织",
    "l4_matched_term": "",
    "l4_term_type": null,
    "evidence": "证据片段",
    "reason": "分类理由",
    "other_suggestion": "",
    "likert_confidence": 5,
    "reviewer_comment": "分类正确，中国图书馆学会为职业共同体组织",
    "suggested_correction": ""
}
```

| 字段分组 | 字段 | 含义 |
|----------|------|------|
| **元数据** | `sentence_id`, `entity_id`, `sentence` | 来源句编号、实体编号、完整上下文 |
| **实体** | `entity`, `normalized_name` | 原文表述与规范化名称 |
| **有效性** | `valid_entity`, `invalid_reason` | Agent 3 判定结果 |
| **分类** | `l1`, `l2`, `l2_label`, `l3_type_code`, `l3_label` | L1/L2/L3 层级分类 |
| **L4** | `l4_matched_term`, `l4_term_type` | 术语底库匹配结果 |
| **依据** | `evidence`, `reason`, `other_suggestion` | 分类证据与理由 |
| **审查** | `likert_confidence` (1-5), `reviewer_comment`, `suggested_correction` | Agent 4 Likert 评分 |

### 汇总统计 (`output/entities/{name}_summary.json`)

L1 分布、L3 类型分布、Likert 五点分布、平均分等聚合指标。

### 评价关系 (`output/evaluative_relation/relation_full_{num}.jsonl`)

由 **Agent 1** 产出，每行一个句子的评价关系结果。每条关系字段为：

```json
{
    "sentence_id": "1",
    "sentence": "完整上下文句",
    "subject": "_paper_author",
    "object": "1_e1",
    "aspect": "研究水平",
    "opinion": "偏低",
    "evidence": "研究水平偏低"
}
```

| 字段 | 含义 |
|------|------|
| `subject` | 评价主体：`_paper_author`（本文作者）、`_cite[N]`（第N篇被引文献）、具体人名、`_unknown` |
| `object` | 被评价对象的最终实体 ID（已由管道重映射，可直接关联 `output/entities/` 中的 `entity_id`）；无法解析时为 `_missing_entity`（附 `object_text` 原文） |
| `aspect` | 评价方面；没有则为 `null` |
| `opinion` | 评价表达（完整片段） |
| `evidence` | 支持该评价的最小原文片段 |

`mid_data/{name}_extracted.json` 中同样保存了按句分组的关系与 `has_evaluation` 标记，供断点续传与 `run_relation.py --from-agent1` 读取。

### 中间数据 (`mid_data/{name}_extracted.json`)

Agent 1 评价关系 + Agent 2 实体抽取的合并中间结果（stage = `agent1_relation_agent2_entity_extraction`），用于调试和断点续传。

---

## 环境变量参考

| 变量 | 说明 | 回退链 |
|------|------|--------|
| `DEEPSEEK_API_KEY_RELATION` | Agent 1 评价关系 Key | → `DEEPSEEK_API_KEY` → `B2_LLM_API_KEY` |
| `DEEPSEEK_API_KEY_EXTRACTION` | Agent 2 实体抽取 Key | → `DEEPSEEK_API_KEY_L1` → `B2_LLM_API_KEY` |
| `DEEPSEEK_API_KEY_CLASSIFICATION` | Agent 3 分类 Key | → `DEEPSEEK_API_KEY_L2` → `B2_LLM_API_KEY` |
| `DEEPSEEK_API_KEY_REVIEWER` | Agent 4 审查 Key | → `DEEPSEEK_API_KEY_L3` → `B2_LLM_API_KEY` |
| `DEEPSEEK_API_KEY` | Agent 1 次选 / 通用 Key | — |
| `B2_LLM_API_KEY` | 所有 Agent 的最终兜底 Key | — |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |

---

## 关键配置 (`config.py`)

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `LLMConfig.temperature` | 0.0 | 确保分类一致性 |
| `LLMConfig.max_tokens` | 4096 | 每次请求最大输出 |
| `LLMConfig.timeout` | 60s | API 超时时间 |
| `LLMConfig.enable_thinking_relation` | False | Agent 1 推理模式开关 |
| `LLMConfig.enable_thinking_extraction` | False | Agent 2 推理模式开关 |
| `LLMConfig.enable_thinking_classification` | False | Agent 3 推理模式开关 |
| `LLMConfig.enable_thinking_reviewer` | False | Agent 4 推理模式开关 |
| `PipelineConfig.batch_size` | 12 | Agent 3 每批处理实体数 |
| `PipelineConfig.voting_rounds` | 3 | Voting 轮数 |
| `PipelineConfig.voting_temperature` | 0.3 | Voting 温度 |
| `PipelineConfig.rag_k_examples` | 5 | RAG 检索示例数 |

---

## 数据流转全景

```
input/reviewed_full_*.json           # 输入: 评价句 + 上下文
        │
        ▼
  [Agent 1: 评价关系抽取]  ── 评价对象实体 ──┐
        │                                   │ (作为已知实体)
        ├──────────────────────────────┐    │
        ▼                              ▼    ▼
output/evaluative_relation/      [Agent 2: 实体抽取补充]
  relation_full_{num}.jsonl                 │
                                       ▼
                          mid_data/{name}_extracted.json   # 中间: 合并实体列表
                                       │
                                       ▼
                             [Agent 3: 精分类 + L4匹配]
                                       │
                                       ▼
                             [Agent 4: Likert 审查]
                                       │
                                       ├──────────────────────────┐
                                       ▼                          ▼
                             output/entities/               dynamic_terms.json
                               {name}_result.jsonl          (Likert=5 实体自动入库)
                               {name}_summary.json
```
