# Ontology Agent V4.3 — 三Agent端到端文献知识挖掘系统

> **多智能体协同 · DeepSeek API · 并行批处理 · L1/L2/L3 精分类 + L4 规则匹配 · 评价关系抽取 · 评估体系**

面向情报学/图书馆学领域学术文献，从评价句到结构化知识实体全自动端到端挖掘管道，支持抽取 → 分类 → 审查 → 评价关系抽取的完整链路。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent 1: Entity Extraction (实体抽取 — 高召回)                    │
│  评价句 → L1/L2 候选实体 + 证据片段 + 置信度                        │
│  输出: mid_data/{name}_extracted.json                             │
├──────────────────────────────────────────────────────────────────┤
│  Agent 2: Classification (精分类 — L1/L2/L3 + L4)                  │
│  Step 1: 验证实体有效性 (过滤泛称/类别词)                          │
│  Step 2: L1/L2/L3 层级精标注 (Agent/Artifact/Abstract/Event)       │
│  Step 3: L4 规则匹配 (与 DynamicTermDB 做字符串匹配)                │
├──────────────────────────────────────────────────────────────────┤
│  Agent 3: Review (审查 — Likert 5点量表)                           │
│  从图书馆学专业视角审查分类结果，输出 1-5 分置信度                  │
│  输出: output/entities/{name}_result.jsonl + {name}_summary.json   │
├──────────────────────────────────────────────────────────────────┤
│  Agent 4: Evaluative Relation Extraction (评价关系抽取) [NEW]      │
│  从已抽取实体的句子中识别 评价主体→评价对象 关系                    │
│  输出: output/evaluative_relation/relation_full_{num}.jsonl        │
└──────────────────────────────────────────────────────────────────┘
```

### V4.3 新增特性

| 特性 | 说明 |
|------|------|
| **Thinking Mode** | 支持 DeepSeek 推理模式，每个 Agent 可独立开关 |
| **Voting** | Agent 2 多轮投票机制，减少分类不确定性 |
| **RAG Few-Shot** | 基于动态术语库的相似度检索，为分类提供示例 |
| **评价关系抽取** | 新增 Agent 4，从已分类实体中抽取评价主体-对象关系 |
| **评估体系** | 完整的 eval 模块：标注工具、指标计算、报告生成 |

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
ontology_agent_2_latest_main/
│
├── run.py                              # 【主入口】批量并行处理 三Agent管道
├── run_eval.py                         # 【评估入口】一键评估脚本 (自动加载api.txt)
├── extract_relation.py                 # 【关系抽取入口】从实体结果中抽取评价关系
├── config.py                           # 统一配置系统 (LLM / Pipeline / 路径)
├── api.txt                             # API Key 存储文件 (3行: 抽取/分类/审查)
├── requirements.txt                    # Python 依赖
├── dynamic_terms.json                  # 预加载的动态术语库 (可选)
│
├── pipeline/                           # 核心管道包
│   ├── __init__.py                     # 包定义 (v4.2.0) + 公开API导出
│   ├── llm.py                          # LLM客户端: OpenAI兼容, Thinking/Voting/JSON重试
│   ├── taxonomy.py                     # L1/L2/L3 分类体系层级映射 + 中文标签
│   ├── entity_extraction_agent.py      # Agent 1: 实体抽取 Prompt (LangGPT风格)
│   ├── classification_agent.py         # Agent 2: 精分类 + Voting + RAG + L4匹配
│   ├── reviewer_agent.py               # Agent 3: Likert 5点量表审查 Prompt
│   ├── evaluative_relation_agent.py    # Agent 4: 评价关系抽取 (主体-对象)
│   ├── dynamic_term_db.py              # 线程安全动态术语底库 (trigram检索)
│   └── dual_agent_pipeline.py          # 三Agent管道编排 + JSONL/Summary导出
│
├── eval/                               # 评估模块
│   ├── __init__.py                     # 评估包定义 (v1.0.0)
│   ├── gold_standard.py                # 标注数据模型 (GoldStandard/GoldEntity/GoldSentence)
│   ├── metrics.py                      # F1/Precision/Recall + 混淆矩阵 + 逐类型指标
│   ├── reporter.py                     # Markdown 评测报告生成
│   └── annotation_tool.py             # 命令行交互式标注工具
│
├── src/                                # 向后兼容层
│   └── __init__.py                     # 重新导出 pipeline.* (v2.0.0)
│
├── doc/                                # 文档
│   ├── ARCHITECTURE.md                 # V4.2 架构详细文档 (37 KB)
│   ├── RFREADME.md                     # 评价关系抽取工具文档
│   └── 实体类型分类体系_opencode版.md    # 完整分类体系规格 (63 KB, L1×4 L2×14 L3×59)
│
├── input/                              # 输入: reviewed_full_*.json
├── output/                             # 输出目录
│   ├── entities/                       # Agent 1-3 最终结果 (JSONL + Summary)
│   ├── evaluative_relation/            # Agent 4 评价关系结果
│   └── eval/                           # 评估报告输出
└── mid_data/                           # 中间数据: Agent 1 抽取结果
```

### 核心文件详解

#### 入口脚本

| 文件 | 作用 | 启动方式 |
|------|------|----------|
| [run.py](run.py) | 三Agent管道主入口，批量并行处理所有输入文件 | `python run.py --live` |
| [run_eval.py](run_eval.py) | 一键评估：自动加载 `api.txt`，运行 Pipeline 并计算指标 | `python run_eval.py` |
| [extract_relation.py](extract_relation.py) | 评价关系抽取：从 `output/entities/` 读取实体结果，输出评价主体-对象关系 | `python extract_relation.py --live` |

#### 管道模块 (`pipeline/`)

| 文件 | 职责 |
|------|------|
| [llm.py](pipeline/llm.py) | **LLM 抽象层**：封装 DeepSeek/OpenAI API，支持 Thinking 推理模式、JSON 输出自动重试与修复、多轮 Voting 投票机制、指数退避重连 |
| [taxonomy.py](pipeline/taxonomy.py) | **分类体系**：定义 L1→L2→L3 完整层级映射表，提供 `get_l1_options()`、`get_l2_label()` 等查询工具函数 |
| [entity_extraction_agent.py](pipeline/entity_extraction_agent.py) | **Agent 1**：从评价句中高召回抽取实体，输出 mention、normalized_name、候选 L1/L3、evidence、confidence。严格过滤泛称/类别词 |
| [classification_agent.py](pipeline/classification_agent.py) | **Agent 2**：验证实体有效性 → 标注 L1/L2/L3 → L4 规则匹配。支持 Voting（多轮投票）、RAG（从 TermDB 检索相似示例）、schema 约束输出 |
| [reviewer_agent.py](pipeline/reviewer_agent.py) | **Agent 3**：以"图书馆学专业学长"视角审查 Agent 2 的分类结果，输出 Likert 1-5 置信度 + 修正建议 |
| [evaluative_relation_agent.py](pipeline/evaluative_relation_agent.py) | **Agent 4**：从已标注实体的句子中抽取评价关系（评价主体 → 评价对象 → 评价极性），输出结构化关系三元组 |
| [dynamic_term_db.py](pipeline/dynamic_term_db.py) | **动态术语底库**：线程安全的术语积累和检索系统，支持 trigram 相似度搜索（用于 RAG）、JSON 导入导出、自动去重 |
| [dual_agent_pipeline.py](pipeline/dual_agent_pipeline.py) | **管道编排器**：串联 Agent 1→2→3 的执行流程，管理中间数据写入、Likert 5 分实体入 TermDB、调用导出工具 |

#### 评估模块 (`eval/`)

| 文件 | 职责 |
|------|------|
| [gold_standard.py](eval/gold_standard.py) | 标注数据加载与验证：从 JSONL 读取人工标注的 Gold Standard，支持句子级和实体级标注 |
| [metrics.py](eval/metrics.py) | 指标计算：精确匹配 / 宽松匹配的 F1/Precision/Recall、逐类型指标、混淆矩阵 |
| [reporter.py](eval/reporter.py) | 报告生成：输出 Markdown 格式的量化评估报告 |
| [annotation_tool.py](eval/annotation_tool.py) | 交互式命令行标注工具，用于人工标注 Gold Standard 数据 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

**方式 A — 环境变量 (推荐)：**

```bash
export DEEPSEEK_API_KEY_EXTRACTION="sk-xxx"    # Agent 1 抽取用
export DEEPSEEK_API_KEY_CLASSIFICATION="sk-xxx" # Agent 2 分类用
export DEEPSEEK_API_KEY_REVIEWER="sk-xxx"       # Agent 3 审查用
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # 可选
export LLM_MODEL="deepseek-chat"                     # 可选
```

三个 Agent 使用独立的 API Key，可分别配置以控制成本与并发。

**方式 B — api.txt 文件：**

在项目根目录创建 `api.txt`，每行一个 Key：
```
sk-xxx-extraction
sk-xxx-classification
sk-xxx-reviewer
```
运行 `run_eval.py` 时会自动加载。

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
# 测试模式 (随机抽取 10 个文件)
python run.py --live

# 全量模式 (处理所有输入文件)
python run.py --live --mode full

# 高级: 启用 Voting + RAG + Thinking
python run.py --live --mode full --voting --rag --thinking

# 控制并发
python run.py --live --mode full --concurrency 20

# 限制文件数
python run.py --live --limit 50

# 强制重新处理
python run.py --live --mode full --no-skip

# 评估模式 (需标注数据)
python run.py --eval --gold eval/gold_data.jsonl

# 一键评估 (自动加载 api.txt)
python run_eval.py

# 评价关系抽取
python extract_relation.py --live
```

---

## 命令行参数

### run.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--live` | 启用真实 LLM API (需设置 API Key) | `False` |
| `--mode` | 运行模式: `test` / `full` | `test` |
| `--limit N` | 限制处理文件数, 0=不限制 | `0` |
| `--concurrency N` | 并行处理文件数 | `40` |
| `--no-skip` | 不跳过已有输出, 强制重跑 | `False` |
| `--dynamic-out` | 动态术语库导出路径 | `output/dynamic_terms.json` |
| `--eval` | 启用评估模式 | `False` |
| `--gold` | 标注数据路径 | `eval/gold_data.jsonl` |
| `--eval-output` | 评估报告输出路径 | `eval/report.md` |
| `--thinking` | 启用 DeepSeek 推理模式 | `False` |
| `--no-thinking` | 禁用推理模式 | — |
| `--voting` | 启用 Agent 2 多轮投票 | `False` |
| `--no-voting` | 禁用多轮投票 | — |
| `--rag` | 启用 RAG Few-Shot 示例 | `False` |
| `--dynamic-terms` | 预加载术语库路径 | `dynamic_terms.json` |
| `--max-terms` | 术语库最大加载量, 0=不限制 | `0` |

### extract_relation.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--live` | 启用真实 LLM API | `False` |
| `--limit N` | 限制处理文件数 | `0` |
| `--concurrency N` | 并行处理文件数 | `4` |
| `--no-skip` | 覆盖已有关系输出 | `False` |
| `--input-dir` | 实体 JSONL 输入目录 | `output/entities` |
| `--output-dir` | 关系 JSONL 输出目录 | `output/evaluative_relation` |

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
| **有效性** | `valid_entity`, `invalid_reason` | Agent 2 判定结果 |
| **分类** | `l1`, `l2`, `l2_label`, `l3_type_code`, `l3_label` | L1/L2/L3 层级分类 |
| **L4** | `l4_matched_term`, `l4_term_type` | 术语底库匹配结果 |
| **依据** | `evidence`, `reason`, `other_suggestion` | 分类证据与理由 |
| **审查** | `likert_confidence` (1-5), `reviewer_comment`, `suggested_correction` | Agent 3 Likert 评分 |

### 汇总统计 (`output/entities/{name}_summary.json`)

L1 分布、L3 类型分布、Likert 五点分布、平均分等聚合指标。

### 评价关系 (`output/evaluative_relation/relation_full_{num}.jsonl`)

每行一个句子的评价关系结果，包含 `has_evaluation` 标记和 `relations` 列表（主体-对象-极性三元组）。

### 中间数据 (`mid_data/{name}_extracted.json`)

Agent 1 抽取的原始实体列表，用于调试和断点续传。

---

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY_EXTRACTION` | Agent 1 抽取 Key | — |
| `DEEPSEEK_API_KEY_CLASSIFICATION` | Agent 2 分类 Key | — |
| `DEEPSEEK_API_KEY_REVIEWER` | Agent 3 审查 Key | — |
| `DEEPSEEK_API_KEY` | 回退统一 Key (未分层配置时) | — |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |

---

## 关键配置 (`config.py`)

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `LLMConfig.temperature` | 0.0 | 确保分类一致性 |
| `LLMConfig.max_tokens` | 4096 | 每次请求最大输出 |
| `LLMConfig.timeout` | 60s | API 超时时间 |
| `PipelineConfig.batch_size` | 12 | Agent 2 每批处理实体数 |
| `PipelineConfig.max_l1_per_entity` | 2 | 每个实体最多候选 L1 数 |
| `PipelineConfig.voting_rounds` | 3 | Voting 轮数 |
| `PipelineConfig.voting_temperature` | 0.3 | Voting 温度 |
| `PipelineConfig.rag_k_examples` | 5 | RAG 检索示例数 |

---

## 数据流转全景

```
input/reviewed_full_*.json           # 输入: 评价句 + 上下文
        │
        ▼
  [Agent 1: 实体抽取]
        │
        ▼
mid_data/{name}_extracted.json       # 中间: 候选实体列表
        │
        ▼
  [Agent 2: 精分类 + L4匹配]
        │
        ▼
  [Agent 3: Likert 审查]
        │
        ├──────────────────────────────┐
        ▼                              ▼
output/entities/                   dynamic_terms.json
  {name}_result.jsonl              (Likert=5 实体自动入库)
  {name}_summary.json
        │
        ▼
  [Agent 4: 评价关系抽取]  ← extract_relation.py
        │
        ▼
output/evaluative_relation/
  relation_full_{num}.jsonl
```
