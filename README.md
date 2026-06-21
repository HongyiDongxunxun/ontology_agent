# final_Agent — 三Agent端到端文献知识挖掘系统

> V4.2 · 多智能体协同 · DeepSeek API · 并行批处理 · L1/L2/L3 精分类 + L4 规则匹配

面向情报学领域含 OCR 噪声的学术文献，从评价句到结构化知识实体全自动端到端挖掘管道。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Agent 1: Entity Extraction (实体抽取 — 高召回)               │
│  评价句 → L1/L2 候选实体 + 证据片段 + 置信度                   │
│  输出: mid_data/{name}_extracted.json                        │
├─────────────────────────────────────────────────────────────┤
│  Agent 2: Classification (精分类 — L1/L2/L3)                  │
│  Step 1: 验证实体有效性 (过滤泛称/类别词)                      │
│  Step 2: L1/L2/L3 层级精标注 (Agent/Artifact/Abstract/Event)  │
│  Step 3: L4 规则匹配 (与 DynamicTermDB 做字符串匹配)            │
├─────────────────────────────────────────────────────────────┤
│  Agent 3: Review (审查 — Likert 5点量表)                      │
│  从图书馆学专业视角审查分类结果，输出 1-5 分置信度              │
│  输出: output/{name}_result.jsonl + {name}_summary.json       │
└─────────────────────────────────────────────────────────────┘
```

## 实体分类体系

### 顶层 L1 (4 大类)

| L1 | 说明 | L2 子类 |
|----|------|---------|
| `Agent` | 行为主体 | Person / Organization |
| `Artifact` | 人工制品 | Discursive / Organizational / Empirical / Normative / System / Tool |
| `Abstract` | 抽象实体 | Conceptual / KnowledgeClaim / Methodological / Epistemic / Phenomenon |
| `Event` | 事件 | Event |

### L4 MicroMapping (术语底库)

L4 层采用**规则匹配**：Agent 2 完成 L1/L2/L3 分类后，将实体名称与 `DynamicTermDB`（线程安全动态术语底库）中的术语做精确匹配和包含匹配。术语底库启动时为空，运行过程中 Likert 5分（完全认同）的实体自动积累入库，后续文件受益于前序文件的术语积累。

详见 `pipeline/dynamic_term_db.py` 和 `实体类型分类体系_opencode版.md`。

## 目录结构

```
final_Agent/
├── run.py                          # 主入口 (批量并行处理)
├── config.py                       # 统一配置 (LLM / 管道)
├── requirements.txt                # Python 依赖
├── pipeline/
│   ├── __init__.py                 # 包定义 (v4.2.0)
│   ├── llm.py                      # LLM 客户端 (OpenAI 兼容)
│   ├── taxonomy.py                 # L1/L2/L3 分类体系 + 中文标签
│   ├── entity_extraction_agent.py  # Agent 1: 实体抽取 Prompt + 解析
│   ├── classification_agent.py     # Agent 2: 精分类 Prompt + 规则 L4 匹配
│   ├── reviewer_agent.py           # Agent 3: Likert 审查 Prompt + 解析
│   ├── dynamic_term_db.py          # 线程安全动态术语底库 (L4 规则匹配用)
│   └── dual_agent_pipeline.py      # 三Agent管道编排 + 导出工具
├── input/                          # 输入: reviewed_full_*.json
├── output/                         # 输出: *_result.jsonl + *_summary.json
├── mid_data/                       # 中间: *_extracted.json
├── 实体类型分类体系_opencode版.md    # 分类体系完整文档
└── API文档.txt                      # (仅供参考, 不在版本控制中)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境变量 (真实 LLM 模式)

```bash
export DEEPSEEK_API_KEY_EXTRACTION="sk-xxx"   # Agent 1 抽取用 Key
export DEEPSEEK_API_KEY_CLASSIFICATION="sk-xxx" # Agent 2 分类用 Key
export DEEPSEEK_API_KEY_REVIEWER="sk-xxx"       # Agent 3 审查用 Key
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # 可选, 默认值
export LLM_MODEL="deepseek-chat"                     # 可选, 默认值
```

Agent 1、Agent 2、Agent 3 使用独立的 API Key，可分别配置以控制成本与并发。

### 测试模式运行 (随机抽取 10 个文件)

```bash
python run.py --live
```

### 全量模式运行

```bash
python run.py --live --mode full
```

### 控制并发数

```bash
python run.py --live --mode full --concurrency 20
```

### 限制处理文件数

```bash
python run.py --live --limit 50
```

### 强制重新处理 (不跳过已有输出)

```bash
python run.py --live --mode full --no-skip
```

## 输入格式

`input/` 目录下为 `reviewed_full_*.json` 文件，每条记录包含:

```json
{
    "1": {
        "previous_sentence": "前一句",
        "evaluative_sentence": "评价句主体",
        "next_sentence": "后一句"
    }
}
```

程序自动将前后句拼接为完整上下文送入模型。

## 输出格式

项目输出分为两层: 中间数据 (Agent 1) 和最终结果 (Agent 2)。

### 中间数据 (`mid_data/{name}_extracted.json`)

Agent 1 实体抽取的中间结果, 每条提取出的实体包含:

| 字段 | 类型 | 含义 |
|------|------|------|
| `entity_id` | `str` | 实体编号, 格式 `{句号}_e{序号}`, 如 `3_e2` 表示第3句的第2个实体 |
| `mention` | `str` | 从原文精确摘取的实体表述, 禁止改写 |
| `normalized_name` | `str` | 规范化实体名 (去冗余修饰、去括号等), 无特殊情况等于 `mention` |
| `candidate_l1` | `list[str]` | 候选顶层类别, 如 `["Agent", "Abstract"]` — 召回优先, 允许一个实体有多个候选 L1 |
| `candidate_l3` | `str` | 候选叶子类型代码, 如 `"scholar"`、`"book"`、`"theory"` |
| `evidence` | `str` | 原句中证明该实体存在的文本片段 |
| `is_specific_entity` | `bool` | 是否为具体实体: `true`=可唯一识别, `false`=泛称/类别词 (如"科学家""图书馆") |
| `confidence` | `float` | Agent 1 自评置信度, 范围 0~1 |
| `uncertainty` | `str` | 不确定性说明, 如 `"可兼为concept"` 表示该实体可能跨类归属, 无则空字符串 |

### 最终 JSONL (`output/{name}_result.jsonl`)

每行一个实体分类结果:

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

#### 字段说明

**元数据**

| 字段 | 类型 | 含义 |
|------|------|------|
| `sentence_id` | `str` | 来源句编号, 对应输入 JSON 的 key |
| `entity_id` | `str` | 实体编号 (继承自 Agent 1) |
| `sentence` | `str` | 完整上下文句 (前句 + 评价句 + 后句拼接) |

**实体标记**

| 字段 | 类型 | 含义 |
|------|------|------|
| `entity` | `str` | 实体原文表述 (同 `mention`) |
| `normalized_name` | `str` | 规范化实体名 |

**有效性验证**

| 字段 | 类型 | 含义 |
|------|------|------|
| `valid_entity` | `bool` | `true`=有效具体实体, `false`=被 Agent 2 判定为无效 (泛称/类别词/非实体) |
| `invalid_reason` | `str` | 无效原因, 如 `"泛称群体，不是具体可指称实体"`, 有效实体时为空 |

**L1/L2/L3 层级分类**

| 字段 | 类型 | 含义 |
|------|------|------|
| `l1` | `str` | 顶层类别: `Agent` / `Artifact` / `Abstract` / `Event` |
| `l2` | `str` | 第二层子类代码, 如 `Person`、`Discursive`、`KnowledgeClaim` |
| `l2_label` | `str` | L2 中文标签, 如 `"个人"`、`"话语性著作"`、`"知识主张层"` |
| `l3_type_code` | `str` | 第三层叶子类型代码 (最终确定类型), 如 `"scholar"`、`"book"`、`"theory"` |
| `l3_label` | `str` | L3 中文标签, 如 `"知识生产者"`、`"专著"`、`"理论"` |

**L4 MicroMapping**

| 字段 | 类型 | 含义 |
|------|------|------|
| `l4_matched_term` | `str` | 与 DynamicTermDB 中匹配到的术语; 未匹配时为空 |
| `l4_term_type` | `str?` | 匹配到的术语类型代码; 未匹配时为 `null` |

> L4 采用**规则匹配**：Agent 2 分类完成后，对 valid_entity=true 的实体执行精确匹配和包含匹配。不依赖 LLM。

**分类依据**

| 字段 | 类型 | 含义 |
|------|------|------|
| `evidence` | `str` | Agent 2 判断分类所使用的原句证据片段 |
| `reason` | `str` | 简短分类理由说明 |
| `other_suggestion` | `str` | 当 L3 为 `Other` 时, 必须填写的建议类型描述 (供体系迭代); 非 `Other` 时为空 |

**Agent 3 Likert 审查**

| 字段 | 类型 | 含义 |
|------|------|------|
| `likert_confidence` | `int` | Likert 5点量表: 1=完全不认同 ~ 5=完全认同 |
| `reviewer_comment` | `str` | 图书馆学专业学长审查理由 |
| `suggested_correction` | `str` | likert_confidence ≤ 2 时的修正建议

### 汇总统计 (`output/{name}_summary.json`)

| 字段 | 类型 | 含义 |
|------|------|------|
| `base_name` | `str` | 输入文件名 (不含扩展名) |
| `total_entities` | `int` | 总实体数 |
| `valid_entities` | `int` | 有效实体数 |
| `invalid_entities` | `int` | 无效实体数 |
| `l1_distribution` | `dict` | L1 分布, 如 `{"Abstract": 20, "Agent": 10}` |
| `l3_type_distribution` | `dict` | L3 类型分布, 如 `{"concept": 8, "scholar": 5}` |
| `likert_distribution` | `dict` | Likert 5点分布, 如 `{"5_完全认同": 71, "4_认同": 48}` |
| `likert_average` | `float` | Likert 平均分 |

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--live` | 启用真实 LLM API (需设置 3 个 API Key) | `False` |
| `--mode` | 运行模式: `test` / `full` | `test` (随机 10 文件) |
| `--limit N` | 限制处理文件数, 0=不限制 | `0` |
| `--concurrency N` | 并行处理文件数 | `40` |
| `--no-skip` | 不跳过已有输出, 强制重跑 | `False` |
| `--dynamic-out` | 动态术语库导出路径 | `output/dynamic_terms.json` |

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY_EXTRACTION` | Agent 1 抽取 Key | — |
| `DEEPSEEK_API_KEY_CLASSIFICATION` | Agent 2 分类 Key | — |
| `DEEPSEEK_API_KEY_REVIEWER` | Agent 3 审查 Key | — |
| `DEEPSEEK_API_KEY` | 回退统一 Key (未分层时) | — |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |

## 关键配置 (`config.py`)

- `LLMConfig.temperature`: 0.0 (确保分类一致)
- `LLMConfig.max_tokens`: 4096
- `PipelineConfig.batch_size`: 12 (Agent 2 每批处理实体数)
- `PipelineConfig.max_l1_per_entity`: 2 (每个实体最多进入几个候选 L1)
- L4 术语底库: 由 `DynamicTermDB` 在运行时从 Likert 5 分实体中动态积累，初始为空
