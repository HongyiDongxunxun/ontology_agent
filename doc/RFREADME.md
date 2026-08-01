# 评论索引实体抽取工具

这是一个独立的本地批处理工具，用于从综述论文的“评价语句”中抽取具体实体，并依据评论索引实体分类体系完成两阶段分类：

1. 第一阶段：从原始评价语句中高召回抽取具体实体，并判断候选 L1。
2. 第二阶段：基于第一阶段结果，按 L1 分组调用模型，复核实体有效性，并输出唯一 L2 和 L3。

当前目录已加入项目 `.gitignore`，不会进入主项目提交。代码运行时不读取原始 Markdown 分类方案，而是使用 `entity_taxonomy.py` 中内置的分类常量。把这个目录单独发给同事即可使用。

## 设计目标

- **召回优先**：第一阶段尽量找全具体实体，允许一个实体有多个候选 L1。
- **精分类准确**：第二阶段只在对应 L1 的标签空间里选择 L2/L3，降低一次性分类混淆。
- **实体边界严格**：只抽具体可指称对象，不抽“研究人员”“图书馆”“某论文”“图书馆数据库”等泛称、占位词或类别词。
- **可批量运行**：支持 JSONL、CSV、TXT 输入，输出 JSONL，便于后续进入数据库、标注工具或人工复核流程。
- **可追溯**：每条最终结果保留原句、实体、证据片段、分类理由和置信度。

## 目录结构

```text
review_entity_extraction_tool/
├── README.md                 # 使用说明，建议先读这个文件
├── PSEUDOCODE.md             # 两阶段流程伪代码
├── entity_taxonomy.py        # 内置实体分类体系，不依赖 Markdown
├── prompts.py                # LangGPT 风格提示词生成逻辑
├── extract_entities.py       # 命令行入口，负责读文件、调 API、写结果
├── PROMPTS.generated.md      # 已导出的完整提示词，便于人工检查或转发
└── sample_input.jsonl        # 最小输入示例
```

## 核心流程

```text
评价语句文件
  ↓
第一阶段 L1 粗抽取
  - 输入：原始评价语句
  - 输出：句子中的具体实体、候选 L1、证据片段
  - 特点：召回优先，允许多个 L1
  ↓
第二阶段 L2/L3 精分类
  - 输入：原句 + 第一阶段实体 + 候选 L1
  - 输出：valid_entity、L1、L2、L3、理由、置信度
  - 特点：按 L1 分组，只在当前 L1 的标签范围内分类
  ↓
最终 JSONL 结果
```

例如，第一阶段可能把“开放获取”标为候选 `Abstract` 和 `Event`。第二阶段会分别在两个 L1 范围内复核。如果原句语境指有组织行动，`Event > Event > movement` 会更可能成立；如果只是概念或理念，则更可能归入 `Abstract`。

## 输入格式

支持三种输入：

- JSONL：每行包含 `id` 和 `sentence`，也兼容 `text`、`content`、`评价语句` 字段。
- CSV：列名同上。
- TXT：每行一条评价语句。

示例 JSONL：

```jsonl
{"id":"s1","sentence":"建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。"}
{"id":"s2","sentence":"新版《图书馆学概论》观点更新颖,内容更充实,结构更合理。"}
```

CSV 示例：

```csv
id,sentence
s1,建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。
s2,新版《图书馆学概论》观点更新颖,内容更充实,结构更合理。
```

TXT 示例：

```text
建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。
新版《图书馆学概论》观点更新颖,内容更充实,结构更合理。
```

建议优先使用 JSONL，因为 JSONL 能稳定保留句子 id，后续更容易和原始数据回连。

## 环境变量

```bash
export LLM_API_KEY="你的 API Key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="你的模型名"
```

也可以在命令行传入 `--api-key`、`--base-url`、`--model`。

这个脚本调用的是 OpenAI-compatible Chat Completions 接口，请确认你的服务地址支持：

```text
POST {LLM_BASE_URL}/chat/completions
```

如果供应商不支持 `response_format={"type":"json_object"}`，运行时加 `--no-json-mode`。

## 运行

```bash
python review_entity_extraction_tool/extract_entities.py \
  --input /path/to/sentences.jsonl \
  --output /path/to/entity_results.jsonl \
  --batch-size 12
```

输出文件：

- `/path/to/entity_results.jsonl.stage1.jsonl`：第一阶段 L1 候选实体。
- `/path/to/entity_results.jsonl`：第二阶段 L2/L3 最终分类。
- `/path/to/entity_results.jsonl.errors.jsonl`：失败批次。

注意：脚本会覆盖同名输出文件。正式批量运行前，建议先用 10-30 条样本测试提示词表现。

## 分阶段运行

默认 `--stage all` 会连续执行两阶段。如果第一阶段结果已经生成，可以只跑第二阶段，节省 API 调用。

只跑第一阶段：

```bash
python review_entity_extraction_tool/extract_entities.py \
  --stage l1 \
  --input /path/to/sentences.jsonl \
  --output /path/to/entity_results.jsonl
```

第一阶段输出示例：

```json
{
  "sentence_id": "s1",
  "sentence": "建国以前在谱学研究领域颇有建树的学者有潘光旦、罗香林等人。",
  "entities": [
    {
      "entity_id": "s1_e1",
      "mention": "潘光旦",
      "normalized_name": "潘光旦",
      "aliases": [],
      "candidate_l1": ["Agent"],
      "evidence": "颇有建树的学者有潘光旦",
      "is_specific_entity": true,
      "uncertainty": "",
      "confidence": 0.95
    }
  ]
}
```

基于已有第一阶段结果只跑第二阶段：

```bash
python review_entity_extraction_tool/extract_entities.py \
  --stage l2 \
  --stage1-input /path/to/entity_results.jsonl.stage1.jsonl \
  --output /path/to/entity_results.jsonl
```

这种模式适合：

- 第一阶段抽取结果已经人工看过；
- 第二阶段提示词改了，需要重跑 L2/L3；
- API 中途失败，只想从已有 L1 结果继续。

导出完整 LangGPT 提示词：

```bash
python review_entity_extraction_tool/extract_entities.py \
  --write-prompts review_entity_extraction_tool/PROMPTS.generated.md
```

导出的 `PROMPTS.generated.md` 不参与运行，只是便于人工检查、复制给同事或放入标注说明文档。真正运行时，脚本会从 `prompts.py` 动态生成提示词。

## 结果字段

最终 JSONL 每行是一个实体分类：

```json
{
  "sentence_id": "s1",
  "entity_id": "s1_e1",
  "sentence": "原句",
  "entity": "实体",
  "normalized_name": "规范化实体名",
  "valid_entity": true,
  "invalid_reason": "",
  "l1": "Agent",
  "l2": "Person",
  "l2_label": "个人",
  "l3_type_code": "scholar",
  "l3_label": "知识生产者",
  "evidence": "分类证据",
  "reason": "分类理由",
  "other_suggestion": "",
  "confidence": 0.92
}
```

字段含义：

- `sentence_id`：输入句子的 id。
- `entity_id`：第一阶段为实体生成的稳定编号。
- `sentence`：原始评价语句。
- `entity`：原句中的实体表述。
- `normalized_name`：规范化实体名，无法规范化时等于 `entity`。
- `valid_entity`：第二阶段复核后，该实体是否有效。
- `invalid_reason`：无效原因，例如“泛称群体，不是具体可指称实体”。
- `l1`：顶层标签，取值为 `Agent`、`Artifact`、`Abstract`、`Event`。
- `l2`：第二层标签代码，例如 `Person`、`DiscursiveWork`、`KnowledgeClaim`。
- `l3_type_code`：第三层叶子类型代码，例如 `scholar`、`book`、`theory`。
- `evidence`：模型用于判断的原句证据片段。
- `reason`：简短分类理由。
- `other_suggestion`：当 L3 为 `Other` 或 `other` 时必须填写。
- `confidence`：模型自评置信度，范围通常为 0-1。

## 建议参数

- 召回优先：`--batch-size 8` 到 `12`，降低单次上下文压力。
- 低随机性：`--temperature 0.1`。
- 供应商不支持 JSON mode 时：加 `--no-json-mode`。
- 模糊实体保留多个 L1：默认 `--max-l1-per-entity 2`。

## 常用参数说明

```text
--input              输入文件，支持 JSONL/CSV/TXT。
--output             最终输出 JSONL 文件路径。
--stage              all/l1/l2，默认 all。
--stage1-input       只跑第二阶段时，传入已有 stage1 JSONL。
--batch-size         单次 API 调用处理多少条句子或实体。
--max-l1-per-entity  每个实体最多进入几个候选 L1 进行第二阶段分类。
--base-url           OpenAI-compatible API base URL。
--model              模型名。
--temperature        建议 0.0-0.2，避免分类飘。
--max-tokens         单次响应最大 token 数。
--timeout            单次请求超时时间，单位秒。
--max-retries        请求失败后的重试次数。
--no-json-mode       供应商不支持 JSON mode 时使用。
--write-prompts      只导出提示词，不调用 API。
```

## 如何判断结果质量

建议正式批量前先抽样检查 50-100 条，重点看四类问题：

1. 是否漏掉具体实体，例如人名、书名、系统名、理论名。
2. 是否误抽泛称，例如“研究人员”“相关文献”“图书馆数据库”。
3. L1 是否过早排除边界实体，例如“开放获取”这类可能跨 `Abstract/Event` 的对象。
4. L2/L3 是否把载体和内容混淆，例如把政策文件 `policy` 误标成政策举措 `policy_initiative`。

如果召回不足，可以调小 `--batch-size`，或在第一阶段提示词中增加正向示例。如果误抽泛称较多，应优先加强第一阶段的禁止规则。

## 分类体系维护

分类体系在 `entity_taxonomy.py` 中维护，核心结构是：

```python
TAXONOMY = [
    {
        "l1": "Agent",
        "l2": "Person",
        "type_code": "scholar",
        "label": "知识生产者",
        "definition": "...",
        "distinguish": "...",
        "examples": "巴巴拉·奎恩特（B. Quint）；潘光旦；罗香林",
    }
]
```

修改分类体系后，请重新导出提示词：

```bash
python review_entity_extraction_tool/extract_entities.py \
  --write-prompts review_entity_extraction_tool/PROMPTS.generated.md
```

不要让脚本运行时读取 Markdown 分类表。这样做是为了保证批处理时分类体系稳定、可复现，也避免 Markdown 表格格式变化导致程序失败。

## 错误处理

如果某个批次 API 调用失败，脚本不会直接丢失上下文，而是写入：

```text
输出文件.errors.jsonl
```

错误记录包含：

- `stage`：失败阶段，`l1` 或 `l2_l3`。
- `batch_index`：失败批次编号。
- `error`：错误信息。
- `records`：该批次输入数据。

常见原因：

- API key 或 base URL 错误。
- 模型不支持 JSON mode，需要加 `--no-json-mode`。
- batch 太大，模型输出被截断。
- 模型返回了非 JSON 文本。
- 网络超时。

处理建议：

1. 先查看 `.errors.jsonl`。
2. 把 `--batch-size` 调小，例如从 12 改为 5。
3. 必要时增大 `--max-tokens`。
4. 如果是 JSON mode 兼容问题，加 `--no-json-mode`。

## 给同事的最小上手流程

1. 准备输入文件，推荐 JSONL：

```jsonl
{"id":"s1","sentence":"评价语句 1"}
{"id":"s2","sentence":"评价语句 2"}
```

2. 设置 API 环境变量：

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="..."
export LLM_MODEL="..."
```

3. 先跑样例：

```bash
python review_entity_extraction_tool/extract_entities.py \
  --input review_entity_extraction_tool/sample_input.jsonl \
  --output /tmp/entity_results.jsonl \
  --batch-size 3
```

4. 检查三个文件：

```text
/tmp/entity_results.jsonl.stage1.jsonl
/tmp/entity_results.jsonl
/tmp/entity_results.jsonl.errors.jsonl
```

5. 确认样例表现符合预期后，再替换成正式输入文件。

## 与主项目的关系

这个目录是独立批处理工具，不依赖 Next.js、FastAPI、PostgreSQL 或 Docker Compose。它不会自动写入数据库，也不会启动后端服务。

如果后续要接入 CHSE 主项目，建议把最终 JSONL 作为中间数据，再单独写导入脚本进入 PostgreSQL。不要直接把这个实验脚本嵌入线上 API 流程，除非已经完成队列、限流、失败重试、审计日志和人工复核设计。
