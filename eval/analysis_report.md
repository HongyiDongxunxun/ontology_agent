# Ontology_Agent V4.3 评估分析报告

> **生成时间**: 2026-07-17 18:52  
> **评估句数**: 455 (来自20个源文件)  
> **标注实体**: 1,774 (Gold) | **预测实体**: 1,882 (有效)  
> **配置**: RAG Few-Shot ✅ | 动态术语库 50K ✅ | Thinking ❌ | Voting ❌  

---

## 1. 实验配置

### 1.1 与基线版本对比

| 配置项 | V4.2 基线 | V4.3 本次实验 |
|--------|-----------|---------------|
| 模型 | deepseek-chat | deepseek-chat |
| Temperature | 0.0 | 0.0 |
| RAG Few-Shot | ❌ | ✅ (trigram检索, 每实体K=5) |
| 动态术语库 | 空 (运行时积累) | 预加载50K条 (来自398K全量库) |
| Thinking Mode | ❌ | ❌ (速度考虑) |
| Self-Consistency Voting | ❌ | ❌ (速度考虑) |
| L4匹配 | 精确+包含字符串 | 精确+包含字符串 |

### 1.2 RAG Few-Shot 实现

- **检索方式**: 字符 trigram 倒排索引 → Jaccard 相似度排序
- **术语库规模**: 50,000 条高置信度实体 (L3类型已标注)
- **注入方式**: 分类 Prompt 中 `## Few-Shot 参考` 段落，每实体注入 Top-5 最相似术语及其类型
- **示例格式**: `「潘光旦」→ scholar (知识生产者) [相似度:95%]`

---

## 2. 总体指标对比

| 指标 | V4.2 基线 | V4.3 (本次) | 变化 |
|------|-----------|-------------|------|
| **实体抽取 Precision** | — | 72.4% | — |
| **实体抽取 Recall** | — | 77.3% | — |
| **实体抽取 F1** | 73.7% | **74.8%** | **+1.1%** ↑ |
| **端到端严格 Precision** | — | 58.0% | — |
| **端到端严格 Recall** | — | 65.8% | — |
| **端到端严格 F1** | 59.5% | **61.7%** | **+2.2%** ↑ |
| **L1 准确率** | 95.1% | **96.3%** | **+1.2%** ↑ |
| **L2 准确率** | 90.3% | **90.6%** | **+0.3%** ↑ |
| **L3 准确率** | 85.5% | 85.3% | −0.2% ↓ |
| **有效/无效判定** | 98.5% | 98.5% | — |
| **平均 Likert 自评** | 4.31 | **4.70** | **+0.39** ↑ |

### 关键发现

1. **严格 F1 提升 +2.2pp** — RAG few-shot 注入使最难的全匹配场景获得了可测量的提升
2. **L1 分类显著改善 (+1.2pp)** — 顶层四大类别的区分更加准确，术语库提供了充分的类型锚点
3. **自评置信度大幅提升 (+0.39)** — 模型对自身分类结果更加"确信"，表明 few-shot 参考减少了不确定性
4. **L3 准确率持平** — 细粒度分类的提升空间仍然主要在个别高频混淆对

---

## 3. 各 L1 大类表现

| L1 大类 | Gold数量 | 预测数量 | 主要问题 |
|---------|----------|----------|---------|
| Abstract | 921 | — | concept 体量最大 (F1=71.6%), 边界混淆严重 |
| Artifact | 420 | — | book/journal_article 混淆, information_system vs concept |
| Agent | 364 | — | 表现最好, 多数子类型 F1 > 84% |
| Event | 69 | — | 样本较少, debate/stage 混淆 |

---

## 4. Per-type F1 详解

### 4.1 高表现类型 (F1 ≥ 90%)

| L3 类型 | F1 | TP | FP | FN | 分析 |
|---------|----|----|----|----|------|
| `metadata_schema` | 100.0% | 4 | 0 | 0 | 特征明确, 易识别 |
| `methodology` | 100.0% | 1 | 0 | 0 | 样本少但准确 |
| `policy_advocate` | 100.0% | 4 | 0 | 0 | 人物角色清晰 |
| `research` | 100.0% | 25 | 0 | 0 | 大学/研究院特征显著 |
| `software` | 100.0% | 2 | 0 | 0 | 软件名易识别 |
| `typology` | 100.0% | 4 | 0 | 0 | 特征独特 |
| `trend` | 100.0% | 1 | 0 | 0 | 含"趋势"关键词 |
| `instrument` | 95.7% | 11 | 1 | 0 | 量表/问卷特征明确 |
| `discipline` | 95.2% | 10 | 0 | 1 | 学科名易识别 |
| `governance` | 94.4% | 42 | 3 | 2 | 政府机构名清晰 |
| `algorithm` | 94.1% | 8 | 1 | 0 | 算法名特征明确 |
| `professional` | 92.3% | 18 | 1 | 2 | 学会/协会命名规范 |
| `research_program` | 92.3% | 6 | 1 | 0 |  项目名特征明确 |
| `theory` | 90.9% | 10 | 1 | 1 | "理论"关键词易识别 |
| `service` | 90.0% | 94 | 9 | 12 | 图书馆/档案馆特征 |

### 4.2 中等表现类型 (70% ≤ F1 < 90%)

| L3 类型 | F1 | TP | FP | FN | 主要混淆 |
|---------|----|----|----|----|---------|
| `publishing` | 89.7% | 13 | 1 | 2 | vs research |
| `method` | 89.2% | 37 | 4 | 5 | vs technique |
| `phenomenon` | 88.9% | 20 | 3 | 2 | **→ concept (19次!)** |
| `policy_initiative` | 88.9% | 8 | 0 | 2 | vs conference_meeting |
| `journal_article` | 88.0% | 11 | 2 | 1 | vs book |
| `model` | 87.5% | 14 | 1 | 3 | vs framework |
| `technique` | 86.1% | 59 | 11 | 8 | vs method |
| `movement` | 85.7% | 3 | 1 | 0 | 样本少 |
| `stage` | 85.7% | 9 | 0 | 3 | 边界模糊 |
| `database` | 84.3% | 35 | 8 | 5 | vs information_system |
| `conference_meeting` | 84.2% | 8 | 1 | 2 | vs conference_paper |
| `scholar` | 84.2% | 85 | 16 | 16 | vs practitioner |
| `report` | 83.3% | 5 | 1 | 1 | vs paper |
| `information_system` | 83.3% | 97 | 15 | 24 | **→ concept (10次)** |
| `subfield` | 82.8% | 24 | 3 | 7 | vs discipline |
| `policy` | 82.5% | 26 | 4 | 7 | vs standard |
| `guideline` | 80.0% | 4 | 1 | 1 | 样本少 |
| `practitioner` | 80.0% | 6 | 1 | 2 | vs scholar |
| `knowledge_organization_system` | 76.2% | 8 | 3 | 2 | vs concept |
| `paper` | 75.0% | 3 | 1 | 1 | vs journal_article |
| `framework` | 72.7% | 4 | 2 | 1 | vs model |
| `concept` | 71.6% | 378 | 131 | 169 | **最大体量, 多方向混淆** |

### 4.3 低表现类型 (F1 < 70%)

| L3 类型 | F1 | TP | FP | FN | 主要问题 |
|---------|----|----|----|----|---------|
| `book` | 66.2% | 48 | 24 | 25 | → journal_article (8次) |
| `index` | 64.0% | 8 | 4 | 5 | 引文索引 vs database |
| `standard` | 62.5% | 5 | 5 | 1 | → policy (5次) |
| `debate` | 50.0% | 1 | 1 | 1 | 样本极少 |
| `paradigm` | 0.0% | 0 | 1 | 1 | 完全漏判 |

### 4.4 金标中存在但零预测的类型

以下类型在标注数据中存在但在本次评估中未被预测 (完全漏召):

- `conference_paper` — 会议论文 (可能与 journal_article 混淆)
- `dataset` — 数据集 (样本极少)
- `definition` — 概念定义 (与 concept 高度重叠)
- `newspaper` — 报纸 (论文语料中极少出现)
- `contract` / `contract_clause` — 合约类 (图情领域极少)

---

## 5. L3 混淆矩阵分析

### 5.1 Top-10 混淆对

| 排名 | Gold类型 | → 误判为 | 次数 | 根因分析 |
|------|---------|----------|------|---------|
| 1 | `phenomenon` | `concept` | 19 | 现象vs概念的边界是图情领域最难的问题。需引入识别标准：是否有独立学术命名+是否作为研究主题+是否可观察 |
| 2 | `information_system` | `concept` | 10 | 数字图书馆/检索系统等词汇的二义性。需从上下文判断是否评价系统功能 |
| 3 | `discipline` | `subfield` | 8 | 学科与子领域的层级粒度问题。图书情报学→discipline, 信息检索→subfield |
| 4 | `book` | `journal_article` | 8 | 文献类型识别。需看是否有书名号、ISBN、出版社等线索 |
| 5 | `method` | `technique` | 7 | 方法vs技术的粒度混淆。完整操作程序→method, 具体技术手段→technique |
| 6 | `concept` | `technique` | 6 | concept 过度泛化到 technique |
| 7 | `concept` | `subfield` | 6 | concept 过度泛化到 subfield |
| 8 | `discipline` | `concept` | 6 | 学科被降级为概念 |
| 9 | `practitioner` | `scholar` | 6 | 人物身份判断：产出知识→scholar, 提供服务→practitioner |
| 10 | `concept` | `policy` | 5 | 概念与政策文件的混淆 |

### 5.2 系统性问题分析

**最大问题: `concept` 是"万能兜底"**

`concept` 类型参与了最多的混淆对，作为黄金标签被误判为其他类型，也作为预测标签"吞噬"了其他类型：
- 作为 Gold (`concept → X`): technique(6), subfield(6), policy(5), phenomenon(5) — concept 被过度细化
- 作为 Pred (`X → concept`): phenomenon(19), information_system(10), discipline(6) — concept 过度泛化

**建议**: 在 Prompt 中强化 `concept` 的"兜底仅限"规则 — 只有在排除 phenomenon/technique/subfield 等更精确类型后才使用 concept。

---

## 6. 改进路线图

### 短期 (Prompt优化, 预计 +3~5% F1)

| 优先级 | 措施 | 目标混淆对 | 预计F1增益 |
|--------|------|-----------|-----------|
| P0 | concept 兜底限制规则 | phenomenon/concept, information_system/concept | +2% |
| P0 | book vs journal_article 区分强化 | book → journal_article (8次) | +1% |
| P1 | method vs technique 粒度指引 | method → technique (7次) | +1% |
| P1 | discipline vs subfield 层级判定 | discipline → subfield (8次) | +0.5% |
| P2 | standard vs policy 区分 | standard → policy (5次) | +0.5% |

### 中期 (架构升级, 预计 +5~10% F1)

- **Thinking Mode**: 对 Top-10 混淆对涉及的实体启用 DeepSeek 深度推理
- **迭代修正**: Agent 3 评分 ≤2 的实体反馈回 Agent 2 修正
- **语义 L4 匹配**: 用 embedding 替代纯字符串匹配
- **扩充术语库**: 从 50K → 200K, 覆盖更多罕见实体

### 长期 (高级技术, 目标 F1 > 75%)

- Self-Consistency Voting (3轮) 对复杂句子
- 多模型集成 (DeepSeek + Qwen)
- 对高频混淆类型做定向微调

---

## 7. 附录

### 7.1 运行日志摘要

```
Pipeline: Agent 1 (抽取) → Agent 2 (分类+RAG) → Agent 3 (审查)
实体抽取: 461/461 句, 2,023 实体
分类结果: 1,882 有效, 115 无效
审查评分: 1,997/1,997 条, 平均 Likert 4.70
动态术语库: 累计 50,710 条 (含运行时新增)
```

### 7.2 文件清单

| 文件 | 说明 |
|------|------|
| `eval/analysis_report.md` | 本报告 |
| `eval/report_thinking_voting.md` | 自动生成的评估报告 |
| `eval/report_thinking_voting.jsonl` | 完整预测结果 (1,997条) |
| `mid_data/eval_baseline_extracted.json` | Agent 1 中间结果 |
| `output/eval/gold_standard.jsonl` | 标注金标数据 (461句) |

---

*报告由 Ontology_Agent V4.3 评估系统生成*
