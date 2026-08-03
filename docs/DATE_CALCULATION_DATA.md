# 日期计算数据构建规范

## 1. 目的

当前日期题的主要失败不是“模型不会做减法”，而是模型经常把两个实体、两个版本或两个文档的日期配错，然后对错误的日期完成了正确的计算。例如：

- 把 FreshQA/FreshLLMs 的提交日期与 BrowseComp 的其他论文日期混在一起；
- 把 NIST AI RMF Playbook 当成 Generative AI Profile；
- 找不到 Go 官方 Release History 时，退回到第三方文章的近似日期；
- 已经看到两个日期，但输出“约 5 个月 7 天”，没有按题目要求计算自然日。

因此，提示词只能改善行为，不能替代“实体—字段—来源—日期—计算结果”的结构化数据。日期计算数据应同时训练/验证检索绑定和计算流程，不能只保存一条最终答案。

## 2. 推荐的数据单元

采用 UTF-8 JSONL，一行一个问题。建议使用如下结构：

```json
{
  "schema_version": "date-calculation.v1",
  "id": "DATE-001",
  "question": "Go 1.26 发布到 go1.25.12 发布相隔多少天？",
  "task_type": "date_difference",
  "as_of": "2026-07-25",
  "source_policy": "official_required",
  "entities": [
    {
      "id": "A",
      "name": "Go 1.26",
      "kind": "release",
      "source_url": "https://go.dev/blog/go1.26",
      "field": "release_date",
      "value": "2026-02-10",
      "precision": "day"
    },
    {
      "id": "B",
      "name": "go1.25.12",
      "kind": "release",
      "source_url": "https://go.dev/doc/devel/release",
      "field": "release_date",
      "value": "2026-07-07",
      "precision": "day"
    }
  ],
  "evidence": [
    {
      "entity_id": "A",
      "field": "release_date",
      "url": "https://go.dev/blog/go1.26",
      "quote": "...",
      "retrieved_at": "2026-07-25T00:00:00Z",
      "content_sha256": "..."
    },
    {
      "entity_id": "B",
      "field": "release_date",
      "url": "https://go.dev/doc/devel/release",
      "quote": "...",
      "retrieved_at": "2026-07-25T00:00:00Z",
      "content_sha256": "..."
    }
  ],
  "calculation": {
    "operation": "date_diff_days",
    "inputs": ["A.release_date", "B.release_date"],
    "inclusive": false,
    "expected": 147,
    "formula": "abs(B - A).days"
  },
  "final_answer": "Go 1.26 于 2026-02-10 发布，go1.25.12 于 2026-07-07 发布，相隔 147 天。",
  "gold": {
    "required_facts": ["2026-02-10", "2026-07-07", "147 天"],
    "forbidden_facts": [],
    "fact_aliases": {
      "147 天": ["147 days", "147 calendar days"]
    }
  }
}
```

关键约束：

1. `entities[].value` 必须是机器可计算的 ISO 日期，不把“约 5 个月”作为原始标签。
2. 每个日期必须绑定 `entity_id + field + url + quote`，不能只有 URL 列表。
3. `calculation.inputs` 只能引用实体字段；计算结果由脚本生成，不能手工填写后不校验。
4. 明确 `inclusive`、时区和精度。只知道月份时不能伪造某一天，也不能进入精确天数题。
5. `as_of` 是检索和答案时点约束。它不能改变历史事实，但应防止把截止日期之后的版本或发布日期当成截止日前状态。

`gold.fact_aliases` 只用于评测表达归一化，不是答案提示。它适合记录人工确认过的跨语言或同义表达，例如 `2026-07-07` 的英文日期写法、`147 天` 与 `147 days`。不要把不确定的近义词放进去，否则相似度会被人为抬高。

## 3. 数据构建流程

### 第一步：从失败轨迹抽题

优先从 `WR-041`–`WR-050` 这类失败中抽取问题，保留原问题、模型查询、选中的 URL、最终回答和失败原因。不要只根据模型答案反推标准答案。

### 第二步：建立实体和字段表

先写实体表，再写答案。日期字段至少区分：

- `release_date`：正式发布日；
- `paper_submission_date`：论文首次提交日；
- `publication_date`：网页或论文发表日；
- `effective_date`：标准或规则生效日；
- `announcement_date`：公告日；
- `planned_date`：计划日期。

这些字段不能互相替代。一个问题涉及两个对象时必须有两个实体 ID，即使它们来自同一个网站。

### 第三步：保存原文证据

打开原始页面，保存能够直接支持字段的短引文、页面 URL、抓取时间和内容哈希。对官方页面无法抽取的情况，记录页面状态和原因；不要用搜索摘要或模型总结填充 `quote`。

### 第四步：用程序计算

标准结果应由 Python `datetime.date` 计算：

```python
from datetime import date

start = date.fromisoformat("2026-02-10")
end = date.fromisoformat("2026-07-07")
assert abs((end - start).days) == 147
```

月份差、自然日差、是否包含首尾日期必须分成不同的 `task_type`，不能让模型自行选择算法。

### 第五步：做独立复核和反例

每条题至少复核两次：一次复核来源和实体绑定，一次复核日期与计算。为每条题添加至少一个硬负例，例如：

- 同一项目的公告日期 vs 正式发布日期；
- 同一机构的 RMF 1.0 vs Generative AI Profile；
- 同一论文作者的不同论文日期；
- 同一版本系列的 1.25.12 vs 1.26。

反例的目标是检测“来源看起来相关，但对象不对”，而不是增加无关难度。

## 4. 数据规模与切分

第一版建议构建 300–500 条：

| 类型 | 建议数量 | 重点 |
|---|---:|---|
| 两日期自然日差 | 100 | 版本、论文、标准、公告 |
| 日期先后/同日判断 | 60 | 同月、跨年、同日 |
| 发布/生效/公告字段区分 | 60 | 同一对象多个日期 |
| 实体—来源绑定 | 60 | 相似标题、同机构、多版本 |
| 截止日期与未来状态 | 40 | planned、release candidate、正式版 |
| 缺证据/冲突证据 | 40 | 应标记不确定，不能猜 |

训练、开发和测试集应按实体或来源域切分，而不是随机按句子切分。否则模型只记住某个网站的日期格式，无法验证检索和绑定能力。

## 5. 哪些问题靠数据，哪些问题靠工程

### 必须用工程约束解决

- 两个日期的减法、四舍五入和首尾日期规则；
- 禁止把模型自由生成的数字直接当计算结果；
- `entity_id → field → source span` 的绑定；
- 截止日期判断和“计划/已发布”状态判断；
- 当证据缺少时输出未确认，而不是补全答案。

### 需要数据覆盖

- 不同网站对发布日期字段的写法；
- 同一实体多个相近文档的区分；
- 版本号、补丁号和论文版本号的识别；
- 中英文日期、月份和时区表达；
- 官方页面正文不可抽取时的替代来源和拒答边界。

### 主要由提示词改善

- 先列出两个实体及其日期，再计算；
- 最终答案同时给出两个原始日期和差值；
- 只引用直接支持该实体字段的证据；
- 不把搜索摘要、标题或规划器内容当作事实。

## 6. 进入 harness 前的校验门

每次数据变更必须通过以下门：

1. JSON Schema/字段校验通过；
2. 所有 `calculation.inputs` 可解析且结果与 `expected` 一致；
3. 每个实体日期有直接引文和 URL；
4. 至少一个人工或独立脚本复核；
5. 用未参与构建的完整测试集运行，不能只跑新增题；
6. 报告同时保留 `similarity_score`、事实覆盖、禁止事实命中和证据覆盖，不能只看一个总分。

当前最值得优先构建的是“两个实体、两个官方页面、日期差”的数据族。它能直接覆盖 WR-041、WR-042、WR-043、WR-044、WR-050 暴露的实体错配和计算错误，也是后续加入确定性日期计算器最容易验证的一组。
