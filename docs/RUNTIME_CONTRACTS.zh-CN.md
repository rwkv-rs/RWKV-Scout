# RWKV-ECRA 正式运行态合约

## 命名规则

运行态对象统一使用：

```json
{"contract":"rwkv.ecra.runtime.<domain>"}
```

`contract` 表示对象的长期领域身份，不表示实验轮次、代码发布号或模型版本。因此生产路径禁止输出 `schema_version`，合约名禁止包含 `.v1`、`.v2`、`.v4`。旧值只允许出现在 `scripts/migrate_runtime_contracts.py` 的历史输入映射表和不可改写的 `prompt`/`raw_model_output` 审计字符串里。

## 合约目录

| 合约 | 唯一职责 | 生产者 | 主要消费者 |
|---|---|---|---|
| `rwkv.ecra.runtime.task-plan` | 保存用户目标、事实记录和字段身份 | Task Planner + Controller 规范化器 | Planner、Retrieval Query Plan、Evidence Ledger、Evidence Resolution、Evidence Review、Writer |
| `rwkv.ecra.runtime.tool-call` | 保存一次已验证的工具调用 | Tool Protocol | Orchestrator、Tool Registry |
| `rwkv.ecra.runtime.retrieval-query-plan` | 保存主查询及互补查询路线，不保存事实 | Retrieval Query Planner | Web Retrieval、Routing State |
| `rwkv.ecra.runtime.retrieval-object` | 保存请求对象和来源对象身份 | Retrieval Object Contract | Evidence extraction、Ledger、Resolution |
| `rwkv.ecra.runtime.evidence-ledger` | 按 Task Record 保存已落地的候选证据记录 | Evidence Ledger | Routing、Evidence Review、Writer context builder |
| `rwkv.ecra.runtime.evidence-record-set` | 对一次模型调用可见的有界证据记录投影 | Controller | Planner、Evidence Resolution |
| `rwkv.ecra.runtime.evidence-resolution` | 保存字段闭包、缺失和冲突控制状态，不保存改写后的事实 | Evidence Resolver | Evidence Review、Writer |
| `rwkv.ecra.runtime.evidence-review` | 保存 `finish` 或结构化 `gap` | Evidence Reviewer | Plan Revision、Orchestrator |
| `rwkv.ecra.runtime.planner-evidence` | 保存 Planner 可见的有界证据摘要 | Retrieval State | Planner |
| `rwkv.ecra.runtime.retrieval-routing-state` | 保存查询、冻结路线和证据 revision | Retrieval State | Planner、Plan Revision |
| `rwkv.ecra.runtime.retrieval-infrastructure` | 保存抓取/抽取基础设施故障 | Retrieval State | Planner、审计 |
| `rwkv.ecra.runtime.model-extraction-diagnostics` | 保存模型抽取链完整性 | Web Retrieval | Retrieval State、审计 |
| `rwkv.ecra.runtime.retrieval-event-ledger` | 保存一次任务的检索操作历史 | Retrieval Ledger | Planner、审计 |

所有常量只在 `agent/runtime_contracts.py` 定义。业务模块不得再复制字符串常量。

R-ST 数据中的 `oracle_evidence_assessment` 是离线监督标注，包含 `supported_facts` 等答案真值；它不是运行态 `evidence_review`，不得进入线上模型请求或运行状态。这个独立命名用于防止 oracle 信息污染生产链路。

## 每次 RWKV 调用的输入与输出

| 调用 | 只允许输入 | 只允许输出 | 明确禁止 |
|---|---|---|---|
| Task Planner | 用户原问题、当前环境摘要 | 完整 Task Plan | 来源、事实答案、工具、完成判断 |
| Retrieval Query Planner | 用户目标、Task Plan 目标记录、主查询、查询历史、可信当前时间 | Retrieval Query Plan 的新增查询行 | 事实、URL/域名猜测、输入外硬锚点 |
| Page Evidence Extractor | 单一来源的原文 chunk、Task Record/Field ID | 可回定位原文 offset 的 quote | fallback 改写、跨来源拼接 |
| Evidence Resolver | 完整 Task Plan、Writer 将看到的同一批 exact spans | Evidence Resolution | 自由文本事实、摘要、替换 quote |
| Evidence Reviewer | 原问题、完整 Task Plan、同一 Evidence Record Set、Evidence Resolution、Routing State | `finish` 或闭合 Review Gap | 新事实、新查询、答案正文 |
| Plan Revision | 原问题、Task Plan、完整 Review Gap、冻结路线、Routing State | 下一次工具调用或完成动作 | 重新猜测丢失的 gap 原因 |
| Answer Writer | 原问题、完整 Task Plan、已落地证据文本、独立的 Evidence Resolution 控制通道 | 最终答案 | 未绑定页面正文、Evidence Review 输出作为事实 |

## 演进策略

- 兼容增加：在同一个正式合约中增加可选字段，并让消费者对未知字段保持封闭读取。
- 不兼容变化：先实现显式迁移器和双读审计，再切换唯一生产者；不通过给名字追加数字来并存多套运行态。
- 领域含义改变：使用新的语义名称，例如从“查询执行结果”变为“证据记录”时必须换合约，不能复用旧名字。
- 历史回放：先运行 `scripts/migrate_runtime_contracts.py`。模型原始输入/输出字符串永不改写。

## 代码级不变量

1. 每个运行态对象只允许一个 `contract`，不得同时出现 `schema_version`。
2. Task Record ID、Field ID、Evidence Record ID 由 Controller 分配，模型不得发明或改名。
3. 语义状态只能由拥有该决策的 RWKV 阶段产生；确定性代码只验证、拒绝、投影和传输。
4. Evidence Review 与 Evidence Resolution 是控制通道，不能被拼接进证据事实通道。
5. 任何历史兼容逻辑不得出现在正常生产者里。
