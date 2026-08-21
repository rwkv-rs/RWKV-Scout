# Round 39：Final Writer 指令负载冻结 A/B

## 问题

Round 30 全 100 题因果遥测显示：错误样本并不是简单因为 Writer 上下文过短。零分题的字段饱和率和 `record_key` 比例反而高于高分题，说明这些上游标签不能作为事实门禁。与此同时，当前 Final Writer 固定指令为 2650 字符（约 396 个英文词），历史最佳 Round 17 的固定指令只有 1243 字符（约 187 个英文词）。新增指令主要要求模型处理候选记录、未绑定片段、对象冲突和字段路由，可能占用 RWKV 的注意力并提高拒答或错误绑定概率。

Round 39 只回答一个问题：在完全相同的最终证据包、采样参数、停止边界和 seed 下，把 Final Writer 固定指令恢复为 Round 17 的较短版本，是否提高最终答案质量。

## 唯一变量

- A：保留当前 Round 35/37 冻结请求中的完整 Writer 固定指令。
- B：只把 `CURRENT RUNTIME` 之前的固定 Writer 指令替换为从 Round 17 Trace 提取并校验一致的短指令。

以下内容保持逐字节一致：

- 用户问题；
- runtime 时间；
- research material、来源、quote、顺序和所有候选元数据；
- `max_tokens`、temperature、top-p、top-k、penalty 和 stop；
- endpoint、模型及同题 seed；
- 不调用 Planner、检索、Cross Validation 或其他 Agent 工具。

## 非作弊边界

- 运行时数据集不含 gold/reference；
- 不按 case ID 或题目内容改变提示词；
- 不选择、删除、排序或改写证据；
- 不规则判断事实；
- 不修改 RWKV 原始最终输出；
- 标准答案只在生成完成后用于人工语义评分。

## 晋级门槛

B 只有同时满足以下条件，才允许修改生产 Writer 指令并进入自然 12 题 canary：

- 12/12 两组都有模型输出且无运行错误；
- B 人工语义总分高于 A；
- 不出现单题下降 40 分及以上；
- Fortnite、Warframe、Android 等历史困难题不发生明显整体退化；
- pg_dump、OIDC、Mamba 等既有正确能力不回退；
- 输出不是复读或截断伪改善。

冻结 A/B 只能证明 Writer 指令的局部因果效果，不能直接代表完整 Agent 质量。通过后仍须运行自然 12 题和全 100 题。
