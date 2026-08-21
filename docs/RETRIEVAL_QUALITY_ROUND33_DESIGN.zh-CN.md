# Round 33：Planner Routing Packet v3

## 结论

Round 32 的 Atomic Evidence Record 解决了证据记录被拆散的问题，但 12 题 Trace 暴露出更早的失效层：Planner 明确看见已完成/冻结路径后，仍大量复制相同请求。

离线 Trace 统计（不读取标准答案）：

- 12 题共 102 次有效 Planner 工具决策；
- 69 次是模型再次生成已经生成过的完整请求；
- 37 次生成的请求在当次输入中已明确标记为 `frozen`；
- 3 个多任务点用例全部只路由到 P1；
- frozen-repeat 请求平均约 7.4K 输入 token，其他请求平均约 6.3K。

这不是“缺少更多停止符”。停止符只定义生成终点，不能修复 Planner 对状态的注意与交接。当前 Planner packet 同时重复发送：Evidence Record、Source Locator span、object metadata、retrieval bindings 和 route history；同一个旧查询可在多个位置重复出现，形成强锚点。

## 单一语义变量

Round 33 只修改 Planner 的只读状态投影，不修改：

- 搜索 provider、候选排序、抓取和清洗；
- Evidence Extractor、Claim Ledger、Atomic Evidence Record；
- Cross Validation；
- Writer packet、Writer prompt 和最终答案；
- request-level temperature；
- G1i 生成边界和停止符；
- Controller 的重试/截止逻辑。

## 新投影合同

Planner 仍收到四类必要事实，但每项只出现一次：

1. `freshness`：当前时间和问题时间策略。
2. `evidence_records`：每条候选记录只保留 task point、对象标识、字段、标题、URL、日期和短原文 quote。
3. `source_locators`：只保留没有被 Evidence Record 表示的来源；这种来源保留一个短原文 span，供 extraction 为空时重构查询。
4. `retrieval_ledger`：所有可见请求按完整 request identity 去重，保留 action、operation、arguments、task point、状态、错误和 frozen 状态。

删除 Planner packet 中不属于下一步决策的重复传输：

- 完整 `retrieval_bindings`；
- 完整 `object_alignments`；
- Evidence Record 已包含的重复 Source Locator；
- 同一来源的第二份 evidence span；
- 最新 outcome 对完整旧请求的回显。

`retrieval_ledger` 放在 Runtime state 最后，使冻结路径紧邻恢复指令。代码只报告“发生过什么”，不生成替代查询、不判断事实正确性，也不选择下一步；下一工具、查询、task point 和停止仍由 RWKV 生成。

## 设计边界

本轮不是以删模块为目标，而是恢复职责：

- State 负责保存完整事实和 provenance；
- Planner packet 负责传递下一决策需要的有界状态；
- Planner 负责模型化选择；
- Controller 只阻止完全相同请求再次执行；
- Writer 只消费最终证据包并由 RWKV 输出答案。

`rwkv-skills` 只作为边界行为参考。Round 33 不引入其 Math-500 NoCoT sampling、flower marker 或 Markdown fence stop；这些都没有被当前 Retrieval Trace 证明必要。

## 验证门槛

1. 现有 Planner/Runtime/Prompt 合同测试全部通过。
2. 同一已执行查询在一次 Planner Runtime state 中只出现一次。
3. Evidence Record 与 Source Locator 同 URL 时不重复。
4. 无 Evidence Record 的来源仍保留一个有界原文 span。
5. Round 30 frozen protocol replay 保持 100% 调用语义 parity。
6. 同一 12 题 canary 与 Round 32 配对比较：
   - frozen request repeat 数下降；
   - Planner 平均输入 token 下降；
   - 不能牺牲人工语义分、答案返回率或工程稳定性。
7. 只有 canary 通过后才运行 100 题。
