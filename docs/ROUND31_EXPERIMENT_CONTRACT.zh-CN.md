# Round 31 实验合同

## 单一语义变量

Round 31 只评估 **Markdown 可见文本的精确 quote 定位**：Page Evidence 仍由 RWKV 从原始 chunk 中选择事实和连续 span；模型输入与 Round 30 保持一致。程序在原始逐字定位失败时，可以检查模型 quote 是否与去除 Markdown 链接地址和显示符后的可见文本逐字一致，并将该 span 映射回原始 chunk。禁止模糊相似度放行、摘要拼接、事实纠错、版本判断和答案改写。

最初的 v1 canary 曾把全部 Page Evidence 输入改成可见文本投影。12 题人工配对虽有局部提升，但 Python、Warframe 和 Mamba 回退，而且该变化同时改变了模型抽取分布，无法归因到 locator，故未晋级。v2 已恢复原始 chunk 输入，只保留精确 locator 作为本轮单一变量。

额外 Evidence Selector 属于未晋级实验：冻结 12 题 A/B 没有提升，并删除过正确证据，已在 Round 31 前撤回。CV 与 Writer 直接共享同一 revision 的完整证据包。RRF 排序、provider、工具目录、问题文本、参考时间、Planner/Writer 温度和模型保持 Round 30 设置。错误语义持久化、完整 request identity、常见 JSON 外壳归一化、角色/JSON 停止符属于工程合同修复，必须先通过 frozen replay parity，不作为新的答案选择模块。

## 固定流程

1. Round 30 的 100 题 Planner 输出 frozen replay；要求执行调用 100% parity。
2. Round 30 的 216 条被拒 quote 做生产 locator 回放；只统计 `markdown_visible_exact`，不得用模糊匹配扩大数字。
3. 预注册的 12 题 live canary；问题逐字来自自然 100 题。v1 结果只作为被拒实验留档；v2 通过后才可晋级。
4. 只有前三步没有工程错误且没有明显类别退化，才运行自然 100 题。
5. 全 100 题逐题人工语义评分、配对比较和 bootstrap 区间完成后才能判断晋级。

## 四层指标

- Acquisition：候选召回、fetch 成功、正文成功。
- Record：对象身份、字段完整性、quote、跨记录混合率。
- Answer：冻结人工标准下的逐题语义分。
- Efficiency：模型调用、token、延迟、重复率；本轮只记录，不以牺牲质量换效率。

## 晋级门槛

- 100 题均有最终输出，0 个运行错误。
- 人工均分高于 Round 17 的 50.15。
- `>=90` 高于 Round 25 的 26 题；`>=60` 高于 Round 25 的 45 题。
- direct-page 均分至少 98，procedure 至少 75。
- current、security、game、release 无明显回退。
- Writer 上下文平均 4K–6K tokens。
- duplicate-boundary 终止率低于 30%。
- 运行时不能读取 gold、reference、case ID 或人工标准。

## 停止协议

当前 RWKV 未接受 Search-R1 式检索策略训练，因此推理 harness 必须提供有限停止边界：JSON 调用在代码围栏或下一角色边界停止；自由文本仅在带换行的下一角色边界停止；token 0/EOS 与 max token 是最终资源边界。`rwkv-skills` 只作为已验证行为参考，不照搬裸角色 stop 或 Math NoCoT 采样。检索何时完成仍由 RWKV 输出 `finish_task`，Cross Validation 只由 RWKV 二选一决定 finish/replan；程序不得据答案内容替模型停止或改写最终输出。
