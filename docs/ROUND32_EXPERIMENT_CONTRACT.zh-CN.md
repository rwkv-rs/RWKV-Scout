# Round 32 实验合同

## 实验变量

唯一变量：**grounded Page Evidence candidate 在 Ledger 与 Writer Packet 中保持原子 Evidence Record，不再按空 `record_key` 的 source object/task point 合并。**

本轮不自动拆分 grounded quote；每个候选保留输入中的原始连续 quote 和可逆 source locator。Assembler 只生成 opaque span identity，并在精确重复视图之间合并 RWKV 已给出的 task-point/field binding。

旧的8个页面块上限迁移为：候选 Evidence Record 最多24个、继续服从现有精确 evidence token budget；无候选记录时的页面/fallback上限不变。该迁移是记录计量单位变化的必要组成，不是新增排序策略。

## 固定项

- 数据集、问题文本及顺序；
- 模型、endpoint、request-level temperature和其他采样参数；
- 4 workers；
- Planner、CV、replan和Writer判断语义（只同步新的候选块名称/边界文本）；
- provider、Tavily key池、RRF状态、抓取与清洗；
- Page Evidence输入、prompt、输出schema和`supported`判断；
- stop与max token；
- 最终RWKV输出原样返回。

## 不作弊证明

- runtime不读取reference、manual score或case ID；
- Assembler输入只有已grounded source span及其RWKV标签；
- Assembler输出只包含原始连续子串、坐标和opaque identity；
- 不存在版本比较、日期比较、答案修复或题目专用分支；
- 人工评分仅在运行结束后执行。

## 执行门禁

```text
单元测试
→ Round 30 frozen packet replay
→ Round 31 同一12题 canary
→ 人工逐题配对
→ 通过才运行自然100题
```

失败时保存完整结果并撤回变量，不以局部修补继续同一Round。
