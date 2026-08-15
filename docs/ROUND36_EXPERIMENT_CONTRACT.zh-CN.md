# Round 36 实验合同

## 预注册变量

实验名：`writer_field_label_visibility_ablation`

唯一允许变化：Writer Evidence Packet不再显示Page Evidence产生的候选`field_keys`自然语言提示行；运行后Trace仍保留这些字段用于审计。

## 冻结项

- 数据集：`data/evaluation/retrieval_round31_quality_canary12_20260814.json`
- 题目文本、题序和case ID；
- 模型、endpoint、request-level sampling policy和seed行为；
- 4 workers、50 max steps、单题1800秒；
- Planner与replan提示；
- 工具目录、provider、Tavily key池、RRF、fetch和chunk流程；
- Task Plan、task binding、Page Evidence、Evidence Record Assembler；
- Writer source顺序、token预算、quote、URL、record key和最终提示；
- Cross Validation与协议纠正；
- 所有停止边界；
- 运行时禁止读取gold/reference/历史答案/人工分数。

## Canary晋级门槛

必须同时满足：

- 12/12有RWKV最终答案；
- 0运行错误；
- 人工语义均分至少达到Round 32的47.92；
- `>=90`至少2题；
- `>=60`至少4题；
- pg_dump与GitHub OIDC不得回退；
- 不出现单题相对Round 32下降40分以上的严重回退；
- 机制检查确认Writer文本不含field label行，Trace仍保留`field_keys`；
- 最终Writer上下文平均保持在4K–6K tokens。

未达到门槛则停止，不运行全100；结果和失败链路仍完整保留。

## 运行后完整性判定

本轮实现触及了Cross Validation与Writer共享的Evidence Packet，而合同把Cross Validation列为冻结项。因此即使最终均分达到数值门槛，也按实验合同污染处理：不运行全100，不作单变量因果宣称。后续必须先分离CV视图与Writer视图，或者把“共享finalization packet”明确预注册为实验变量。
