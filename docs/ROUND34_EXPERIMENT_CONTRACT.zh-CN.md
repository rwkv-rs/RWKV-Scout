# Round 34 实验合同

## 单一变量

多任务点检索在 Planner 未明确输出合法 `task_point_id` 时保持 unbound，不执行独立 Task Point Binder。

## 固定条件

- 与 Round 32 基线相同的模型、端点、工具目录、Prompt 主结构、采样参数和 canary 12题；
- 4 worker；
- 不读取 gold/reference；
- 不修改或清洗 RWKV 最终输出；
- 不混入候选排序、工具拆分、停止符、Writer 或 Cross Validation 改动。

## 执行顺序

1. Targeted tests；
2. Full Python regression；
3. Frozen protocol/replay checks；
4. 与 Round 32 完全相同的12题 canary；
5. 逐题读取 Task Plan、Planner、Tool Result、Page Evidence、Record、Writer Packet 和最终答案；
6. 通过 canary 质量与机制门槛后才运行全100题。

## 机制指标

- 多任务点路径中 unbound 比例；
- 独立 Binder 调用必须为0；
- P2/P3 grounded evidence record 是否出现；
- exact duplicate 数量；
- Planner/总模型调用量；
- Page Evidence 是否使用完整用户目标；
- 12题逐题人工语义分。

## 晋级门槛

- 12/12 有最终答案，0工程错误；
- 多任务点样本不再由 Controller/Binder 全部强制写成 P1；
- 至少两个多任务点样本出现后续任务点候选记录，或最终答案明确补齐后续字段；
- 人工均分不低于 Round 32 的47.92；
- 不出现单题严重新回退且不能由检索随机性解释；
- 若机制未生效或质量回退，恢复 Round 32 行为，不运行全100题。
