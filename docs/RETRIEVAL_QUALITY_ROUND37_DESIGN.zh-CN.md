# Round 37：Final Writer 字段标签可见性同 seed 配对实验

## 要回答的问题

Round 36 从共享 Evidence Packet 删除候选 `field_keys` 标签后，12题均分由 Round 32 的47.92升至55.00，但该修改同时改变了Cross Validation输入，进而改变了replan和后续检索；因此不能归因为Writer。

Round 37只检验一个问题：在检索、Cross Validation、证据顺序和原始span完全相同的条件下，Final Writer是否应看到上游候选字段标签。

## 两个配对条件

### A：seeded baseline

- 共享 Evidence Packet保留`RWKV candidate field bindings for this span ...`；
- Cross Validation与Final Writer都读取原始共享Packet；
- 使用实验专用固定seed `424242`。

### B：seeded writer-only ablation

- 共享 Evidence Packet仍保留字段标签，Cross Validation输入不变；
- 仅在调用Final Writer前，从上下文副本中移除完整的字段标签元数据行；
- `selected_evidence`、`packed_chunks[].field_keys`、Trace、quote、URL、顺序、预算全部保留；
- 使用与A相同的实验专用固定seed `424242`。

## 因果边界

固定seed只用于配对实验，目的是让相同prompt和采样参数产生可复现输出。生产replanner继续不设置seed，并根据失败类型使用request-level temperature；不会把固定seed部署为生产策略。

字段标签是上游RWKV候选抽取的路由元数据，不是事实。B条件不删除证据、不判断事实、不修改模型输出，只减少Final Writer可见的非事实标签。

## 冻结项

- 数据集、题序和case ID；
- 4 workers、50 steps、单题1800秒；
- 模型、endpoint、工具目录、provider、Tavily key池；
- Task Plan、Planner、replan、Cross Validation和协议纠正；
- RRF、fetch、Page Evidence、Evidence Record Assembler；
- Evidence Packet中的原文、来源、顺序和token预算；
- request-level采样策略以及实验seed；
- gold/reference在运行时不可见；
- 结构化stop边界冻结为换行围栏加下一角色边界。

## 实验执行更正

顺序运行两套live full-chain即使固定seed，系统时间、搜索结果和网页内容仍可能变化，不能证明Writer-only因果。因此第一阶段改成对Round 35最终Writer请求做frozen A/B：同一个已经保留的prompt只删除完整字段标签行，不重新检索。运行输入先导出为不含gold/reference的manifest。

如果frozen A/B显示改善，再运行一次不固定seed的自然12题canary验证生产效果；不能将frozen分数直接视为最终Agent质量。

## 运行结果

Frozen A/B人工语义均分从43.75升至45.00，但Fortnite回退40分，违反严重回退门槛；3题改善、2题回退、7题持平。全局隐藏字段标签因此被拒绝，实验代码已从生产路径恢复，未运行自然12题或全100。完整结果见`outputs/round37_quality_review_20260814/ROUND37_FROZEN_WRITER_AB_REVIEW.md`。

## 晋级条件

先比较同seed A/B的12题逐题结果。B只有同时满足以下条件才进入自然100题：

- 12/12有最终答案，0运行错误；
- 上游Task Plan、工具请求、工具结果、CV决策和最终Writer前Evidence Packet在每题可验证一致；
- B人工语义均分高于A；
- pg_dump和GitHub OIDC不回退；
- 不出现单题40分以上严重回退；
- Final Writer输入确实不含标签，而CV输入仍含标签；
- 不修改RWKV最终输出。

若上游链路不一致，则本轮配对失败，不能进行质量归因，也不运行全100。
