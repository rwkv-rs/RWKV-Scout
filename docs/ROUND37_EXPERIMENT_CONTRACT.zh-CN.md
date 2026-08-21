# Round 37 实验合同

实验：`writer_only_field_label_visibility_seeded_pair`

唯一语义变量：字段绑定路由标签是否进入Final Writer提示。共享Evidence Packet、Cross Validation和审计状态保持相同。

实验专用固定项：`RWKV_ECRA_LLM_SEED=424242`。该值不属于生产配置，不得据此改变生产replan seed策略。

运行顺序：

1. 从Round 35 Trace导出不含gold/reference的12个最终Writer请求；
2. 对同一个保留请求直接运行标签可见/标签隐藏同seed A/B，不执行检索工具；
3. 实现仅作用于Final Writer输入副本的标签投影；
4. 通过定向测试、全套测试和frozen packet检查；
5. 人工比较12组答案；
6. 只有frozen Writer A/B改善后，才运行生产配置的自然12题canary；
7. 只有自然canary通过质量门槛，才允许自然100题。

禁止：读取gold/reference、按题目特殊处理、规则选择事实、规则改写答案、改变CV输入、同时修改sampling/retrieval/ranking/context预算。

## 失效实验记录

最初启动的顺序full-chain seeded baseline未冻结系统时间；Planner环境和Freshness Packet会随运行时刻改变，因此无法与后续variant构成严格单变量配对。该运行在1题完成后停止并保留为失效样本，不计分。固定seed不能替代完整的模型可见输入冻结。

Frozen Writer A/B随后按合同完成。Variant均分仅提高1.25分且Fortnite严重回退40分，未通过晋级门槛；运行时变体已恢复，不执行自然canary或全100。
