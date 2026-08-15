# Round 36：Writer Field-Label Visibility Ablation

## 背景

Round 35 的cardinality-aware task binding机制按设计生效，但12题人工语义均分44.58，低于Round 32的47.92，不能晋级全量100题。完整Trace显示，Page Evidence经常把只支持一个字段的原文短引声明为支持Task Record中的全部字段。例如Chrome的桌面版本短引同时被标记为“版本号、发布时间、安全问题数量”，ZZZ的主页长段也把版本、日期和主题全部绑定。Writer Packet随后以显眼的自然语言再次展示这些`field_keys`，容易让RWKV把低可靠度路由标签误认为已经验证的事实合同。

Round 30冻结全100审计中，属于多字段Task Record的572个选中候选里，508个（88.8%）声明支持全部字段。高字段饱和组人工均分40.14，低饱和组57.50；bootstrap区间仍跨0，因此这只是值得做单变量消融的全局疑点，不是已经证明的因果结论。

## 唯一语义变量

只改变Writer可见Evidence Packet：

- 不再渲染`RWKV candidate field bindings for this span ...`这一行；
- 原始quote、URL、标题、record key、task-record binding、source object、日期/版本字面marker、排序顺序和上下文预算全部不变；
- `field_keys`继续完整保存在`selected_evidence`、`packed_chunks`、Trace和审计产物中；
- Page Evidence输出合同和解析逻辑不变；
- 不删除、降权或重排任何候选；
- 不改变Planner、provider、RRF、fetch、Record Assembler、Cross Validation、Writer提示、sampling或stop边界；
- 不读取gold/reference，不修改RWKV最终输出。

该变量不是规则替RWKV判断字段，而是停止把未逐字段验证的上游标签作为Writer提示。RWKV仍根据原始span自行判断每条证据支持什么。

## 不在本轮处理

- 不合并Task Plan记录；
- 不新增字段级Binder；
- 不修Warframe工具选择；
- 不修改replan temperature；
- 不增加停止符；
- 不实现post-draft CV；
- 不调整RRF、BM25或source authority；
- 不运行全100，除非同一12题canary达到预注册门槛。

## 预期观测

可能改善：Chrome、ZZZ、Bun、Python、Ubuntu等“quote只支持字段子集、标签却饱和”的题。

可能回退：pg_dump、OIDC等一个短引确实同时支持全部字段的题；如果RWKV依赖字段标签而非原文理解，移除后会漏答。

因此必须用逐题配对结果判断，不能只看平均分。

## 验证顺序

1. 定向单元测试：Writer文本不含field label行，但Trace仍保留有界去重后的`field_keys`。
2. frozen packet replay：除该行外，Evidence Packet其他文本、候选顺序、quote和metadata保持一致。
3. 全Python回归。
4. 服务器相同模型、相同12题、4并发、50 steps。
5. 逐题完整链路人工语义评分。
6. 仅通过门槛后，才允许自然100题。

## 运行后合同审计更正

实际调用链显示`Orchestrator._resolved_writer_context()`构建的是一个由Cross Validation和Final Writer共享的不可变Evidence Packet。因此本轮在`_source_context_block()`删除字段标签，不只改变Final Writer输入，也改变了Cross Validation输入；CV决策变化后还会进一步改变replan和后续检索。

所以Round 36不是预注册的“Writer-only消融”，而是“共享finalization packet字段标签消融”。该差异在运行前本应通过整条调用链检查发现，是实验合同缺陷。Round 36结果可以用于发现机制和提出下一轮假设，但不得用于宣称Writer-only因果效果，也不得据此直接晋级全100。
