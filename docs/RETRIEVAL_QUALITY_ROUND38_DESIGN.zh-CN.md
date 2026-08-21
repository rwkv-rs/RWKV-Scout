# Round 38：Page Evidence 字段粒度实验

## Round 37后的根因

Round 35的144个原始RWKV正样本中，144个都面对多字段Task Record；其中124个（86.11%）直接声明支持该记录的全部字段。`raw_field_keys`与解析后`field_keys`为144/144完全一致，说明饱和绑定在Page Evidence的RWKV原始输出中已经出现，不是Parser、Evidence Record Assembler或Claim Ledger后续扩张造成。

多数失败题达到100%全字段绑定：NASA、ZZZ、Fortnite、Python、Chrome、Ubuntu。Writer隐藏标签只能在部分题改善，并导致Fortnite严重回退；因此不能在Writer层补偿。

## 待验证假设

当前Page Evidence一次向RWKV展示一个Task Record的所有字段，并要求一个JSON对象同时返回`field_keys`。模型容易把“这个span与记录相关”简化成“这个span支持记录全部字段”。

候选方案不是用规则删字段，而是把RWKV判断粒度改为：同一chunk针对一个Task Record的一个字段独立判断`SUPPORTED/UNSUPPORTED`并返回原文span。Assembler只合并同一精确span上由RWKV分别确认的字段。

## 实验顺序

1. 用Round 35全部124个原始全字段候选构建无gold/reference的frozen prompt集；
2. 每个源事件同时运行原多字段prompt和逐字段prompt，prompt组内使用同一seed；
3. 先观察逐字段判断是否产生真实选择性，而不是仍然所有字段全部SUPPORTED；
4. 若机制有效，再修改Page Evidence生产fan-out；
5. 通过frozen page replay、定向测试、全套测试后运行自然12题；
6. 只有自然12题达到质量门槛才运行全100。

## 边界

- 字段是否支持仍由RWKV判断；
- 代码只校验字段名属于Task Plan并验证quote来自原chunk；
- 不读取标准答案，不按题目或字段类型写规则；
- 不修改RWKV最终答案；
- 不改变检索、RRF、来源排序、Cross Validation、Writer或sampling；
- 代价是Page Evidence请求数约按字段数增加，质量验证优先，性能后续通过并发和批处理优化。

## 机制实验结果

覆盖全部124个Round 35原始全字段候选，执行124个同seed原合同请求和344个逐字段请求，共468次completion：

- 逐字段后仍有111/124（89.52%）源候选的所有字段都被判为SUPPORTED；
- 仅12/124（9.68%）产生字段选择性；
- 1个源候选全部字段UNSUPPORTED；
- 123/124组逐字段协议全部有效，唯一失败为NASA请求发生长度截断并开始复制prompt；
- 原始全字段率为86.11%，逐字段并未降低饱和，反而略升。

结论：该模型在chunk局部抽取阶段把“相关记录”近似为“字段支持”，仅缩小字段列表不能形成可靠的逐字段蕴含判断。生产实现被否决，不增加三倍调用。字段标签只能视为RWKV候选路由声明，不能承担事实解析合同。

下一步应回到完整链路：Warframe首先在Planner工具路由失效；Fortnite/Chrome/Android主要在Acquisition与当前记录召回失效；ZZZ/Python/Ubuntu/Mamba则需要在完整候选集合上由RWKV比较记录，而不是继续要求chunk-local extractor判断全局字段。更合理的下一候选是一个有界的、RWKV生成的Evidence Record Resolution视图，复用Claim Ledger候选，不让确定性代码选择事实。
