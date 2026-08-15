# Round 35：按记录基数隔离 Planner 路由绑定合同

## Round 34 为什么不能晋级

Round 34 删除了多记录Task Point Binder，但同时把所有Planner请求的函数调用示例改成不含`task_point_id`，并将task-point指导全局改为optional。该变化影响了单记录题的工具选择分布；Fortnite连续选择不适用的`github_release`并从70分降到0分。因此Round 34既有机制改善，也有实验变量污染。

## 唯一语义变量

Planner的路由绑定合同由Task Plan中的事实记录数量决定：

- 一条事实记录：恢复已验证的G1i调用形状，函数调用示例显式包含唯一现有ID；若模型省略，Controller只附加这个唯一ID，不改变工具和参数。
- 多条事实记录：当一个检索可能支持多个记录时允许省略ID；缺失ID保持unbound，由RWKV Page Evidence根据真实页面span绑定记录与字段。

该变量只管理检索请求到任务记录的路由身份，不判断事实是否正确，也不修改RWKV最终答案。

## 冻结不变

- Task Plan生成提示与Schema；
- 模型、端点、温度、seed策略和停止边界；
- 工具目录、provider、RRF、fetch和清洗；
- Page Evidence提示与Evidence Record Assembler；
- Cross Validation、Writer Packet与最终Writer；
- request identity、重复路径和资源边界；
- canary 12题文本与运行参考时间策略。

## 预期

1. Fortnite等单记录题恢复带`P1`的稳定函数调用形状，不再受Round 34全局optional示例影响；
2. NASA、Chrome、Warframe仍可避免独立Binder的P1首项偏置；
3. 独立Binder调用保持0；
4. 不引入任何gold、题目特例或Controller事实规则。
