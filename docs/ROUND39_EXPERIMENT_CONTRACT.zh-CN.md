# Round 39 实验合同

实验名：`frozen_writer_instruction_load_seeded_pair`

唯一语义变量：Final Writer 在 `CURRENT RUNTIME` 前读取当前长指令，还是历史最佳 Round 17 的短指令。

执行顺序：

1. 读取 Round 37 已冻结且明确不含 gold/reference 的 12 个 Final Writer 请求；
2. 校验每个请求的当前指令前缀完全相同；
3. 对每个请求生成原始 baseline 和仅替换固定前缀的 variant；
4. 同题使用相同 seed `424242`，4 workers，直接调用同一 RWKV completion endpoint；
5. 保存每个请求和答案的 SHA256、采样参数、stop、usage 与原始输出；
6. 生成完成后人工逐题配对评分；
7. 通过门槛后才允许修改生产代码和运行自然 canary。

禁止：运行时读取标准答案、case 特化、规则选事实、改变证据、改变采样、改变 stop、改变最终输出、执行 Agent 工具、把冻结结果冒充完整 Agent 回归结果。

`rwkv-skills` 只作为停止和采样边界的行为参考，不是本轮必须照搬的合同。本轮沿用请求本身已经冻结的实际生产参数。
