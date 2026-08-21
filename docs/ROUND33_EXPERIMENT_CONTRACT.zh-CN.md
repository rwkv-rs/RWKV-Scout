# Round 33 实验合同

## 实验变量

唯一质量变量：`Planner Routing Packet v3`。

模型、endpoint、工具目录、问题文本、参考时间、温度策略、provider、RRF、fetch、Evidence Record、Cross Validation 和 Writer 全部保持 Round 32 不变。

## 顺序

```text
targeted tests
→ full Python regression
→ Round 30 frozen protocol replay
→ 与 Round 32 完全相同的 12 题 canary（4 workers）
→ 下载完整 Trace
→ 逐题人工语义评分与最早失效层分析
→ 通过晋级门槛后才运行完整 100 题
```

## 必须记录

- 每题 Planner 输入、输出、temperature、tool call；
- route identity、frozen route、重复请求；
- candidate/fetch/body、Evidence Record、Writer packet；
- 最终答案及人工语义分；
- Planner prompt tokens、模型调用、延迟和错误；
- 对 Round 32 的逐题配对差值。

运行期不读取 gold/reference。标准答案只在任务完成后的独立人工审计阶段使用。

## 停止与晋级

canary 若出现工程错误、答案返回率下降、人工语义明显回退，或 frozen repeats 没有下降，则停止 Round 33，不运行全 100。通过 canary 后仍按既定历史门槛判断是否晋级，不能仅凭平均分或单题提升晋级。
