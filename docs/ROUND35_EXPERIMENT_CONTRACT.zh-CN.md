# Round 35 实验合同

## 单一变量

根据Task Plan事实记录基数选择Planner路由绑定合同：单记录显式唯一ID，多记录允许unbound。

## 执行顺序

1. Targeted tests；
2. Full Python regression；
3. Frozen Planner/tool protocol replay；
4. Planner消息合同检查：单记录保留带唯一ID的G1i形状，多记录保留unbound形状；
5. 与Round 32/34相同的12题canary，4 worker；
6. 逐题读取Task Plan、Planner、工具结果、Page Evidence、Record、Writer Packet、CV和最终答案；
7. 达到门槛后才运行全100题。

## 固定条件

- 不读取gold/reference；
- 不使用题目ID、参考域名或标准答案影响运行；
- 不修改、删句、补写或筛选RWKV最终答案；
- 不同时改变provider、RRF、Page Evidence、Writer、CV、采样或stop；
- 服务器最多4 worker。

## 晋级门槛

- 12/12有最终答案，0工程错误；
- 独立Binder调用为0；
- 多记录题至少两个样本的P2/P3进入候选Writer记录；
- 人工语义均分不低于Round 32的47.92；
- Fortnite等单记录题不得出现Round 34式严重路由回退；
- 无法由检索波动解释的单题严重回退即否决；
- 未达标时不运行100题。
