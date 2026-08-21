# RWKV-Scout 架构约定

完整的当前架构见 [`docs/ARCHITECTURE_HANDOFF.zh-CN.md`](docs/ARCHITECTURE_HANDOFF.zh-CN.md)，正式运行态对象见 [`docs/RUNTIME_CONTRACTS.zh-CN.md`](docs/RUNTIME_CONTRACTS.zh-CN.md)。

## 目录职责

```text
api.py
├── HTTP 路由和兼容性响应格式
app/
├── models.py                 # HTTP/任务传输模型
└── services/
    ├── task_runner.py        # 后台任务生命周期与配置覆盖
    └── workspace_files.py    # 文件、路径和报告读写
agent/
├── orchestrator.py           # 单一有界检索循环和任务编排
├── state.py                  # 单任务状态与上下文投影
├── slm_scheduler.py          # 可选的本地 SLM 调度
├── planner.py                # 任务计划和下一步动作路由
├── runtime_contracts.py      # 唯一正式运行态合约目录
├── retrieval_query_plan.py   # 可选的有界互补查询计划（默认关闭）
├── evidence_ledger.py        # 按 Task Record 保存落地证据
├── evidence_records.py       # 组装原子 Evidence Record
├── evidence_resolution.py    # 全局证据字段闭包
├── retrieval_loop.py         # 检索结果合并与去重
├── page_evidence.py          # 单页正文取证
└── retrieval_synthesis.py    # 证据上下文和最终回答合成
tools/
└── *_search.py               # 外部数据源适配器，只返回结构化证据
workflows/
└── *_flow.py                 # 长文本分析和报告生成
utils/
└── 基础设施：配置、网络、任务存储、日志、分块和指标
```

## 依赖方向

```text
HTTP routes → app services → agent/workflows → tools/clients → utils
```

- app/services 可以调用领域流程，但不应该注册 FastAPI 路由。
- agent 只负责状态、规划和编排，不直接解析 HTTP 请求。
- tools 不应修改用户文件或执行网页中的指令，只返回带来源的不可信证据。
- utils 不依赖 api.py，避免基础设施反向依赖入口文件。
- 新增联网来源时，优先新增工具适配器，不要把供应商逻辑写入 Orchestrator。

## 模型运行时边界

```text
agent/workflows
      ↓
clients/llm_client.py, clients/slm_client.py
      ↓
runtime.ModelBackend
      ├─ runtime.direct_rwkv     # 直接加载 checkpoint，主路径
      └─ runtime.compat          # OpenAI 兼容服务，迁移适配器
```

工作流不得直接依赖 OpenAI SDK、`/v1/chat/completions` 或具体模型服务器。模型
协议、prompt transcript、token usage 和健康检查由 `runtime/` 负责；云端 SDK
只在显式选择云端 provider 时按可选依赖加载。

## 当前重构的兼容边界

- /api/v1/* 和 /frontend-api/* 地址保持不变。
- 没有 wigolo 或云端 API 时，仍可使用现有无 Key 搜索。
- 前端继续读取原有任务、事件和报告 JSON 结构。
- 文件路径统一经过 app.services.workspace_files 校验。

## 当前检索执行链

生产请求使用一个共享状态的有界循环。Task Plan 只定义用户要回答的事实记录；
RWKV 决定是否继续检索以及下一次搜索方向，代码负责协议验证、原文落地、
状态投影、循环上限和最终回答协议。

```text
Task Plan (controller-owned record_id/field_id)
    ↓
Planner → [optional Retrieval Query Plan] → Web Retrieval → Page Evidence
    ↓                                      ↓
Retrieval Routing State              Evidence Ledger
    └──────────────→ Evidence Record Set ←──────────────┘
                           ↓
                  Evidence Resolution
                           ↓
                    Evidence Review
                    ├── gap → Plan Revision
                    └── finish → Answer Writer
```

Task Plan、Retrieval Event Ledger、网页正文证据和最终合成属于同一条链路；
不再维护并行的 Fork/Runner 默认架构。实验脚本可以覆盖搜索预算，不能改变
生产链路的证据边界。
## Unified research runtime

The production web path is one repeatable research loop. A finish request is
validated against the shared task evidence store; when coverage is incomplete,
the controller returns a bounded gap report and the model enters the same
`web_search` loop again. There is no model-visible recovery/page-fetch branch.

```text
Planner (existing RWKV prompt contract)
    -> web_search(query)
        -> parallel providers
        -> parallel page fetch
        -> parallel page/chunk evidence work
        -> shared EvidenceStore
        -> compact routing observation
    -> finish_task
        -> coverage check
        -> gap report -> web_search again
        -> final synthesis when complete
```

`AgentState.retrieval` is the task-scoped shared state. It owns query history,
deduplicated source bodies, chunk provenance and coverage metadata. The model
transcript is only a routing view; it is not the evidence database.

Concurrency is layered and bounded: network provider/page workers are
independent from page-evidence workers, and every RWKV request still passes
through the workspace-wide model request gate.
