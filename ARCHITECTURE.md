# RWKV-ECRA 架构约定

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
├── orchestrator.py           # 流程编排，不拥有底层文件实现
├── state.py                  # 单任务状态与上下文投影
├── slm_scheduler.py          # 本地 SLM 批量调度
├── planner.py                # 查询/动作路由
└── retrieval_*.py            # 检索循环与答案合成
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
