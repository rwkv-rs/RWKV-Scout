# 模型运行时迁移

RWKV-Scout 的工作流不再把 OpenAI 协议当成模型抽象。工作流只依赖
`runtime.backend.ModelBackend`，目前有两个实现：

- `direct_rwkv`：项目进程直接加载 checkpoint，通过 Albatross 的 `reference.rwkv7` 推理；
- `openai_compat`：仅作为现有本地模型服务的过渡适配器。

## 启用直接推理

在 `config.json` 的 `MODEL_RUNTIME` 中配置本地引擎和 checkpoint：

```json
{
  "MODEL_RUNTIME": {
    "backend": "direct_rwkv",
    "direct_rwkv": {
      "engine_root": "../Albatross/faster2_251201",
      "model_path": "/home/chase/weights/rwkv7-g1h-1.5b-20260710-ctx10240",
      "vocab_path": "",
      "device": "cuda",
      "max_tokens": 768,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "stop_tokens": [0]
    }
  }
}
```

也可以用环境变量覆盖路径：

```bash
RWKV_ECRA_MODEL_BACKEND=direct_rwkv
RWKV_ECRA_RWKV_ENGINE_ROOT=/home/chase/GitHub/Albatross/faster2_251201
RWKV_ECRA_RWKV_MODEL_PATH=/home/chase/weights/rwkv7-g1h-1.5b-20260710-ctx10240
```

模型只在第一次请求时加载，单个请求使用独立 RWKV state；当前批量接口先串行执行，后续再增加安全的同长度批处理。这样可以先确保 state 不串线，再优化吞吐。

## 按模型 profile 选择运行时

`config.json` 已提供 `local_direct_1p5b` profile。前端或 `AnalyzeRequest.model_key` 选择这个 profile 时，任务会同时切换到 `direct_rwkv`，不会把空的 `base_url` 当成 HTTP 地址。引擎和 checkpoint 路径仍建议通过 `RWKV_ECRA_RWKV_ENGINE_ROOT`、`RWKV_ECRA_RWKV_MODEL_PATH` 注入，以便不同机器使用不同权重。

`auto` 只会在引擎目录和 checkpoint 路径都已配置时选择直连；否则继续使用兼容服务，便于迁移期间保留旧 profile。

直接运行时要求 CUDA-enabled PyTorch 和 Albatross 引擎目录。未配置 checkpoint 时，运行时会报告明确的本地配置错误，不会悄悄切回云端 API。

可以先只检查配置，再执行一次真实生成：

```bash
python -m scripts.model_backend_smoke --probe-only
python -m scripts.model_backend_smoke --prompt $'User:\n1+1=?\n\nAssistant:' --max-tokens 16
# 验证前端同样使用的直连 profile
python -m scripts.model_backend_smoke --model-key local_direct_1p5b --max-tokens 8
```

## 兼容路径

迁移期间将 `backend` 设为 `openai_compat` 即可继续调用当前本地服务。这个路径只存在于 `runtime/compat.py`，业务工作流和 `LLMClient` 不再直接依赖其 HTTP 格式。
