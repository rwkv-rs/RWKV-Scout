# 模型运行时迁移

项目的工作流现在依赖内部 `ModelBackend`，不再把 OpenAI 协议当作模型抽象。

- `direct_rwkv`：项目直接加载 RWKV checkpoint，通过 Albatross 推理，不需要模型 API 服务；
- `openai_compat`：仅作为现有本地服务的过渡兼容层。

## 配置直连推理

在 `config.json` 中设置：

```json
{
  "MODEL_RUNTIME": {
    "backend": "direct_rwkv",
    "direct_rwkv": {
      "engine_root": "../Albatross/faster2_251201",
      "model_path": "/home/chase/weights/rwkv7-g1h-1.5b-20260710-ctx10240",
      "device": "cuda",
      "max_tokens": 768,
      "temperature": 0.0,
      "stop_tokens": [0]
    }
  }
}
```

也可以通过环境变量配置：

```bash
RWKV_ECRA_MODEL_BACKEND=direct_rwkv
RWKV_ECRA_RWKV_ENGINE_ROOT=/home/chase/GitHub/Albatross/faster2_251201
RWKV_ECRA_RWKV_MODEL_PATH=/home/chase/weights/rwkv7-g1h-1.5b-20260710-ctx10240
```

模型只在第一次请求时加载，每个请求拥有独立的 RWKV state。当前批量推理先串行执行，等 state 隔离验证稳定后再加入同长度批处理。

## 按模型 profile 选择运行时

`config.json` 已提供 `local_direct_1p5b` profile。前端或 `AnalyzeRequest.model_key` 选择该 profile 时，任务会同时切换到 `direct_rwkv`，不会把空的 `base_url` 当成 HTTP 地址。引擎和 checkpoint 路径建议通过 `RWKV_ECRA_RWKV_ENGINE_ROOT`、`RWKV_ECRA_RWKV_MODEL_PATH` 注入，以适配不同机器的权重位置。

`auto` 只有在引擎目录和 checkpoint 路径都已配置时才选择直连，否则继续使用兼容服务，便于迁移期间保留旧 profile。

验证方式：

```bash
python -m scripts.model_backend_smoke --probe-only
python -m scripts.model_backend_smoke --max-tokens 64
# 验证前端同样使用的直连 profile
python -m scripts.model_backend_smoke --model-key local_direct_1p5b --max-tokens 8
```

直连模式要求 CUDA 版 PyTorch、Albatross 引擎目录和本地 checkpoint。配置不完整时会明确返回 `not_ready`，不会静默切回云端 API。

OpenAI SDK 已从基础依赖中移除；只有明确使用云端 provider 时才安装：

```bash
pip install -e ".[cloud]"
```
