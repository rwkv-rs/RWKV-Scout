# RWKV-ECRA

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module boundaries and dependency rules.
See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for reproducible runs and [docs/external_projects/wigolo.md](docs/external_projects/wigolo.md) for the wigolo evaluation record.

Language: [中文](README.md) | English

## Features and Demonstrations

Currently, the primary function is long-text analysis, which can achieve results similar to DeepResearch. However, unlike common DeepResearch systems, it does not make decisions but solely presents the analysis.

After enabling the frontend, you can input research topics one by one to queue them up for processing;
![](./Imgs/example.png)

The final report presents an analysis of the provided content with source citations. More examples can be found in the `data/output` directory. To avoid copyright issues, the `input` directory only contains open-access papers from arXiv.
![](./Imgs/example1.png)


## Usage

### 1. Environment Preparation

Python 3.10 or newer is recommended.

Install the pinned dependencies required for the current code:

```bash
pip install -r requirements.txt
```

> Current production experiments use the exact local RWKV 13.3B contract: model `rwkv7-g1i_preview4922-13.3b-20260720-ctx12288`, endpoint `http://172.21.122.93:29613/v1`, and context length `12288`. The compatibility endpoint is retained for that historical deployment, but the application now has a project-owned `direct_rwkv` backend that loads a checkpoint in-process. See [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md) before switching a model profile to direct inference. The former RWKV 7.2B contract remains as a separate historical baseline and must not be mixed with 13.3B results.

> LLM calls based on Volcengine and Baidu AI Studio remain optional adapters. The base installation does not require the OpenAI SDK; install `pip install -e ".[cloud]"` only when a cloud provider is explicitly selected.

### Optional: local wigolo web retrieval

The project can use [wigolo](https://github.com/KnockOutEZ/wigolo) as a local web retrieval backend. The default mode is `auto`:

- use the local wigolo REST service when it is available;
- fall back to the existing keyless public search if wigolo is not installed, not running, or fails;
- no Tavily, OpenAI, or other cloud API key is required for this path.

After initializing wigolo, start its local REST service:

    npx wigolo init
    npx wigolo serve

The default endpoint is `http://127.0.0.1:3333`. Configure it with:

    RWKV_ECRA_WIGOLO_MODE=auto
    WIGOLO_BASE_URL=http://127.0.0.1:3333

    # Always use the existing keyless provider
    RWKV_ECRA_WIGOLO_MODE=off

    # Fail instead of falling back when wigolo is unavailable
    RWKV_ECRA_WIGOLO_MODE=only

Web content remains marked as untrusted evidence and cannot act as system instructions. Search results, excerpts, citation IDs, and scores are retained for later report rendering.

### 2. Model runtime configuration

The workflow layer uses an internal `ModelBackend` contract. Set `MODEL_RUNTIME.backend` to `direct_rwkv` to load a local RWKV checkpoint through the Albatross engine without a model API server. Keep `openai_compat` only as a migration or remote-service adapter.

See [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md) for paths, environment overrides, readiness checks, and a real generation smoke test.

### 3. Parameter Configuration Guide

Below are the parameter descriptions found in `config.json`.

> Parameters that need to be configured before running

| Parameter Name | Function | Options |
| --- | --- | --- |
| `LLM_PROVIDER` | Runtime model provider | `local_7b` `local_13b` `local_direct_1p5b` `baidu` `volcengine` |
| `API_KEYS.baidu` | API_Key for Baidu AI Studio | `Any valid Key` (Optional if not used) |
| `API_KEYS.volcengine` | API_Key for Volcengine | `Any valid Key` (Optional if not used) |
| `API_KEYS.tavily` | API_Key for tavily search engine | `Any valid Key` |

For other pre-configured modifiable parameters, please check the Appendix.

### 4. CLI Startup

```bash
cd RWKV-ECRA
python main.py
```

After starting, you can enter commands in the terminal, for example:

```
Help me look into the dynamics of RWKV-based research, and see if there is anything currently not directly related to RWKV, but might support RWKV research or be supported by RWKV in the future. Be sure not to only look at local files, but also perform a web search.

```

### 5. GUI Startup

```bash
python api.py
```

Once started, the backend service runs at `http://0.0.0.0:8787` by default, and the frontend is already adapted to this port.

Then, open another terminal and enter `RWKV-ECRA/frontend`:

```bash
npm install
npm run dev
```

> Note: Node.js needs to be installed in advance. Please refer to the official Node.js website for installation instructions.

Once started, the frontend runs at `http://127.0.0.1:5177` by default.

Before deployment, run `python -m scripts.preflight --dataset data/evaluation/dynamic.jsonl`.
The backend exposes `/healthz`, `/readyz`, `/api/v1/metrics/operational`, and a
Prometheus-compatible `/metrics`. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for dynamic evaluation, blind review, and rollback gates.

## Future Optimization Plans

* Expand the direct RWKV runtime with safe same-length batching and a dedicated local worker process.
* Optimize performance and execution paths.
* Beautify the frontend and refine logging. Currently, the frontend lacks visual appeal and the logic could be more elegant.
* Support LLMs and search engines from more sources.
* Fix some existing citation parsing and passing issues in the near future.

## Appendix

### 1. Other Modifiable Parameters

> Parts with default parameters set that can be customized

| Parameter Name | Function | Options |
| --- | --- | --- |
| `LLM_ENDPOINTS.volcengine.base_url` | Volcengine model API base URL | `https://ark.cn-beijing.volces.com/api/v3` |
| `LLM_ENDPOINTS.volcengine.model` | Volcengine model name, check Volcengine documentation for details | `doubao-seed-2-0-lite-260428` |
| `LLM_ENDPOINTS.volcengine.reasoning_effort` | Volcengine model reasoning effort level | `medium` |
| `LLM_ENDPOINTS.baidu.base_url` | Baidu AI Studio model API base URL | `"https://aistudio.baidu.com/llm/lmapi/v3"` |
| `LLM_ENDPOINTS.baidu.model` | Baidu AI Studio model name | `ernie-5.1` |
| `LLM_ENDPOINTS.baidu.max_completion_tokens` | Baidu maximum output token limit | Must be less than `65536` |
| `LLM_ENDPOINTS.baidu.enable_web_search` | Whether to enable the built-in web search for the Ernie model | `true` |
| `SEARCH_CONFIG.search_depth` | Tavily search depth parameter | `advanced` |
| `SEARCH_CONFIG.max_results` | Tavily maximum returned web pages | `10` |
| `SEARCH_CONFIG.time_range` | Tavily search time range | `year` `month` `week` `day` `none` |
| `SEARCH_CONFIG.chunks_per_source` | Number of chunks per search source | `5` |
| `DATA_PIPELINE.input_directory` | Workspace input path | `./data/input` |
| `DATA_PIPELINE.output_directory` | Workspace output path | `./data/output` |
| `DATA_PIPELINE.checkpoint_directory` | Checkpoint storage path | `"./data/checkpoints"` |
| `DATA_PIPELINE.debug_directory` | RWKV model debug log output path | `./data/debug_slm` |
| `DATA_PIPELINE.enable_debug_slm` | Whether to enable RWKV model debug log output | `false` `true` |
| `DATA_PIPELINE.allowed_extensions` | Allowed input file extensions. PDF and other formats are not currently supported; consider converting them first. | `[".txt", ".md"]` |
| `DATA_PIPELINE.max_chunk_tokens` | RWKV model max input tokens | Any integer (1600~2400 is recommended) |
| `DATA_PIPELINE.overlap_ratio` | Context overlap ratio during RWKV processing | Any float (0.05 is recommended) |
| `DATA_PIPELINE.reduce_group_size` | Batch size during RWKV's secondary summary merge | Any integer (less than 4 is recommended) |
| `DATA_PIPELINE.reduce_target_chunks` | Target number of chunks during reduce | `1` |
| `DATA_PIPELINE.reduce_max_tokens` | Max input tokens for the LLM summarization after RWKV reaches its compression limit | Any integer, min(32k, model max context/2) is recommended |
| `DATA_PIPELINE.slm_reduce_steps` | Maximum compression steps for RWKV | Any integer, 2 is recommended |
| `DATA_PIPELINE.llm_safe_window_tokens` | Maximum input safe window for the LLM | `60000` |
| `DATA_PIPELINE.map_focus` | Instruction for chunked summarization | `"保持原意压缩，提取核心逻辑，严格保留所有事实性内容"` |
| `DATA_PIPELINE.reduce_rule` | Instruction for merged summarization | `"保持原意压缩，去重并合并同类逻辑，绝对保留事实性数据和原始结论"` |
| `DATA_PIPELINE.map_focus_en` | English instruction for chunked summarization | `"Compress while maintaining original meaning, extract core logic, strictly preserve all factual content"` |
| `DATA_PIPELINE.reduce_rule_en` | English instruction for merged summarization | `"Compress while maintaining original meaning, deduplicate and merge similar logic, absolutely preserve factual data and original conclusions"` |
| `DATA_PIPELINE.english_ratio_threshold` | Text is judged as an English document if the English ratio exceeds this threshold, otherwise Chinese | `0.5` |
| `DATA_PIPELINE.reduce_max_tokens_internal` | Maximum input limit during the internal merge phase | `3500` |
| `DATA_PIPELINE.slm_repeat_threshold` | SLM repetition penalty threshold | `5` |
| `AGENT_CONFIG.max_files_per_batch` | Maximum files processed per round | `10` |
| `AGENT_CONFIG.max_error_retries` | Maximum error retries | `3` |
| `AGENT_CONFIG.memory_truncate_length` | Memory truncation length | `60000` |
| `SLM_CONFIG.endpoint` | RWKV API endpoint | `"http://192.168.0.82:8080/v1/chat/completions"` |
| `SLM_CONFIG.password` | RWKV API password (leave empty if none) | `"rwkv-skills"` |
| `SLM_CONFIG.concurrency` | RWKV maximum concurrency | Integer. For the 7.2B model with 24GB VRAM, setting this to 16GB is optimal. |
| `TRACKING.enable` | Whether to enable log tracking | `true` |
| `TRACKING.enable_slm_log` | Whether to track RWKV processing logs | `false` |
| `TRACKING.log_dir` | Directory path to store logs | `"./logs"` |
| `MODEL_RUNTIME.backend` | Project-owned model execution backend | `direct_rwkv` or `openai_compat` |
| `MODEL_RUNTIME.direct_rwkv.engine_root` | Albatross checkout containing `reference/rwkv7.py` | Local path |
| `MODEL_RUNTIME.direct_rwkv.model_path` | RWKV checkpoint path, with or without `.pth` | Local path |

```
