# Reproducible experiments

## Repeatable deployment

Install the pinned Python dependencies from `requirements.txt`, then start the
API with `python api.py` (or the `rwkv-ecra-api` console entry point). The
frontend remains a separate Vite build. wigolo is optional and is intentionally
not a Python package dependency; install it with its own documented Node
workflow only when the local retrieval candidate is being evaluated.
The API host, port and worker count come from `SERVER` in `config.json` and
can be overridden with `RWKV_ECRA_API_HOST`, `RWKV_ECRA_API_PORT` and
`RWKV_ECRA_API_WORKERS`.

The project separates runtime tasks from experiment evidence. Every task writes:

- `data/output/<task_id>/run_manifest.json`: model, endpoint, code revision, configuration hash, dataset/prompt versions and experiment hypothesis;
- `data/output/<task_id>/events.jsonl`: ordered query rewrites, tool calls/results, page evidence, model prompts/outputs, citations, validation and errors;
- `data/output/<task_id>/retrieval_report.jsonl`: retrieval payload and final answer.

The normalized trace keeps `citations` for sources actually exposed to the
final answer and `retrieval_citations` for the complete provider result set;
this prevents unused search noise from being counted as answer citations while
keeping the full acquisition trail available for replay.

The API exposes the normalized replay view at:

```text
GET /frontend-api/history/<task_id>/trace
```

The same replay contract can be checked from CI or an operations shell:

```powershell
.venv\Scripts\python.exe -m scripts.validate_run_trace `
  data/output/TASK_... --output-directory data/output
```

The command exits with code `2` when required trace fields, event ordering,
terminal events, secret checks or the concurrency gate are invalid.

Operational probes are available at `/healthz` and `/readyz`. The readiness
response reports the configured RWKV model contract, writable input/output
directories and a short `/v1/models` probe; it returns `not_ready` when the
configured local RWKV service is unavailable, but never returns an API key.
The JSON operational metrics are available at `/api/v1/metrics/operational`
and the secret-free Prometheus view is `/metrics`. Run the same checks before
deployment with:

```powershell
.venv\Scripts\python.exe -m scripts.preflight --dataset data/evaluation/dynamic.jsonl
```

The preflight exits non-zero when the exact model contract, writable
directories, model service or (when requested) reference gate is not ready.
Each long-running task also has a configurable wall-clock budget
(`RUNTIME.analysis_timeout_seconds`), a workspace-wide atomic-file lease concurrency gate and
bounded model/network I/O timeouts. The `runtime_budget` and `runtime_gate`
events make queue wait, completion and timeout behavior replayable.

Use the deployment smoke before enabling multiple API workers. It exercises the
same budget and workspace lease without requiring the model service:

```powershell
.venv\Scripts\python.exe -m scripts.runtime_smoke `
  --tasks 8 --workers 4 --hold-ms 50 `
  --output data/output/runtime-smoke.json
```

## Generate a dataset

```powershell
.venv\Scripts\python.exe -m scripts.generate_eval_dataset --count 60 --seed 20260726 --output data/evaluation/dynamic.jsonl
```

The generator can expand the same versioned dataset without overwriting
reviewed samples; use a new seed so question IDs cannot collide:

```powershell
.venv\Scripts\python.exe -m scripts.generate_eval_dataset --append `
  --count 20 --seed 20260746 --output data/evaluation/dynamic.jsonl
```

Each case carries domain, persona, task type, L1-L5 difficulty, acceptance/rejection criteria, risk checks, expected source types and a dataset version. The generated cases are intended to be executed, not only displayed.

Reference answers are human-reviewable data. Add one only after checking the
source text and provide a human reviewer id; the command does not ask the
model to create its own gold answer. Citations must retain evidence text, and
the validator derives a text-excerpt locator when a provider does not expose
paragraph offsets:

```powershell
.venv\Scripts\python.exe -m scripts.set_eval_reference data/evaluation/dynamic.jsonl DYN-20260726-0001 `
  --answer "Reviewed answer" `
  --reviewer-id "human-reviewer-1" `
  --key-fact "Atomic fact that must be present" `
  --citation "https://official.example/docs|Official documentation|Supporting source excerpt"
```

Before using a dataset for promotion, validate that every reference is human
reviewed, has citation evidence and is fresh enough for time-sensitive cases:

```powershell
.venv\Scripts\python.exe -m scripts.validate_references data/evaluation/dynamic.jsonl
```

To make the pending work reviewable in batches, export a queue from a
completed artifact. The model draft is explicitly marked as review-only:

```powershell
.venv\Scripts\python.exe -m scripts.export_reference_queue `
  data/evaluation/dynamic.jsonl `
  --artifact data/output/experiments/search-provider-baseline.json `
  --output data/output/experiments/reference-review-queue.jsonl
```

After a completed run, promote its normalized trace into the sample so that
the search queries, page evidence and navigation history are versioned with
the human-reviewed answer. The trace may be the response from
`/frontend-api/history/<task_id>/trace` or an experiment artifact containing a
`trace` field:

```powershell
.venv\Scripts\python.exe -m scripts.promote_reference_trace `
  data/evaluation/dynamic.jsonl DYN-20260726-0001 `
  data/output/TASK_.../trace.json `
  --answer "Reviewed answer" `
  --reviewer-id "human-reviewer-1" `
  --key-fact "Atomic fact that must be present" `
  --citation "https://official.example/docs|Official documentation|Supporting source excerpt"
```

## Run baseline and candidate

The active production model contract is configured in `config.json` under `LLM_ENDPOINTS.local_13b` and `EXPERIMENT`: model `rwkv7-g1i_preview4922-13.3b-20260720-ctx12288`, endpoint `http://172.21.122.93:29613/v1`, context length `12288`. Use the same dataset and model for both runs; change only `--search-action`. The former `local_7b` profile remains a separately validatable historical baseline and must not be compared as if it were the same-model candidate:

```powershell
.venv\Scripts\python.exe -m scripts.run_dynamic_eval data/evaluation/dynamic.jsonl `
  --experiment-id search-provider-ab --variant baseline `
  --search-action search_web_keyless `
  --output data/output/experiments/search-provider-baseline.json

.venv\Scripts\python.exe -m scripts.run_dynamic_eval data/evaluation/dynamic.jsonl `
  --experiment-id search-provider-ab --variant candidate `
  --search-action search_web_wigolo `
  --baseline-search-action search_web_keyless `
  --output data/output/experiments/search-provider-candidate.json
```

If no model service or wigolo daemon is available, the run still records structured failures and provider fallback behavior. Such a run is an operational diagnostic, not a quality win.

To run a same-model ranking ablation, pass one strategy JSON per variant. The
runner rejects a command that changes both the search action and a strategy
variable, and the strategy is persisted in every manifest:

```powershell
.venv\Scripts\python.exe -m scripts.run_paired_eval data/evaluation/dynamic.jsonl `
  --experiment-id ranking-ab-13b --baseline-action search_web_keyless `
  --candidate-action search_web_keyless --limit 1 --repeats 2 `
  --baseline-strategy-config data/evaluation/strategies/default.json `
  --candidate-strategy-config data/evaluation/strategies/evidence-quality.json `
  --changed-variable ranking_strategy `
  --hypothesis "evidence-quality ranking improves relevant captured evidence without changing retrieval or model"
```

`evidence_quality.v1` is a candidate only: it combines query-term relevance,
captured body availability, cross-round support and provider rank. It must be
measured against `candidate_support_then_rank.v1` on the same model and dataset
before it can be enabled as a production default.

For repeated paired experiments, use the orchestration wrapper. It runs both
variants with the same dataset/model, swaps execution order on the second
repeat, compares each pair, and writes a manifest of every command and result:
The manifest also records per-metric repeat direction (`positive`, `negative`,
`mixed` or `unknown`) so an isolated improvement cannot be mistaken for a
stable promotion signal.

```powershell
.venv\Scripts\python.exe -m scripts.run_paired_eval data/evaluation/dynamic.jsonl `
  --experiment-id search-provider-ab `
  --baseline-action search_web_keyless `
  --candidate-action search_web_wigolo `
  --repeats 2 --limit 10 `
  --changed-variable search_action `
  --hypothesis "wigolo improves primary-source recall without citation or latency regression"
```

To inspect retrieval, extraction, ranking and citation integrity without making
any RWKV synthesis request, use retrieval-only mode:

```powershell
.venv\Scripts\python.exe -m scripts.run_dynamic_eval data/evaluation/dynamic.jsonl `
  --limit 1 --variant baseline --search-action search_web_keyless `
  --retrieval-only --output data/output/experiments/retrieval-smoke.json
```

This mode intentionally writes an empty answer and never creates a reference
answer. It is useful for validating network/search wiring before the local
model service is installed; its artifact must not be used as a quality win.

To persist a trace-derived report for one completed artifact, including
aggregate and domain/persona/task/difficulty groups:

```powershell
.venv\Scripts\python.exe -m scripts.summarize_experiment `
  data/output/experiments/rwkv-13b-full-baseline.json `
  --output data/output/experiments/rwkv-13b-full-baseline-summary.json
```

## Compare and decide

```powershell
.venv\Scripts\python.exe -m scripts.compare_experiment_runs `
  data/output/experiments/search-provider-baseline.json `
  data/output/experiments/search-provider-candidate.json `
  --group-by difficulty `
  --model-judge data/output/experiments/model-judge.json `
  --output data/output/experiments/search-provider-comparison.json
```

The decision helper only emits one of `保留基线`, `继续实验`, `局部采用`, `全量采用` or `回滚候选方案`. Missing references, invalid traces, failed risk gates and unimplemented metrics keep the decision at `继续实验`.
The comparison report also contains paired deltas, an approximate 95% confidence
interval, a failure-case report grouped by stage and a non-mutating rollback
recommendation. For stability checks, run both variants
more than once with the same dataset, set `--repeat-id`, and reverse
`--execution-order` for the second pass; do not combine those runs into a
promotion decision unless the direction is consistent.

Human evaluation uses a blind A/B rubric rather than asking the model to grade
itself. Record all seven scores (1-5) with `scripts.record_human_review.py` and
pass the JSONL file to the comparison command with `--human-reviews`; prompts,
answers, reviewer identity and notes remain in the review artifact.
Build the reviewer packet and keep the unblinding key separately:

```powershell
.venv\Scripts\python.exe -m scripts.build_blind_review `
  data/output/experiments/search-provider-baseline.json `
  data/output/experiments/search-provider-candidate.json `
  --packet data/output/experiments/blind-packet.json `
  --key data/output/experiments/blind-key.json
```

After both A/B reviews are recorded, join the operator-held key before
aggregating by baseline/candidate:

```powershell
.venv\Scripts\python.exe -m scripts.unblind_human_reviews `
  data/output/experiments/human-reviews-blind.jsonl `
  data/output/experiments/blind-key.json `
  --output data/output/experiments/human-reviews-unblinded.jsonl
```

A model-assisted judge is available as supplementary evidence only. It stores
the prompt, visible output, parsed scores and failure reason, and must not
replace human review or the reference gate:

```powershell
.venv\Scripts\python.exe -m scripts.score_model_judge `
  data/output/experiments/search-provider-baseline.json `
  data/output/experiments/search-provider-candidate.json `
  --output data/output/experiments/model-judge.json
```
