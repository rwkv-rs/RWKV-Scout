# External project evaluation: wigolo

## Candidate record

- Repository: <https://github.com/KnockOutEZ/wigolo>
- Snapshot checked: `e3a9adaf1fa5090c7fde42f35124095a2d97b8d9` (main, 2026-07-27)
- License: GNU Affero General Public License v3.0; the repository also declares a maintainer trademark.
- Scope: local-first web search, fetch, crawl, extraction, cache, similarity search and research loops exposed through MCP, REST and SDK surfaces.
- Runtime note: the project documents Node.js 20 or newer and a local browser/on-device model download during setup.

## What was evaluated

wigolo solves web acquisition and navigation, not RWKV answer quality, experiment design, citation adjudication, or domain-specific evaluation. Its useful boundary for this repository is:

1. receive a short search query and bounded result/fetch parameters;
2. return result metadata and page content/evidence;
3. preserve provider citation identifiers and source spans when available;
4. leave synthesis, trust decisions, citation validation and final answer generation to RWKV-Scout.

The local adapter in `tools/web_search_wigolo.py` uses only the documented local REST boundary. It does not copy wigolo source code or make wigolo a hard dependency. `auto` falls back to the existing keyless provider; `only` fails explicitly; `off` keeps the original provider. This keeps the core pipeline replaceable and avoids an unreviewed dependency expansion.

## Independent validation plan

The candidate must be compared with the existing keyless provider using the same RWKV model, endpoint, context length, dataset version and prompt version. The only changed variable is `search_action`:

- baseline: `search_web_keyless`
- candidate: `search_web_wigolo`

For each case, `scripts/run_dynamic_eval.py` records the full run and `scripts/compare_experiment_runs.py` compares Recall@K, MRR, nDCG, evidence recall, citation metrics, duplicate/invalid results, tool calls, retries and latency. Cases without reference URLs or key facts remain unknown rather than being scored as wins.

## Current decision

**Continue experiment.** The adapter is accepted as an optional acquisition provider because it has a bounded interface, explicit fallback and preserved evidence metadata. It is not evidence that wigolo improves the production system until a same-model, same-dataset baseline/candidate experiment shows stable gains without citation or latency regression.
