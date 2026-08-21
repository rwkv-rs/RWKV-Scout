# Current acceptance audit

This is an evidence-based status record for the RWKV-Scout production goal. It
is intentionally not a claim that the system is ready while the human
reference set and blind review gates are incomplete.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Exact active RWKV 13.3B model contract | `config.json`, `config.validate_experiment_model_contract()`, unit test | Verified after explicit model switch |
| Replayable end-to-end trace | `data/output/experiments/rwkv-13b-full-postfix-baseline.json`; 60/60 replay traces valid; all 60 runs completed with final answers and answer-level citations | Verified for real 13.3B flow |
| Citation URL/evidence/locator checks | `utils/citation_validator.py`, search-page filtering, selected-context citation scope, locator coverage metric | Structural checks verified; source truth still requires reviewed references |
| Dynamic test generation and expansion | 60-case `dynamic.jsonl`; `--append` preserves reviewed rows and rejects ID collisions | Verified |
| Stage metrics and grouping | `utils/experiment_metrics.py`; domain/persona/task/difficulty grouping, paired deltas and P50/P95/P99 | Implemented |
| Repeated and order-swapped experiments | `data/output/experiments/evidence-quality-ab-13b-paired-manifest.json`; 2 repeats, real 13.3B synthesis, baseline-first/candidate-first, mixed stability direction | Verified; candidate not promoted |
| Blind human review | blind packet, separate key, review recorder, unblinding command | Implemented; no completed review set yet |
| Model-assisted review | supplementary judge artifact stores prompt/output/scores/reason | Implemented; not a promotion gate |
| Production runtime controls | workspace lease, cross-process file leases, timeout, degradation, health/readiness, operational metrics | Smoke verified |
| Real RWKV answer-quality comparison | `rwkv-13b-full-postfix-baseline.json` and `rwkv-13b-full-postfix-baseline-summary.json` (60 real samples); 0 model failures, P95 43.6s, citation completeness 100%, structural citation accuracy 83.3% | Baseline verified; factual quality gate pending references |
| Human-reviewed reference gate | 60/60 dynamic samples are `pending`; ready count is 0 | Not ready |
| Production promotion decision | comparison reports conservatively return `继续实验` while gates are incomplete | Correctly blocked |

The remaining production blockers are human-reviewed reference answers/source
freshness, blind A/B scores, and a stable candidate decision on those reviewed
metrics. The active 13.3B service and real synthesis path are now verified;
existing 7.2B artifacts remain historical evidence only. The current full
baseline also shows retrieval-quality work remains: body availability is
49.6%, duplicate-fragment rate is 67.3%, and important-information
truncation is 28.3%.
