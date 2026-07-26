#!/usr/bin/env python3
"""Reproduce the legacy RWKV7 Three Body 1 final-answer baseline end to end.

This single file runs both stages of the earlier RWKV7 Three Body 1
material-QA experiment:

1. Generate per-chunk RWKV outputs with the fixed top3 multi-state logit mix.
2. Rerank final answers with RWKV logprob + query quote score.

Expected result on the included 30 structured tasks:

    evidence_qa/query_sqrt_mean -> 24/30 exact, 25/30 contain

For the current best 41-task example, use
reproduce_rwkv7_threebody1_triggered_candidates.py.

Expected command on the remote machine:

    cd /home/codex/work/dev2
    /home/codex/miniconda3/bin/python \
      subprojects/rwkv7-long-context/examples/prepare_threebody1_chunks_tasks.py

    CUDA_VISIBLE_DEVICES=1 /home/codex/miniconda3/bin/python \
      subprojects/rwkv7-long-context/examples/reproduce_rwkv7_threebody1_final_answer.py \
      --albatross-dir /home/codex/work/dev2/Albatross/faster3a_2605 \
      --model /dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth

Default input/output files are under subprojects/rwkv7-long-context/runs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_RWKV_OUTPUTS_JSONL = RUNS_DIR / "threebody1_repro_rwkv7_per_chunk_outputs.jsonl"
DEFAULT_RWKV_SUMMARY_JSON = RUNS_DIR / "threebody1_repro_rwkv7_per_chunk_summary.json"
DEFAULT_TASKS_JSONL = RUNS_DIR / "threebody1_structured_task_candidates_chunks1000_overlap3.jsonl"
DEFAULT_CHUNKS_JSONL = RUNS_DIR / "threebody1_chunks_1000_overlap3.jsonl"
DEFAULT_OUT_JSONL = RUNS_DIR / "threebody1_repro_rwkv7_final_answer_outputs.jsonl"
DEFAULT_OPTIONS_JSONL = RUNS_DIR / "threebody1_repro_rwkv7_final_answer_options.jsonl"
DEFAULT_SUMMARY_JSON = RUNS_DIR / "threebody1_repro_rwkv7_final_answer_summary.json"

SCORE_METHODS = ("sum", "mean", "sqrt_mean", "first", "query_sqrt_mean")
TOP3_MULTI_STATE_COEFFS = (0.1985609382390976, 0.3641647398471832, 0.03268775716423988)
MAX_NEW_TOKENS_PER_CHUNK = 8
QUERY_STOP_SUBSTRINGS = (
    "多少",
    "什么",
    "第几",
    "几号",
    "是多少",
    "是什么",
    "叫什么",
    "分别",
    "哪些",
    "请",
    "根据",
    "下文",
    "回答",
)
PROMPT_QUERY_STOP_SUBSTRINGS = (
    "多少",
    "什么",
    "第几",
    "几号",
    "是多少",
    "是什么",
    "叫什么",
    "请",
    "根据",
    "回答",
)
STOP_CHARS = ('"', "}", "\n")
RWKV7_TORCH_EXTENSION_DIR_NAMES = {
    "rwkv7_v3a_ops",
    "rwkv7_fast_ops_fp16",
    "rwkv7_fast_ops_fp32io16",
    "rwkv7_wkv_fp16_v2",
    "rwkv7_wkv_fp32_v2",
}
TOP3_AUX_SPECS = (
    {
        "id": "user_only_no_bos_cond_minus",
        "prefix": "User:\n",
        "add_bos_zero": False,
        "direction": "cond_minus_aux",
        "suffix_mode": "main",
    },
    {
        "id": "extract_pos_direct_minus_aux",
        "prefix": "User: 请从材料中抽取明确答案，直接给出JSON。\n",
        "add_bos_zero": False,
        "direction": "minus_aux",
        "suffix_mode": "pos_direct",
    },
    {
        "id": "null_bias_neg_strict_plus",
        "prefix": "User: 没有依据时必须回答null，只输出JSON。\n",
        "add_bos_zero": False,
        "direction": "plus_aux",
        "suffix_mode": "neg_strict",
    },
)


@dataclass(frozen=True)
class Candidate:
    task_id: str
    chunk_id: int
    answer: str
    raw_answer: str
    quote_found: bool
    quote_text: str | None


@dataclass
class PrefillOutput:
    state: list[torch.Tensor]
    logits: torch.Tensor
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RWKV7 per-chunk outputs and reproduce final-answer rerank on Three Body 1 structured QA."
    )
    parser.add_argument("--albatross-dir", default="/home/codex/work/dev2/Albatross/faster3a_2605")
    parser.add_argument("--model", default="/dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth")
    parser.add_argument("--tasks-jsonl", default=str(DEFAULT_TASKS_JSONL))
    parser.add_argument("--chunks-jsonl", default=str(DEFAULT_CHUNKS_JSONL))
    parser.add_argument("--rwkv-outputs-jsonl", default=str(DEFAULT_RWKV_OUTPUTS_JSONL))
    parser.add_argument("--rwkv-summary", default=str(DEFAULT_RWKV_SUMMARY_JSON))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUT_JSONL))
    parser.add_argument("--options-jsonl", default=str(DEFAULT_OPTIONS_JSONL))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--reuse-rwkv-outputs", action="store_true")
    parser.add_argument("--per-chunk-batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--max-options", type=int, default=48)
    parser.add_argument("--prompt-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--final-prefill-mode", choices=("active_b1t1", "b1tn", "bntn"), default="bntn")
    parser.add_argument("--hybrid-query-weight", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp32io16")
    parser.add_argument("--emb", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--batched-rkv", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc")
    parser.add_argument("--lowrank-weight", choices=("orig", "transpose", "both"), default="both")
    parser.add_argument("--orig-linear-groups", default="att_c2c,ffn_key,head")
    parser.add_argument("--torch-extensions-dir", default="")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_txt(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text.replace("\r\n", "\n")).strip()


def load_structured_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    for line_number, row in enumerate(load_jsonl(path), 1):
        if row.get("answer_format") not in {"scalar_number_string", "scalar_string"}:
            continue
        if not isinstance(row.get("answer"), str):
            continue
        row = dict(row)
        row["positive_chunks"] = set(row["positive_chunks"])
        row["_source_line"] = line_number
        tasks.append(row)
    if not tasks:
        raise RuntimeError(f"no structured tasks selected from {path}")
    return tasks


def parse_answer_prefix(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "null"
    if text.startswith("null"):
        return "null"
    if text.startswith('"'):
        text = text[1:].strip()
    positions = [text.find(char) for char in STOP_CHARS if text.find(char) >= 0]
    if positions:
        text = text[: min(positions)]
    return text.strip() or "null"


def answer_from_output_row(row: dict[str, Any]) -> str:
    if row.get("extracted_answer") is not None:
        return parse_answer_prefix(row.get("extracted_answer"))
    return parse_answer_prefix(row.get("greedy_completion"))


def normalize_answer_for_format(answer: str, answer_format: str | None) -> str | None:
    answer = str(answer).strip()
    if not answer or answer == "null":
        return "null"
    if answer_format == "scalar_number_string":
        if re.search(r"[A-Za-z]", answer):
            return None
        numbers = re.findall(r"\d+(?:\.\d+)?", answer)
        if len(numbers) == 1:
            return numbers[0]
        if len(numbers) > 1:
            return None
    return answer


def quote_window(text: str, answer: str, *, max_chars: int = 180) -> dict[str, Any] | None:
    """Return the same compact quote window used by the experiment harness."""
    if not answer or answer == "null":
        return None
    position = text.find(answer)
    if position < 0:
        return None
    half = max_chars // 2
    start = max(0, position - half)
    end = min(len(text), position + len(answer) + half)

    line_start = text.rfind("\n", 0, position)
    if line_start >= 0 and position - line_start <= half:
        start = line_start + 1
    line_end = text.find("\n", position + len(answer))
    if line_end >= 0 and line_end - position <= half:
        end = line_end

    return {
        "quote": text[start:end].strip(),
        "answer_char_start": position,
        "answer_char_end": position + len(answer),
        "quote_char_start": start,
        "quote_char_end": end,
    }


def collect_candidates(
    output_rows: Iterable[dict[str, Any]],
    chunks_by_id: dict[int, dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for row in output_rows:
        raw_answer = answer_from_output_row(row)
        if raw_answer == "null":
            continue
        task_id = str(row.get("task_id", ""))
        task = tasks_by_id.get(task_id)
        answer = normalize_answer_for_format(raw_answer, task.get("answer_format") if task else None)
        if answer is None or answer == "null":
            continue
        chunk_id = int(row.get("chunk_id", -1))
        chunk = chunks_by_id.get(chunk_id, {})
        text = str(chunk.get("text", ""))
        quote = quote_window(text, raw_answer) or quote_window(text, answer)
        grouped[task_id].append(
            Candidate(
                task_id=task_id,
                chunk_id=chunk_id,
                answer=answer,
                raw_answer=raw_answer,
                quote_found=quote is not None,
                quote_text=str(quote["quote"]) if quote is not None else None,
            )
        )
    return grouped


def query_ngrams(question: str) -> set[str]:
    text = re.sub(r"[？?，,。、《》：:（）()\s]", "", str(question))
    for substring in QUERY_STOP_SUBSTRINGS:
        text = text.replace(substring, "")
    grams: set[str] = set()
    for size in range(2, 7):
        for start in range(0, len(text) - size + 1):
            grams.add(text[start : start + size])
    return grams


def prompt_query_ngrams(question: str) -> set[str]:
    text = re.sub(r"[？?，,。、《》：:（）()\s]", "", str(question))
    for substring in PROMPT_QUERY_STOP_SUBSTRINGS:
        text = text.replace(substring, "")
    grams: set[str] = set()
    for size in range(2, 7):
        for start in range(0, len(text) - size + 1):
            grams.add(text[start : start + size])
    return grams


def query_overlap_score(question: str, quote_text: str) -> float:
    return sum(len(gram) - 1 for gram in query_ngrams(question) if gram in quote_text)


def prompt_query_overlap_score(question: str, quote_text: str) -> float:
    return sum(len(gram) - 1 for gram in prompt_query_ngrams(question) if gram in quote_text)


def query_quote_scores(candidates: list[Candidate], question: str) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        if not candidate.quote_found or not candidate.quote_text:
            continue
        score = 1.0 + 0.1 * query_overlap_score(question, candidate.quote_text)
        if len(candidate.answer) > 24:
            score -= 0.5
        scores[candidate.answer] += score
    return dict(scores)


def candidate_rank_score(candidate: Candidate, question: str) -> tuple[float, int]:
    quote = candidate.quote_text or ""
    overlap = prompt_query_overlap_score(question, quote)
    return (overlap + (2 if candidate.quote_found else 0), -candidate.chunk_id)


def build_candidate_lines(candidates: list[Candidate], question: str, max_candidates: int) -> list[str]:
    ranked = sorted(candidates, key=lambda candidate: candidate_rank_score(candidate, question), reverse=True)
    lines = []
    for index, candidate in enumerate(ranked[:max_candidates], 1):
        quote = (candidate.quote_text or "").replace("\n", " ")
        if len(quote) > 180:
            quote = quote[:180] + "..."
        lines.append(
            f"{index}. chunk={candidate.chunk_id} answer={json.dumps(candidate.answer, ensure_ascii=False)} "
            f"quote={json.dumps(quote, ensure_ascii=False)}"
        )
    return lines


def build_answer_options(candidates: list[Candidate], question: str, max_options: int) -> list[str]:
    ranked = sorted(candidates, key=lambda candidate: candidate_rank_score(candidate, question), reverse=True)
    options: list[str] = []
    seen: set[str] = set()
    for candidate in ranked:
        answer = str(candidate.answer)
        if answer in seen:
            continue
        seen.add(answer)
        options.append(answer)
        if len(options) >= max_options:
            break
    if "null" not in seen:
        options.append("null")
    return options


def build_evidence_prompt(task: dict[str, Any], candidates: list[Candidate], max_candidates: int) -> str:
    question = str(task["question"])
    lines = build_candidate_lines(candidates, question, max_candidates)
    evidence = "\n".join(lines) if lines else "（没有非null候选）"
    return (
        "User: 下文是从长文中抽出的候选证据：\n"
        f"{evidence}\n"
        f"请根据上文回答：{question}（如果上文没有明确答案，回答\"null\"）\n\n"
        'Assistant: {"answer": "'
    )


def evidence_f1(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    strict_tp = strict_fp = strict_fn = 0
    nonnull_tp = nonnull_fp = nonnull_fn = 0
    for row in rows:
        label = str(row.get("label", ""))
        prediction = answer_from_output_row(row)
        positive = label != "null"
        predicted_nonnull = prediction != "null"
        correct = positive and predicted_nonnull and (prediction == label or label in prediction)

        if correct:
            strict_tp += 1
        elif positive:
            strict_fn += 1
            if predicted_nonnull:
                strict_fp += 1
        elif predicted_nonnull:
            strict_fp += 1

        if positive and predicted_nonnull:
            nonnull_tp += 1
        elif positive:
            nonnull_fn += 1
        elif predicted_nonnull:
            nonnull_fp += 1

    return {
        "strict": prf(strict_tp, strict_fp, strict_fn),
        "nonnull_detection": prf(nonnull_tp, nonnull_fp, nonnull_fn),
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def clear_rwkv7_torch_extension_locks(torch_extensions_dir: str = "") -> None:
    roots: list[Path] = []
    if torch_extensions_dir:
        roots.append(Path(torch_extensions_dir).expanduser())
    env_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path.home() / ".cache" / "torch_extensions")

    seen_roots: set[Path] = set()
    extension_dirs: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        if root.name in RWKV7_TORCH_EXTENSION_DIR_NAMES and root.is_dir():
            extension_dirs.add(root)
        for dirname in RWKV7_TORCH_EXTENSION_DIR_NAMES:
            extension_dirs.update(path for path in root.glob(f"**/{dirname}") if path.is_dir())

    removed: list[str] = []
    for extension_dir in sorted(extension_dirs):
        for lock_path in [extension_dir / "lock", *extension_dir.glob("*.lock")]:
            if not lock_path.is_file():
                continue
            try:
                lock_path.unlink()
                removed.append(str(lock_path))
            except OSError as exc:
                print(json.dumps({"event": "lock_remove_failed", "path": str(lock_path), "error": str(exc)}), flush=True)

    if removed:
        print(
            json.dumps({"event": "cleared_rwkv7_torch_extension_locks", "count": len(removed), "paths": removed}),
            flush=True,
        )


def setup_rwkv7(args: argparse.Namespace):
    if args.torch_extensions_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = args.torch_extensions_dir
    clear_rwkv7_torch_extension_locks(args.torch_extensions_dir)
    sys.path.insert(0, args.albatross_dir)
    os.chdir(args.albatross_dir)

    import rwkv7_fast_v3a as v3a
    from rwkv.utils import PIPELINE

    v3a.MODEL_PATH = args.model
    v3a.WKV_MODE = args.wkv
    v3a.EMB_DEVICE = args.emb
    v3a.RKV_MODE = args.batched_rkv
    v3a.CMIX_SPARSE = args.cmix_sparse
    v3a.LOWRANK_WEIGHT = args.lowrank_weight
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(args.orig_linear_groups)
    torch.set_grad_enabled(False)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    token_device = "cpu" if model.emb_cpu else args.device
    return model, tokenizer, token_device


def copy_batch_rows_to_active(full_state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    return [
        full_state[0][:, :, rows, :].contiguous(),
        full_state[1][:, rows, :, :, :].contiguous(),
        full_state[2][rows].contiguous(),
    ]


def copy_active_rows_to_batch(full_state: list[torch.Tensor], rows: list[int], active_state: list[torch.Tensor]) -> None:
    row_tensor = torch.tensor(rows, dtype=torch.long, device=full_state[2].device)
    full_state[0].index_copy_(2, row_tensor, active_state[0])
    full_state[1].index_copy_(1, row_tensor, active_state[1])
    full_state[2].index_copy_(0, row_tensor, active_state[2])


def clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [part.clone() for part in state]


def repeat_state_rows(state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    row_tensor = torch.tensor(rows, dtype=torch.long, device=state[2].device)
    return [
        state[0].index_select(2, row_tensor).contiguous(),
        state[1].index_select(1, row_tensor).contiguous(),
        state[2].index_select(0, row_tensor).contiguous(),
    ]


def select_state_rows(state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    return [
        state[0][:, :, rows, :].contiguous(),
        state[1][:, rows, :, :, :].contiguous(),
        state[2][rows].contiguous(),
    ]


@torch.inference_mode()
def parallel_prefill(model: Any, token_lists: list[list[int]], token_device: str) -> PrefillOutput:
    batch_size = len(token_lists)
    lengths = [len(tokens) for tokens in token_lists]
    max_len = max(lengths)
    final_state = model.zero_state(batch_size)
    active_rows = list(range(batch_size))
    active_state = model.zero_state(batch_size)
    last_logits: torch.Tensor | None = None
    started = time.perf_counter()

    for pos in range(max_len):
        active_rows = [row for row in active_rows if pos < lengths[row]]
        if not active_rows:
            break
        active_tokens = [token_lists[row][pos] for row in active_rows]
        tokens = torch.tensor(active_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        logits = model.forward(tokens, active_state)
        if last_logits is None:
            last_logits = torch.empty((batch_size, logits.size(-1)), dtype=logits.dtype, device=logits.device)

        finished_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 == lengths[row]]
        if finished_local:
            finished_rows = [active_rows[local_pos] for local_pos in finished_local]
            finished_local_tensor = torch.tensor(finished_local, dtype=torch.long, device=active_state[2].device)
            finished_rows_tensor = torch.tensor(finished_rows, dtype=torch.long, device=last_logits.device)
            finished_state = [
                active_state[0].index_select(2, finished_local_tensor).contiguous(),
                active_state[1].index_select(1, finished_local_tensor).contiguous(),
                active_state[2].index_select(0, finished_local_tensor).contiguous(),
            ]
            row_tensor = torch.tensor(finished_rows, dtype=torch.long, device=final_state[2].device)
            final_state[0].index_copy_(2, row_tensor, finished_state[0])
            final_state[1].index_copy_(1, row_tensor, finished_state[1])
            final_state[2].index_copy_(0, row_tensor, finished_state[2])
            last_logits.index_copy_(
                0,
                finished_rows_tensor,
                logits.index_select(0, finished_local_tensor.to(logits.device)),
            )

        keep_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 < lengths[row]]
        if keep_local:
            active_rows = [active_rows[local_pos] for local_pos in keep_local]
            active_state = select_state_rows(active_state, keep_local)
        else:
            active_rows = []

        if (pos + 1) % 500 == 0:
            print(
                f"parallel_prefill pos={pos + 1}/{max_len} active={len(active_rows)} "
                f"elapsed_s={time.perf_counter() - started:.3f}",
                flush=True,
            )

    if last_logits is None:
        raise RuntimeError("empty prefill")
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    print(f"parallel_prefill done batch={batch_size} max_len={max_len} elapsed_s={elapsed_s:.3f}")
    return PrefillOutput(
        state=final_state,
        logits=last_logits,
        stats={
            "mode": "active_b1t1",
            "batch": batch_size,
            "groups": None,
            "max_group": None,
            "max_len": max_len,
            "elapsed_s": elapsed_s,
        },
    )


@torch.inference_mode()
def bntn_prefill(model: Any, token_lists: list[list[int]], token_device: str, mode: str) -> PrefillOutput:
    """Prefill prompts with Albatross v3a exact-length BnTn groups."""

    batch_size = len(token_lists)
    groups_by_len: dict[int, list[int]] = defaultdict(list)
    for row, tokens in enumerate(token_lists):
        if not tokens:
            raise RuntimeError(f"empty prompt tokens at row {row}")
        groups_by_len[len(tokens)].append(row)

    final_state = model.zero_state(batch_size)
    last_logits: torch.Tensor | None = None
    started = time.perf_counter()
    group_count = 0
    max_group_size = 0
    max_len = max(groups_by_len)

    for _token_len, rows in sorted(groups_by_len.items(), key=lambda item: item[0], reverse=True):
        group_count += 1
        max_group_size = max(max_group_size, len(rows))
        tokens = torch.tensor([token_lists[row] for row in rows], dtype=torch.long, device=token_device)
        group_state = model.zero_state(len(rows))
        logits = model.forward(tokens, group_state)
        if last_logits is None:
            last_logits = torch.empty((batch_size, logits.size(-1)), dtype=logits.dtype, device=logits.device)
        copy_active_rows_to_batch(final_state, rows, group_state)
        row_tensor = torch.tensor(rows, dtype=torch.long, device=last_logits.device)
        last_logits.index_copy_(0, row_tensor, logits.to(last_logits.device))

    if last_logits is None:
        raise RuntimeError("empty prefill")
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    print(
        f"bntn_prefill done batch={batch_size} groups={group_count} max_group={max_group_size} "
        f"max_len={max_len} elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    return PrefillOutput(
        state=final_state,
        logits=last_logits,
        stats={
            "mode": mode,
            "batch": batch_size,
            "groups": group_count,
            "max_group": max_group_size,
            "max_len": max_len,
            "elapsed_s": elapsed_s,
        },
    )


def final_prefill(
    model: Any,
    token_lists: list[list[int]],
    token_device: str,
    mode: str,
) -> PrefillOutput:
    if mode == "active_b1t1":
        return parallel_prefill(model, token_lists, token_device)
    if mode in {"b1tn", "bntn"}:
        return bntn_prefill(model, token_lists, token_device, mode)
    raise ValueError(f"unknown final prefill mode: {mode}")


def build_context_prefix(chunk_text: str) -> str:
    return f"User: 下文：\n{clean_txt(chunk_text)}\n"


def build_structured_variants(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for task in tasks:
        variants.append(
            {
                "id": f"base_answer_or_null__{task['id']}",
                "style_id": "base_answer_or_null",
                "task_id": task["id"],
                "task": task,
                "suffix": (
                    f"请根据上文，用JSON回答：{task['question']}（只答答案或\"null\"）\n\n"
                    "Assistant: {\"answer\": \""
                ),
            }
        )
    return variants


def render_aux_suffix(spec: dict[str, Any], variant: dict[str, Any]) -> str:
    mode = str(spec.get("suffix_mode", "main"))
    if mode == "main":
        return str(variant["suffix"])
    task = variant["task"]
    question = str(task["question"])
    if mode == "pos_direct":
        return f"请根据上文，用JSON回答：{question}\n\nAssistant: {{\"answer\": \""
    if mode == "neg_strict":
        return f"请根据上文，用JSON回答：{question}（没有明确出现就回答\"null\"）\n\nAssistant: {{\"answer\": \""
    raise ValueError(f"unknown aux suffix mode: {mode}")


def label_for_variant(chunk: dict[str, Any], variant: dict[str, Any]) -> str:
    task = variant["task"]
    return str(task["answer"]) if int(chunk["chunk_id"]) in task["positive_chunks"] else "null"


def direction_logits(cond_logits: torch.Tensor, aux_logits: torch.Tensor, direction: str) -> torch.Tensor:
    cond = cond_logits.float()
    aux = aux_logits.float()
    if direction == "cond_minus_aux":
        return cond - aux
    if direction == "minus_aux":
        return -aux
    if direction == "plus_aux":
        return aux
    raise ValueError(f"unknown direction: {direction}")


def mixed_logits_from_dirs(
    cond_logits: torch.Tensor,
    aux_logits_list: list[torch.Tensor],
    aux_specs: Iterable[dict[str, Any]],
    coeffs: Iterable[float],
) -> torch.Tensor:
    mixed = cond_logits.float()
    for aux_logits, spec, coeff in zip(aux_logits_list, aux_specs, coeffs):
        if coeff != 0.0:
            mixed = mixed + coeff * direction_logits(cond_logits, aux_logits, str(spec["direction"]))
    return mixed


@torch.inference_mode()
def continuation_prefill(
    model: Any,
    initial_state: list[torch.Tensor],
    token_lists: list[list[int]],
    token_device: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Continue per-row states with exact-length BnTn groups.

    Padding would alter RWKV state, so rows are grouped only when their
    continuation token length is identical.
    """

    batch_size = len(token_lists)
    groups_by_len: dict[int, list[int]] = defaultdict(list)
    for row, tokens in enumerate(token_lists):
        if not tokens:
            raise RuntimeError(f"empty continuation tokens at row {row}")
        groups_by_len[len(tokens)].append(row)

    final_state = clone_state(initial_state)
    last_logits: torch.Tensor | None = None
    started = time.perf_counter()
    group_count = 0
    max_group_size = 0
    max_len = max(groups_by_len)

    for _token_len, rows in sorted(groups_by_len.items(), key=lambda item: item[0], reverse=True):
        group_count += 1
        max_group_size = max(max_group_size, len(rows))
        tokens = torch.tensor([token_lists[row] for row in rows], dtype=torch.long, device=token_device)
        group_state = select_state_rows(initial_state, rows)
        logits = model.forward(tokens, group_state)
        if last_logits is None:
            last_logits = torch.empty((batch_size, logits.size(-1)), dtype=logits.dtype, device=logits.device)
        copy_active_rows_to_batch(final_state, rows, group_state)
        row_tensor = torch.tensor(rows, dtype=torch.long, device=last_logits.device)
        last_logits.index_copy_(0, row_tensor, logits.to(last_logits.device))

    if last_logits is None:
        raise RuntimeError("empty continuation prefill")
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    print(
        f"continuation_prefill done batch={batch_size} groups={group_count} max_group={max_group_size} "
        f"max_len={max_len} elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    return final_state, last_logits


@torch.inference_mode()
def multi_state_decode(
    model: Any,
    tokenizer: Any,
    cond_state: list[torch.Tensor],
    cond_logits: torch.Tensor,
    aux_states: list[list[torch.Tensor]],
    aux_logits_list: list[torch.Tensor],
    aux_specs: Iterable[dict[str, Any]],
    coeffs: Iterable[float],
    max_new_tokens: int,
    token_device: str,
) -> tuple[list[list[int]], list[str]]:
    batch_size = cond_logits.size(0)
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    texts = ["" for _ in range(batch_size)]
    active_rows = list(range(batch_size))
    active_cond_state = clone_state(cond_state)
    active_aux_states = [clone_state(state) for state in aux_states]
    next_tokens = torch.argmax(mixed_logits_from_dirs(cond_logits, aux_logits_list, aux_specs, coeffs), dim=-1).cpu().tolist()
    started = time.perf_counter()

    for _step in range(max_new_tokens):
        forward_rows = []
        forward_tokens = []
        for row in active_rows:
            token = int(next_tokens[row])
            if token == 0:
                continue
            generated[row].append(token)
            forward_rows.append(row)
            forward_tokens.append(token)
        if not forward_rows:
            break

        active_cond = copy_batch_rows_to_active(active_cond_state, forward_rows)
        active_aux = [copy_batch_rows_to_active(state, forward_rows) for state in active_aux_states]
        token_tensor = torch.tensor(forward_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        next_cond_logits = model.forward(token_tensor, active_cond)
        next_aux_logits = [model.forward(token_tensor, state) for state in active_aux]
        copy_active_rows_to_batch(active_cond_state, forward_rows, active_cond)
        for full_state, state in zip(active_aux_states, active_aux):
            copy_active_rows_to_batch(full_state, forward_rows, state)

        sampled = torch.argmax(
            mixed_logits_from_dirs(next_cond_logits, next_aux_logits, aux_specs, coeffs),
            dim=-1,
        ).cpu().tolist()
        for row, token in zip(forward_rows, sampled):
            next_tokens[row] = int(token)
        active_rows = forward_rows

    torch.cuda.synchronize()
    for row, token_ids in enumerate(generated):
        texts[row] = tokenizer.decode(token_ids)
    print(
        f"multi_state_decode done batch={batch_size} tokens={max_new_tokens} elapsed_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return generated, texts


def bucket_row(row: dict[str, Any]) -> str:
    label = str(row["label"])
    completion = str(row["greedy_completion"])
    if label != "null":
        if row["greedy_matches_target"]:
            return "pos_ok_exact"
        if label in completion:
            return "pos_ok_contain"
        if completion.startswith("null"):
            return "pos_wrong_null"
        return "pos_wrong"
    if row["greedy_matches_target"] or completion.startswith("null"):
        return "neg_ok_null"
    return "neg_wrong_nonnull"


def build_batch_aux(
    aux_variant_states: list[list[torch.Tensor]],
    aux_variant_logits_list: list[torch.Tensor],
    variant_source_rows: list[int],
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    states: list[list[torch.Tensor]] = []
    logits: list[torch.Tensor] = []
    row_tensor_cache: torch.Tensor | None = None
    for aux_variant_state, aux_variant_logits in zip(aux_variant_states, aux_variant_logits_list):
        states.append(repeat_state_rows(aux_variant_state, variant_source_rows))
        if row_tensor_cache is None:
            row_tensor_cache = torch.tensor(variant_source_rows, dtype=torch.long, device=aux_variant_logits.device)
        logits.append(aux_variant_logits.index_select(0, row_tensor_cache))
    return states, logits


def generate_rwkv_chunk_outputs(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    token_device: str,
    chunks: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    out_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    variants = build_structured_variants(tasks)
    aux_specs = list(TOP3_AUX_SPECS)
    coeffs = list(TOP3_MULTI_STATE_COEFFS)
    prefixes = [build_context_prefix(str(row["text"])) for row in chunks]
    prefix_token_lists = [[0] + tokenizer.encode(prefix) for prefix in prefixes]
    suffix_token_lists_by_variant = [tokenizer.encode(variant["suffix"]) for variant in variants]
    labels_by_variant_chunk = [[label_for_variant(row, variant) for row in chunks] for variant in variants]
    all_labels = sorted({label for labels in labels_by_variant_chunk for label in labels})
    answer_tokens = {answer: tokenizer.encode(answer) for answer in all_labels}

    print(
        json.dumps(
            {
                "event": "generate_rwkv_outputs_tokenized",
                "chunks": len(chunks),
                "variants": len(variants),
                "flat_cases": len(chunks) * len(variants),
                "aux_state_ids": [spec["id"] for spec in aux_specs],
                "coeffs": coeffs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    prefix_prefill = bntn_prefill(model, prefix_token_lists, token_device, "bntn")
    prefix_state = prefix_prefill.state

    aux_variant_states: list[list[torch.Tensor]] = []
    aux_variant_logits_list: list[torch.Tensor] = []
    for spec in aux_specs:
        aux_prefix_tokens = ([0] if spec["add_bos_zero"] else []) + tokenizer.encode(str(spec["prefix"]))
        aux_suffix_token_lists = [tokenizer.encode(render_aux_suffix(spec, variant)) for variant in variants]
        aux_prefix_prefill = bntn_prefill(model, [aux_prefix_tokens], token_device, "b1tn")
        aux_initial_state = repeat_state_rows(aux_prefix_prefill.state, [0] * len(variants))
        aux_variant_state, aux_variant_logits = continuation_prefill(
            model,
            aux_initial_state,
            aux_suffix_token_lists,
            token_device,
        )
        aux_variant_states.append(aux_variant_state)
        aux_variant_logits_list.append(aux_variant_logits)

    flat_meta = []
    flat_suffix_token_lists = []
    flat_target_token_ids = []
    prefix_source_rows = []
    variant_source_rows = []
    for variant_index, (variant, suffix_tokens) in enumerate(zip(variants, suffix_token_lists_by_variant)):
        for chunk_index, row in enumerate(chunks):
            label = labels_by_variant_chunk[variant_index][chunk_index]
            flat_meta.append((variant_index, chunk_index, label, variant["task_id"], variant["style_id"]))
            flat_suffix_token_lists.append(suffix_tokens)
            flat_target_token_ids.append(answer_tokens[label])
            prefix_source_rows.append(chunk_index)
            variant_source_rows.append(variant_index)

    out_rows = []
    started = time.perf_counter()
    for batch_start in range(0, len(flat_suffix_token_lists), args.per_chunk_batch_size):
        batch_end = min(batch_start + args.per_chunk_batch_size, len(flat_suffix_token_lists))
        cond_initial_state = repeat_state_rows(prefix_state, prefix_source_rows[batch_start:batch_end])
        cond_final_state, cond_logits = continuation_prefill(
            model,
            cond_initial_state,
            flat_suffix_token_lists[batch_start:batch_end],
            token_device,
        )
        batch_aux_states, batch_aux_logits = build_batch_aux(
            aux_variant_states,
            aux_variant_logits_list,
            variant_source_rows[batch_start:batch_end],
        )
        generated_ids, decoded_texts = multi_state_decode(
            model,
            tokenizer,
            cond_final_state,
            cond_logits,
            batch_aux_states,
            batch_aux_logits,
            aux_specs,
            coeffs,
            MAX_NEW_TOKENS_PER_CHUNK,
            token_device,
        )
        for local_pos, (gen_ids, gen_text) in enumerate(zip(generated_ids, decoded_texts)):
            flat_index = batch_start + local_pos
            variant_index, chunk_index, label, task_id, style_id = flat_meta[flat_index]
            target_ids = flat_target_token_ids[flat_index]
            row = {
                "variant_id": variants[variant_index]["id"],
                "variant_index": variant_index,
                "style_id": style_id,
                "task_id": task_id,
                "chunk_id": chunks[chunk_index]["chunk_id"],
                "label": label,
                "target_token_ids": target_ids,
                "coeffs": coeffs,
                "aux_state_ids": [spec["id"] for spec in aux_specs],
                "greedy_token_ids": gen_ids,
                "greedy_completion": gen_text,
                "greedy_matches_target": gen_ids[: len(target_ids)] == target_ids,
            }
            row["bucket"] = bucket_row(row)
            out_rows.append(row)
        torch.cuda.empty_cache()
        print(
            json.dumps({"event": "rwkv_output_batch_done", "batch_start": batch_start, "batch_end": batch_end}),
            flush=True,
        )

    bucket_counts: dict[str, int] = {}
    for row in out_rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1

    write_jsonl(out_path, out_rows)
    summary = {
        "stage": "rwkv_per_chunk_generation",
        "model": args.model,
        "wkv": args.wkv,
        "chunks": len(chunks),
        "variants": len(variants),
        "flat_cases": len(flat_meta),
        "style_id": "base_answer_or_null",
        "aux_specs": aux_specs,
        "coeffs": coeffs,
        "formula": "mixed = cond + 0.1985609382390976*(cond-user_main) + 0.3641647398471832*(-extract_pos_direct) + 0.03268775716423988*null_strict",
        "max_new_tokens": MAX_NEW_TOKENS_PER_CHUNK,
        "bucket_counts": bucket_counts,
        "elapsed_s": time.perf_counter() - started,
        "out_jsonl": str(out_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "rwkv_per_chunk_generation_done", **summary}, ensure_ascii=False), flush=True)
    return out_rows


@torch.inference_mode()
def score_option_batch(
    model: Any,
    base_state: list[torch.Tensor],
    base_logits: torch.Tensor,
    entries: list[dict[str, Any]],
    token_device: str,
) -> None:
    first_log_probs = torch.log_softmax(base_logits.float(), dim=-1)
    active_entries: list[dict[str, Any]] = []
    active_first_tokens: list[int] = []
    active_base_rows: list[int] = []

    for entry in entries:
        token_ids = entry["option_token_ids"]
        base_row = int(entry["prompt_row_local"])
        first_token = int(token_ids[0])
        first_logprob = float(first_log_probs[base_row, first_token].item())
        entry["token_logprobs"] = [first_logprob]
        entry["first_logprob"] = first_logprob
        if len(token_ids) > 1:
            active_entries.append(entry)
            active_first_tokens.append(first_token)
            active_base_rows.append(base_row)

    if not active_entries:
        return

    active_state = copy_batch_rows_to_active(base_state, active_base_rows)
    previous_tokens = active_first_tokens
    max_len = max(len(entry["option_token_ids"]) for entry in active_entries)
    for pos in range(1, max_len):
        token_tensor = torch.tensor(previous_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        logits = model.forward(token_tensor, active_state)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        keep_local = []
        next_previous_tokens = []
        for local_pos, entry in enumerate(active_entries):
            token_ids = entry["option_token_ids"]
            if pos >= len(token_ids):
                continue
            target_token = int(token_ids[pos])
            entry["token_logprobs"].append(float(log_probs[local_pos, target_token].item()))
            if pos + 1 < len(token_ids):
                keep_local.append(local_pos)
                next_previous_tokens.append(target_token)

        if not keep_local:
            break
        active_entries = [active_entries[local_pos] for local_pos in keep_local]
        active_state = select_state_rows(active_state, keep_local)
        previous_tokens = next_previous_tokens

    torch.cuda.synchronize()


def option_score(row: dict[str, Any], method: str, hybrid_query_weight: float) -> float:
    score_sum = float(row["logprob_sum"])
    token_count = len(row["token_logprobs"])
    if method == "sum":
        return score_sum
    if method == "mean":
        return score_sum / token_count
    if method == "sqrt_mean":
        return score_sum / math.sqrt(token_count)
    if method == "first":
        return float(row["first_logprob"])
    if method == "query_sqrt_mean":
        return float(row["logprob_sqrt_mean"]) + hybrid_query_weight * float(row["query_quote_score"])
    raise ValueError(f"unknown score method: {method}")


def evaluate_predictions(rows: list[dict[str, Any]], tasks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["score_method"])].append(row)

    summary: dict[str, Any] = {}
    for method, method_rows in sorted(grouped.items()):
        exact = 0
        contain = 0
        null_predictions = 0
        for row in method_rows:
            answer = str(tasks_by_id[row["task_id"]]["answer"])
            prediction = str(row["prediction"])
            is_exact = prediction == answer
            is_contain = prediction != "null" and answer in prediction
            exact += int(is_exact)
            contain += int(is_exact or is_contain)
            null_predictions += int(prediction == "null")
        total = len(method_rows)
        summary[f"evidence_qa/{method}"] = {
            "task_count": total,
            "exact": exact,
            "contain": contain,
            "exact_rate": exact / total if total else 0,
            "contain_rate": contain / total if total else 0,
            "null_predictions": null_predictions,
        }
    return summary


def main() -> None:
    args = parse_args()
    rwkv_outputs_path = Path(args.rwkv_outputs_jsonl).resolve()
    rwkv_summary_path = Path(args.rwkv_summary).resolve()
    tasks_path = Path(args.tasks_jsonl).resolve()
    chunks_path = Path(args.chunks_jsonl).resolve()
    out_path = Path(args.out_jsonl).resolve()
    options_path = Path(args.options_jsonl).resolve()
    summary_path = Path(args.summary).resolve()

    tasks = load_structured_tasks(tasks_path)
    chunks = load_jsonl(chunks_path)
    chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}
    tasks_by_id = {str(task["id"]): task for task in tasks}

    model, tokenizer, token_device = setup_rwkv7(args)
    if args.reuse_rwkv_outputs:
        output_rows = load_jsonl(rwkv_outputs_path)
    else:
        output_rows = generate_rwkv_chunk_outputs(
            args,
            model,
            tokenizer,
            token_device,
            chunks,
            tasks,
            rwkv_outputs_path,
            rwkv_summary_path,
        )

    candidates_by_task = collect_candidates(output_rows, chunks_by_id, tasks_by_id)
    query_scores_by_task = {
        task_id: query_quote_scores(candidates, str(tasks_by_id[task_id]["question"]))
        for task_id, candidates in candidates_by_task.items()
        if task_id in tasks_by_id
    }

    flat_prompts: list[str] = []
    option_entries: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        question = str(task["question"])
        candidates = candidates_by_task.get(task_id, [])
        prompt_row = len(flat_prompts)
        flat_prompts.append(build_evidence_prompt(task, candidates, args.max_candidates))
        for option in build_answer_options(candidates, question, args.max_options):
            option_entries.append({"prompt_row": prompt_row, "task_id": task_id, "option": option})

    token_lists = [[0] + tokenizer.encode(prompt) for prompt in flat_prompts]
    for entry in option_entries:
        token_ids = tokenizer.encode(str(entry["option"]))
        if not token_ids:
            raise RuntimeError(f"empty option tokens: {entry['option']!r}")
        entry["option_token_ids"] = token_ids

    prompt_token_lengths = [len(tokens) for tokens in token_lists]
    print(
        json.dumps(
            {
                "event": "tokenized",
                "tasks": len(tasks),
                "option_entries": len(option_entries),
                "min_prompt_tokens": min(prompt_token_lengths),
                "max_prompt_tokens": max(prompt_token_lengths),
                "avg_prompt_tokens": sum(prompt_token_lengths) / len(prompt_token_lengths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    final_score_started = time.perf_counter()
    final_prefill_batches: list[dict[str, Any]] = []
    for batch_start in range(0, len(token_lists), args.prompt_batch_size):
        batch_tokens = token_lists[batch_start : batch_start + args.prompt_batch_size]
        batch_end = batch_start + len(batch_tokens)
        batch_entries = [
            entry
            for entry in option_entries
            if batch_start <= int(entry["prompt_row"]) < batch_end
        ]
        for entry in batch_entries:
            entry["prompt_row_local"] = int(entry["prompt_row"]) - batch_start

        prefill = final_prefill(model, batch_tokens, token_device, args.final_prefill_mode)
        final_prefill_batches.append({"batch_start": batch_start, "batch_end": batch_end, **prefill.stats})
        for score_start in range(0, len(batch_entries), args.score_batch_size):
            score_option_batch(
                model,
                prefill.state,
                prefill.logits,
                batch_entries[score_start : score_start + args.score_batch_size],
                token_device,
            )
    final_logprob_elapsed_s = time.perf_counter() - final_score_started

    option_rows = []
    for entry in option_entries:
        token_logprobs = entry["token_logprobs"]
        score_sum = float(sum(token_logprobs))
        token_count = len(token_logprobs)
        task_id = str(entry["task_id"])
        option = str(entry["option"])
        option_rows.append(
            {
                "task_id": task_id,
                "option": option,
                "option_token_ids": entry["option_token_ids"],
                "token_logprobs": token_logprobs,
                "logprob_sum": score_sum,
                "logprob_mean": score_sum / token_count,
                "logprob_sqrt_mean": score_sum / math.sqrt(token_count),
                "first_logprob": float(entry["first_logprob"]),
                "query_quote_score": float(query_scores_by_task.get(task_id, {}).get(option, 0.0)),
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in option_rows:
        by_task[str(row["task_id"])].append(row)

    prediction_rows = []
    for task_id, rows in sorted(by_task.items()):
        task = tasks_by_id[task_id]
        for method in SCORE_METHODS:
            best = max(rows, key=lambda row: option_score(row, method, args.hybrid_query_weight))
            prediction = normalize_answer_for_format(str(best["option"]), task.get("answer_format"))
            prediction_rows.append(
                {
                    "score_method": method,
                    "task_id": task_id,
                    "question": task["question"],
                    "answer": task["answer"],
                    "prediction": prediction if prediction is not None else str(best["option"]),
                    "best_option": best["option"],
                    "best_option_scores": {
                        "logprob_sum": best["logprob_sum"],
                        "logprob_mean": best["logprob_mean"],
                        "logprob_sqrt_mean": best["logprob_sqrt_mean"],
                        "first_logprob": best["first_logprob"],
                        "query_quote_score": best["query_quote_score"],
                        "hybrid_query_weight": args.hybrid_query_weight,
                    },
                    "option_count": len(rows),
                }
            )

    write_jsonl(options_path, option_rows)
    write_jsonl(out_path, prediction_rows)
    summary = {
        "reproduction": "rwkv7_threebody1_final_answer_evidence_qa_query_sqrt_mean",
        "inputs": {
            "rwkv_outputs_jsonl": str(rwkv_outputs_path),
            "rwkv_summary": str(rwkv_summary_path),
            "tasks_jsonl": str(tasks_path),
            "chunks_jsonl": str(chunks_path),
        },
        "outputs": {
            "predictions_jsonl": str(out_path),
            "options_jsonl": str(options_path),
            "summary": str(summary_path),
        },
        "model": args.model,
        "albatross_dir": args.albatross_dir,
        "wkv": args.wkv,
        "final_prefill_mode": args.final_prefill_mode,
        "hybrid_query_weight": args.hybrid_query_weight,
        "score_formula": "logprob_sum / sqrt(answer_token_count) + hybrid_query_weight * query_quote_score",
        "tasks": len(tasks),
        "chunks": len(chunks),
        "per_chunk_rows": len(output_rows),
        "option_rows": len(option_rows),
        "prompt_tokens": {
            "min": min(prompt_token_lengths),
            "max": max(prompt_token_lengths),
            "avg": sum(prompt_token_lengths) / len(prompt_token_lengths),
        },
        "final_prefill": {
            "mode": args.final_prefill_mode,
            "batches": final_prefill_batches,
        },
        "final_logprob_elapsed_s": final_logprob_elapsed_s,
        "evidence_metrics": evidence_f1(output_rows),
        "final_answer": evaluate_predictions(prediction_rows, tasks_by_id),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
