"""Verified recursive data utilities for RWKV retrieval-agent training.

The runtime agent never imports this module.  It builds and validates private
training task bundles whose public surface contains only an instruction and a
frozen retrieval environment.  Oracle trajectories and verifier contracts are
kept under ``private/`` and are used only after a rollout has finished.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "rwkv-retrieval-rst-task.v1"
REPORT_VERSION = "rwkv-retrieval-rst-validation.v1"
REQUIRED_FILES = (
    "task.json",
    "instruction.md",
    "environment/sources.jsonl",
    "private/contract.json",
    "private/oracle_trajectory.json",
    "private/reference_answer.md",
)

PRIVATE_LEAK_PATTERNS = (
    re.compile(r"private[/\\]", re.IGNORECASE),
    re.compile(r"oracle_trajectory", re.IGNORECASE),
    re.compile(r"reference_answer", re.IGNORECASE),
    re.compile(r"verifier contract", re.IGNORECASE),
    re.compile(r"hidden test", re.IGNORECASE),
)


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
        }


@dataclass(slots=True)
class ValidationReport:
    bundle: str
    task_id: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        path: str = "",
    ) -> None:
        self.issues.append(
            ValidationIssue(code=code, message=message, severity=severity, path=path)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_VERSION,
            "bundle": self.bundle,
            "task_id": self.task_id,
            "accepted": self.accepted,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "provenance": self.provenance,
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[\w\u3400-\u9fff]+", text, flags=re.UNICODE))


def normalized_tokens(value: Any) -> list[str]:
    return normalize_text(value).split()


def token_windows(value: Any, size: int) -> set[tuple[str, ...]]:
    tokens = normalized_tokens(value)
    if size < 1 or len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def ngram_jaccard(left: Any, right: Any, size: int = 5) -> float:
    left_windows = token_windows(left, size)
    right_windows = token_windows(right, size)
    if not left_windows or not right_windows:
        return 0.0
    return len(left_windows & right_windows) / len(left_windows | right_windows)


def load_evaluation_prompts(paths: Iterable[Path]) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() == ".jsonl":
            values: Any = _read_jsonl(path)
        else:
            values = _read_json(path)
            if isinstance(values, dict):
                values = values.get("cases") or values.get("tasks") or values.get("data") or []
        if not isinstance(values, list):
            continue
        for index, row in enumerate(values):
            if not isinstance(row, Mapping):
                continue
            prompt = str(
                row.get("query")
                or row.get("question")
                or row.get("prompt")
                or row.get("instruction")
                or ""
            ).strip()
            if not prompt:
                continue
            prompts.append(
                {
                    "id": str(row.get("case_id") or row.get("id") or f"{path.stem}:{index}"),
                    "path": str(path),
                    "text": prompt,
                }
            )
    return prompts


def contamination_scan(
    instruction: str,
    evaluation_prompts: Sequence[Mapping[str, str]],
    *,
    exact_window_size: int = 13,
    jaccard_ngram_size: int = 5,
) -> dict[str, Any]:
    instruction_windows = token_windows(instruction, exact_window_size)
    exact_matches: list[dict[str, Any]] = []
    nearest: list[dict[str, Any]] = []
    for row in evaluation_prompts:
        prompt = str(row.get("text") or "")
        overlap = instruction_windows & token_windows(prompt, exact_window_size)
        if overlap:
            exact_matches.append(
                {
                    "id": str(row.get("id") or ""),
                    "path": str(row.get("path") or ""),
                    "window_count": len(overlap),
                    "example": " ".join(next(iter(overlap))),
                }
            )
        nearest.append(
            {
                "id": str(row.get("id") or ""),
                "path": str(row.get("path") or ""),
                "jaccard_5gram": round(
                    ngram_jaccard(instruction, prompt, jaccard_ngram_size), 6
                ),
            }
        )
    nearest.sort(key=lambda item: (-float(item["jaccard_5gram"]), item["id"]))
    return {
        "exact_window_size": exact_window_size,
        "exact_matches": exact_matches,
        "max_5gram_jaccard": nearest[0]["jaccard_5gram"] if nearest else 0.0,
        "nearest": nearest[:5],
    }


def _contains(haystack: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(haystack)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            parsed = datetime.combine(
                datetime.strptime(text, "%Y-%m-%d").date(),
                time.min,
                tzinfo=timezone.utc,
            )
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _derived_value(claim: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> str | None:
    derivation = claim.get("derivation")
    if not isinstance(derivation, Mapping):
        return None
    operation = str(derivation.get("operation") or "")
    if operation != "date_add_years":
        return None
    source_ref = str(derivation.get("evidence_ref") or "")
    source_value = str(derivation.get("source_value") or "")
    years = int(derivation.get("years") or 0)
    source = sources.get(source_ref) or {}
    if not source_value or not _contains(str(source.get("content") or ""), source_value):
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", source_value)
    if not match:
        return None
    return f"{int(match.group(1)) + years:04d}-{match.group(2)}-{match.group(3)}"


def _bundle_digest(bundle: Path) -> str:
    rows: list[dict[str, str]] = []
    for relative in REQUIRED_FILES:
        path = bundle / relative
        if path.exists() and path.is_file():
            rows.append({"path": relative, "sha256": _sha256_file(path)})
    return _sha256_bytes(_canonical_json(rows).encode("utf-8"))


def validate_bundle(
    bundle: Path,
    *,
    evaluation_prompts: Sequence[Mapping[str, str]] = (),
    contamination_warning_threshold: float = 0.20,
) -> ValidationReport:
    bundle = bundle.resolve()
    report = ValidationReport(bundle=str(bundle))
    for relative in REQUIRED_FILES:
        path = bundle / relative
        if not path.exists() or not path.is_file():
            report.add("missing_required_file", f"required file is missing: {relative}", path=relative)
    if not report.accepted:
        return report

    try:
        task = _read_json(bundle / "task.json")
        sources_list = _read_jsonl(bundle / "environment/sources.jsonl")
        contract = _read_json(bundle / "private/contract.json")
        trajectory = _read_json(bundle / "private/oracle_trajectory.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("invalid_bundle_json", str(exc))
        return report

    instruction = (bundle / "instruction.md").read_text(encoding="utf-8").strip()
    reference_answer = (bundle / "private/reference_answer.md").read_text(
        encoding="utf-8"
    ).strip()
    if not isinstance(task, Mapping):
        report.add("invalid_task_metadata", "task.json must contain an object", path="task.json")
        return report
    report.task_id = str(task.get("task_id") or "")
    if task.get("schema_version") != SCHEMA_VERSION:
        report.add(
            "schema_version_mismatch",
            f"expected {SCHEMA_VERSION}, got {task.get('schema_version')!r}",
            path="task.json",
        )
    if not report.task_id:
        report.add("task_id_missing", "task_id is required", path="task.json")
    snapshot_time = _parse_timestamp(task.get("snapshot_time"))
    if snapshot_time is None:
        report.add(
            "snapshot_time_invalid",
            "task snapshot_time must be an ISO date or timestamp",
            path="task.json",
        )
    if not instruction:
        report.add("instruction_empty", "public instruction is empty", path="instruction.md")
    for pattern in PRIVATE_LEAK_PATTERNS:
        if pattern.search(instruction):
            report.add(
                "private_contract_leak",
                f"public instruction exposes private implementation: {pattern.pattern}",
                path="instruction.md",
            )

    sources: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(sources_list):
        source_id = str(source.get("source_id") or "")
        if not source_id:
            report.add(
                "source_id_missing",
                f"source row {index} has no source_id",
                path="environment/sources.jsonl",
            )
            continue
        if source_id in sources:
            report.add(
                "duplicate_source_id",
                f"duplicate source id: {source_id}",
                path="environment/sources.jsonl",
            )
        sources[source_id] = source
        for field_name in (
            "url",
            "title",
            "authority",
            "published_at",
            "retrieved_at",
            "content",
        ):
            if not str(source.get(field_name) or "").strip():
                report.add(
                    "source_field_missing",
                    f"{source_id} is missing {field_name}",
                    path="environment/sources.jsonl",
                )
        published_at = _parse_timestamp(source.get("published_at"))
        retrieved_at = _parse_timestamp(source.get("retrieved_at"))
        if published_at is None:
            report.add(
                "source_published_at_invalid",
                f"{source_id} published_at is not an ISO date or timestamp",
                path="environment/sources.jsonl",
            )
        if retrieved_at is None:
            report.add(
                "source_retrieved_at_invalid",
                f"{source_id} retrieved_at is not an ISO date or timestamp",
                path="environment/sources.jsonl",
            )
        if published_at is not None and retrieved_at is not None and published_at > retrieved_at:
            report.add(
                "source_published_after_retrieval",
                f"{source_id} is published after it was retrieved",
                path="environment/sources.jsonl",
            )
        if retrieved_at is not None and snapshot_time is not None and retrieved_at > snapshot_time:
            report.add(
                "source_retrieved_after_snapshot",
                f"{source_id} was retrieved after the task snapshot",
                path="environment/sources.jsonl",
            )

    claims = contract.get("claims") if isinstance(contract, Mapping) else None
    if not isinstance(claims, list) or not claims:
        report.add(
            "contract_claims_missing",
            "private contract must contain at least one claim",
            path="private/contract.json",
        )
        claims = []

    fact_bindings: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    events = trajectory.get("events") if isinstance(trajectory, Mapping) else None
    if not isinstance(events, list):
        report.add(
            "oracle_events_missing",
            "oracle trajectory must contain an events array",
            path="private/oracle_trajectory.json",
        )
        events = []
    event_types = Counter(str(event.get("type") or "") for event in events if isinstance(event, Mapping))
    for required_type in ("task_plan", "tool_call", "oracle_evidence_assessment", "final"):
        if not event_types.get(required_type):
            report.add(
                "oracle_stage_missing",
                f"oracle trajectory has no {required_type} stage",
                path="private/oracle_trajectory.json",
            )
    final_events = [
        event
        for event in events
        if isinstance(event, Mapping) and str(event.get("type") or "") == "final"
    ]
    if not final_events or str(final_events[-1].get("content") or "").strip() != reference_answer:
        report.add(
            "oracle_final_mismatch",
            "last oracle final event must equal private/reference_answer.md",
            path="private/oracle_trajectory.json",
        )
    for event in events:
        if not isinstance(event, Mapping) or event.get("type") != "oracle_evidence_assessment":
            continue
        content = event.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        point_status = (
            content.get("task_record_status")
            or content.get("task_point_status")
            or {}
        )
        if not isinstance(point_status, Mapping):
            continue
        for point_id, row in point_status.items():
            if not isinstance(row, Mapping):
                continue
            for fact in row.get("supported_facts") or []:
                if isinstance(fact, Mapping):
                    fact_bindings[str(point_id)].append(fact)

    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            report.add(
                "claim_not_object",
                f"claim {index} must be an object",
                path="private/contract.json",
            )
            continue
        claim_id = str(claim.get("claim_id") or "")
        expected = str(claim.get("expected") or "")
        field_name = str(claim.get("field") or "")
        point_id = str(claim.get("point_id") or claim_id)
        if not claim_id or claim_id in claim_ids:
            report.add(
                "claim_id_invalid",
                f"claim id is missing or duplicated: {claim_id!r}",
                path="private/contract.json",
            )
        claim_ids.add(claim_id)
        if not expected or not field_name:
            report.add(
                "claim_value_missing",
                f"{claim_id or index} needs field and expected",
                path="private/contract.json",
            )
            continue
        evidence_refs = [str(value) for value in claim.get("evidence_refs") or []]
        if not evidence_refs:
            report.add(
                "claim_evidence_missing",
                f"{claim_id} has no evidence_refs",
                path="private/contract.json",
            )
        for source_ref in evidence_refs:
            if source_ref not in sources:
                report.add(
                    "claim_source_unknown",
                    f"{claim_id} references unknown source {source_ref}",
                    path="private/contract.json",
                )
        quotes = claim.get("quotes") or []
        if not isinstance(quotes, list) or not quotes:
            report.add(
                "claim_quote_missing",
                f"{claim_id} has no source quote",
                path="private/contract.json",
            )
        for quote in quotes if isinstance(quotes, list) else []:
            if not isinstance(quote, Mapping):
                report.add(
                    "claim_quote_invalid",
                    f"{claim_id} quote must be an object",
                    path="private/contract.json",
                )
                continue
            source_ref = str(quote.get("evidence_ref") or "")
            text = str(quote.get("text") or "")
            source_text = str((sources.get(source_ref) or {}).get("content") or "")
            if not text or not _contains(source_text, text):
                report.add(
                    "claim_quote_not_grounded",
                    f"{claim_id} quote is not present in {source_ref}",
                    path="private/contract.json",
                )
        derived = _derived_value(claim, sources)
        source_contains_value = any(
            _contains(str((sources.get(ref) or {}).get("content") or ""), expected)
            for ref in evidence_refs
        )
        if derived is not None and derived != expected:
            report.add(
                "derived_value_mismatch",
                f"{claim_id} derives {derived}, expected {expected}",
                path="private/contract.json",
            )
        if not source_contains_value and derived is None:
            report.add(
                "expected_value_not_discoverable",
                f"{claim_id} expected value is neither sourced nor deterministically derived",
                path="private/contract.json",
            )
        if bool(claim.get("required", True)) and not _contains(reference_answer, expected):
            report.add(
                "reference_answer_missing_claim",
                f"reference answer omits {claim_id}: {expected}",
                path="private/reference_answer.md",
            )
        if bool(claim.get("required", True)) and not any(
            f"[{source_ref}]" in reference_answer for source_ref in evidence_refs
        ):
            report.add(
                "reference_answer_missing_citation",
                f"reference answer does not cite evidence for {claim_id}",
                path="private/reference_answer.md",
            )
        bindings = fact_bindings.get(point_id) or []
        matching_binding = any(
            str(fact.get("field") or "") == field_name
            and str(fact.get("value") or "") == expected
            and str(fact.get("evidence_ref") or "") in evidence_refs
            and _contains(
                str(
                    (
                        sources.get(str(fact.get("evidence_ref") or "")) or {}
                    ).get("content")
                    or ""
                ),
                str(fact.get("quote") or ""),
            )
            for fact in bindings
        )
        if not matching_binding:
            report.add(
                "oracle_fact_binding_missing",
                f"oracle evidence-review does not bind {claim_id}",
                path="private/oracle_trajectory.json",
            )
        if not bool(claim.get("may_be_public", False)) and _contains(instruction, expected):
            report.add(
                "answer_leaked_in_instruction",
                f"public instruction contains private expected value for {claim_id}",
                path="instruction.md",
            )

    forbidden_values = [str(value) for value in contract.get("forbidden_values") or []]
    for value in forbidden_values:
        if value and _contains(reference_answer, value):
            report.add(
                "reference_answer_contains_forbidden_value",
                f"reference answer contains forbidden value: {value}",
                path="private/reference_answer.md",
            )

    behavior = contract.get("behavior") or {}
    if isinstance(behavior, Mapping):
        tool_calls = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("type") == "tool_call"
        ]
        max_tool_calls = behavior.get("max_tool_calls")
        if max_tool_calls is not None and len(tool_calls) > int(max_tool_calls):
            report.add(
                "oracle_tool_budget_exceeded",
                f"oracle uses {len(tool_calls)} tool calls, budget is {max_tool_calls}",
                path="private/oracle_trajectory.json",
            )
        allowed_urls = {str(value) for value in behavior.get("allowed_urls") or []}
        if bool(behavior.get("direct_urls_only")):
            for event in tool_calls:
                args = event.get("args") or {}
                url = str(args.get("url") or "") if isinstance(args, Mapping) else ""
                if not url or url not in allowed_urls:
                    report.add(
                        "oracle_violates_direct_url_scope",
                        f"direct URL task used unapproved tool call: {args}",
                        path="private/oracle_trajectory.json",
                    )

    contamination = contamination_scan(instruction, evaluation_prompts)
    if contamination["exact_matches"]:
        report.add(
            "benchmark_exact_window_overlap",
            "instruction shares a normalized 13-token window with held-out evaluation",
            path="instruction.md",
        )
    if float(contamination["max_5gram_jaccard"]) >= contamination_warning_threshold:
        report.add(
            "benchmark_similarity_high",
            (
                "instruction has high 5-gram similarity to held-out evaluation: "
                f"{contamination['max_5gram_jaccard']}"
            ),
            severity="warning",
            path="instruction.md",
        )

    report.metrics = {
        "source_count": len(sources),
        "claim_count": len(claims),
        "oracle_event_count": len(events),
        "oracle_tool_call_count": event_types.get("tool_call", 0),
        "oracle_evidence_assessment_count": event_types.get("oracle_evidence_assessment", 0),
        "instruction_tokens": len(normalized_tokens(instruction)),
        "answer_tokens": len(normalized_tokens(reference_answer)),
        "contamination": contamination,
    }
    report.provenance = {
        "bundle_sha256": _bundle_digest(bundle),
        "task_sha256": _sha256_file(bundle / "task.json"),
        "instruction_sha256": _sha256_file(bundle / "instruction.md"),
        "sources_sha256": _sha256_file(bundle / "environment/sources.jsonl"),
        "contract_sha256": _sha256_file(bundle / "private/contract.json"),
        "oracle_sha256": _sha256_file(bundle / "private/oracle_trajectory.json"),
        "reference_answer_sha256": _sha256_file(
            bundle / "private/reference_answer.md"
        ),
    }
    return report


def discover_bundles(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "task.json").is_file():
        return [root]
    return sorted(path.parent for path in root.rglob("task.json"))


def validate_pool(
    root: Path,
    *,
    evaluation_paths: Iterable[Path] = (),
    report_dir: Path | None = None,
) -> dict[str, Any]:
    prompts = load_evaluation_prompts(evaluation_paths)
    reports = [
        validate_bundle(bundle, evaluation_prompts=prompts)
        for bundle in discover_bundles(root)
    ]
    task_ids = [report.task_id for report in reports if report.task_id]
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        for report in reports:
            if report.task_id in duplicates:
                report.add("duplicate_pool_task_id", f"duplicate task id: {report.task_id}")
    manifest = {
        "schema_version": "rwkv-retrieval-rst-pool.v1",
        "root": str(root.resolve()),
        "evaluation_prompt_count": len(prompts),
        "task_count": len(reports),
        "accepted_count": sum(report.accepted for report in reports),
        "rejected_count": sum(not report.accepted for report in reports),
        "tasks": [report.to_dict() for report in reports],
    }
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cases_dir = report_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        for report in reports:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", report.task_id or "unknown")
            (cases_dir / f"{safe_id}.json").write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return manifest


def select_diverse_tasks(
    manifest: Mapping[str, Any],
    *,
    limit: int,
    max_per_parent: int = 4,
    max_per_domain: int = 160,
    max_per_family: int = 320,
    max_per_operator: int = 80,
) -> list[dict[str, Any]]:
    accepted = [
        row
        for row in manifest.get("tasks") or []
        if isinstance(row, Mapping) and row.get("accepted") is True
    ]
    candidates: list[dict[str, Any]] = []
    for row in accepted:
        bundle = Path(str(row.get("bundle") or ""))
        try:
            task = _read_json(bundle / "task.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append({"report": dict(row), "task": task})
    candidates.sort(
        key=lambda item: (
            int(item["task"].get("repair_count") or 0),
            int(item["task"].get("round") or 0),
            str(item["task"].get("task_id") or ""),
        )
    )
    counters: dict[str, Counter[str]] = {
        "parent": Counter(),
        "domain": Counter(),
        "family": Counter(),
        "operator": Counter(),
    }
    caps = {
        "parent": max_per_parent,
        "domain": max_per_domain,
        "family": max_per_family,
        "operator": max_per_operator,
    }
    selected: list[dict[str, Any]] = []
    for item in candidates:
        task = item["task"]
        keys = {
            "parent": str(task.get("parent_id") or task.get("task_id") or "root"),
            "domain": str(task.get("domain") or "unknown"),
            "family": str(task.get("rewrite_family") or "unknown"),
            "operator": str(task.get("rewrite_operator") or "unknown"),
        }
        if any(counters[name][key] >= caps[name] for name, key in keys.items()):
            continue
        selected.append(
            {
                "task_id": str(task.get("task_id") or ""),
                "bundle": str(item["report"].get("bundle") or ""),
                "round": int(task.get("round") or 0),
                "parent_id": str(task.get("parent_id") or ""),
                "domain": keys["domain"],
                "rewrite_family": keys["family"],
                "rewrite_operator": keys["operator"],
                "bundle_sha256": (item["report"].get("provenance") or {}).get(
                    "bundle_sha256", ""
                ),
            }
        )
        for name, key in keys.items():
            counters[name][key] += 1
        if len(selected) >= limit:
            break
    return selected


def _trajectory_messages(
    instruction: str, events: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an RWKV retrieval agent. Use tools, bind every answer-ready fact "
                "to source text, distinguish sourced facts from deterministic derivations, "
                "and never claim that a source is official or that a procedure was tested "
                "unless the evidence says so."
            ),
            "loss_mask": False,
        },
        {"role": "user", "content": instruction, "loss_mask": False},
    ]
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "name": str(event.get("tool") or "retrieval"),
                    "content": _canonical_json(event.get("content") or {}),
                    "loss_mask": False,
                }
            )
            continue
        if event_type in {"task_plan", "tool_call", "oracle_evidence_assessment", "final"}:
            content = event.get("content")
            if event_type == "tool_call":
                content = {
                    "tool": event.get("tool"),
                    "args": event.get("args") or {},
                }
            rendered = content if isinstance(content, str) else _canonical_json(content or {})
            messages.append(
                {
                    "role": "assistant",
                    "stage": event_type,
                    "content": rendered,
                    "loss_mask": bool(event.get("trainable", True)),
                }
            )
    return messages


def export_training_records(
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    rl_rows: list[dict[str, Any]] = []
    for row in manifest.get("tasks") or []:
        if not isinstance(row, Mapping) or row.get("accepted") is not True:
            continue
        bundle = Path(str(row.get("bundle") or ""))
        task = _read_json(bundle / "task.json")
        contract = _read_json(bundle / "private/contract.json")
        trajectory = _read_json(bundle / "private/oracle_trajectory.json")
        instruction = (bundle / "instruction.md").read_text(encoding="utf-8").strip()
        events = [event for event in trajectory.get("events") or [] if isinstance(event, Mapping)]
        task_id = str(task.get("task_id") or "")
        common = {
            "task_id": task_id,
            "task_group_id": str(task.get("task_group_id") or task_id),
            "round": int(task.get("round") or 0),
            "domain": str(task.get("domain") or "unknown"),
            "rewrite_family": str(task.get("rewrite_family") or "unknown"),
            "rewrite_operator": str(task.get("rewrite_operator") or "unknown"),
            "bundle_sha256": (row.get("provenance") or {}).get("bundle_sha256", ""),
        }
        trajectory_rows.append(
            {
                **common,
                "lane": "trajectory_sft",
                "messages": _trajectory_messages(instruction, events),
                "verifier_reward": 1.0,
            }
        )
        context: list[dict[str, Any]] = [
            {"role": "user", "content": instruction}
        ]
        for event in events:
            event_type = str(event.get("type") or "")
            if event_type == "tool_result":
                context.append(
                    {
                        "role": "tool",
                        "name": str(event.get("tool") or "retrieval"),
                        "content": event.get("content") or {},
                    }
                )
                continue
            if event_type in {"task_plan", "tool_call", "oracle_evidence_assessment", "final"}:
                target = event.get("content")
                if event_type == "tool_call":
                    target = {"tool": event.get("tool"), "args": event.get("args") or {}}
                if bool(event.get("trainable", True)):
                    stage_rows.append(
                        {
                            **common,
                            "lane": "stage_sft",
                            "stage": event_type,
                            "context": list(context),
                            "target": target,
                        }
                    )
                context.append(
                    {
                        "role": "assistant",
                        "stage": event_type,
                        "content": target,
                    }
                )
        for rejected in trajectory.get("rejected_attempts") or []:
            if not isinstance(rejected, Mapping):
                continue
            preference_rows.append(
                {
                    **common,
                    "lane": "preference",
                    "stage": str(rejected.get("stage") or ""),
                    "prompt": rejected.get("prompt") or {},
                    "chosen": rejected.get("chosen") or {},
                    "rejected": rejected.get("rejected") or {},
                    "failure_class": str(rejected.get("failure_class") or ""),
                }
            )
        rl_rows.append(
            {
                **common,
                "lane": "verifier_rl",
                "public_instruction": instruction,
                "environment_sources": str(
                    (bundle / "environment/sources.jsonl").resolve()
                ),
                "private_verifier_contract": str(
                    (bundle / "private/contract.json").resolve()
                ),
                "reward_dimensions": list(contract.get("reward_dimensions") or []),
            }
        )

    outputs = {
        "trajectory_sft.jsonl": trajectory_rows,
        "stage_sft.jsonl": stage_rows,
        "preference.jsonl": preference_rows,
        "verifier_rl.jsonl": rl_rows,
    }
    for name, rows in outputs.items():
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "rwkv-retrieval-rst-export.v1",
        "trajectory_sft": len(trajectory_rows),
        "stage_sft": len(stage_rows),
        "preference": len(preference_rows),
        "verifier_rl": len(rl_rows),
        "files": {
            name: {
                "rows": len(rows),
                "sha256": _sha256_file(output_dir / name),
            }
            for name, rows in outputs.items()
        },
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
