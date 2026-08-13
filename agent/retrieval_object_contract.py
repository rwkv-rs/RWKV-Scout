"""Observable object identity shared by every retrieval stage.

The contract preserves identities already written by RWKV, a provider, or a
fetched source.  It never infers which candidate is correct, current, or
complete and it never authors an answer.  Its purpose is to prevent an object
from silently changing between planning, tool execution, evidence extraction,
the ledger, and final context construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from urllib.parse import unquote, urlparse

from agent.task_plan_contract import compact_task_plan
from utils.web_retrieval import normalize_url


OBJECT_CONTRACT_VERSION = "retrieval-object.v2"

_STRICT_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.@+-]+$")


def _append_target(
    output: list[dict[str, str]],
    object_id: str,
    object_type: str,
    literal: str,
    basis: str,
) -> None:
    object_id = str(object_id or "").strip().casefold()
    if not object_id or any(row["object_id"] == object_id for row in output):
        return
    output.append(
        {
            "object_id": object_id,
            "object_type": str(object_type or "resource"),
            "literal": _text(literal, 300),
            "basis": str(basis or "explicit_literal"),
        }
    )


def _text(value: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _github_full_name(value: Any) -> str:
    """Read an explicit GitHub owner/repository without consuming suffix text."""

    text = str(value or "").strip()
    if not text:
        return ""
    url_match = re.search(
        r"https?://(?:www\.)?github\.com/([^/\s?#]+/[^/\s?#]+)",
        text,
        flags=re.IGNORECASE,
    )
    if url_match:
        return unquote(url_match.group(1)).removesuffix(".git").strip("/")
    api_match = re.search(
        r"https?://api\.github\.com/repos/([^/\s?#]+/[^/\s?#]+)",
        text,
        flags=re.IGNORECASE,
    )
    if api_match:
        return unquote(api_match.group(1)).removesuffix(".git").strip("/")
    if urlparse(text).scheme in {"http", "https"}:
        return ""
    bare_match = re.search(
        r"(?<![\w.-])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?![\w./-])",
        text,
    )
    if not bare_match:
        return ""
    return bare_match.group(1).removesuffix(".git").strip("/")


def github_repository_target(value: Any) -> str:
    """Return an explicit owner/repository token found anywhere in a request."""

    return _github_full_name(value)


def explicit_object_targets(value: Any) -> list[dict[str, str]]:
    """Extract only typed resource identifiers literally present in text.

    These identifiers are transport scope, not factual answers.  In
    particular, no project name, package name or domain is guessed from
    general prose.
    """

    text = str(value or "")
    output: list[dict[str, str]] = []
    repository = _github_full_name(text)
    if repository:
        _append_target(
            output,
            f"github:{repository}",
            "github_repository",
            repository,
            "explicit_owner_repository",
        )

    registry_patterns = (
        (
            "crates",
            "rust_crate",
            r"https?://(?:www\.)?crates\.io/crates/([^/\s?#]+)",
        ),
        (
            "pypi",
            "python_package",
            r"https?://(?:www\.)?pypi\.org/project/([^/\s?#]+)",
        ),
        (
            "npm",
            "npm_package",
            r"https?://(?:www\.)?npmjs\.com/package/([^\s?#]+)",
        ),
    )
    for namespace, object_type, pattern in registry_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            literal = unquote(match.group(1)).strip("/.,;:!?，。；：！？")
            if literal:
                _append_target(
                    output,
                    f"{namespace}:{literal}",
                    object_type,
                    literal,
                    "explicit_registry_url",
                )

    for match in re.finditer(
        r"\b(?:arXiv\s*:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?\b",
        text,
        flags=re.IGNORECASE,
    ):
        _append_target(
            output,
            f"arxiv:{match.group(1)}",
            "scholarly_work",
            match.group(0),
            "explicit_arxiv_id",
        )
    for match in re.finditer(r"\bCVE-\d{4}-\d{4,}\b", text, flags=re.IGNORECASE):
        _append_target(
            output,
            f"cve:{match.group(0)}",
            "security_advisory",
            match.group(0),
            "explicit_cve_id",
        )
    return output


def _registry_subject_target(
    text: str,
    subject: str,
) -> list[dict[str, str]]:
    """Bind a strict RWKV subject to a registry named in the same task."""

    literal = str(subject or "").strip()
    if not literal or not _STRICT_IDENTIFIER_RE.fullmatch(literal):
        return []
    folded = str(text or "").casefold()
    output: list[dict[str, str]] = []
    for marker, namespace, object_type in (
        ("crates.io", "crates", "rust_crate"),
        ("pypi", "pypi", "python_package"),
        ("npm", "npm", "npm_package"),
    ):
        if marker not in folded:
            continue
        _append_target(
            output,
            f"{namespace}:{literal}",
            object_type,
            literal,
            "explicit_registry_plus_rwkv_subject",
        )
    return output


def object_alignment(
    requested_targets: Any,
    source_object: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare typed transport IDs without deciding semantic correctness."""

    requested = [
        dict(row)
        for row in (requested_targets or [])
        if isinstance(row, Mapping) and str(row.get("object_id") or "").strip()
    ]
    source_id = str((source_object or {}).get("source_object_id") or "").casefold()
    requested_ids = [str(row["object_id"]).casefold() for row in requested]
    if not requested_ids:
        relation = "not_explicitly_scoped"
    elif not source_id:
        relation = "source_identity_unavailable"
    elif source_id in requested_ids:
        relation = "exact"
    else:
        source_namespace = source_id.split(":", 1)[0]
        same_namespace = [
            value
            for value in requested_ids
            if value.split(":", 1)[0] == source_namespace
        ]
        relation = "conflict" if same_namespace else "unresolved"
    return {
        "relation": relation,
        "requested_object_ids": requested_ids,
        "source_object_id": source_id,
        "transport_only": True,
    }


def rwkv_subject_alignment(requested_subject: Any, observed_subject: Any) -> dict[str, Any]:
    """Expose a strict identifier disagreement authored by RWKV itself."""

    requested = _text(requested_subject, 300)
    observed = _text(observed_subject, 300)
    if not requested or not observed:
        relation = "unresolved"
    elif not (
        _STRICT_IDENTIFIER_RE.fullmatch(requested)
        and _STRICT_IDENTIFIER_RE.fullmatch(observed)
    ):
        relation = "unresolved"
    else:
        normalize = lambda item: re.sub(r"[-_.+]+", "-", item.casefold()).strip("-")
        relation = "exact" if normalize(requested) == normalize(observed) else "conflict"
    return {
        "relation": relation,
        "requested_subject": requested,
        "observed_source_subject": observed,
        "labels_authored_by": "rwkv",
    }


def merge_mapping_rows(*values: Any, limit: int = 16) -> list[dict[str, Any]]:
    """Merge observable mapping rows without selecting a preferred meaning.

    Retrieval may encounter the same source through several model-authored
    requests.  Keeping only the first singular ``object_alignment`` or
    ``retrieval_request`` silently discards that history.  This helper performs
    transport-level deduplication only; consumers such as RWKV may inspect the
    resulting observations and decide what they mean.
    """

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, Mapping) or not row:
                continue
            marker = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if marker in seen:
                continue
            seen.add(marker)
            output.append(deepcopy(dict(row)))
            if len(output) >= max(1, int(limit or 1)):
                return output
    return output


def merge_candidate_observations(
    *values: Any,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """Merge repeated grounded spans while preserving every route binding.

    The identity is the observable source span (chunk ID plus exact quote), not
    whichever task point happened to retrieve it first.  Scalar labels remain
    model-authored; only set-like provenance fields are unioned.
    """

    output: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        rows = value if isinstance(value, list) else []
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            quote = str(raw.get("quote") or "").strip()
            if not quote:
                continue
            identity = (str(raw.get("chunk_id") or ""), quote)
            incoming = deepcopy(dict(raw))
            current = by_identity.get(identity)
            if current is None:
                object_alignments = merge_mapping_rows(
                    incoming.get("object_alignments"),
                    incoming.get("object_alignment"),
                )
                if object_alignments:
                    incoming["object_alignments"] = object_alignments
                rwkv_subject_alignments = merge_mapping_rows(
                    incoming.get("rwkv_subject_alignments"),
                    incoming.get("rwkv_subject_alignment"),
                )
                if rwkv_subject_alignments:
                    incoming["rwkv_subject_alignments"] = rwkv_subject_alignments
                by_identity[identity] = incoming
                output.append(incoming)
                if len(output) >= max(1, int(limit or 1)):
                    return output
                continue

            for key in (
                "task_record_ids",
                "claim_ids",
                "task_point_ids",
                "field_keys",
            ):
                existing_values = list(current.get(key) or [])
                incoming_values = list(incoming.get(key) or [])
                merged_values = list(
                    dict.fromkeys(
                        str(item)
                        for item in [*existing_values, *incoming_values]
                        if str(item).strip()
                    )
                )
                if merged_values:
                    current[key] = merged_values[:16]
            object_alignments = merge_mapping_rows(
                current.get("object_alignments"),
                current.get("object_alignment"),
                incoming.get("object_alignments"),
                incoming.get("object_alignment"),
            )
            if object_alignments:
                current["object_alignments"] = object_alignments
            rwkv_subject_alignments = merge_mapping_rows(
                current.get("rwkv_subject_alignments"),
                current.get("rwkv_subject_alignment"),
                incoming.get("rwkv_subject_alignments"),
                incoming.get("rwkv_subject_alignment"),
            )
            if rwkv_subject_alignments:
                current["rwkv_subject_alignments"] = rwkv_subject_alignments
            for key, item in incoming.items():
                if current.get(key) in (None, "", [], {}) and item not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    current[key] = deepcopy(item)
            if incoming.get("supported") is True:
                current["supported"] = True
            if incoming.get("source_grounded") is True:
                current["source_grounded"] = True
    return output


def task_record_contract(
    task_plan: Mapping[str, Any] | None,
    task_record_id: str,
) -> dict[str, Any]:
    """Project one RWKV-authored task record without filling missing semantics."""

    point_id = _text(task_record_id, 120)
    plan = compact_task_plan(task_plan)
    point = next(
        (
            row
            for row in plan.get("atomic_points") or []
            if isinstance(row, Mapping) and _text(row.get("id"), 120) == point_id
        ),
        None,
    )
    if not isinstance(point, Mapping):
        return {"task_record_id": point_id} if point_id else {}
    requested_subject = _text(point.get("subject"), 400)
    point_descriptor = "\n".join(
        value
        for value in (
            _text(point.get("question"), 1200),
            requested_subject,
        )
        if value
    )
    requested_targets = explicit_object_targets(point_descriptor)

    # A multi-object user goal must not be copied wholesale into every atomic
    # task record.  That previously made both ``owner/repo-a`` and
    # ``owner/repo-b`` look exact for each point.  A single explicit goal-level
    # identifier is a safe transport fallback when the RWKV point omitted the
    # literal; multiple identifiers remain unresolved until RWKV names one in
    # the point itself.
    if not requested_targets:
        goal_targets = explicit_object_targets(_text(plan.get("goal"), 1200))
        if len(goal_targets) == 1:
            requested_targets = goal_targets

    registry_descriptor = "\n".join(
        value
        for value in (
            point_descriptor,
            _text(plan.get("goal"), 1200),
        )
        if value
    )
    for target in _registry_subject_target(registry_descriptor, requested_subject):
        _append_target(
            requested_targets,
            target["object_id"],
            target["object_type"],
            target["literal"],
            target["basis"],
        )
    return {
        "task_record_id": point_id,
        "question": _text(point.get("question"), 1200),
        "requested_subject": requested_subject,
        "requested_relation": _text(point.get("relation"), 240),
        "requested_fields": [
            _text(value, 160)
            for value in point.get("fields") or []
            if _text(value, 160)
        ][:16],
        "time_scope": _text(point.get("time_scope"), 40) or "unspecified",
        "set_semantics": _text(point.get("set_semantics"), 40) or "single",
        "requested_object_targets": requested_targets,
    }


def retrieval_request_contract(
    action: str,
    arguments: Mapping[str, Any] | None,
    *,
    task_record_id: str = "",
    task_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable audit identity for one model-authored tool request."""

    args = deepcopy(dict(arguments or {}))
    task_record = task_record_contract(task_plan, task_record_id)
    if not task_record_id:
        # ``task_point_id`` is optional model-authored trace metadata and G1i
        # often omits it.  Object transport must not disappear as a result.
        # A singleton task plan supplies an unambiguous object scope while
        # explicitly retaining ``evidence_binding=False``; this never assigns
        # a page span to that task record.  For a multi-point plan, only one
        # literal goal-level identifier is safe to carry without choosing a
        # point on RWKV's behalf.
        compact = compact_task_plan(task_plan)
        points = [
            row
            for row in compact.get("atomic_points") or []
            if isinstance(row, Mapping)
        ]
        if len(points) == 1:
            point_id = _text(points[0].get("id"), 120)
            singleton = task_record_contract(task_plan, point_id)
            task_record = {
                "task_record_id": "",
                "requested_object_targets": list(
                    singleton.get("requested_object_targets") or []
                ),
                "object_scope_basis": "singleton_task_plan",
                "object_scope_task_record_id": point_id,
                "evidence_binding": False,
            }
        else:
            goal_targets = explicit_object_targets(_text(compact.get("goal"), 1200))
            if len(goal_targets) == 1:
                task_record = {
                    "task_record_id": "",
                    "requested_object_targets": goal_targets,
                    "object_scope_basis": "single_explicit_goal_object",
                    "evidence_binding": False,
                }
    # The same semantic request must retain one identity even when JSON key
    # insertion order differs between a model call, an audit replay and a
    # recovered session.
    digest_payload = {
        "tool": str(action or ""),
        "arguments": args,
        "task_record_id": str(task_record_id or ""),
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": OBJECT_CONTRACT_VERSION,
        "request_id": f"R-{digest}",
        "tool": _text(action, 120),
        "arguments": args,
        "task_record_id": _text(task_record_id, 120),
        "requested_object": task_record,
    }


def source_object_contract(
    item: Mapping[str, Any],
    *,
    connector: str = "",
) -> dict[str, Any]:
    """Describe the source's explicit identity without matching it to a task."""

    row = item if isinstance(item, Mapping) else {}
    source_type = _text(
        row.get("source_object_type")
        or row.get("content_type")
        or row.get("evidence_kind")
        or connector
        or "web_page",
        120,
    ).casefold()
    full_name = _text(row.get("full_name"), 300) or _github_full_name(
        row.get("api_url") or row.get("url")
    )
    doi = _text(row.get("doi"), 300).casefold()
    arxiv_id = _text(row.get("arxiv_id") or row.get("arxiv"), 200).casefold()
    location = _text(row.get("location") or row.get("resolved_location"), 300)
    url = normalize_url(str(row.get("url") or row.get("api_url") or ""))
    parsed_url = urlparse(url) if url else None
    host = (
        parsed_url.netloc.casefold().removeprefix("www.")
        if parsed_url is not None
        else ""
    )
    path_parts = (
        [unquote(part) for part in parsed_url.path.split("/") if part]
        if parsed_url is not None
        else []
    )

    if not arxiv_id and url:
        parsed = urlparse(url)
        if parsed.netloc.casefold().removeprefix("www.") in {
            "arxiv.org",
            "export.arxiv.org",
        }:
            arxiv_match = re.search(
                r"/(?:abs|pdf|html)/([^/?#]+)",
                parsed.path,
                flags=re.IGNORECASE,
            )
            if arxiv_match:
                arxiv_id = unquote(arxiv_match.group(1)).removesuffix(".pdf").casefold()

    registry_object_id = ""
    registry_object_type = ""
    if host == "crates.io" and len(path_parts) >= 2 and path_parts[0] == "crates":
        registry_object_id = f"crates:{path_parts[1].casefold()}"
        registry_object_type = "rust_crate"
    elif host == "pypi.org" and len(path_parts) >= 2 and path_parts[0] == "project":
        registry_object_id = f"pypi:{path_parts[1].casefold()}"
        registry_object_type = "python_package"
    elif host == "npmjs.com" and len(path_parts) >= 2 and path_parts[0] == "package":
        package_name = "/".join(path_parts[1:3] if path_parts[1].startswith("@") else path_parts[1:2])
        registry_object_id = f"npm:{package_name.casefold()}"
        registry_object_type = "npm_package"

    if full_name:
        object_id = f"github:{full_name.casefold()}"
        source_type = "github_repository"
    elif registry_object_id:
        object_id = registry_object_id
        source_type = registry_object_type
    elif doi:
        object_id = f"doi:{doi.removeprefix('https://doi.org/')}"
        source_type = "scholarly_work"
    elif arxiv_id:
        object_id = f"arxiv:{arxiv_id.removeprefix('arxiv:')}"
        source_type = "scholarly_work"
    elif location and str(connector or "").casefold().startswith("weather"):
        object_id = f"weather:{location.casefold()}"
        source_type = "weather_location"
    elif url:
        object_id = f"url:{url.casefold()}"
    else:
        title = _text(row.get("title"), 500)
        object_id = (
            "source:" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:20]
            if title
            else ""
        )

    # Only explicit release/version identifiers may become a source-record
    # identity. Publication timestamps remain observable metadata but are not
    # unique record keys: using them here previously merged unrelated rows
    # that happened to share a date.
    explicit_record_values = [
        row.get("source_record_id"),
        row.get("tag_name"),
        row.get("version"),
        row.get("release"),
    ]
    source_record_id = next(
        (_text(value, 300) for value in explicit_record_values if _text(value, 300)),
        "",
    )
    return {
        "schema_version": OBJECT_CONTRACT_VERSION,
        "source_object_id": object_id,
        "source_object_type": source_type or "web_page",
        "source_record_id": source_record_id,
        "source_url": url,
        "provider": _text(row.get("provider") or row.get("source") or connector, 160),
    }


def attach_result_object_contract(
    result: Mapping[str, Any],
    *,
    action: str,
    arguments: Mapping[str, Any] | None,
    task_record_id: str = "",
    task_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach request/source identities to a tool result as transport metadata."""

    output = deepcopy(dict(result))
    request = retrieval_request_contract(
        action,
        arguments,
        task_record_id=task_record_id,
        task_plan=task_plan,
    )
    connector = _text(output.get("connector"), 120)
    rows = []
    for raw in output.get("results") or []:
        if not isinstance(raw, Mapping):
            continue
        row = deepcopy(dict(raw))
        row["retrieval_request"] = deepcopy(request)
        existing = row.get("source_object")
        row["source_object"] = (
            deepcopy(dict(existing))
            if isinstance(existing, Mapping)
            else source_object_contract(row, connector=connector)
        )
        request_alignment = object_alignment(
            request.get("requested_object", {}).get("requested_object_targets"),
            row["source_object"],
        )
        existing_alignment = row.get("object_alignment")
        if (
            request_alignment.get("relation") == "not_explicitly_scoped"
            and isinstance(existing_alignment, Mapping)
            and existing_alignment
        ):
            # A provider pipeline such as web_search may already have compared
            # the candidate against the literal user/task object set.  An
            # omitted optional task-point ID must not erase that observation.
            row["object_alignment"] = deepcopy(dict(existing_alignment))
        else:
            row["object_alignment"] = request_alignment
        rows.append(row)
    output["results"] = rows
    output["retrieval_request"] = request
    output["object_contract_version"] = OBJECT_CONTRACT_VERSION
    return output


__all__ = [
    "OBJECT_CONTRACT_VERSION",
    "attach_result_object_contract",
    "explicit_object_targets",
    "github_repository_target",
    "merge_candidate_observations",
    "merge_mapping_rows",
    "object_alignment",
    "retrieval_request_contract",
    "rwkv_subject_alignment",
    "source_object_contract",
    "task_record_contract",
]
