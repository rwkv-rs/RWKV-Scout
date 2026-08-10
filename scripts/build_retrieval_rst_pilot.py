"""Build a contamination-resistant pilot pool for retrieval-agent training.

All names, products, versions, dates, and URLs in this pilot are fictional.
The bundles teach transferable behavior without copying evaluation answers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


OPERATOR_CARDS: list[dict[str, Any]] = [
    {
        "family": "task_and_time_decomposition",
        "operator": "add_temporal_role_separation",
        "teaches": ["historical_vs_current", "field_complete_planning"],
        "extension": "Add distinct reveal, launch, anniversary, and current-version fields.",
        "shortcut_rejection": ["do not substitute launch for reveal", "do not label a derived date as quoted"],
    },
    {
        "family": "task_and_time_decomposition",
        "operator": "add_deterministic_date_derivation",
        "teaches": ["sourced_fact_vs_derivation", "date_calculator_use"],
        "extension": "Require one date computed from a sourced date and label the derivation.",
        "shortcut_rejection": ["reject unsourced arithmetic", "reject contradictory date labels"],
    },
    {
        "family": "retrieval_and_source_choice",
        "operator": "add_stale_conflicting_source",
        "teaches": ["freshness", "authority", "conflict_resolution"],
        "extension": "Add an older plausible page that conflicts with a newer authoritative source.",
        "shortcut_rejection": ["reject newest-looking snippet without timestamp comparison"],
    },
    {
        "family": "retrieval_and_source_choice",
        "operator": "add_missing_official_source",
        "teaches": ["source_role_calibration", "practical_stop_condition"],
        "extension": "Remove the presumed official guide while retaining a well-supported community procedure.",
        "shortcut_rejection": ["do not call community guidance official", "do not claim hands-on verification"],
    },
    {
        "family": "retrieval_and_source_choice",
        "operator": "add_direct_url_scope",
        "teaches": ["direct_page", "bounded_tool_use"],
        "extension": "Give one explicit page and require an answer from that page only.",
        "shortcut_rejection": ["reject open-web expansion", "reject uncited external details"],
    },
    {
        "family": "evidence_and_context",
        "operator": "add_navigation_noise",
        "teaches": ["page_cleaning", "exact_span_selection"],
        "extension": "Surround a short answer-bearing span with menus, login text, and unrelated sections.",
        "shortcut_rejection": ["reject navigation text as evidence"],
    },
    {
        "family": "evidence_and_context",
        "operator": "add_context_budget_pressure",
        "teaches": ["claim_scoped_context", "answer_ready_fact_binding"],
        "extension": "Add many relevant-looking chunks while keeping one exact answer span per field.",
        "shortcut_rejection": ["reject omission of required fields", "reject stale duplicate spans"],
    },
    {
        "family": "protocol_and_recovery",
        "operator": "add_cross_validation_protocol_repair",
        "teaches": ["strict_json", "same_temperature_repair", "supported_facts"],
        "extension": "Include one masked malformed attempt and one valid corrected cross-validation object.",
        "shortcut_rejection": ["never train loss on malformed attempt", "reject empty supported_facts"],
    },
    {
        "family": "protocol_and_recovery",
        "operator": "add_strategy_changing_replan",
        "teaches": ["replan", "avoid_duplicate_query", "missing_field_focus"],
        "extension": "Make the first retrieval insufficient and require a changed source strategy.",
        "shortcut_rejection": ["reject repeated equivalent query", "reject controller-authored answer"],
    },
    {
        "family": "answer_grounding",
        "operator": "add_source_role_trap",
        "teaches": ["official_vs_community", "no_fake_testing_claim"],
        "extension": "Add a useful community page whose wording tempts the writer to call it official or tested.",
        "shortcut_rejection": ["reject unsupported source labels", "reject fabricated environment versions"],
    },
    {
        "family": "answer_grounding",
        "operator": "add_internal_consistency_check",
        "teaches": ["single_value_per_field", "final_answer_consistency"],
        "extension": "Require the same date/version value in heading, prose, and summary.",
        "shortcut_rejection": ["reject mutually inconsistent values"],
    },
    {
        "family": "answer_grounding",
        "operator": "add_partial_evidence_answer",
        "teaches": ["answer_supported_part", "calibrated_uncertainty"],
        "extension": "Support some requested fields and leave another genuinely unavailable.",
        "shortcut_rejection": ["reject fabricated missing field", "reject empty blanket refusal"],
    },
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _fact(
    *,
    field: str,
    value: str,
    evidence_ref: str,
    quote: str,
) -> dict[str, str]:
    return {
        "field": field,
        "value": value,
        "evidence_ref": evidence_ref,
        "quote": quote,
    }


def _claim(
    claim_id: str,
    point_id: str,
    field: str,
    expected: str,
    evidence_ref: str,
    quote: str,
    *,
    derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "claim_id": claim_id,
        "point_id": point_id,
        "field": field,
        "expected": expected,
        "evidence_refs": [evidence_ref],
        "quotes": [{"evidence_ref": evidence_ref, "text": quote}],
        "required": True,
        "may_be_public": False,
    }
    if derivation:
        row["derivation"] = dict(derivation)
    return row


def _source(
    source_id: str,
    *,
    url: str,
    title: str,
    authority: str,
    published_at: str,
    content: str,
    retrieved_at: str = "2035-09-20T12:00:00Z",
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "url": url,
        "title": title,
        "authority": authority,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "content": content,
    }


def _plan(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "goal": "Answer every requested field from frozen evidence without inventing facts.",
        "atomic_points": [dict(point) for point in points],
        "completion_rule": "Every supported field has an exact value, source ref, and grounded quote.",
    }


def _cross_validation(
    point_facts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    missing_fields: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    missing_fields = missing_fields or {}
    return {
        "decision": "finish",
        "missing_points": [],
        "conflicts": [],
        "task_point_status": {
            point_id: {
                "status": "supported",
                "evidence_refs": sorted(
                    {str(fact["evidence_ref"]) for fact in facts}
                ),
                "supported_facts": [dict(fact) for fact in facts],
                "missing_fields": list(missing_fields.get(point_id) or []),
            }
            for point_id, facts in point_facts.items()
        },
        "next_focus": "",
        "reason": "All required fields have answer-ready source bindings.",
    }


def _bundle(
    root: Path,
    *,
    task: Mapping[str, Any],
    instruction: str,
    sources: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reference_answer: str,
    rejected_attempts: Sequence[Mapping[str, Any]] = (),
) -> Path:
    bundle = root / str(task["task_id"])
    bundle.mkdir(parents=True, exist_ok=False)
    _write_json(bundle / "task.json", dict(task))
    (bundle / "instruction.md").write_text(instruction.strip() + "\n", encoding="utf-8")
    _write_jsonl(bundle / "environment/sources.jsonl", sources)
    _write_json(bundle / "private/contract.json", dict(contract))
    _write_json(
        bundle / "private/oracle_trajectory.json",
        {
            "schema_version": "rwkv-retrieval-oracle-trajectory.v1",
            "events": [dict(event) for event in events],
            "rejected_attempts": [dict(row) for row in rejected_attempts],
        },
    )
    (bundle / "private/reference_answer.md").write_text(
        reference_answer.strip() + "\n", encoding="utf-8"
    )
    return bundle


def _task(
    task_id: str,
    *,
    group: str,
    round_number: int,
    parent_id: str | None,
    domain: str,
    family: str,
    operator: str,
    capability_tags: Sequence[str],
    snapshot_time: str = "2035-09-20T12:00:00Z",
) -> dict[str, Any]:
    return {
        "schema_version": "rwkv-retrieval-rst-task.v1",
        "task_id": task_id,
        "task_group_id": group,
        "round": round_number,
        "parent_id": parent_id,
        "original_parent_id": parent_id or task_id,
        "domain": domain,
        "rewrite_family": family,
        "rewrite_operator": operator,
        "repair_count": 0,
        "snapshot_time": snapshot_time,
        "allowed_tools": ["web_search", "direct_page", "date_diff"],
        "capability_tags": list(capability_tags),
        "training_lanes": ["stage_sft", "trajectory_sft", "preference", "verifier_rl"],
        "public_files": ["instruction.md", "environment/sources.jsonl"],
        "private_files": [
            "private/contract.json",
            "private/oracle_trajectory.json",
            "private/reference_answer.md",
        ],
    }


def build_current_version_seed(root: Path) -> Path:
    task = _task(
        "rrst_seed_lumen_current_001",
        group="rrst_group_lumen_001",
        round_number=0,
        parent_id=None,
        domain="fictional_product_updates",
        family="bootstrap",
        operator="bootstrap_current_version",
        capability_tags=["freshness", "version_theme", "fact_binding"],
    )
    sources = [
        _source(
            "S1",
            url="https://updates.project-lumen.invalid/v1-8",
            title="Project Lumen 1.8: Glass Harbor",
            authority="official",
            published_at="2035-03-01",
            content="Official archive. Version 1.8, Glass Harbor, launched on 2035-03-01.",
        ),
        _source(
            "S2",
            url="https://updates.project-lumen.invalid/v2-4",
            title="Project Lumen 2.4: Orbit After Rain",
            authority="official",
            published_at="2035-09-18",
            content=(
                "Official live update. Project Lumen version 2.4 launched on 2035-09-18. "
                "The update title and theme are Orbit After Rain."
            ),
        ),
    ]
    facts = [
        _fact(field="current_version", value="2.4", evidence_ref="S2", quote="version 2.4 launched on 2035-09-18"),
        _fact(field="current_theme", value="Orbit After Rain", evidence_ref="S2", quote="The update title and theme are Orbit After Rain"),
        _fact(field="current_release_date", value="2035-09-18", evidence_ref="S2", quote="version 2.4 launched on 2035-09-18"),
    ]
    instruction = (
        "As of the snapshot date, identify Project Lumen's current live version, its release "
        "date, and its theme. Use the frozen update sources, distinguish current from archived "
        "information, and cite the supporting source."
    )
    answer = (
        "Project Lumen's current live release is **version 2.4**, launched on **2035-09-18**, "
        "with the theme **Orbit After Rain** [S2]. The 1.8 page is an older archive, not the "
        "current release [S1]."
    )
    points = [{"id": "P1", "objective": "Bind current version, release date, and theme."}]
    events = [
        {"type": "task_plan", "content": _plan(points)},
        {"type": "tool_call", "tool": "web_search", "args": {"query": "Project Lumen current official update"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources}},
        {"type": "cross_validation", "content": _cross_validation({"P1": facts})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "current_version", "2.4", "S2", "version 2.4 launched on 2035-09-18"),
            _claim("C2", "P1", "current_release_date", "2035-09-18", "S2", "version 2.4 launched on 2035-09-18"),
            _claim("C3", "P1", "current_theme", "Orbit After Rain", "S2", "The update title and theme are Orbit After Rain"),
        ],
        "forbidden_values": ["1.8 is the current release"],
        "behavior": {"max_tool_calls": 2},
        "reward_dimensions": ["field_coverage", "freshness", "fact_binding", "citation_grounding"],
    }
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer)


def build_temporal_child(root: Path) -> Path:
    task = _task(
        "rrst_r1_lumen_temporal_001",
        group="rrst_group_lumen_001",
        round_number=1,
        parent_id="rrst_seed_lumen_current_001",
        domain="fictional_product_updates",
        family="task_and_time_decomposition",
        operator="add_temporal_role_separation",
        capability_tags=["historical_vs_current", "derived_date", "conflict_resolution", "internal_consistency"],
    )
    sources = [
        _source(
            "S1",
            url="https://news.project-lumen.invalid/first-reveal",
            title="Project Lumen first reveal",
            authority="official",
            published_at="2032-02-03",
            content="Project Lumen was first publicly revealed on 2032-02-03. This is an announcement, not the live launch.",
        ),
        _source(
            "S2",
            url="https://news.project-lumen.invalid/global-launch",
            title="Project Lumen global launch",
            authority="official",
            published_at="2033-06-12",
            content="Project Lumen entered public live service on 2033-06-12 across all supported platforms.",
        ),
        _source(
            "S3",
            url="https://events.project-lumen.invalid/first-anniversary",
            title="First Anniversary Week",
            authority="official",
            published_at="2034-05-30",
            content="The first anniversary celebration begins on 2034-06-12 and runs for seven days.",
        ),
        _source(
            "S4",
            url="https://community.lumen-watch.invalid/latest-version",
            title="Lumen Watch version list",
            authority="community",
            published_at="2035-03-02",
            content="Community archive last checked in March 2035: version 1.8, Glass Harbor, appears current.",
        ),
        _source(
            "S5",
            url="https://updates.project-lumen.invalid/v2-4",
            title="Project Lumen 2.4: Orbit After Rain",
            authority="official",
            published_at="2035-09-18",
            content="Official live update. Project Lumen version 2.4 launched on 2035-09-18. The current theme is Orbit After Rain.",
        ),
    ]
    reveal = _fact(field="first_reveal_date", value="2032-02-03", evidence_ref="S1", quote="first publicly revealed on 2032-02-03")
    anniversary = _fact(field="first_anniversary_date", value="2034-06-12", evidence_ref="S3", quote="first anniversary celebration begins on 2034-06-12")
    current_version = _fact(field="current_version", value="2.4", evidence_ref="S5", quote="version 2.4 launched on 2035-09-18")
    current_theme = _fact(field="current_theme", value="Orbit After Rain", evidence_ref="S5", quote="The current theme is Orbit After Rain")
    instruction = (
        "For Project Lumen, report the first public reveal date, the first anniversary event "
        "date, and the current live version theme at the snapshot time. Keep historical events "
        "separate from current-version information, resolve stale conflicts, and cite each field."
    )
    answer = (
        "- First public reveal: **2032-02-03** [S1].\n"
        "- First anniversary event: **2034-06-12** [S3]. This is the event date, not the original launch date.\n"
        "- Current live release: **version 2.4**, themed **Orbit After Rain** [S5].\n\n"
        "The community page is stale and still lists version 1.8, so it does not determine the current state [S4]."
    )
    points = [
        {"id": "P1", "objective": "Find first public reveal date."},
        {"id": "P2", "objective": "Find first anniversary event date."},
        {"id": "P3", "objective": "Find current live version and theme."},
    ]
    events = [
        {"type": "task_plan", "content": _plan(points)},
        {"type": "tool_call", "tool": "web_search", "args": {"query": "Project Lumen official reveal launch anniversary"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources[:4]}},
        {"type": "tool_call", "tool": "web_search", "args": {"query": "site:updates.project-lumen.invalid current live update"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": [sources[4]]}},
        {"type": "cross_validation", "content": _cross_validation({"P1": [reveal], "P2": [anniversary], "P3": [current_version, current_theme]})},
        {"type": "final", "content": answer},
    ]
    chosen_cv = _cross_validation({"P1": [reveal], "P2": [anniversary], "P3": [current_version, current_theme]})
    rejected_attempts = [
        {
            "stage": "cross_validation",
            "failure_class": "supported_without_answer_ready_facts",
            "prompt": {"task_points": ["P1", "P2", "P3"], "evidence_refs": ["S1", "S3", "S5"]},
            "rejected": {"decision": "finish", "task_point_status": {"P1": {"status": "supported", "evidence_refs": ["S1"]}}},
            "chosen": chosen_cv,
        },
        {
            "stage": "final",
            "failure_class": "internal_date_conflict",
            "prompt": {"supported_facts": [reveal, anniversary, current_version, current_theme]},
            "rejected": {"content": "The first anniversary was 2033-06-12, therefore it took place on 2034-06-12."},
            "chosen": {"content": answer},
        },
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "first_reveal_date", "2032-02-03", "S1", "first publicly revealed on 2032-02-03"),
            _claim("C2", "P2", "first_anniversary_date", "2034-06-12", "S3", "first anniversary celebration begins on 2034-06-12"),
            _claim("C3", "P3", "current_version", "2.4", "S5", "version 2.4 launched on 2035-09-18"),
            _claim("C4", "P3", "current_theme", "Orbit After Rain", "S5", "The current theme is Orbit After Rain"),
        ],
        "forbidden_values": ["version 1.8 is current", "first anniversary was 2033-06-12"],
        "behavior": {"max_tool_calls": 3},
        "reward_dimensions": ["temporal_roles", "freshness", "conflict_resolution", "internal_consistency", "fact_binding"],
    }
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected_attempts)


def build_derived_date_seed(root: Path) -> Path:
    task = _task(
        "rrst_seed_derived_anniversary_001",
        group="rrst_group_derived_date_001",
        round_number=0,
        parent_id=None,
        domain="fictional_event_calendar",
        family="task_and_time_decomposition",
        operator="add_deterministic_date_derivation",
        capability_tags=["date_add_years", "sourced_vs_derived", "date_diff_tool"],
    )
    sources = [
        _source(
            "S1",
            url="https://docs.orbit-notes.invalid/launch-record",
            title="Orbit Notes launch record",
            authority="official",
            published_at="2030-11-07",
            content="Orbit Notes public service launched on 2030-11-07. No anniversary campaign is documented on this page.",
        )
    ]
    derived = _fact(field="first_anniversary_calendar_date", value="2031-11-07", evidence_ref="S1", quote="public service launched on 2030-11-07")
    instruction = (
        "Using the frozen launch record and the date tool, calculate the first calendar "
        "anniversary of Orbit Notes. Clearly label the result as a deterministic derivation, "
        "not as an announced anniversary event."
    )
    answer = (
        "Orbit Notes launched on **2030-11-07** [S1]. Adding one calendar year gives "
        "**2031-11-07** as its first calendar anniversary. This is a deterministic date "
        "derivation, not evidence that an anniversary event was announced."
    )
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Source launch date and derive one-year anniversary."}])},
        {"type": "tool_call", "tool": "direct_page", "args": {"url": sources[0]["url"]}},
        {"type": "tool_result", "tool": "direct_page", "content": sources[0]},
        {"type": "tool_call", "tool": "date_diff", "args": {"operation": "add_years", "date": "2030-11-07", "years": 1}},
        {"type": "tool_result", "tool": "date_diff", "content": {"result": "2031-11-07"}},
        {"type": "cross_validation", "content": _cross_validation({"P1": [derived]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim(
                "C1",
                "P1",
                "first_anniversary_calendar_date",
                "2031-11-07",
                "S1",
                "public service launched on 2030-11-07",
                derivation={"operation": "date_add_years", "evidence_ref": "S1", "source_value": "2030-11-07", "years": 1},
            )
        ],
        "forbidden_values": ["an official anniversary event begins on 2031-11-07"],
        "behavior": {"max_tool_calls": 2, "allowed_urls": [sources[0]["url"]]},
        "reward_dimensions": ["date_derivation", "epistemic_label", "citation_grounding"],
    }
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer)


def build_howto_authority_seed(root: Path) -> Path:
    task = _task(
        "rrst_seed_portable_bundle_menu_001",
        group="rrst_group_howto_authority_001",
        round_number=0,
        parent_id=None,
        domain="fictional_desktop_howto",
        family="answer_grounding",
        operator="add_source_role_trap",
        capability_tags=["howto", "official_vs_community", "no_fake_testing_claim"],
    )
    sources = [
        _source(
            "S1",
            url="https://docs.nova-desktop.invalid/menu-entries",
            title="Nova Desktop menu entry specification",
            authority="official",
            published_at="2035-01-11",
            content=(
                "Nova Desktop discovers per-user launchers in ~/.local/share/applications. "
                "A launcher requires a [Desktop Entry] section with Type, Name, and Exec. "
                "The specification does not mention PortableBundle packages."
            ),
        ),
        _source(
            "S2",
            url="https://community.nova-users.invalid/portablebundle-menu",
            title="Community guide: add a PortableBundle to the menu",
            authority="community",
            published_at="2035-02-20",
            content=(
                "Community procedure: make the .PortableBundle executable, keep it at a stable "
                "path, and create ~/.local/share/applications/example.desktop. Set Type=Application, "
                "Name to the display name, and Exec to the absolute bundle path. Log out and back in "
                "if the launcher does not refresh. The author did not publish a tested desktop version."
            ),
        ),
    ]
    path_fact = _fact(field="launcher_directory", value="~/.local/share/applications", evidence_ref="S1", quote="per-user launchers in ~/.local/share/applications")
    fields_fact = _fact(field="required_desktop_fields", value="Type, Name, and Exec", evidence_ref="S1", quote="requires a [Desktop Entry] section with Type, Name, and Exec")
    procedure_fact = _fact(
        field="portable_bundle_procedure",
        value=(
            "make the .PortableBundle executable, keep it at a stable path, and create "
            "~/.local/share/applications/example.desktop"
        ),
        evidence_ref="S2",
        quote=(
            "make the .PortableBundle executable, keep it at a stable path, and create "
            "~/.local/share/applications/example.desktop"
        ),
    )
    instruction = (
        "How should a Nova Desktop user add a PortableBundle application to the start menu? "
        "Use the frozen sources, give the minimal launcher procedure, and accurately distinguish "
        "the official menu-entry specification from community PortableBundle guidance."
    )
    answer = (
        "Community procedure: make the `.PortableBundle` executable, keep it at a stable path, "
        "and create `~/.local/share/applications/example.desktop` [S2]. The launcher needs a "
        "`[Desktop Entry]` section with at least `Type`, `Name`, and `Exec`; point `Exec` to the "
        "absolute PortableBundle path [S1].\n\nThe directory and required fields come from the "
        "official Nova Desktop menu-entry specification. The PortableBundle-specific steps are "
        "community guidance, and the sources do not establish that the procedure was tested on a "
        "particular Nova Desktop version [S2]."
    )
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Find official launcher requirements."}, {"id": "P2", "objective": "Find PortableBundle-specific community procedure and label its role."}])},
        {"type": "tool_call", "tool": "web_search", "args": {"query": "Nova Desktop PortableBundle menu launcher specification"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources}},
        {"type": "cross_validation", "content": _cross_validation({"P1": [path_fact, fields_fact], "P2": [procedure_fact]})},
        {"type": "final", "content": answer},
    ]
    rejected_attempts = [
        {
            "stage": "final",
            "failure_class": "fabricated_source_role_and_testing",
            "prompt": {"sources": ["S1 official", "S2 community"]},
            "rejected": {"content": "Official Nova documentation confirms this was tested on Nova 7.2."},
            "chosen": {"content": answer},
        }
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "launcher_directory", "~/.local/share/applications", "S1", "per-user launchers in ~/.local/share/applications"),
            _claim("C2", "P1", "required_desktop_fields", "Type, Name, and Exec", "S1", "requires a [Desktop Entry] section with Type, Name, and Exec"),
            _claim(
                "C3",
                "P2",
                "portable_bundle_procedure",
                (
                    "make the .PortableBundle executable, keep it at a stable path, and create "
                    "~/.local/share/applications/example.desktop"
                ),
                "S2",
                (
                    "make the .PortableBundle executable, keep it at a stable path, and create "
                    "~/.local/share/applications/example.desktop"
                ),
            ),
        ],
        "forbidden_values": ["tested on Nova 7.2", "official PortableBundle guide"],
        "behavior": {"max_tool_calls": 2},
        "reward_dimensions": ["procedure_correctness", "source_role_accuracy", "no_fabricated_testing", "citation_grounding"],
    }
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected_attempts)


def build_direct_page_seed(root: Path) -> Path:
    task = _task(
        "rrst_seed_direct_page_policy_001",
        group="rrst_group_direct_page_001",
        round_number=0,
        parent_id=None,
        domain="fictional_technical_documentation",
        family="retrieval_and_source_choice",
        operator="add_direct_url_scope",
        capability_tags=["direct_page", "bounded_tool_use", "navigation_noise"],
    )
    url = "https://docs.mesh-garden.invalid/policies/default-isolation"
    sources = [
        _source(
            "S1",
            url=url,
            title="Mesh Garden default isolation policy",
            authority="official",
            published_at="2035-04-14",
            content=(
                "Skip to content | Sign in | Products | Images | Search. "
                "Default isolation example: selector: {} applies the policy to every workload in "
                "the namespace. directions: [Inbound, Outbound] activates isolation in both "
                "directions. An empty allow list admits no traffic. Footer | Privacy | Navigation."
            ),
        )
    ]
    selector = _fact(field="selector_semantics", value="every workload in the namespace", evidence_ref="S1", quote="selector: {} applies the policy to every workload in the namespace")
    directions = _fact(field="directions_semantics", value="isolation in both directions", evidence_ref="S1", quote="directions: [Inbound, Outbound] activates isolation in both directions")
    instruction = (
        f"Read only {url} and explain what `selector: {{}}` and `directions: [Inbound, Outbound]` "
        "mean in its default-isolation example. Do not expand to open-web search."
    )
    answer = (
        "On the specified page, `selector: {}` applies the policy to **every workload in the "
        "namespace**, while `directions: [Inbound, Outbound]` activates **isolation in both "
        "directions** [S1]. With the empty allow list described there, no traffic is admitted [S1]."
    )
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Read the explicit page and bind the two requested fields."}])},
        {"type": "tool_call", "tool": "direct_page", "args": {"url": url}},
        {"type": "tool_result", "tool": "direct_page", "content": sources[0]},
        {"type": "cross_validation", "content": _cross_validation({"P1": [selector, directions]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "selector_semantics", "every workload in the namespace", "S1", "selector: {} applies the policy to every workload in the namespace"),
            _claim("C2", "P1", "directions_semantics", "isolation in both directions", "S1", "directions: [Inbound, Outbound] activates isolation in both directions"),
        ],
        "forbidden_values": ["external source"],
        "behavior": {"max_tool_calls": 1, "direct_urls_only": True, "allowed_urls": [url]},
        "reward_dimensions": ["direct_url_scope", "noise_rejection", "field_grounding"],
    }
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer)


def build_pilot(output_root: Path, *, replace: bool) -> dict[str, Any]:
    if output_root.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output_root}")
        shutil.rmtree(output_root)
    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "operator_cards.json", {"schema_version": "rwkv-retrieval-rst-operators.v1", "operators": OPERATOR_CARDS})
    bundles = [
        build_current_version_seed(tasks_root),
        build_temporal_child(tasks_root),
        build_derived_date_seed(tasks_root),
        build_howto_authority_seed(tasks_root),
        build_direct_page_seed(tasks_root),
    ]
    manifest = {
        "schema_version": "rwkv-retrieval-rst-build.v1",
        "task_count": len(bundles),
        "tasks": [str(path.relative_to(output_root)) for path in bundles],
        "fictional_data_only": True,
        "benchmark_answers_used": False,
        "paper_method": "Recursive Synthesis for Long-Horizon Terminal Tasks, arXiv:2608.05466v1",
    }
    _write_json(output_root / "build_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "training/retrieval_rst/pilot_v1",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    manifest = build_pilot(args.output_root.resolve(), replace=args.replace)
    print(json.dumps({"output_root": str(args.output_root.resolve()), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
