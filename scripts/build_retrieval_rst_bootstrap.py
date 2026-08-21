"""Build a diversified fictional bootstrap pool for retrieval-agent synthesis.

This pool is deliberately independent from evaluation answers.  It exercises
the data factory before any model-generated recursive rewrites are admitted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from scripts.build_retrieval_rst_pilot import (
    OPERATOR_CARDS,
    ROOT,
    _bundle,
    _claim,
    _oracle_evidence_assessment,
    _fact,
    _plan,
    _source,
    _task,
    _write_json,
)


CURRENT_RELEASE_CASES = (
    ("Aster Vale", "3.2", "Cobalt Tides", "2040-02-14", "2.7", "Amber Relay", "2039-10-01"),
    ("Copper Finch", "5.1", "Paper Constellations", "2041-06-03", "4.8", "Quiet Furnace", "2041-01-19"),
    ("Juniper Circuit", "2.6", "Lanterns Below", "2042-09-27", "2.1", "Marble Signal", "2042-04-08"),
    ("Morrow Atlas", "8.0", "The Long Meridian", "2043-12-11", "7.5", "Northbound Echo", "2043-07-22"),
)

TEMPORAL_CASES = (
    ("Blue Orchard", "2034-01-17", "2035-05-09", "2036-05-09", "4.3", "Clouds Remember", "2040-08-21"),
    ("Signal Harbor", "2035-03-28", "2036-10-02", "2037-10-02", "6.1", "A Map of Embers", "2041-02-13"),
    ("Velvet Engine", "2036-07-04", "2037-11-16", "2038-11-16", "3.9", "After the Brass Rain", "2042-05-30"),
    ("Willow Transit", "2037-09-19", "2038-12-07", "2039-12-07", "9.2", "Last Train to Dawn", "2043-03-18"),
)

DIRECT_PAGE_CASES = (
    (
        "Quartz Mesh",
        "https://docs.quartz-mesh.invalid/policy/empty-selector",
        "selector: {}",
        "every service in the workspace",
        "modes: [Receive, Send]",
        "filtering in both directions",
    ),
    (
        "Cedar Queue",
        "https://docs.cedar-queue.invalid/consumers/wildcard",
        "consumer: *",
        "all registered consumers in the group",
        "ack: explicit",
        "messages remain pending until acknowledged",
    ),
    (
        "Opal Registry",
        "https://manual.opal-registry.invalid/scopes/default",
        "scope: global",
        "every project in the organization",
        "inherit: false",
        "child projects do not inherit the rule",
    ),
    (
        "Pine Relay",
        "https://reference.pine-relay.invalid/routes/fallback",
        "match: any",
        "requests not claimed by an earlier route",
        "retry: 0",
        "failed deliveries are not retried",
    ),
)

HOWTO_CASES = (
    ("Nova Desktop", "PortableBundle", "Nova", "portablebundle"),
    ("Aster Shell", "RunCapsule", "Aster", "runcapsule"),
    ("Birch Workspace", "LaunchPack", "Birch", "launchpack"),
    ("Cobalt Session", "SingleFileApp", "Cobalt", "singlefileapp"),
)

DERIVED_DATE_CASES = (
    ("Orbit Notes", "2030-11-07", 1),
    ("Marble Calendar", "2031-04-18", 2),
    ("Cinder Mail", "2032-08-26", 1),
    ("Lighthouse Board", "2033-12-03", 3),
)

BOOTSTRAP_SNAPSHOT = "2045-01-15T12:00:00Z"


def _task_id(prefix: str, index: int) -> str:
    return f"rrst_bootstrap_{prefix}_{index:03d}"


def build_current_release(root: Path, index: int, values: tuple[str, ...]) -> Path:
    name, current, theme, released, stale, stale_theme, stale_date = values
    task_id = _task_id("current", index)
    task = _task(
        task_id,
        group=task_id,
        round_number=0,
        parent_id=None,
        domain="fictional_product_updates",
        family="bootstrap",
        operator="bootstrap_current_release",
        capability_tags=["freshness", "version_theme", "fact_binding"],
        snapshot_time=BOOTSTRAP_SNAPSHOT,
    )
    sources = [
        _source(
            "S1",
            url=f"https://archive.{name.lower().replace(' ', '-')}.invalid/{stale}",
            title=f"{name} {stale}: {stale_theme}",
            authority="official",
            published_at=stale_date,
            content=f"Official archive: {name} version {stale}, {stale_theme}, launched on {stale_date}.",
            retrieved_at=BOOTSTRAP_SNAPSHOT,
        ),
        _source(
            "S2",
            url=f"https://updates.{name.lower().replace(' ', '-')}.invalid/{current}",
            title=f"{name} {current}: {theme}",
            authority="official",
            published_at=released,
            content=(
                f"Official live update: {name} version {current} launched on {released}. "
                f"The release title and theme are {theme}."
            ),
            retrieved_at=BOOTSTRAP_SNAPSHOT,
        ),
    ]
    version_fact = _fact(
        field="current_version",
        value=current,
        evidence_ref="S2",
        quote=f"version {current} launched on {released}",
    )
    date_fact = _fact(
        field="current_release_date",
        value=released,
        evidence_ref="S2",
        quote=f"version {current} launched on {released}",
    )
    theme_fact = _fact(
        field="current_theme",
        value=theme,
        evidence_ref="S2",
        quote=f"release title and theme are {theme}",
    )
    instruction = (
        f"At the frozen snapshot, identify {name}'s current live version, release date, and "
        "theme. Resolve any archived-version conflict and cite the answer-bearing source."
    )
    answer = (
        f"{name}'s current live release is **version {current}**, launched on **{released}**, "
        f"with the theme **{theme}** [S2]. Version {stale} is an older archived release [S1]."
    )
    points = [{"id": "P1", "objective": "Bind the current release fields to the newest official source."}]
    events = [
        {"type": "task_plan", "content": _plan(points)},
        {"type": "tool_call", "tool": "web_search", "args": {"query": f"{name} current official release"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources}},
        {"type": "oracle_evidence_assessment", "content": _oracle_evidence_assessment({"P1": [version_fact, date_fact, theme_fact]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "current_version", current, "S2", f"version {current} launched on {released}"),
            _claim("C2", "P1", "current_release_date", released, "S2", f"version {current} launched on {released}"),
            _claim("C3", "P1", "current_theme", theme, "S2", f"release title and theme are {theme}"),
        ],
        "forbidden_values": [f"version {stale} is current"],
        "behavior": {"max_tool_calls": 2},
        "reward_dimensions": ["field_coverage", "freshness", "fact_binding", "citation_grounding"],
    }
    rejected = [{
        "stage": "final",
        "failure_class": "stale_source_selected_as_current",
        "prompt": {"source_order": ["S1", "S2"]},
        "rejected": {"content": f"{name} is currently on version {stale}, {stale_theme}."},
        "chosen": {"content": answer},
    }]
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected)


def build_temporal(root: Path, index: int, values: tuple[str, ...]) -> Path:
    name, reveal, launch, anniversary, current, theme, current_date = values
    task_id = _task_id("temporal", index)
    slug = name.lower().replace(" ", "-")
    task = _task(
        task_id,
        group=task_id,
        round_number=0,
        parent_id=None,
        domain="fictional_live_service_history",
        family="bootstrap",
        operator="bootstrap_temporal_roles",
        capability_tags=["reveal_vs_launch", "anniversary_event", "current_version", "internal_consistency"],
        snapshot_time=BOOTSTRAP_SNAPSHOT,
    )
    sources = [
        _source("S1", url=f"https://news.{slug}.invalid/reveal", title=f"{name} reveal", authority="official", published_at=reveal, content=f"{name} was first publicly revealed on {reveal}; this was not the live launch.", retrieved_at=BOOTSTRAP_SNAPSHOT),
        _source("S2", url=f"https://news.{slug}.invalid/launch", title=f"{name} launch", authority="official", published_at=launch, content=f"{name} entered public live service on {launch}.", retrieved_at=BOOTSTRAP_SNAPSHOT),
        _source("S3", url=f"https://events.{slug}.invalid/anniversary-one", title=f"{name} first anniversary", authority="official", published_at=anniversary, content=f"The first anniversary event for {name} begins on {anniversary}.", retrieved_at=BOOTSTRAP_SNAPSHOT),
        _source("S4", url=f"https://updates.{slug}.invalid/current", title=f"{name} current release", authority="official", published_at=current_date, content=f"The current live release is {name} version {current}, launched on {current_date}, with theme {theme}.", retrieved_at=BOOTSTRAP_SNAPSHOT),
    ]
    reveal_fact = _fact(field="first_reveal_date", value=reveal, evidence_ref="S1", quote=f"first publicly revealed on {reveal}")
    launch_fact = _fact(field="launch_date", value=launch, evidence_ref="S2", quote=f"entered public live service on {launch}")
    anniversary_fact = _fact(field="first_anniversary_event_date", value=anniversary, evidence_ref="S3", quote=f"first anniversary event for {name} begins on {anniversary}")
    current_fact = _fact(field="current_version", value=current, evidence_ref="S4", quote=f"current live release is {name} version {current}")
    theme_fact = _fact(field="current_theme", value=theme, evidence_ref="S4", quote=f"with theme {theme}")
    instruction = (
        f"For {name}, report the first public reveal date, launch date, first anniversary "
        "event date, and current live version theme. Keep historical roles separate from current state."
    )
    answer = (
        f"- First public reveal: **{reveal}** [S1].\n"
        f"- Public live launch: **{launch}** [S2].\n"
        f"- First anniversary event: **{anniversary}** [S3].\n"
        f"- Current live release: **version {current}**, themed **{theme}** [S4]."
    )
    points = [
        {"id": "P1", "objective": "Separate first reveal from launch."},
        {"id": "P2", "objective": "Find the dated first-anniversary event."},
        {"id": "P3", "objective": "Bind current version and theme."},
    ]
    events = [
        {"type": "task_plan", "content": _plan(points)},
        {"type": "tool_call", "tool": "web_search", "args": {"query": f"{name} official reveal launch first anniversary"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources[:3]}},
        {"type": "tool_call", "tool": "web_search", "args": {"query": f"site:updates.{slug}.invalid current live release"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources[3:]}},
        {"type": "oracle_evidence_assessment", "content": _oracle_evidence_assessment({"P1": [reveal_fact, launch_fact], "P2": [anniversary_fact], "P3": [current_fact, theme_fact]})},
        {"type": "final", "content": answer},
    ]
    claims = [
        _claim("C1", "P1", "first_reveal_date", reveal, "S1", f"first publicly revealed on {reveal}"),
        _claim("C2", "P1", "launch_date", launch, "S2", f"entered public live service on {launch}"),
        _claim("C3", "P2", "first_anniversary_event_date", anniversary, "S3", f"first anniversary event for {name} begins on {anniversary}"),
        _claim("C4", "P3", "current_version", current, "S4", f"current live release is {name} version {current}"),
        _claim("C5", "P3", "current_theme", theme, "S4", f"with theme {theme}"),
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": claims,
        "forbidden_values": [f"first anniversary event was {launch}", f"first reveal was {launch}"],
        "behavior": {"max_tool_calls": 3},
        "reward_dimensions": ["temporal_roles", "field_coverage", "internal_consistency", "fact_binding"],
    }
    rejected = [{
        "stage": "final",
        "failure_class": "temporal_role_conflation",
        "prompt": {"required_fields": [claim["field"] for claim in claims]},
        "rejected": {"content": f"{name} was revealed, launched, and first celebrated on {launch}."},
        "chosen": {"content": answer},
    }]
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected)


def build_direct_page(root: Path, index: int, values: tuple[str, ...]) -> Path:
    name, url, key_one, value_one, key_two, value_two = values
    task_id = _task_id("direct", index)
    task = _task(
        task_id,
        group=task_id,
        round_number=0,
        parent_id=None,
        domain="fictional_technical_documentation",
        family="bootstrap",
        operator="bootstrap_direct_page",
        capability_tags=["direct_page", "navigation_noise", "bounded_tool_use"],
        snapshot_time=BOOTSTRAP_SNAPSHOT,
    )
    source = _source(
        "S1",
        url=url,
        title=f"{name} reference",
        authority="official",
        published_at=f"204{index}-01-1{index}",
        content=(
            f"Skip to content | Log in | Images | Products. {key_one} means {value_one}. "
            f"{key_two} means {value_two}. Related tutorials | Account | Footer navigation."
        ),
        retrieved_at=BOOTSTRAP_SNAPSHOT,
    )
    fact_one = _fact(field="first_setting_meaning", value=value_one, evidence_ref="S1", quote=f"{key_one} means {value_one}")
    fact_two = _fact(field="second_setting_meaning", value=value_two, evidence_ref="S1", quote=f"{key_two} means {value_two}")
    instruction = f"Read only {url} and explain what `{key_one}` and `{key_two}` mean. Do not expand to open-web search."
    answer = f"On the specified page, `{key_one}` means **{value_one}**, while `{key_two}` means **{value_two}** [S1]."
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Read the explicit page and bind both requested settings."}])},
        {"type": "tool_call", "tool": "direct_page", "args": {"url": url}},
        {"type": "tool_result", "tool": "direct_page", "content": source},
        {"type": "oracle_evidence_assessment", "content": _oracle_evidence_assessment({"P1": [fact_one, fact_two]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "first_setting_meaning", value_one, "S1", f"{key_one} means {value_one}"),
            _claim("C2", "P1", "second_setting_meaning", value_two, "S1", f"{key_two} means {value_two}"),
        ],
        "forbidden_values": ["according to an external source"],
        "behavior": {"max_tool_calls": 1, "direct_urls_only": True, "allowed_urls": [url]},
        "reward_dimensions": ["direct_url_scope", "noise_rejection", "field_grounding"],
    }
    rejected = [{
        "stage": "tool_call",
        "failure_class": "direct_url_scope_expansion",
        "prompt": {"explicit_url": url},
        "rejected": {"tool": "web_search", "args": {"query": f"{name} setting explanation"}},
        "chosen": {"tool": "direct_page", "args": {"url": url}},
    }]
    return _bundle(root, task=task, instruction=instruction, sources=[source], contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected)


def build_howto(root: Path, index: int, values: tuple[str, ...]) -> Path:
    desktop, bundle, vendor, bundle_slug = values
    task_id = _task_id("howto", index)
    slug = desktop.lower().replace(" ", "-")
    task = _task(
        task_id,
        group=task_id,
        round_number=0,
        parent_id=None,
        domain="fictional_desktop_howto",
        family="bootstrap",
        operator="bootstrap_source_role_howto",
        capability_tags=["official_vs_community", "procedure", "no_fake_testing_claim"],
        snapshot_time=BOOTSTRAP_SNAPSHOT,
    )
    launcher_path = "~/.local/share/applications"
    desktop_file = f"{bundle_slug}-example.desktop"
    sources = [
        _source(
            "S1",
            url=f"https://docs.{slug}.invalid/menu-entry-spec",
            title=f"{desktop} menu entry specification",
            authority="official",
            published_at=f"204{index}-02-10",
            content=(
                f"{desktop} discovers per-user launchers in {launcher_path}. A launcher requires "
                "a [Desktop Entry] section containing Type, Name, and Exec."
            ),
            retrieved_at=BOOTSTRAP_SNAPSHOT,
        ),
        _source(
            "S2",
            url=f"https://community.{slug}.invalid/{bundle_slug}-menu",
            title=f"Community guide: add {bundle} to the menu",
            authority="community",
            published_at=f"204{index}-03-12",
            content=(
                f"Community procedure: make the .{bundle} file executable, keep it at a stable "
                f"absolute path, and create {launcher_path}/{desktop_file}. Set Exec to that "
                f"absolute file path. The author did not publish a tested {vendor} version."
            ),
            retrieved_at=BOOTSTRAP_SNAPSHOT,
        ),
    ]
    path_fact = _fact(field="launcher_directory", value=launcher_path, evidence_ref="S1", quote=f"per-user launchers in {launcher_path}")
    fields_fact = _fact(field="required_fields", value="Type, Name, and Exec", evidence_ref="S1", quote="containing Type, Name, and Exec")
    procedure_value = f"make the .{bundle} file executable, keep it at a stable absolute path, and create {launcher_path}/{desktop_file}"
    procedure_fact = _fact(field="bundle_procedure", value=procedure_value, evidence_ref="S2", quote=procedure_value)
    instruction = (
        f"How should a {desktop} user add a {bundle} application to the start menu? Give the "
        "minimal procedure and distinguish official launcher requirements from community guidance."
    )
    answer = (
        f"Community procedure: **make the `.{bundle}` file executable, keep it at a stable absolute "
        f"path, and create `{launcher_path}/{desktop_file}`** [S2]. The launcher belongs under "
        f"`{launcher_path}` and needs `Type`, `Name`, and `Exec` [S1]. The launcher requirements "
        f"are official; the {bundle}-specific procedure is community guidance and is not documented "
        f"as tested on a particular {vendor} version [S2]."
    )
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Find official launcher requirements."}, {"id": "P2", "objective": f"Find and correctly label {bundle}-specific guidance."}])},
        {"type": "tool_call", "tool": "web_search", "args": {"query": f"{desktop} {bundle} start menu launcher"}},
        {"type": "tool_result", "tool": "web_search", "content": {"results": sources}},
        {"type": "oracle_evidence_assessment", "content": _oracle_evidence_assessment({"P1": [path_fact, fields_fact], "P2": [procedure_fact]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim("C1", "P1", "launcher_directory", launcher_path, "S1", f"per-user launchers in {launcher_path}"),
            _claim("C2", "P1", "required_fields", "Type, Name, and Exec", "S1", "containing Type, Name, and Exec"),
            _claim("C3", "P2", "bundle_procedure", procedure_value, "S2", procedure_value),
        ],
        "forbidden_values": [f"official {bundle} guide", f"tested on {vendor} 7.2"],
        "behavior": {"max_tool_calls": 2},
        "reward_dimensions": ["procedure_correctness", "source_role_accuracy", "no_fabricated_testing"],
    }
    rejected = [{
        "stage": "final",
        "failure_class": "fabricated_source_role_and_testing",
        "prompt": {"source_roles": {"S1": "official", "S2": "community"}},
        "rejected": {"content": f"The official {bundle} guide confirms this was tested on {vendor} 7.2."},
        "chosen": {"content": answer},
    }]
    return _bundle(root, task=task, instruction=instruction, sources=sources, contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected)


def build_derived_date(root: Path, index: int, values: tuple[Any, ...]) -> Path:
    name, launch, years = values
    task_id = _task_id("derived", index)
    source_date = date.fromisoformat(launch)
    result = source_date.replace(year=source_date.year + int(years)).isoformat()
    ordinal = {1: "first", 2: "second", 3: "third"}[int(years)]
    slug = name.lower().replace(" ", "-")
    task = _task(
        task_id,
        group=task_id,
        round_number=0,
        parent_id=None,
        domain="fictional_event_calendar",
        family="bootstrap",
        operator="bootstrap_deterministic_date_derivation",
        capability_tags=["date_add_years", "sourced_vs_derived", "epistemic_label"],
        snapshot_time=BOOTSTRAP_SNAPSHOT,
    )
    source = _source(
        "S1",
        url=f"https://records.{slug}.invalid/launch",
        title=f"{name} launch record",
        authority="official",
        published_at=launch,
        content=f"{name} public service launched on {launch}. No anniversary campaign is documented here.",
        retrieved_at=BOOTSTRAP_SNAPSHOT,
    )
    result_fact = _fact(field="calendar_anniversary_date", value=result, evidence_ref="S1", quote=f"public service launched on {launch}")
    instruction = (
        f"Using the frozen launch record and date tool, calculate {name}'s {ordinal} calendar "
        "anniversary. Label it as a deterministic date derivation, not an announced event."
    )
    answer = (
        f"{name} launched on **{launch}** [S1]. Adding **{years} calendar year"
        f"{'s' if int(years) != 1 else ''}** gives **{result}**, its {ordinal} calendar anniversary. "
        "This is a deterministic derivation; the source does not announce an anniversary event."
    )
    events = [
        {"type": "task_plan", "content": _plan([{"id": "P1", "objective": "Source the launch date and derive the requested calendar anniversary."}])},
        {"type": "tool_call", "tool": "direct_page", "args": {"url": source["url"]}},
        {"type": "tool_result", "tool": "direct_page", "content": source},
        {"type": "tool_call", "tool": "date_diff", "args": {"operation": "add_years", "date": launch, "years": years}},
        {"type": "tool_result", "tool": "date_diff", "content": {"result": result}},
        {"type": "oracle_evidence_assessment", "content": _oracle_evidence_assessment({"P1": [result_fact]})},
        {"type": "final", "content": answer},
    ]
    contract = {
        "schema_version": "rwkv-retrieval-verifier-contract.v1",
        "claims": [
            _claim(
                "C1",
                "P1",
                "calendar_anniversary_date",
                result,
                "S1",
                f"public service launched on {launch}",
                derivation={"operation": "date_add_years", "evidence_ref": "S1", "source_value": launch, "years": years},
            )
        ],
        "forbidden_values": [f"an official anniversary event begins on {result}"],
        "behavior": {"max_tool_calls": 2, "allowed_urls": [source["url"]]},
        "reward_dimensions": ["date_derivation", "epistemic_label", "citation_grounding"],
    }
    rejected = [{
        "stage": "final",
        "failure_class": "derived_fact_presented_as_sourced_event",
        "prompt": {"source_date": launch, "operation": f"add {years} years"},
        "rejected": {"content": f"The official source announces an anniversary event on {result}."},
        "chosen": {"content": answer},
    }]
    return _bundle(root, task=task, instruction=instruction, sources=[source], contract=contract, events=events, reference_answer=answer, rejected_attempts=rejected)


def build_bootstrap(output_root: Path, *, replace: bool) -> Mapping[str, Any]:
    if output_root.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output_root}")
        shutil.rmtree(output_root)
    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "operator_cards.json",
        {"schema_version": "rwkv-retrieval-rst-operators.v1", "operators": OPERATOR_CARDS},
    )
    bundles: list[Path] = []
    for index, values in enumerate(CURRENT_RELEASE_CASES, start=1):
        bundles.append(build_current_release(tasks_root, index, values))
    for index, values in enumerate(TEMPORAL_CASES, start=1):
        bundles.append(build_temporal(tasks_root, index, values))
    for index, values in enumerate(DIRECT_PAGE_CASES, start=1):
        bundles.append(build_direct_page(tasks_root, index, values))
    for index, values in enumerate(HOWTO_CASES, start=1):
        bundles.append(build_howto(tasks_root, index, values))
    for index, values in enumerate(DERIVED_DATE_CASES, start=1):
        bundles.append(build_derived_date(tasks_root, index, values))
    manifest = {
        "schema_version": "rwkv-retrieval-rst-build.v1",
        "pool_role": "verified_bootstrap",
        "task_count": len(bundles),
        "tasks": [str(path.relative_to(output_root)) for path in bundles],
        "fictional_data_only": True,
        "benchmark_answers_used": False,
        "model_service_used": False,
        "paper_method": "Recursive Synthesis for Long-Horizon Terminal Tasks, arXiv:2608.05466v1",
    }
    _write_json(output_root / "build_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "training/retrieval_rst/bootstrap_v1",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    manifest = build_bootstrap(args.output_root.resolve(), replace=args.replace)
    print(json.dumps({"output_root": str(args.output_root.resolve()), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
