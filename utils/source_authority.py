"""Explainable source-authority policy for web retrieval.

This module is deliberately a routing/admission layer, not a truth oracle.
It prevents a third-party page from satisfying a request for an official
source merely because the page contains the requested organisation's name.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlparse



_NON_AUTHORITY_HOSTS = {
    "bing.com", "duckduckgo.com", "google.com", "yahoo.com",
    "wikipedia.org", "wikidata.org", "reddit.com", "facebook.com",
    "linkedin.com", "x.com", "twitter.com", "youtube.com",
    "github.com", "gitlab.com", "medium.com", "stackoverflow.com",
}
_ENTITY_STOP_TERMS = {
    "about", "according", "advisory", "answer", "current", "date",
    "documentation", "find", "latest", "official", "published", "release",
    "security", "source", "stable", "version", "what", "when", "where",
    "which", "with", "发布", "日期", "官网", "官方", "最新", "版本", "来源",
}
_OFFICIAL_MARKERS = (
    "official", "official site", "official documentation", "project documentation",
    "官网", "官方网站", "官方文档", "官方发布",
)
_DOCUMENTATION_MARKERS = (
    "documentation", "docs", "release notes", "releases", "changelog",
    "文档", "发布说明", "发行说明",
)
_DOMAIN_SCOPE_STOP_LABELS = {
    "api", "blog", "co", "com", "dev", "developer", "developers", "docs",
    "documentation", "download", "downloads", "edu", "gov", "help", "int",
    "io", "net", "news", "org", "release", "releases", "status", "support",
    "www",
}


def hostname(url: Any) -> str:
    try:
        host = (urlparse(str(url or "")).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""
    return host.removeprefix("www.")


def domain_matches(url: Any, domain: str) -> bool:
    host = _canonical_domain(hostname(url))
    expected = _canonical_domain(domain)
    return bool(host and expected and (host == expected or host.endswith("." + expected)))


def _is_institutional_host(host: str) -> bool:
    """Recognize public-sector and academic domain conventions globally."""

    value = str(host or "").casefold().rstrip(".")
    if not value:
        return False
    return bool(
        re.search(r"\.(?:gov|edu|mil|int)$", value)
        or re.search(r"\.(?:gov|edu|mil|ac|go|gob)\.[a-z]{2,}$", value)
        or value.endswith((".gc.ca", ".gouv.fr"))
    )


def _normalise_domains(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    normalized: list[str] = []
    for value in values:
        domain = _canonical_domain(value)
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def _canonical_domain(value: Any) -> str:
    return str(value or "").casefold().strip().removeprefix("www.").rstrip(".")


def explicit_domains(value: Any) -> list[str]:
    """Extract domains that the user explicitly supplied in text or URLs."""

    output: list[str] = []
    pattern = re.compile(
        r"(?:https?://|site:)?((?:[a-z0-9](?:[a-z0-9-]{0,62})\.)+[a-z]{2,63})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(str(value or "")):
        domain = _canonical_domain(match.group(1))
        if domain and domain not in output:
            output.append(domain)
    return output


def required_domains_for_task_point(
    task_plan: Mapping[str, Any] | None,
    task_point_id: str = "",
    *,
    fallback_query: str = "",
) -> list[str]:
    """Narrow a request-wide official-domain set to one atomic claim.

    A comparison can legitimately require several first-party domains.  The
    retrieval route must not silently send every claim to the first domain in
    that global list.  Explicit point metadata wins; otherwise domains are
    selected only when their entity label has an unambiguous lexical match in
    the active point.  If no safe match exists the full set is returned, which
    lets provider discovery search broadly without inventing a site filter.
    """

    plan = task_plan if isinstance(task_plan, Mapping) else {}
    global_domains = _normalise_domains(plan.get("required_domains"))
    if len(global_domains) <= 1:
        return global_domains

    point_id = str(task_point_id or "").strip()
    point = next(
        (
            value
            for value in plan.get("atomic_points") or []
            if isinstance(value, Mapping)
            and str(value.get("id") or "").strip() == point_id
        ),
        None,
    )
    if isinstance(point, Mapping):
        declared = _normalise_domains(point.get("required_domains"))
        if declared:
            scoped = [domain for domain in declared if domain in global_domains]
            return scoped or declared
        descriptor = " ".join(
            str(value or "")
            for value in (
                point.get("task"),
                point.get("objective"),
                *(point.get("evidence_needed") or []),
                *(point.get("acceptance_criteria") or []),
            )
        ).strip()
    else:
        descriptor = str(fallback_query or "").strip()
    if not descriptor:
        return global_domains

    explicit = explicit_domains(descriptor)
    explicitly_scoped = [domain for domain in global_domains if domain in explicit]
    if explicitly_scoped:
        return explicitly_scoped

    entity_tokens = _entity_tokens(descriptor)
    compact_descriptor = re.sub(r"[^a-z0-9]", "", descriptor.casefold())
    scored: list[tuple[int, str]] = []
    for domain in global_domains:
        labels = [
            label
            for label in re.findall(r"[a-z0-9]+", domain)
            if label not in _DOMAIN_SCOPE_STOP_LABELS
        ]
        compact_domain = "".join(labels)
        score = 0
        for token in entity_tokens:
            if len(token) >= 3 and (
                token in compact_domain
                or any(len(label) >= 3 and label in token for label in labels)
            ):
                score += 3
        # A few project names are valid two/three-letter words (Go, WHO).
        # Require their characteristic capitalization to avoid treating
        # ordinary prose such as "go to" or "who released" as an entity.
        for label in labels:
            if len(label) == 2 and re.search(rf"\b{re.escape(label.title())}\b", descriptor):
                score += 2
            elif len(label) == 3 and re.search(rf"\b{re.escape(label.upper())}\b", descriptor):
                score += 2
            elif len(label) >= 4 and label in compact_descriptor:
                score += 1
        if score:
            scored.append((score, domain))
    if not scored:
        return global_domains
    best = max(score for score, _ in scored)
    return [domain for score, domain in scored if score == best]


def _entity_tokens(value: Any) -> list[str]:
    output: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,63}", str(value or "")):
        token = raw.casefold().replace("_", "").replace("-", "")
        if token in _ENTITY_STOP_TERMS or token.isdigit():
            continue
        if len(token) < 4 and not (len(token) >= 2 and raw.isupper()):
            continue
        if token not in output:
            output.append(token)
    return output[:16]


def infer_candidate_authority_domains(
    query: str,
    provider_results: list[Mapping[str, Any]],
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Infer a first-party domain only from strong, explainable signals.

    This is a bootstrap for an official-source task whose plan omitted a
    domain.  It does not determine truth.  The selected host is subsequently
    crawled and its retained page spans still pass the normal claim gates.
    """

    entities = _entity_tokens(query)
    hosts: dict[str, dict[str, Any]] = {}
    for provider in provider_results:
        provider_name = str(provider.get("provider") or "unknown")
        for item in provider.get("results") or []:
            if not isinstance(item, Mapping):
                continue
            host = hostname(item.get("url"))
            if not host or any(host == blocked or host.endswith("." + blocked) for blocked in _NON_AUTHORITY_HOSTS):
                continue
            title_snippet = " ".join(
                str(item.get(key) or "") for key in ("title", "snippet")
            ).casefold()
            compact_host = re.sub(r"[^a-z0-9]", "", host)
            host_hits = [token for token in entities if token in compact_host]
            text_hits = [token for token in entities if token in re.sub(r"[^a-z0-9]", "", title_snippet)]
            institutional = _is_institutional_host(host)
            official_claim = any(marker in title_snippet for marker in _OFFICIAL_MARKERS)
            documentation_claim = any(marker in title_snippet for marker in _DOCUMENTATION_MARKERS)
            topical = bool(host_hits or text_hits)
            if not topical:
                continue
            row = hosts.setdefault(
                host,
                {
                    "domain": host,
                    "providers": set(),
                    "host_entity_hits": set(),
                    "text_entity_hits": set(),
                    "institutional": False,
                    "official_marker": False,
                    "documentation_marker": False,
                },
            )
            row["providers"].add(provider_name)
            row["host_entity_hits"].update(host_hits)
            row["text_entity_hits"].update(text_hits)
            row["institutional"] = bool(row["institutional"] or institutional)
            row["official_marker"] = bool(row["official_marker"] or official_claim)
            row["documentation_marker"] = bool(row["documentation_marker"] or documentation_claim)

    ranked: list[dict[str, Any]] = []
    for row in hosts.values():
        score = 0
        reasons: list[str] = []
        if row["host_entity_hits"]:
            score += 5
            reasons.append("entity_in_host")
        if row["official_marker"]:
            score += 3
            reasons.append("official_result_label")
        if row["documentation_marker"]:
            score += 2
            reasons.append("first_party_content_label")
        if row["institutional"]:
            score += 6
            reasons.append("institutional_domain")
        if len(row["providers"]) >= 2:
            score += 1
            reasons.append("provider_consensus")
        # A commercial/content host merely mentioning the entity is not
        # enough. Require either an entity-bearing host or an institutional
        # domain, plus a second independent first-party signal.
        strong_identity = bool(row["host_entity_hits"] or row["institutional"])
        if not strong_identity or score < 7:
            continue
        ranked.append(
            {
                "domain": row["domain"],
                "score": score,
                "reasons": reasons,
                "providers": sorted(row["providers"]),
            }
        )
    ranked.sort(key=lambda item: (-int(item["score"]), str(item["domain"])))
    return ranked[: max(1, min(int(limit or 2), 4))]


def resolve_source_policy(query: str, constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve source requirements from the task plan, with safe fallbacks."""

    constraints = constraints or {}
    plan = constraints.get("task_plan") if isinstance(constraints, Mapping) else {}
    plan = plan if isinstance(plan, Mapping) else {}
    policy = str(plan.get("source_policy") or constraints.get("source_policy") or "").casefold()
    planned_domains = _normalise_domains(plan.get("required_domains"))
    constrained_domains = _normalise_domains(constraints.get("required_domains"))
    query_domains = explicit_domains(query)
    required = planned_domains or constrained_domains or query_domains
    domain_source = (
        "model_plan"
        if planned_domains
        else "runtime_constraint"
        if constrained_domains
        else "explicit_query"
        if query_domains
        else "unresolved"
    )
    text = " ".join(
        str(value or "")
        for value in (query, plan.get("goal"), plan.get("task_mode"))
    ).casefold()

    explicit_official = any(
        marker in text
        for marker in ("official", "officially", "官网", "官方", "权威", "normative", "iso record")
    )
    explicit_official = explicit_official or any(
        marker in text for marker in ("according to", "reported by", "reported")
    )
    if required and explicit_official and policy in {"", "open_web", "primary_preferred"}:
        policy = "official_required"
    elif not policy:
        policy = "official_required" if required and explicit_official else "primary_preferred" if required else "open_web"
    if policy in {"official", "official_only", "official_required", "primary_official"}:
        policy = "official_required"
    elif policy in {"direct", "direct_page", "direct_required"}:
        policy = "direct_required"
    elif policy in {"community", "community_only", "community_required"}:
        policy = "community_required"
    elif policy not in {"primary_preferred", "open_web"}:
        policy = "open_web"
    return {
        "mode": policy,
        "required_domains": required,
        "required": bool(required and policy in {"official_required", "direct_required", "community_required"}),
        "domain_source": domain_source,
    }


def authority_for_url(url: Any, query: str = "", constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = resolve_source_policy(query, constraints)
    host = hostname(url)
    required_domains = policy["required_domains"]
    required_match = any(domain_matches(url, domain) for domain in required_domains)
    institutional = _is_institutional_host(host)
    if required_match:
        label = {
            "direct_required": "direct_required",
            "community_required": "community_required",
        }.get(policy["mode"], "official_required")
        rank = 4
    elif institutional:
        label, rank = "institutional", 2
    else:
        label, rank = "third_party", 0
    return {
        "host": host,
        "label": label,
        "rank": rank,
        "required": policy["required"],
        "required_domains": required_domains,
        "satisfied": (required_match if policy["required"] else rank >= 1),
    }


def annotate_source(item: Mapping[str, Any], query: str = "", constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(item)
    authority = authority_for_url(value.get("url"), query, constraints)
    value["authority"] = authority
    value["source_kind"] = "official" if authority["label"].startswith("official") else value.get("source_kind") or "web"
    return value


__all__ = [
    "annotate_source",
    "authority_for_url",
    "domain_matches",
    "explicit_domains",
    "hostname",
    "infer_candidate_authority_domains",
    "required_domains_for_task_point",
    "resolve_source_policy",
]
