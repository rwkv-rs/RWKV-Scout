"""Explainable source-authority policy for web retrieval.

This module is deliberately a routing/admission layer, not a truth oracle.
It prevents a third-party page from satisfying a request for an official
source merely because the page contains the requested organisation's name.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse


# Small, explicit defaults for the domains already exercised by ECRA's web
# tasks.  A task plan may provide a more specific required_domains list.
_DOMAIN_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("miit", "工信部", "工业和信息化部", "telecommunications industry"), ("miit.gov.cn",)),
    (("population statistics", "人口统计", "国家统计局", "nbs"), ("stats.gov.cn",)),
    (("cisa", "known exploited vulnerabilities", "kev catalog"), ("cisa.gov",)),
    (("firefox", "mozilla"), ("mozilla.org",)),
    (("ubuntu", "usn", "security notices"), ("ubuntu.com",)),
    (("html living standard", "whatwg", "html standard"), ("html.spec.whatwg.org", "whatwg.org")),
    (("ecmascript", "tc39"), ("tc39.es", "ecma-international.org")),
    (("c++ standard", "c++", "iso/iec 14882"), ("isocpp.org", "iso.org")),
    (("kubernetes cve", "kubernetes cv", "kubernetes security"), ("kubernetes.io", "kubernetes.dev", "cisa.gov", "nvd.nist.gov")),
)

# Official projects may retain historical or alternate hostnames in
# model-generated task plans. These are exact project-specific aliases; they
# do not turn arbitrary third-party domains into official sources.
_OFFICIAL_DOMAIN_ALIASES: dict[str, str] = {
    "golang.org": "go.dev",
    "swe-bench.com": "swebench.com",
    "curl.haxx.se": "curl.se",
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
    domain = str(value or "").casefold().strip().removeprefix("www.").rstrip(".")
    return _OFFICIAL_DOMAIN_ALIASES.get(domain, domain)


def resolve_source_policy(query: str, constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve source requirements from the task plan, with safe fallbacks."""

    constraints = constraints or {}
    plan = constraints.get("task_plan") if isinstance(constraints, Mapping) else {}
    plan = plan if isinstance(plan, Mapping) else {}
    policy = str(plan.get("source_policy") or constraints.get("source_policy") or "").casefold()
    required = _normalise_domains(plan.get("required_domains") or constraints.get("required_domains"))
    text = " ".join(
        str(value or "")
        for value in (query, plan.get("goal"), plan.get("task_mode"))
    ).casefold()

    if not required:
        for terms, domains in _DOMAIN_HINTS:
            if any(term.casefold() in text for term in terms):
                required = list(domains)
                break

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
    elif policy not in {"primary_preferred", "open_web"}:
        policy = "open_web"
    return {
        "mode": policy,
        "required_domains": required,
        "required": bool(required and policy == "official_required"),
    }


def authority_for_url(url: Any, query: str = "", constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = resolve_source_policy(query, constraints)
    host = hostname(url)
    required_domains = policy["required_domains"]
    required_match = any(domain_matches(url, domain) for domain in required_domains)
    institutional = host.endswith((".gov", ".gov.cn", ".edu", ".edu.cn"))
    known_official = any(
        domain_matches(url, domain)
        for _, domains in _DOMAIN_HINTS
        for domain in domains
    )
    if required_match:
        label, rank = "official_required", 4
    elif known_official:
        label, rank = "official_known", 3
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


__all__ = ["annotate_source", "authority_for_url", "domain_matches", "hostname", "resolve_source_policy"]
