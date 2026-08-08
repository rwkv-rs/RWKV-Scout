"""Deterministic query constraints shared by discovery and evidence gates.

The functions in this module intentionally know nothing about products or
websites.  RWKV still decides what to search for and what a page means; these
helpers only preserve literal identities supplied by the user and reject
obviously unrelated search results before they consume the evidence budget.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import unquote


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_VERSION_RE = re.compile(r"(?<![\d.])v?(\d+\.\d+(?:\.\d+)?)(?![\d.])", re.IGNORECASE)
_NAMED_ID_RE = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,}|CWE-\d+|GHSA-[A-Za-z0-9-]+|RFC\s*\d+|PEP\s*\d+)\b",
    re.IGNORECASE,
)
_SITE_RE = re.compile(r"(?:^|\s)site:[A-Za-z0-9.-]+", re.IGNORECASE)
_LATIN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[+_.-][A-Za-z0-9]+)*")
_CJK_RE = re.compile(r"[\u3400-\u9fff]{2,}")
_TERM_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "find", "for", "from", "give", "how", "in", "into",
    "is", "it", "latest", "look", "me", "of", "official", "on", "or",
    "page", "please", "recent", "report", "search", "show", "site", "source",
    "sources", "tell", "that", "the", "this", "to", "use", "using", "was",
    "were", "what", "when", "where", "which", "who", "why", "with", "would",
    "answer", "exact", "information", "request", "requested", "result", "results",
}
_CJK_STOP_PHRASES = (
    "为什么", "是什么", "哪一个", "哪一天", "哪些", "如何", "怎么", "请",
    "查找", "搜索", "检索", "给出", "告诉", "说明", "根据", "按照", "使用",
    "官方", "页面", "来源", "结果", "回答", "信息", "当前", "最新", "核对",
)
_RETRIEVAL_PREFIXES = (
    re.compile(
        r"^\s*(?:(?:please|kindly)\s+)?(?:(?:can|could|would)\s+you\s+)?"
        r"(?:help\s+(?:me\s+)?(?:to\s+)?)?"
        r"(?:find|search(?:\s+for)?|look\s+up|locate|retrieve)\s+(?:me\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:(?:请|请你|麻烦|麻烦你|劳驾)\s*)?"
        r"(?:(?:帮我|帮忙)\s*)?"
        r"(?:查一下|查找|搜索|检索|查询|搜一下|找出|找到)\s*[:：，,]?\s*"
    ),
)

_PROCEDURE_ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "install": re.compile(r"\binstall(?:ation|ing|ed)?\b|安装|装(?:上|好)?", re.IGNORECASE),
    "enable": re.compile(
        r"\benabl(?:e|ed|es|ing)\b|\bactivat(?:e|ed|es|ing)\b|\bturn\s+on\b|"
        r"开启|启用|打开|(?:怎么|如何|怎样).{0,8}开",
        re.IGNORECASE,
    ),
    "disable": re.compile(
        r"\bdisabl(?:e|ed|es|ing)\b|\bdeactivat(?:e|ed|es|ing)\b|\bturn\s+off\b|禁用|关闭",
        re.IGNORECASE,
    ),
    "configure": re.compile(
        r"\bconfigur(?:e|ed|es|ing|ation)\b|\bsettings?\b|(?<!\w)set(?!\w)|"
        r"(?:^|[_-])set(?:[_-]|$)|配置|设置",
        re.IGNORECASE,
    ),
    "build": re.compile(
        r"(?im)(?:^|[.!?]\s+)(?:build|compile)\b|"
        r"\b(?:how\s+(?:do|can|to)\b.{0,24}|to|when|by|should|must|can|need(?:s)?\s+to)"
        r"\s+(?:build|building|compile|compiling)\b|"
        r"\b(?:build|building|compile|compiling|built|compiled)\s+(?:from|with|using)\b|构建|编译",
        re.IGNORECASE,
    ),
    "create": re.compile(
        r"\bcreat(?:e|ed|es|ing)\b|\bwrit(?:e|es|ing|ten)\b|创建|建立|编写|(?:怎么|如何|怎样).{0,8}写",
        re.IGNORECASE,
    ),
    "run": re.compile(r"\brun(?:s|ning)?\b|\bexecut(?:e|ed|es|ing|ion)\b|\binvok(?:e|ed|es|ing)\b|运行|执行|调用", re.IGNORECASE),
    "start": re.compile(r"\bstart(?:s|ed|ing)?\b|\blaunch(?:es|ed|ing)?\b|启动", re.IGNORECASE),
    "stop": re.compile(r"\bstop(?:s|ped|ping)?\b|\bshut\s*down\b|停止", re.IGNORECASE),
    "check": re.compile(
        r"\bcheck(?:s|ed|ing)?\b|\bverif(?:y|ies|ied|ying|ication)\b|"
        r"\bidentify(?:ing|ied|ies)?\b|\bdetermin(?:e|ed|es|ing)\b|检查|验证|确认|识别",
        re.IGNORECASE,
    ),
    "connect": re.compile(r"\bconnect(?:s|ed|ing|ion)?\b|连接", re.IGNORECASE),
    "calculate": re.compile(r"\bcalculat(?:e|ed|es|ing|ion)\b|\bcomput(?:e|ed|es|ing|ation)\b|计算", re.IGNORECASE),
}

_PROCEDURE_ACTION_COMPATIBILITY: dict[str, set[str]] = {
    "install": {"install", "build"},
    # Enabling a build-time feature is commonly expressed by installing a
    # variant, building with a flag, configuring a setting, or disabling the
    # inverse feature.  A check/verification action is deliberately excluded.
    "enable": {"enable", "disable", "configure", "build", "install"},
    "disable": {"disable", "configure", "stop"},
    "configure": {"configure", "enable", "disable"},
    "build": {"build", "configure"},
    "create": {"create"},
    "run": {"run", "start"},
    "start": {"start", "run", "enable"},
    "stop": {"stop", "disable"},
    "check": {"check"},
    "connect": {"connect", "configure"},
    "calculate": {"calculate"},
}


def procedure_action_families(value: Any) -> list[str]:
    """Return domain-neutral action roles explicitly expressed in text."""

    text = str(value or "")
    return [
        family
        for family, pattern in _PROCEDURE_ACTION_PATTERNS.items()
        if pattern.search(text)
    ]


def procedure_actions_compatible(requested: Any, observed: Any) -> bool:
    """Check whether a source/answer performs the procedure action requested."""

    requested_set = {str(value) for value in (requested or []) if str(value)}
    observed_set = {str(value) for value in (observed or []) if str(value)}
    if not requested_set:
        return True
    return any(
        bool(_PROCEDURE_ACTION_COMPATIBILITY.get(action, {action}) & observed_set)
        for action in requested_set
    )


def explicit_fact_anchors(value: Any) -> list[str]:
    """Return version/standard identifiers explicitly supplied by the user.

    URLs are removed first so host versions, IP addresses, and dated paths do
    not accidentally become answer-identity constraints.
    """

    text = _URL_RE.sub(" ", str(value or ""))
    output: list[str] = []
    for match in _NAMED_ID_RE.finditer(text):
        anchor = re.sub(r"\s+", "", match.group(0)).casefold()
        if anchor not in output:
            output.append(anchor)
    for match in _VERSION_RE.finditer(text):
        anchor = match.group(1).casefold()
        if anchor not in output:
            output.append(anchor)
    return output


def source_contains_anchor(value: Any, anchor: str) -> bool:
    text = str(value or "")
    normalized = str(anchor or "").casefold()
    if not normalized:
        return True
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", normalized):
        return bool(
            re.search(
                rf"(?<![\d.])v?{re.escape(normalized)}(?![\d.])",
                text,
                flags=re.IGNORECASE,
            )
        )
    compact = re.sub(r"\s+", "", text).casefold()
    return normalized in compact


def source_contains_all_anchors(value: Any, anchors: list[str]) -> bool:
    return all(source_contains_anchor(value, anchor) for anchor in anchors)


def strip_search_operators(value: Any) -> str:
    """Remove transport-only operators before topical comparison."""

    text = _URL_RE.sub(" ", str(value or ""))
    return " ".join(_SITE_RE.sub(" ", text).split())


def semantic_search_focus(value: Any) -> str:
    """Remove only transport-level retrieval imperatives from a query.

    This deliberately preserves the factual topic, requested fields, source
    names, versions and all non-leading verbs.  In particular, procedural
    questions such as ``How do I retrieve a token?`` are left unchanged.
    """

    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        return ""
    site_match = re.match(r"^site:([A-Za-z0-9.-]+)\s+", raw, re.IGNORECASE)
    site_prefix = site_match.group(0).strip() if site_match else ""
    focused = raw[site_match.end():] if site_match else raw
    for pattern in _RETRIEVAL_PREFIXES:
        candidate = pattern.sub("", focused, count=1).strip()
        if candidate != focused and len(candidate) >= 2:
            focused = candidate
            break
    return " ".join(value for value in (site_prefix, focused) if value).strip() or raw


def meaningful_query_terms(value: Any, *, domain: str = "") -> list[str]:
    """Return domain-independent topic terms for coarse candidate admission.

    This is deliberately only a noise gate, not a semantic relevance model.
    For Chinese input we retain whole lexical runs and short n-grams so a
    Chinese query can still admit an English-title/Chinese-snippet result when
    either representation is present.
    """

    text = strip_search_operators(value)
    domain_parts = set(re.findall(r"[a-z0-9]+", str(domain or "").casefold()))
    anchors = set(explicit_fact_anchors(text))
    output: list[str] = []

    def add(term: str) -> None:
        normalized = term.casefold().strip("._+-")
        if (
            not normalized
            or normalized in _TERM_STOP
            or normalized in domain_parts
            or normalized in anchors
            or normalized in output
        ):
            return
        output.append(normalized)

    for token in _LATIN_TERM_RE.findall(text):
        add(token)

    cjk_text = text
    for phrase in _CJK_STOP_PHRASES:
        cjk_text = cjk_text.replace(phrase, " ")
    for run in _CJK_RE.findall(cjk_text):
        if len(run) <= 12:
            add(run)
        for size in (2, 3):
            for index in range(max(0, len(run) - size + 1)):
                add(run[index:index + size])
    return output[:48]


def literal_query_anchors(value: Any) -> list[str]:
    """Return explicit technical tokens whose spelling carries identity.

    Mixed-case protocol names, command flags, and multi-part identifiers are
    stronger than generic topic words.  Natural title-case words are excluded,
    so this remains useful outside software questions without turning every
    capitalized noun into a mandatory keyword.
    """

    text = strip_search_operators(value)
    output: list[str] = []
    patterns = (
        r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9_-]+",
        r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b",
        r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+){2,}\b",
        r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*\b",
        r"\b[A-Z][A-Z0-9]{2,}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            token = match.group(0).casefold()
            if token not in _TERM_STOP and token not in output:
                output.append(token)
    return output[:16]


def locator_contains_anchor(value: Any, anchor: str) -> bool:
    """Match an explicit identity in titles/URLs, including common URL slugs.

    Source bodies remain subject to :func:`source_contains_anchor`, which
    requires the literal value.  The relaxed form is only for ranking URLs such
    as ``django-52-released`` for a user-supplied ``5.2``.
    """

    if source_contains_anchor(value, anchor):
        return True
    normalized = str(anchor or "").casefold()
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", normalized):
        return False
    parts = [re.escape(part) for part in normalized.split(".")]
    slug = r"[-_.]?".join(parts)
    return bool(re.search(rf"(?<!\d)v?{slug}(?!\d)", unquote(str(value or "")), re.IGNORECASE))


def _locator_versions(value: Any) -> set[str]:
    """Extract plausible software-style versions from a title/URL."""

    text = unquote(str(value or ""))
    output = {match.group(1).casefold() for match in _VERSION_RE.finditer(text)}
    for match in re.finditer(r"(?<!\d)v?(\d{1,3})[_-](\d{1,3})(?:[_-](\d{1,3}))?(?!\d)", text, re.I):
        major = int(match.group(1))
        if major >= 1900:
            continue
        output.add(".".join(part for part in match.groups() if part is not None).casefold())
    return output


def candidate_relevance(
    row: Mapping[str, Any] | str,
    query: Any,
    *,
    domain: str = "",
    constraint_query: Any | None = None,
) -> dict[str, Any]:
    """Return an explainable, conservative topical-admission decision.

    A result is rejected only when a non-trivial query has no adequate lexical
    overlap, or when its title/URL advertises a different explicit version than
    the one requested.  Semantic support is still decided later by RWKV against
    the fetched body and ClaimLedger.
    """

    if isinstance(row, Mapping):
        locator = " ".join(str(row.get(key) or "") for key in ("title", "snippet", "url"))
        identity_locator = " ".join(str(row.get(key) or "") for key in ("title", "url"))
    else:
        locator = str(row or "")
        identity_locator = locator
    folded = unquote(locator).casefold()
    folded_punctuation = re.sub(r"[._/]+", "-", folded)
    terms = meaningful_query_terms(query, domain=domain)
    term_hits = [
        term
        for term in terms
        if term in folded or re.sub(r"[._]+", "-", term) in folded_punctuation
    ]
    # Topical terms may come from an RWKV-refined search query, but hard
    # identities must come from the user's task.  Otherwise one hallucinated
    # protocol header or version in a search query can exclude the correct
    # page before it is fetched.
    hard_query = query if constraint_query is None else constraint_query
    anchors = explicit_fact_anchors(hard_query)
    anchor_hits = [anchor for anchor in anchors if locator_contains_anchor(identity_locator, anchor)]
    literal_anchors = literal_query_anchors(hard_query)
    literal_hits = [anchor for anchor in literal_anchors if anchor in folded]
    expected_versions = {anchor for anchor in anchors if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", anchor)}
    advertised_versions = _locator_versions(identity_locator)
    version_conflict = bool(
        expected_versions
        and advertised_versions
        and not expected_versions.intersection(advertised_versions)
    )
    required_hits = 0 if not terms else 1 if len(terms) <= 2 else 2
    effective_required_hits = 0 if literal_hits else required_hits
    raw_threshold_satisfied = len(term_hits) >= required_hits
    anchor_only = bool(anchors and not terms)
    related = not version_conflict and (
        len(term_hits) >= effective_required_hits
        and (not anchor_only or len(anchor_hits) == len(anchors))
        and (not literal_anchors or bool(literal_hits))
    )
    return {
        "related": related,
        "terms": terms,
        "term_hits": term_hits,
        "required_term_hits": required_hits,
        "effective_required_term_hits": effective_required_hits,
        "raw_threshold_satisfied": raw_threshold_satisfied,
        "anchors": anchors,
        "anchor_hits": anchor_hits,
        "anchor_satisfied": bool(anchors) and len(anchor_hits) == len(anchors),
        "literal_anchors": literal_anchors,
        "literal_hits": literal_hits,
        "literal_satisfied": not literal_anchors or bool(literal_hits),
        "version_conflict": version_conflict,
        "advertised_versions": sorted(advertised_versions),
        "score": round(len(term_hits) + 4.0 * len(anchor_hits), 4),
    }


__all__ = [
    "candidate_relevance",
    "explicit_fact_anchors",
    "locator_contains_anchor",
    "literal_query_anchors",
    "meaningful_query_terms",
    "procedure_action_families",
    "procedure_actions_compatible",
    "source_contains_all_anchors",
    "source_contains_anchor",
    "semantic_search_focus",
    "strip_search_operators",
]
