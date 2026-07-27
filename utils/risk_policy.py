"""Deterministic safety policy for high-risk information-retrieval tasks."""

from __future__ import annotations

from typing import Any


HIGH_RISK_DOMAINS = {
    "medicine_literacy": "medical information",
    "legal_information": "legal information",
    "finance_business": "financial information",
    "public_policy": "public policy information",
}

_WARNING_TERMS = (
    "not medical advice",
    "not legal advice",
    "not financial advice",
    "consult a qualified",
    "consult a professional",
    "seek professional",
    "仅供参考",
    "不构成医疗建议",
    "不构成法律意见",
    "不构成投资建议",
    "请咨询专业",
    "专业人员确认",
    "风险提示",
)


def risk_context(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    domain = str(metadata.get("domain") or "").strip()
    label = HIGH_RISK_DOMAINS.get(domain, "")
    return {
        "high_risk": bool(label),
        "domain": domain,
        "label": label,
        "risk_checks": [str(item) for item in metadata.get("risk_checks") or [] if str(item).strip()],
        "acceptance_criteria": [
            str(item) for item in metadata.get("acceptance_criteria") or [] if str(item).strip()
        ],
        "rejection_criteria": [
            str(item) for item in metadata.get("rejection_criteria") or [] if str(item).strip()
        ],
    }


def validate_risk_answer(answer: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    policy = risk_context(metadata)
    if not policy["high_risk"]:
        return {
            "validator_version": "risk-policy.v1",
            "high_risk": False,
            "valid": True,
            "warning_present": None,
            "issues": [],
        }
    text = str(answer or "").casefold()
    warning_present = any(term.casefold() in text for term in _WARNING_TERMS)
    issues = [] if warning_present else ["missing_professional_risk_boundary"]
    return {
        "validator_version": "risk-policy.v1",
        "high_risk": True,
        "domain": policy["domain"],
        "warning_present": warning_present,
        "valid": not issues,
        "issues": issues,
    }
