"""Structured current weather-warning feed from China's NMC."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urljoin

from utils.network_fetch import NetworkFetchError, fetch_json


ALERT_PAGE_URL = "https://www.nmc.cn/publish/alarm.html"
ALERT_API_URL = "https://www.nmc.cn/rest/findAlarm"
_ALERT_RE = re.compile(r"发布(.+?)(红色|橙色|黄色|蓝色)预警(?:信号)?$")


def _warning_shape(title: str) -> tuple[str, str]:
    match = _ALERT_RE.search(str(title or ""))
    if not match:
        return ("其他", "未标注")
    return (match.group(1).strip(), match.group(2))


def get_current_weather_alerts(query: str = "", max_results: int = 20, **_: Any) -> dict[str, Any]:
    """Return a bounded, source-labelled snapshot of active NMC warnings."""

    limit = max(5, min(int(max_results or 20), 50))
    try:
        payload = fetch_json(
            ALERT_API_URL,
            params={
                "pageNo": 1,
                "pageSize": max(limit, 50),
                "signaltype": "",
                "signallevel": "",
                "province": "",
            },
            timeout=30,
            headers={"Referer": ALERT_PAGE_URL},
        )
    except (NetworkFetchError, ValueError) as exc:
        return {
            "status": "error",
            "provider": "China National Meteorological Centre",
            "error_class": "weather_alert_feed",
            "provider_errors": [f"{type(exc).__name__}: {exc}"],
            "results": [],
        }

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    page = data.get("page") if isinstance(data.get("page"), dict) else {}
    warnings = [row for row in page.get("list") or [] if isinstance(row, dict)]
    provincial = [row for row in data.get("provinceAlarms") or [] if isinstance(row, dict)]
    stat = data.get("stat") if isinstance(data.get("stat"), dict) else {}
    if not warnings and not provincial:
        return {
            "status": "no_results",
            "provider": "China National Meteorological Centre",
            "query": query,
            "results": [],
            "sources": [ALERT_PAGE_URL, ALERT_API_URL],
        }

    shapes = Counter(_warning_shape(str(row.get("title") or "")) for row in warnings)
    level_names = {"r": "红色", "o": "橙色", "y": "黄色", "b": "蓝色"}
    administrative_names = {
        "province": "省级",
        "city": "市级",
        "county": "县级",
        "省": "省级",
        "市": "市级",
        "县": "县级",
    }
    totals = Counter()
    for counts in stat.values():
        if isinstance(counts, dict):
            for key in level_names:
                totals[key] += int(counts.get(key) or 0)
    lines = [
        "中央气象台全国气象预警实时列表快照。",
        f"接口当前返回预警总数：{int(page.get('count') or len(warnings))}。",
    ]
    if totals:
        lines.append(
            "按预警级别合计（涵盖省、市、县三级）："
            + "、".join(f"{level_names[key]}{totals[key]}条" for key in ("r", "o", "y", "b"))
            + "。"
        )
    for level_name, counts in stat.items():
        if not isinstance(counts, dict):
            continue
        rendered = "、".join(
            f"{level_names[key]}{int(counts.get(key) or 0)}条"
            for key in ("r", "o", "y", "b")
        )
        label = administrative_names.get(str(level_name), f"{level_name}级")
        lines.append(f"{label}统计：{rendered}。")
    if shapes:
        lines.append(
            "最近一页出现的预警类型与级别："
            + "、".join(
                f"{kind}{level}（{count}条）"
                for (kind, level), count in shapes.most_common(12)
            )
            + "。"
        )
    if provincial:
        lines.append("最新省级预警：")
        for row in provincial[: min(limit, 20)]:
            lines.append(
                f"- {row.get('issuetime', '')}｜{row.get('title', '')}｜"
                f"{urljoin(ALERT_PAGE_URL, str(row.get('url') or ''))}"
            )
    lines.append("最近发布的预警（仅列最新记录，不等于全部预警）：")
    for row in warnings[:limit]:
        lines.append(
            f"- {row.get('issuetime', '')}｜{row.get('title', '')}｜"
            f"{urljoin(ALERT_PAGE_URL, str(row.get('url') or ''))}"
        )

    evidence = "\n".join(lines)
    return {
        "status": "ok",
        "provider": "China National Meteorological Centre",
        "query": query,
        "count": 1,
        "results": [
            {
                "title": "中央气象台全国气象预警实时列表",
                "url": ALERT_API_URL,
                "source": "China National Meteorological Centre",
                "structured_evidence_text": evidence,
                "evidence_origin": "structured_api_record",
                "evidence_kind": "structured_record",
                "evidence_boundary": "structured_api_record_only",
                "body_verified": True,
                "snapshot_time": str(warnings[0].get("issuetime") or provincial[0].get("issuetime") or ""),
                "warning_count": int(page.get("count") or len(warnings)),
            }
        ],
        "sources": [ALERT_PAGE_URL, ALERT_API_URL],
    }


__all__ = ["get_current_weather_alerts"]
