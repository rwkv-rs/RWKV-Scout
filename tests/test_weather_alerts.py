from unittest.mock import patch

from tools.weather_alerts import ALERT_API_URL, get_current_weather_alerts


def test_weather_alert_feed_becomes_one_bounded_structured_evidence_record():
    payload = {
        "data": {
            "page": {
                "count": 2,
                "list": [
                    {
                        "issuetime": "2026-08-07 10:00:00",
                        "title": "北京市气象台发布暴雨橙色预警",
                        "url": "/publish/alarm/abc.html",
                    },
                    {
                        "issuetime": "2026-08-07 09:00:00",
                        "title": "上海中心气象台发布高温黄色预警信号",
                        "url": "/publish/alarm/def.html",
                    },
                ],
            },
            "provinceAlarms": [],
            "stat": {"省": {"r": 0, "o": 1, "y": 1, "b": 0}},
        }
    }
    with patch("tools.weather_alerts.fetch_json", return_value=payload):
        result = get_current_weather_alerts("全国现在有哪些气象预警？", max_results=10)

    assert result["status"] == "ok"
    assert result["count"] == 1
    record = result["results"][0]
    assert record["url"] == ALERT_API_URL
    assert record["evidence_origin"] == "structured_api_record"
    assert record["body_verified"] is True
    assert "接口当前返回预警总数：2" in record["structured_evidence_text"]
    assert "按预警级别合计（涵盖省、市、县三级）：红色0条、橙色1条、黄色1条、蓝色0条" in record["structured_evidence_text"]
    assert "省级统计" in record["structured_evidence_text"]
    assert "暴雨橙色" in record["structured_evidence_text"]
    assert "高温黄色" in record["structured_evidence_text"]
