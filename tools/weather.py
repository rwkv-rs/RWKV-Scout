"""Free, keyless weather lookup through Open-Meteo."""

import json
from urllib.parse import quote

import requests

from tools.registry import ToolRegistry


WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


@ToolRegistry.register(
    name="get_current_weather",
    phase="ALL",
    signature="""[Tool] get_current_weather
- 功能: 通过无需 API Key 的 Open-Meteo 查询指定城市的当前天气。
- 参数: location (城市名称，例如上海、Shanghai)""",
)
def get_current_weather(location: str = "上海", working_memory=None, agent_state=None, **kwargs) -> str:
    location = (location or "上海").strip()
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    session = requests.Session()
    # The inherited Windows proxy aborts these requests in this environment;
    # use the same direct transport policy as the local vLLM client.
    session.trust_env = False

    geo_response = session.get(
        geo_url,
        params={"name": location, "count": 1, "language": "zh", "format": "json"},
        timeout=(10, 45),
    )
    geo_response.raise_for_status()
    results = geo_response.json().get("results") or []
    if not results:
        return json.dumps({"status": "not_found", "location": location}, ensure_ascii=False)

    place = results[0]
    latitude = place["latitude"]
    longitude = place["longitude"]
    weather_response = session.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=(10, 45),
    )
    weather_response.raise_for_status()
    data = weather_response.json()
    current = data.get("current") or {}
    weather_code = current.get("weather_code")

    result = {
            "status": "ok",
            "location": place.get("name", location),
            "country": place.get("country", ""),
            "admin1": place.get("admin1", ""),
            "current": {
                "time": current.get("time"),
                "temperature_c": current.get("temperature_2m"),
                "apparent_temperature_c": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "weather_code": weather_code,
                "condition": WEATHER_CODES.get(weather_code, "未知天气"),
            },
            "sources": [
                f"{geo_url}?name={quote(location)}&count=1&language=zh&format=json",
                f"{weather_url}?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=auto",
            ],
        }
    result_text = json.dumps(result, ensure_ascii=False)
    if working_memory is not None:
        working_memory[f"Weather_{location}"] = result_text
    if agent_state:
        current = result["current"]
        agent_state.is_finished = True
        agent_state.final_result = (
            f"{result['location']} 当前天气：{current.get('condition')}，"
            f"气温 {current.get('temperature_c')}°C，体感 {current.get('apparent_temperature_c')}°C，"
            f"湿度 {current.get('humidity_percent')}%，风速 {current.get('wind_speed_kmh')} km/h。\n"
            f"数据时间：{current.get('time')}；来源：{result['sources'][1]}"
        )
    return result_text
