# RWKV-ECRA/agent/planner.py
import json
import re
from clients.llm_client import LLMClient
from tools.registry import ToolRegistry

class Planner:
    def __init__(self):
        self.llm = LLMClient()
        
    def plan_next_action(self, user_query: str, analysis_result: dict, env_context: str, phase: str) -> dict:
        normalized_query = (user_query or "").strip().lower()
        paper_cues = ["\u8bba\u6587", "\u6587\u732e", "arxiv", "paper", "series", "\u7cfb\u5217", "\u76f8\u5173\u5de5\u4f5c", "survey"]
        multi_hop_cues = ["\u518d\u67e5", "\u518d\u627e", "\u7136\u540e", "\u63a5\u7740", "\u4ea4\u96c6", "\u518d\u4ece", "readme"]
        if any(cue in normalized_query for cue in multi_hop_cues):
            first_action = "search_papers" if any(cue in normalized_query for cue in paper_cues + ["rwkv", "\u6a21\u578b\u67b6\u6784"]) else "search_web_keyless"
            scope = "series" if any(cue in normalized_query for cue in ["\u7cfb\u5217", "series", "\u76f8\u5173\u5de5\u4f5c", "survey"]) else "paper"
            return {
                "action": "multi_hop_research",
                "args": {"first_action": first_action, "scope": scope},
                "router": "static_multi_hop_cue",
            }
        if any(cue in normalized_query for cue in paper_cues):
            series_cues = ["\u7cfb\u5217", "series", "\u76f8\u5173\u5de5\u4f5c", "survey"]
            scope = "series" if any(cue in normalized_query for cue in series_cues) else "paper"
            return {
                "action": "search_papers",
                "args": {"query": user_query, "scope": scope, "max_results": 8},
                "router": "static_paper_cue",
            }
        # Weather is a dedicated live-data tool. It must take precedence over
        # generic words such as “当前” and “查询”, while closed-world
        # translation prompts were handled above.
        if any(term in normalized_query for term in ["天气", "气温", "温度", "weather", "temperature"]):
            cities = ["上海", "北京", "广州", "深圳", "杭州", "成都", "重庆", "南京", "Shanghai", "Beijing"]
            location = next((city for city in cities if city.lower() in normalized_query), "上海")
            return {"action": "get_current_weather", "args": {"location": location}, "router": "static_weather_cue"}

        live_search_cues = [
            "\u622a\u81f3", "\u5f53\u524d", "\u6700\u65b0", "\u67e5\u627e", "\u641c\u7d22", "\u68c0\u7d22",
            "\u5b98\u65b9", "\u7248\u672c", "\u53d1\u5e03", "\u65e5\u671f", "\u63d0\u4ea4", "\u6587\u6863",
            "github", "release", "version", "official", "benchmark", "pep", "arxiv",
            "pytorch", "python", "ubuntu", "transformers", "codex",
        ]
        if any(cue in normalized_query for cue in live_search_cues):
            return {
                "action": "search_web_keyless",
                "args": {"query": user_query, "max_results": 6, "fetch_pages": 3},
                "router": "static_live_search_cue",
            }

        # Closed-world operations must win over keyword matches. A sentence
        # asking to translate “今天天气很好” contains the word 天气 but is not
        # a live weather request.
        closed_world_cues = [
            "翻译成英文", "翻译为英文", "计算", "格式化", "json", "摘要",
            "翻译", "translate", "summarize", "format",
        ]
        if any(cue in normalized_query for cue in closed_world_cues):
            return {"action": "answer_user", "args": {}, "router": "static_no_search_cue"}

        if normalized_query in {"你好", "您好", "hello", "hi", "hey", "谢谢", "谢谢你"}:
            return {"action": "answer_user", "args": {}}

        research_cues = ["研究", "分析", "报告", "比较", "检索", "查找", "论文", "文件", "资料", "调研", "research", "analyze"]
        if len(normalized_query) <= 60 and not any(cue in normalized_query for cue in research_cues):
            return {"action": "answer_user", "args": {}}

        tool_interfaces = ToolRegistry.get_interfaces_by_phase(phase)
        active_query = analysis_result.get('refined_query', user_query)
        missing_info = analysis_result.get('missing_information', '无')
        
        sys_prompt = f"""基于当前缺口生成工具调用的 JSON 参数。

目标：{active_query}
缺口：{missing_info}

{tool_interfaces}

约束：
1. execute_web_search：query 必须是极简短关键词，禁止长句。
2. verify_keyword_in_file：必须将长实体拆分为简短的核心词组放入 keywords 数组。
3. 必须且只能输出单个 JSON 对象，格式严格如下：
{{"action": "工具名称", "args": {{"参数1": "值1"}}}}"""
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"当前环境状态：\n{env_context}\n\n请直接输出 JSON 工具规划。"}
        ]
        
        llm_response = self.llm.chat_completion(messages).content
        
        print(f"[大模型规划原始输出]:\n{llm_response.strip()}")

        match_dict = re.search(r'\{.*\}', llm_response, re.DOTALL)
        match_list = re.search(r'\[.*\]', llm_response, re.DOTALL)
        
        if match_dict:
            clean_json = match_dict.group(0)
        elif match_list:
            clean_json = match_list.group(0)
        else:
            clean_json = llm_response
            
        # 这里去除了 try...except，解析错误直接向外抛出供主流程捕获
        plan_data = json.loads(clean_json)
        
        if isinstance(plan_data, list) and len(plan_data) > 0:
            plan_data = plan_data[0]
        if not isinstance(plan_data, dict):
            plan_data = {}
        
        action = plan_data.get("action") or plan_data.get("tool_name") or plan_data.get("name") or plan_data.get("tool") or "none"
        args = plan_data.get("args") or plan_data.get("parameters") or plan_data.get("arguments") or {}
        
        return {
            "action": action,
            "args": args
        }
