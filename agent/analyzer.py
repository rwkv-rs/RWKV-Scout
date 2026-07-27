# RWKV-ECRA/agent/analyzer.py
import json
from datetime import datetime
from clients.llm_client import LLMClient

class Analyzer:
    def __init__(self):
        self.llm = LLMClient()
        
    def analyze_intent_and_phase(self, user_query: str, env_context: str) -> dict:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sys_prompt = f"""系统时间：{current_time}
任务：分析用户的初始意图，并基于已挂载的上下文输出状态与行动方向。

【核心架构：双轨执行机制】
你需要首先判定用户的指令属于哪种模式，并严格按对应的模式执行：

▶ 模式 A：【全局泛读 / 无特定实体】(BROAD_ANALYSIS)
- 触发场景：没有要求调查特定对象或特定逻辑。
- 执行策略：不执行实体审计。针对未知的本地工作区文件，应当先进入 DISCOVERY 阶段调用 preview 进行试读。注意：本地工作区中可能包含与本次任务完全无关的干扰文件，你需要根据试读结果，决定是剔除它，还是进入 EXTRACTION 阶段提炼其全文。

▶ 模式 B：【定向深研 / 有特定实体】(DEEP_RESEARCH)
- 触发场景：指令包含具体的专有名词、人物、机构，或要求验证特定逻辑。
- 执行策略：启动严格的防诱导与跨界隔离防御。提取并查证所有实体状态。

【视觉与防脑补绝对红线】
1. 你目前能感知和操作的，只有当前明确列在“环境状态”中的本地工作区文件。绝对不要凭空虚构或引入未列出的文件！
2. 本地工作区的文件之间可能是完全平行且毫无关联的，绝对不要在没有原文依据的情况下强行脑补它们之间的关联。
3. 跨实体验证（防幻觉）：若用户同时提及多个实体，不要假定它们都在同一个文件中。你应当明确在 missing_information 中指出：“需要调用 verify_keyword_in_file 验证某实体是否存在于某文件中”。

【资产与目标对比判定规则】
1. 当你在环境状态中看到某文件的“试读结论”或“缓存资产”时，必须严格将其与【总体目标】进行对比推演。
2. 核心红线：如果用户的任务是“泛读、提取全部文件”，绝不能因为内容平淡就抛弃它。
3. 只有当内容是乱码，或者与特定垂直目标形成绝对的南辕北辙（完全纯垃圾）时，才可将其加入废弃名单，并必须提供对比推演理由。

【状态流转与防死循环绝对红线】
1. 本地文件(DOC_X)：若未读则走 DISCOVERY(试读)，若相关则走 EXTRACTION(全文提炼)。
2. 网络情报(WebFact_xxx)：只要出现在【情报目录大纲】中，代表底层系统【已经完成了全文检索与压缩提炼】！它是直接可用的结论！绝对、绝对不可以再对 WebFact 执行任何试读或提取操作！
3. 当所有目标实体的状态都变为 "确认相关" 或 "确认无关"，且缺口补齐时，必须立即将 next_phase 指向 SYNTHESIS，停止无意义的检索。

输出JSON格式：
{{
  "intent_mode": "BROAD_ANALYSIS 或是 DEEP_RESEARCH",
  "sandbox_evaluation": "针对本地工作区文件的评估，若无则为空",
  "entity_audit": {{
    // 若为 BROAD_ANALYSIS，必须为空字典 {{}}
    // 若为 DEEP_RESEARCH，输出 "实体名": "待检索 / 确认相关 / 确认无关"
    // 注：实体名必须极度精简，仅保留核心名词或动作，禁止填入长句。
  }},
  "abandoned_file_ids": {{
    // 🗑️ 物理屏蔽资源字典。如果是有效文件请留空 {{}}
    // 如果经过对比预定目标，发现某文件属于绝对纯垃圾干扰项，填入此处彻底抛弃。
    // 格式为: "DOC_ID": "详细说明为什么该文件内容与预定目标完全无关，必须被屏蔽"
  }},
  "refined_query": "当前有效脱水目标（若是泛读模式，保持原意即可；深研模式剔除无关项）",
  "missing_information": "指明下一步缺口（需要试读哪些文件，或需要全文提炼哪些文件）",
  "next_phase": "DISCOVERY|EXTRACTION|SYNTHESIS"
}}"""
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"总体目标: {user_query}\n当前环境状态:\n{env_context}"}
        ]
        local_max_tokens = None
        if self.llm.provider == "local":
            # G1h is used as a routing model here. Keep the contract compact
            # so the model cannot spend its whole budget on free-form prose.
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object and no explanation. "
                        "Keys: intent_mode, refined_query, missing_information, next_phase. "
                        "intent_mode must be BROAD_ANALYSIS or DEEP_RESEARCH; "
                        "next_phase must be DISCOVERY, EXTRACTION, or SYNTHESIS."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Goal: {user_query}\nEnvironment summary:\n{env_context[:2400]}",
                },
            ]
            local_max_tokens = 256
        
        try:
            resp = self.llm.chat_completion(messages, max_tokens=local_max_tokens).content
            raw = resp.split("</think>")[-1].strip()
            decoder = json.JSONDecoder()
            for start, char in enumerate(raw):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(raw[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
            raise ValueError("local analyzer did not return a complete JSON object")
        except Exception as e:
            return {
                "intent_mode": "BROAD_ANALYSIS",
                "sandbox_evaluation": "",
                "entity_audit": {},
                "refined_query": user_query,
                "missing_information": "解析异常，请求对未知文件进行提取或检索", 
                "next_phase": "DISCOVERY"
            }
