# RWKV-ECRA/tools/web_search.py
import re
import uuid
import json
import concurrent.futures
from config import API_KEYS, SEARCH_CONFIG, DATA_PIPELINE, get_llm_provider, get_slm_concurrency
from tools.registry import ToolRegistry
from clients.slm_client import SLMClient
from clients.llm_client import LLMClient
from prompts.slm_prompts import build_slm_sequential_summary_prompt, build_slm_reduce_prompt
from utils.chunker import get_token_count, semantic_chunk_text

slm_client = SLMClient()

def _clean_slm_web_output(output: str) -> str:
    if not output:
        return ""
    return output.split("</think>")[-1].strip() if "</think>" in output else output.strip()

def _is_empty_web_fact(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if t in ["无", "None", "none", "NONE", "未找到", "无实质内容", "不相关"]:
        return True
    if t.startswith("未找到与") or t.startswith("未找到相关"):
        return True
    return False

def _assemble_web_search_facts(all_responses: list, prompt_metadata: list) -> tuple:
    sources = {}

    for idx, out in enumerate(all_responses):
        if idx >= len(prompt_metadata):
            break

        clean_out = _clean_slm_web_output(out)
        if _is_empty_web_fact(clean_out):
            continue

        meta = prompt_metadata[idx]
        ref_id = meta["ref_id"]
        source = sources.setdefault(ref_id, {
            "title": meta["title"],
            "url": meta["url"],
            "facts": []
        })

        source["facts"].append(clean_out)

    clean_parts = []
    structured_web_facts = []
    processed_refs = set()

    for ref_id, source in sources.items():
        if not source["facts"]:
            continue

        processed_refs.add(ref_id)
        structured_web_facts.append({
            "ref_id": ref_id,
            "title": source["title"],
            "url": source["url"],
            "content": "Tavily智能多源融合事实"
        })
        for fact in source["facts"]:
            clean_parts.append(f"【网络情报 ^[{ref_id}]^ 】 {source['title']}：{fact}")

    return clean_parts, structured_web_facts, processed_refs

def _generate_search_queries(query: str, active_goal: str) -> list:
    llm = LLMClient()
    sys_prompt = (
        "基于给定的核心实体和全局目标，生成 3 个用于网页搜索的极简短语。\n\n"
        "约束：\n"
        "1. 必须是直接的搜索词组（核心主体+事件/动作）。\n"
        "2. 禁止使用“维度影响”、“分析报告”、“企业经营”等抽象书面词汇。\n\n"
        "仅输出 JSON 字符串数组，例如：[\"词1\", \"词2\", \"词3\"]"
    )
    
    try:
        resp = llm.chat_completion([
            {"role": "system", "content": sys_prompt}, 
            {"role": "user", "content": f"核心实体: {query}\n全局目标: {active_goal}"}
        ]).content
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        if match:
            q_list = json.loads(match.group(0))
            if isinstance(q_list, list) and len(q_list) > 0:
                return [str(q)[:30] for q in q_list[:3]]
    except Exception:
        pass
    return [f"{query}", f"{query} 影响", f"{query} 最新消息"]

@ToolRegistry.register(
    name="execute_web_search",
    # Keep the pre-agent multi-query route callable for compatibility, while
    # hiding it from the current RWKV model tool catalog.
    phase="LEGACY",
    plugin="web.legacy",
    capabilities=("url_discovery", "web_search"),
    retrieval_role="discovery",
    signature="""[Tool] execute_web_search
- 功能: 联网检索外部事实。底层集成了 AI 自动搜索词扩展与多源抓取，用于调查缺乏本地文件支撑的特定实体或逻辑。
- 参数: query (精简的实体名称或搜素短语)"""
)
def execute_web_search(query: str, working_memory: dict = None, tracker=None, agent_state=None, **kwargs) -> str:
    provider = get_llm_provider()
    safe_query = re.sub(r'\W+', '_', query)[:20]
    
    active_goal = agent_state.refined_query if (agent_state and hasattr(agent_state, 'refined_query') and agent_state.refined_query) else kwargs.get("original_goal", "当前主线任务")

    if working_memory is not None and "__web_structured_facts__" not in working_memory:
        working_memory["__web_structured_facts__"] = []

    if provider == "baidu":
        print(f"[Web Search]: 🚀 启动文心原生关联检索 -> 实体: '{query}' ...")
        llm = LLMClient()
        prompt_msg = [
            {"role": "system", "content": "执行深度联网检索。基于原生搜索结果提取与用户目标相关的客观事实，保留可溯源信息。"},
            {"role": "user", "content": f"全局调查目标: {active_goal}\n请全面调查: {query} 是什么？它最近有什么动态？它与我们的调查目标有哪些事实性信息可以参考？"}
        ]
        
        try:
            response_msg = llm.chat_completion(prompt_msg, enable_native_search=True)
            response_text = response_msg.content
            search_results = getattr(response_msg, "search_results", [])
            
            structured_facts = []
            sorted_results = sorted(search_results, key=lambda x: int(x.get("index", 0)) if str(x.get("index", 0)).isdigit() else 0, reverse=True)
            
            for res in sorted_results:
                idx = res.get("index")
                title = res.get("title", f"参考资料_{idx}")
                web_ref_id = f"WEB_REF_BD_{uuid.uuid4().hex[:6]}"
                
                response_text = response_text.replace(f"^[{idx}]^", f"^[{web_ref_id}]^").replace(f"[{idx}]", f"^[{web_ref_id}]^")
                structured_facts.append({"ref_id": web_ref_id, "title": title, "url": res.get("url", ""), "content": "系统原生抓取"})
            
            clean_res = f"来源于《原生搜索引擎》:\n{response_text}"
            
            if working_memory is not None:
                working_memory[f"WebFact_{safe_query}"] = clean_res
                working_memory["__web_structured_facts__"].extend(structured_facts)
                if agent_state:
                    agent_state.memory_catalog[f"WebFact_{safe_query}"] = "状态: 已提炼完成，直接可用，切勿再次提取"
                    status_str = "确认相关 (已完成提炼)"
                    for ent in list(agent_state.entity_audit.keys()):
                        if ent.lower() in query.lower() or query.lower() in ent.lower():
                            agent_state.entity_audit[ent] = status_str
                    
            return f"[系统状态] 实体 '{query}' 联网检索完成。数据已挂载至记忆区并更正实体状态，切勿再次提取。请推进下一步或进入 SYNTHESIS 阶段。"
        except Exception as e:
            return f"[系统异常] 联网检索报错: {str(e)}"
            
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=API_KEYS.get("tavily", ""))
        
        queries_to_run = _generate_search_queries(query, active_goal)
        print(f"[Web Search]: 🧠 AI 动态扩展多维检索词: {queries_to_run}")
        
        s_depth = SEARCH_CONFIG.get("search_depth", "basic")
        s_max_res = SEARCH_CONFIG.get("max_results", 4)
        s_time_range = SEARCH_CONFIG.get("time_range", "month")

        def run_tavily_search(q: str, topic: str):
            try:
                return client.search(
                    query=q, 
                    search_depth=s_depth, 
                    max_results=s_max_res, 
                    include_raw_content=False, 
                    time_range=s_time_range, 
                    search_topic=topic
                ).get("results", [])
            except: return []

        print(f"[Web Search]: 🚀 正在向 Tavily 并发发射检索探针 (深度:{s_depth}, 最大结果:{s_max_res})...")
        all_raw_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for q in queries_to_run:
                futures.append(executor.submit(run_tavily_search, q, "finance"))
                futures.append(executor.submit(run_tavily_search, q, "news"))
                futures.append(executor.submit(run_tavily_search, q, "general"))
            for f in concurrent.futures.as_completed(futures):
                all_raw_results.extend(f.result())
            
        seen_urls = set()
        unique_results = []
        for r in all_raw_results:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)
        
        if not unique_results: return f"[系统状态] 搜索实体 '{query}' 无有效结果返回。"

        prompts = []
        prompt_metadata = []
        max_chunk = DATA_PIPELINE.get("max_chunk_tokens", 800)
        overlap_ratio = DATA_PIPELINE.get("overlap_ratio", 0.05)
        
        clean_parts = []
        structured_web_facts = []
        processed_refs = set()
        short_pages_count = 0
        
        # 1. 预处理分离短文本与长文本
        for r in unique_results:
            title = r.get("title", "未命名网页").strip()
            web_ref_id = f"WEB_REF_TV_{uuid.uuid4().hex[:6]}"
            content = r.get("content", "").strip()
            url = r.get("url", "")
            
            if not content: continue
            
            # 🟢 核心优化：如果 Token < 500，直接免压缩编入事实库
            if get_token_count(content) < 500:
                short_pages_count += 1
                processed_refs.add(web_ref_id)
                structured_web_facts.append({
                    "ref_id": web_ref_id,
                    "title": title,
                    "url": url,
                    "content": "短篇网页原生抓取 (免压缩直通)"
                })
                clean_parts.append(f"【网络情报 ^[{web_ref_id}]^ 】 {title}：\n{content}")
                continue

            # 超过 500 Token，按需切割并推入 Map 队列
            chunks = semantic_chunk_text(content, max_tokens=max_chunk, overlap_ratio=overlap_ratio)
            for idx, chunk in enumerate(chunks):
                chunk_with_title = f"【标题】: {title}\n【片段】: {chunk}"
                map_prompt = build_slm_sequential_summary_prompt(
                    chunk_str=chunk_with_title,
                    current_idx=idx + 1,
                    total_chunks=len(chunks),
                    focus=query,
                    is_english=False
                )
                prompts.append(map_prompt)
                prompt_metadata.append({"ref_id": web_ref_id, "title": title, "url": url})
                
        if short_pages_count > 0:
            print(f"[Web Search]: ⚡ 命中 {short_pages_count} 个短篇网页 (Token < 500)，已安全免压缩直通。")
            
        mapped_responses = []
        task_id = agent_state.task_id if agent_state and getattr(agent_state, "task_id", "") else kwargs.get("task_id")
        slm_scheduler = kwargs.get("slm_scheduler")
        concurrency_limit = get_slm_concurrency()
        
        # 2. 长文本 Map 阶段
        if prompts:
            print(f"[Web Search]: 🛡️ 启动 SLM 全文关联提炼 (共 {len(prompts)} 个分块)...")
            for i in range(0, len(prompts), concurrency_limit):
                prompt_batch = prompts[i:i+concurrency_limit]
                if slm_scheduler:
                    mapped_responses.extend(slm_scheduler.submit(prompt_batch, tracker=tracker, task_id=task_id))
                else:
                    mapped_responses.extend(slm_client.batch_generate(prompt_batch, tracker=tracker, task_id=task_id))
            
        retained_facts = []
        for idx, out in enumerate(mapped_responses):
            if idx >= len(prompt_metadata): break
            clean_out = _clean_slm_web_output(out)
            if _is_empty_web_fact(clean_out): continue
            meta = prompt_metadata[idx]
            retained_facts.append({
                "ref_id": meta["ref_id"],
                "title": meta["title"],
                "url": meta["url"],
                "fact": clean_out
            })

        # 3. 长文本 Reduce 阶段
        reduce_tasks = []
        reduce_metadata = []
        sources_map = {}
        for item in retained_facts:
            ref_id = item["ref_id"]
            if ref_id not in sources_map:
                sources_map[ref_id] = {"title": item["title"], "url": item["url"], "facts": []}
            sources_map[ref_id]["facts"].append(item["fact"])

        reduce_group_size = DATA_PIPELINE.get("reduce_group_size", 4)
        actual_reduce = DATA_PIPELINE.get("reduce_rule", "保持原意压缩，去重并合并同类逻辑，绝对保留事实性数据和原始结论")
        
        for ref_id, source in sources_map.items():
            facts = source["facts"]
            if not facts: continue
            for i in range(0, len(facts), reduce_group_size):
                batch_facts = facts[i:i+reduce_group_size]
                b_text = "\n\n".join([f"片段{j+1}:\n{b}" for j, b in enumerate(batch_facts)])
                prompt = build_slm_reduce_prompt(b_text, actual_reduce, 1, 1, False)
                reduce_tasks.append(prompt)
                reduce_metadata.append({
                    "ref_id": ref_id,
                    "title": source["title"],
                    "url": source["url"]
                })

        all_reduce_responses = []
        if reduce_tasks:
            print(f"[Web Search]: 🛡️ 启动 SLM REDUCE 合并去重 (共 {len(reduce_tasks)} 个分块)...")
            for i in range(0, len(reduce_tasks), concurrency_limit):
                prompt_batch = reduce_tasks[i:i+concurrency_limit]
                if slm_scheduler:
                    all_reduce_responses.extend(slm_scheduler.submit(prompt_batch, tracker=tracker, task_id=task_id))
                else:
                    all_reduce_responses.extend(slm_client.batch_generate(prompt_batch, tracker=tracker, task_id=task_id))

        # 4. 把经过提炼的长文本和原本直通的短文本混合拼接
        clean_parts_r, structured_web_facts_r, processed_refs_r = _assemble_web_search_facts(all_reduce_responses, reduce_metadata)
        clean_parts.extend(clean_parts_r)
        structured_web_facts.extend(structured_web_facts_r)
        processed_refs.update(processed_refs_r)

        status_str = "确认相关 (已完成提炼)"
            
        if not clean_parts: return "[系统状态] 未能从有效网页中提取到客观事实。"
            
        if working_memory is not None:
            working_memory[f"WebFact_{safe_query}"] = "\n\n".join(clean_parts)
            working_memory["__web_structured_facts__"].extend(structured_web_facts)
            if agent_state:
                agent_state.memory_catalog[f"WebFact_{safe_query}"] = f"状态: 已绑定 {len(processed_refs)} 个网页源，严禁再次提取"
                for ent in list(agent_state.entity_audit.keys()):
                    if ent.lower() in query.lower() or query.lower() in ent.lower():
                        agent_state.entity_audit[ent] = status_str
            
        return f"[系统状态] 实体 '{query}' 联网多源检索完成。已载入记忆区并更正状态，切勿再次提取。请推进下一步或进入 SYNTHESIS 阶段。"
        
    except Exception as e:
        return f"[系统异常] 联网搜索报错: {str(e)}"
