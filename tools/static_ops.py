# RWKV-ECRA/tools/static_ops.py
import os
import json
from utils.file_reader import read_local_file
from config import DATA_PIPELINE
from tools.registry import ToolRegistry
from clients.slm_client import SLMClient
from prompts.slm_prompts import build_slm_preview_prompt

slm_client = SLMClient()

@ToolRegistry.register(
    name="search_local_file",
    phase="DISCOVERY",
    signature="""[Tool] search_local_file
- 功能: 基于关键词搜索本地工作区文件，返回文件路径及虚拟ID。传空字符串为全量查询。
- 参数: keyword (搜索关键词)"""
)
def search_local_file(keyword: str = "", path_to_id: dict = None, **kwargs) -> str:
    base_dir = DATA_PIPELINE["input_directory"]
    allowed_exts = DATA_PIPELINE["allowed_extensions"]
    found_info = []
    raw_paths = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("~") or file.startswith("."): continue
            if os.path.splitext(file)[1].lower() in allowed_exts and (not keyword or keyword.lower() in file.lower()):
                full_path = os.path.abspath(os.path.join(root, file))
                raw_paths.append(full_path)
                if path_to_id and full_path in path_to_id:
                    found_info.append(f"- ID: `{path_to_id[full_path]}` | 文件名: {file}")

    if not path_to_id:
        return json.dumps(raw_paths, ensure_ascii=False)
    if not found_info:
        return f"[系统状态] 未找到包含关键词 '{keyword}' 的文件。请尝试全量空词查询。"
    return f"[系统状态] 检索完成，找到 {len(found_info)} 个匹配文件:\n" + "\n".join(found_info)


@ToolRegistry.register(
    name="preview_document_content",
    phase="DISCOVERY",
    signature="""[Tool] preview_document_content
- 功能: [斥候试读] 让小模型读取本地文件片段，判别其主题和文件类型。用于排查本地工作区未知文件是否与任务相关，剔除干扰项。
- 参数: file_ids (待预览的文件ID数组)"""
)
def preview_document_content(file_paths: list = None, actual_file_ids: list = None, agent_state=None, tracker=None, working_memory: dict = None, **kwargs) -> str:
    if not file_paths or not actual_file_ids: return "未传入目标文件路径或ID。"

    from utils.chunker import _smart_truncate
    from utils.asset_manager import get_asset

    res = []
    prompts = []
    valid_files = []
    cached_assets = {}

    for idx, path in enumerate(file_paths):
        fid = actual_file_ids[idx]
        fname = os.path.basename(path)
        try:
            # 检查永久资产库，拦截试读
            asset = get_asset(path)
            if asset and os.path.exists(asset["asset_path"]):
                with open(asset["asset_path"], "r", encoding="utf-8") as f:
                    asset_content = f.read()
                cached_assets[fid] = {
                    "fname": fname,
                    "main_cat": asset["main_cat"],
                    "sub_cat": asset["sub_cat"],
                    "asset_path": asset["asset_path"],
                    "content": asset_content
                }
                print(f"⚡ [试读拦截] {fname} 命中永久知识库，瞬间直通加载，免试读算力消耗。")
                res.append(f"[{fname}] 命中本地资产库，免试读且已直通提取。")
                continue

            text = read_local_file(path)
            if not text.strip():
                raise ValueError("文件无实质内容或为空")

            # 采用 2400 Token 精确前缀截断，屏蔽结尾附录噪音
            preview_text, _ = _smart_truncate(text, max_tokens=2400)

            if len(preview_text) < len(text):
                preview_text += "\n\n...[后续内容较长，截取前2400 Token进行概览评估]..."

            prompts.append(build_slm_preview_prompt(preview_text))
            valid_files.append((fid, fname))
        except Exception as e:
            print(f"⚠️ [试读拦截] {fname} 无法阅读 (原因: {str(e)})，已自动剔除出并发队列。")
            res.append(f"文件 {fname} 读取失败: {str(e)}")
            if working_memory is not None:
                working_memory[f"Preview_{fid}"] = f"读取失败: {str(e)}"
                if agent_state:
                    agent_state.memory_catalog[f"Preview_{fid}"] = f"读取失败: {str(e)}"

    # ======== 写入缓存与早退逻辑 ========
    def _apply_caches_and_return():
        if working_memory is not None:
            from utils.chunker import get_token_count
            for c_fid, asset_info in cached_assets.items():
                working_memory[f"Category_{c_fid}"] = {"main": asset_info["main_cat"], "sub": asset_info["sub_cat"]}
                # 💥 核心修复：命中资产直接赋予 Summary 级别，不走 Preview，防止大模型误判抛弃！
                working_memory[f"Summary_{c_fid}"] = asset_info["content"]
                working_memory[f"Path_{c_fid}"] = asset_info["fname"]
                working_memory[f"AbsPath_{c_fid}"] = asset_info["asset_path"]
                if agent_state:
                    tok_count = get_token_count(asset_info["content"])
                    agent_state.memory_catalog[f"Summary_{c_fid}"] = f"状态: 已加载复用资产库直通提取 [{asset_info['main_cat']}/{asset_info['sub_cat']}] (~{tok_count} Tokens)"
        return "状态返回: 试读与资产对齐结束。请查阅缓存区决定下一步。\n" + "\n".join(res)

    if not prompts:
        return _apply_caches_and_return()

    print(f"[试读斥候]: 正在委派 SLM 全面抽样试读 {len(prompts)} 个未知文件...")

    task_id = agent_state.task_id if agent_state and getattr(agent_state, "task_id", "") else kwargs.get("task_id")
    slm_scheduler = kwargs.get("slm_scheduler")
    if slm_scheduler:
        slm_responses = slm_scheduler.submit(prompts, tracker=tracker, task_id=task_id)
    else:
        slm_responses = slm_client.batch_generate(prompts, tracker=tracker, task_id=task_id)

    files_to_categorize = []

    for i, out in enumerate(slm_responses):
        fid, fname = valid_files[i]
        clean_out = out.split("</think>")[-1].strip() if "</think>" in out else out.strip()
        catalog_desc = clean_out.replace('\n', ' | ')

        if agent_state:
            agent_state.memory_catalog[f"Preview_{fid}"] = f"试读结论: {catalog_desc}"
        if working_memory is not None:
            working_memory[f"Preview_{fid}"] = catalog_desc

        res.append(f"[{fname}] 试读完成，情报已登记。")
        files_to_categorize.append((fid, fname, clean_out))

    # ======== 🔴 核心逻辑 2：大模型批量对齐归类与全局合并 ========
    print(f"[试读斥候]: 正在调用 LLM 为 {len(files_to_categorize)} 个文件按批次提取本地资产分类...")
    from utils.asset_manager import get_all_categories
    from clients.llm_client import LLMClient
    from config import get_llm_concurrency
    import concurrent.futures
    import re

    existing_tree = get_all_categories()
    sub_cats_str = json.dumps(existing_tree, ensure_ascii=False)
    llm = LLMClient()

    def _batch_categorize(batch_data):
        sys_msg = """你是一个专业的知识库资产管理员。请仔细阅读以下多篇文章的概览，严格按照【知识领域】（例如：人工智能、生物医药、金融经济、材料科学等）对它们进行大类(main)和小类(sub)划分。
【红线约束】：
1. 严禁使用“学术论文”、“科研成果”、“技术文献”、“研究报告”等【文件体裁】作为大类！必须提炼其背后的【核心学科或业务领域】。
2. 尽量复用已有的分类体系，保持分类收敛。

请仅输出合法的 JSON，格式为：
{
  "fid_1": {"main": "大类名", "sub": "小类名"},
  "fid_2": {"main": "大类名", "sub": "小类名"}
}"""
        user_msg = f"现有分类树: {sub_cats_str}\n\n"
        for c_fid, c_fname, preview_text in batch_data:
            user_msg += f"--- 文件ID: {c_fid} | 文件名: {c_fname} ---\n{preview_text}\n\n"

        try:
            resp = llm.chat_completion([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]).content
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            return json.loads(match.group(0))
        except Exception:
            return {b[0]: {"main": "综合领域", "sub": "默认分类"} for b in batch_data}

    # 4 篇为一组合并预测
    batch_size = 4
    batches = [files_to_categorize[i:i+batch_size] for i in range(0, len(files_to_categorize), batch_size)]

    preliminary_categories = {}

    import contextvars
    with concurrent.futures.ThreadPoolExecutor(max_workers=get_llm_concurrency()) as executor:
        futures = []
        # ✨ 修改：在循环内部，为每个子任务单独拷贝上下文
        for b in batches:
            ctx = contextvars.copy_context()
            futures.append(executor.submit(ctx.run, _batch_categorize, b))

        for future in concurrent.futures.as_completed(futures):
            res_json = future.result()
            if isinstance(res_json, dict):
                preliminary_categories.update(res_json)

    # 兜底防丢
    for c_fid, _, _ in files_to_categorize:
        if c_fid not in preliminary_categories:
            preliminary_categories[c_fid] = {"main": "综合领域", "sub": "默认分类"}

    # 🔴 全局收敛洗牌：超过 4 篇文件时，启动同义词与异常分类合并
    if len(files_to_categorize) > 4:
        print(f"[试读斥候]: 文件较多 (共 {len(files_to_categorize)} 篇)，正在触发全局类别合并降噪...")
        unique_pairs = set()
        for cat in preliminary_categories.values():
            unique_pairs.add(f"{cat['main']}/{cat['sub']}")

        merge_sys_msg = """你是一个分类对齐专家。以下是并发产生的初步分类列表，可能存在语义重复、粒度不一，或者误用“学术论文”等体裁名作为知识领域的问题。
请合并同义词（如将“AI模型研究”合并入“人工智能”），规范化为统一的【知识领域】大类和小类。
请输出 JSON 映射字典，格式为：
{
  "旧大类/旧小类": {"main": "新大类", "sub": "新小类"}
}"""
        merge_user_msg = "待合并的初步类别列表：\n" + json.dumps(list(unique_pairs), ensure_ascii=False)

        try:
            merge_resp = llm.chat_completion([{"role": "system", "content": merge_sys_msg}, {"role": "user", "content": merge_user_msg}]).content
            match = re.search(r'\{.*\}', merge_resp, re.DOTALL)
            merge_mapping = json.loads(match.group(0))

            # 将映射好的新类写回
            for c_fid, cat in preliminary_categories.items():
                key = f"{cat['main']}/{cat['sub']}"
                if key in merge_mapping:
                    cat["main"] = merge_mapping[key].get("main", cat["main"])
                    cat["sub"] = merge_mapping[key].get("sub", cat["sub"])
        except Exception as e:
            print(f"[合并异常] 类别映射失败，降级使用初步分类: {e}")

    # --- 最终写入 Working Memory ---
    if working_memory is not None:
        for cat_fid, cat in preliminary_categories.items():
            working_memory[f"Category_{cat_fid}"] = {"main": cat["main"], "sub": cat["sub"]}

    return _apply_caches_and_return()


@ToolRegistry.register(
    name="verify_keyword_in_file",
    phase="ALL",
    signature="""[Tool] verify_keyword_in_file
- 功能: [精准验真] 物理级全文检索，验证特定文件中是否真实提及了关键词。用于打破关联幻觉（例如查验A文档是否提到B实体）。
- 参数: file_ids (目标文件虚拟ID数组), keywords (待验证的关键词字符串数组)"""
)
def verify_keyword_in_file(file_ids: list = None, keywords: list = None, agent_state=None, **kwargs) -> str:
    if not file_ids or not isinstance(file_ids, list):
        return "[系统状态] 执行失败：未传入有效的目标文件ID数组(file_ids)。"
    if not keywords or not isinstance(keywords, list):
        return "[系统状态] 执行失败：未提供需验证的关键词列表(keywords)。"

    import os
    import concurrent.futures
    from utils.file_reader import read_local_file

    results = []
    global_found_kws = set()

    # 提前统一小写化搜索词，避免循环中重复转换
    lower_kws = [(kw, str(kw).lower()) for kw in keywords]

    def _process_single_file(fid):
        if agent_state and hasattr(agent_state, 'id_to_path') and fid in agent_state.id_to_path:
            path = agent_state.id_to_path[fid]
        else:
            return (fid, None, f"📄 【{fid}】 验证跳过: ID无效或已被屏蔽。", set())

        fname = os.path.basename(path)
        try:
            # 瞬间读入内存
            text = read_local_file(path)
            # ⚡ 核心优化：一次性将全文转为小写，后续用高度优化的 C 底层方法暴力扫
            lower_text = text.lower()

            file_res = [f"📄 【{fname}】 验真结果:"]
            local_found = set()

            for kw, l_kw in lower_kws:
                # ⚡ 极其底层的 C 语言级别统计，比正则快几十倍
                count = lower_text.count(l_kw)

                if count == 0:
                    file_res.append(f"  - 关键词 '{kw}': 出现 0 次")
                else:
                    local_found.add(str(kw))
                    # 找到第一次出现的位置，提取上下文
                    first_idx = lower_text.find(l_kw)
                    start = max(0, first_idx - 30)
                    end = min(len(text), first_idx + len(kw) + 30)
                    context_snippet = text[start:end].replace('\n', ' ')

                    file_res.append(f"  - 关键词 '{kw}': 出现 {count} 次。片段: \"...{context_snippet}...\"")

            return (fid, fname, "\n".join(file_res), local_found)

        except Exception as e:
            return (fid, fname, f"📄 【{fname}】 验证读取失败: {str(e)}", set())

    # 虽然字符串匹配极快，但保留多线程并发读盘，应对未来挂载机械硬盘(HDD)或网络路径(NAS)的场景
    max_workers = min(len(file_ids), 32)
    res_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {executor.submit(_process_single_file, fid): fid for fid in file_ids}
        for future in concurrent.futures.as_completed(future_to_fid):
            fid = future_to_fid[future]
            try:
                f_id, fname, res_text, found_kws = future.result()
                res_dict[f_id] = res_text
                global_found_kws.update(found_kws)
            except Exception as e:
                res_dict[fid] = f"📄 【{fid}】 执行异常: {str(e)}"

    for fid in file_ids:
        if fid in res_dict:
            results.append(res_dict[fid])

    if agent_state and hasattr(agent_state, 'entity_audit') and agent_state.entity_audit:
        for kw in keywords:
            if str(kw) not in global_found_kws:
                for ent in list(agent_state.entity_audit.keys()):
                    if str(kw).lower() in ent.lower() or ent.lower() in str(kw).lower():
                        agent_state.entity_audit[ent] = f"确认无关 (在指定的 {len(file_ids)} 个文件中均未检出，已打破强制关联)"

    return "[系统状态] 全文验真执行完毕：\n\n" + "\n\n".join(results)
