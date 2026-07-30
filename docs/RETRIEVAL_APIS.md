# 检索 API 插件

项目把检索源统一为两类工具：`discovery` 负责发现候选资源，`evidence` 负责打开一个确定的资源并返回可引用证据。模型负责选择工具和参数；控制器只负责校验协议、执行、去重、切分和汇总。

当前已注册的精确检索源：

| 插件 | 发现工具 | 证据工具 | 适用场景 | API Key |
| --- | --- | --- | --- | --- |
| Crossref | `search_crossref` | `fetch_crossref_record` | 论文 DOI、作者、期刊、年份、出版信息 | 不需要 |
| GitHub REST | `search_github_rest` | `fetch_github_rest` | 仓库、代码、默认分支、语言、简介、文件内容 | 公共资源不需要；有 `GITHUB_TOKEN`/`GH_TOKEN` 时提高限额 |
| MediaWiki | `search_mediawiki` | `fetch_mediawiki_page` | Wikipedia/Wikidata/Commons 页面、模板和结构化条目 | 不需要 |
| Bing / 无 Key 搜索 | `search_web_keyless` | `fetch_web_url` | 广泛发现网页 | 不需要，但依赖当前网络代理和搜索结果质量 |
| Tavily | `search_web_tavily` | `fetch_web_url` | 可选的通用搜索 | 需要 `TAVILY_API_KEY` |

每个证据工具都必须返回一个确定页面或结构化资源，随后统一进入：

```text
evidence page → page_evidence chunks (默认 2048 tokens) → parallel candidates → RWKV 汇总 → completion judge
```

新增检索源时，原则上只需：

1. 在 `tools/` 新增 discovery/evidence 实现，并声明 `role`、参数 schema 和返回格式。
2. 在 `tools/builtin.py` 注册工具。
3. 为 discovery、evidence 和网络异常补充单元测试。
4. 将能力描述加入 planner catalog，让模型能够自主选择；不要在 orchestrator 中写死 URL 或供应商路由。

网络请求统一经过 `utils/network_fetch.py`。WSL 会自动读取 Windows 的代理设置，也可以用 `RWKV_ECRA_HTTP_PROXY`、`RWKV_ECRA_HTTPS_PROXY` 显式覆盖；任何 Key 都只从环境变量读取，不写入仓库。

真实回归输入见 `data/evaluation/manual_real_queries.json`，最终结果见 `data/evaluation/manual_real_results_api_six_final.json`。完整的逐阶段事件仍保存在对应任务的 `data/output/<task_id>/events.jsonl` 中。
## Model-facing contract

The real-time RWKV loop exposes one retrieval capability: `web_search(query)`.
`finish_task` is a terminal control action, not a retrieval provider. Tavily,
keyless Bing/HTML search, GitHub REST, Crossref, MediaWiki, URL fetching,
Markdown conversion, chunking and parallel-candidate extraction are internal
adapters executed by the bounded `web_search` transaction. They remain
registered for backend compatibility and direct tests, but they are not added
to the Planner's model-visible catalog.
