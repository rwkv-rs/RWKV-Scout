import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Braces,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Combine,
  Database,
  FileSearch,
  FileText,
  GitBranch,
  Globe2,
  Layers3,
  MessageSquare,
  Route,
  ScanSearch,
  Search,
  Split,
  Wrench,
} from "lucide-react";

import { cn } from "@/lib/utils";

const EVENT_DEFS = {
  user_input: { label: "用户输入", group: "输入", icon: MessageSquare, tone: "neutral" },
  run_started: { label: "任务启动", group: "输入", icon: Activity, tone: "neutral" },
  task_plan: { label: "任务计划", group: "规划", icon: Route, tone: "plan" },
  task_replan: { label: "重新拆分计划", group: "规划", icon: GitBranch, tone: "plan" },
  model_tool_decision: { label: "RWKV 工具决策", group: "模型决策", icon: Bot, tone: "model" },
  query_candidates: { label: "检索查询候选", group: "检索", icon: Search, tone: "search" },
  tool_call: { label: "工具调用", group: "检索", icon: Wrench, tone: "search" },
  tool_result: { label: "工具结果", group: "检索", icon: CheckCircle2, tone: "result" },
  progress: { label: "运行进度", group: "输入", icon: Activity, tone: "neutral" },
  page_fetch: { label: "网页抓取", group: "网页", icon: Globe2, tone: "web" },
  page_extract: { label: "网页正文提取", group: "网页", icon: FileText, tone: "web" },
  content_extract: { label: "正文提取", group: "网页", icon: FileText, tone: "web" },
  page_chunk: { label: "网页 Chunk", group: "网页", icon: Split, tone: "chunk" },
  page_chunk_candidate: { label: "Chunk 候选", group: "网页", icon: ScanSearch, tone: "chunk" },
  page_candidate_merge: { label: "网页证据聚合", group: "网页", icon: Combine, tone: "result" },
  ranking: { label: "候选排序", group: "聚合", icon: Layers3, tone: "aggregate" },
  context_build: { label: "模型上下文构建", group: "聚合", icon: Database, tone: "aggregate" },
  synthesis_start: { label: "开始总结", group: "总结", icon: Braces, tone: "model" },
  synthesis: { label: "RWKV 总结输出", group: "总结", icon: Bot, tone: "model" },
  completion_judgement: { label: "完成性判断", group: "判断", icon: CheckCircle2, tone: "judge" },
  completion_pending: { label: "证据仍不完整", group: "判断", icon: CircleDot, tone: "warning" },
  citation_validation: { label: "引用校验", group: "判断", icon: FileSearch, tone: "judge" },
  risk_validation: { label: "风险校验", group: "判断", icon: CheckCircle2, tone: "judge" },
  provider_error: { label: "提供方错误", group: "错误", icon: AlertTriangle, tone: "error" },
  error: { label: "执行错误", group: "错误", icon: AlertTriangle, tone: "error" },
  final: { label: "最终结果", group: "结束", icon: CheckCircle2, tone: "final" },
};

const GROUPS = ["全部", "输入", "规划", "模型决策", "检索", "网页", "聚合", "总结", "判断", "错误", "结束"];

function definitionFor(event) {
  return EVENT_DEFS[event?.type] || { label: event?.type || "未知事件", group: "其他", icon: Activity, tone: "neutral" };
}

function parseMaybeJson(value) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string" || !value.trim()) return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function pretty(value) {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function compact(value, length = 180) {
  const text = typeof value === "string" ? value : pretty(value);
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function jsonBlock(label, value, options = {}) {
  const { open = false, className = "" } = options;
  if (value === undefined || value === null || value === "") return null;
  return (
    <details className={cn("trace-json-block", className)} open={open}>
      <summary>
        <ChevronDown className="size-3" aria-hidden="true" />
        <span>{label}</span>
      </summary>
      <pre>{pretty(value)}</pre>
    </details>
  );
}

function Field({ label, value, mono = false, wide = false }) {
  if (value === undefined || value === null || value === "") return null;
  const renderedValue = value && typeof value === "object" && value.$$typeof
    ? value
    : typeof value === "string"
      ? value
      : pretty(value);
  return (
    <div className={cn("trace-field", wide && "trace-field-wide")}>
      <span>{label}</span>
      <strong className={mono ? "font-mono" : ""}>{renderedValue}</strong>
    </div>
  );
}

function StatusPill({ value }) {
  if (value === undefined || value === null || value === "") return null;
  const normalized = String(value).toLowerCase();
  const tone = normalized.includes("error") || normalized.includes("fail") || normalized.includes("incomplete")
    ? "error"
    : normalized.includes("ok") || normalized.includes("complete")
      ? "success"
      : "neutral";
  return <span className={cn("trace-status-pill", `is-${tone}`)}>{String(value)}</span>;
}

function PlanView({ plan }) {
  if (!plan || typeof plan !== "object") return jsonBlock("计划原始 JSON", plan);
  return (
    <div className="trace-detail-stack">
      <div className="trace-fields">
        <Field label="目标" value={plan.goal} wide />
        <Field label="完成规则" value={plan.completion_rule} wide />
      </div>
      {Array.isArray(plan.atomic_points) ? (
        <div className="trace-points">
          {plan.atomic_points.map((point, index) => (
            <div className="trace-point" key={`${point.id || "point"}-${index}`}>
              <div className="trace-point-head">
                <span className="trace-point-id">{point.id || `P${index + 1}`}</span>
                <StatusPill value={point.status || "pending"} />
              </div>
              <p>{point.objective || "未提供目标"}</p>
              {Array.isArray(point.evidence_needed) && point.evidence_needed.length ? (
                <div className="trace-chip-row">
                  {point.evidence_needed.map((item, itemIndex) => (
                    <span className="trace-chip" key={`${item}-${itemIndex}`}>{item}</span>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {jsonBlock("计划原始 JSON", plan)}
    </div>
  );
}

function ToolResultView({ event }) {
  const result = parseMaybeJson(event.result);
  const rows = Array.isArray(result?.results) ? result.results : [];
  return (
    <div className="trace-detail-stack">
      <div className="trace-fields">
        <Field label="状态" value={<StatusPill value={result?.status || event.execution_status} />} />
        <Field label="提供方" value={result?.provider} />
        <Field label="查询" value={result?.query || result?.provider_query} wide />
        <Field label="结果数" value={result?.count ?? rows.length} mono />
        <Field label="真实网络" value={result?.real_network === undefined ? undefined : String(result.real_network)} />
      </div>
      {Array.isArray(result?.provider_errors) && result.provider_errors.length ? (
        <div className="trace-alert">{result.provider_errors.join("\n")}</div>
      ) : null}
      {rows.length ? (
        <div className="trace-result-list">
          <div className="trace-subtitle">返回的候选网页（这里只是发现结果，正文需再抓取）</div>
          {rows.map((row, index) => (
            <div className="trace-result-row" key={`${row.url || "result"}-${index}`}>
              <span className="trace-result-index">{index + 1}</span>
              <div className="min-w-0">
                <strong>{row.title || row.url || "未命名页面"}</strong>
                {row.url ? <a href={row.url} target="_blank" rel="noreferrer">{row.url}</a> : null}
                {row.snippet ? <p>{row.snippet}</p> : null}
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {jsonBlock("工具原始返回", result || event.result)}
    </div>
  );
}

function EventDetail({ event }) {
  const type = event.type;
  const data = event.data || {};
  const candidate = event.candidate || {};
  const pageData = type === "page_candidate_merge" ? data : null;

  if (type === "task_plan" || type === "task_replan") return <PlanView plan={data} />;

  if (type === "model_tool_decision") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="阶段" value={event.phase} />
          <Field label="动作" value={event.action} mono />
          <Field label="任务点" value={event.task_point_id || "模型未提供"} mono />
          <Field label="解析错误" value={event.planner_error} />
        </div>
        {jsonBlock("模型选择的工具参数", event.args, { open: true })}
        {jsonBlock("RWKV 原始决策输出", event.raw_model_output)}
      </div>
    );
  }

  if (type === "tool_call") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="阶段" value={event.phase} />
          <Field label="动作" value={event.action} mono />
          <Field label="查询轮次" value={event.round} mono />
        </div>
        {jsonBlock("发送给工具的参数", event.args, { open: true })}
      </div>
    );
  }

  if (type === "tool_result") return <ToolResultView event={event} />;

  if (type === "progress") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="步骤" value={event.step} mono />
          <Field label="阶段" value={event.phase} />
        </div>
        <div className="trace-answer-box"><div className="trace-subtitle">系统进度</div><p>{event.message || event.content}</p></div>
        {jsonBlock("进度事件完整 JSON", event)}
      </div>
    );
  }

  if (type === "query_candidates") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="来源" value={event.source} />
          <Field label="动作" value={event.action} mono />
          <Field label="耗时" value={event.duration_ms ? `${event.duration_ms} ms` : undefined} mono />
        </div>
        <div className="trace-chip-row">
          {(event.queries || []).map((query, index) => <span className="trace-chip" key={`${query}-${index}`}>{query}</span>)}
        </div>
        {jsonBlock("查询模型原始输出", event.raw_model_output)}
      </div>
    );
  }

  if (type === "page_fetch" || type === "page_extract" || type === "content_extract") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="网页 URL" value={event.url} wide />
          <Field label="状态" value={<StatusPill value={event.status} />} />
          <Field label="阶段" value={event.phase} />
          <Field label="动作" value={event.action} mono />
          <Field label="正文字符数" value={event.body_chars ?? event.excerpt_chars} mono />
          <Field label="耗时" value={event.duration_ms ? `${event.duration_ms} ms` : undefined} mono />
        </div>
        {jsonBlock("网页阶段完整 JSON", event, { open: true })}
      </div>
    );
  }

  if (type === "page_chunk") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="网页 URL" value={event.url} wide />
          <Field label="证据查询上下文" value={event.evidence_query} wide />
          <Field label="Chunk" value={`${event.chunk_id || "—"} / ${Number(event.chunk_index ?? 0) + 1}`} mono />
          <Field label="Token 数" value={event.chunk_tokens} mono />
          <Field label="字符数" value={event.chunk_chars} mono />
        </div>
        {jsonBlock("送入 Chunk 候选模型的网页正文", event.chunk_text, { open: true, className: "trace-long-text" })}
      </div>
    );
  }

  if (type === "page_chunk_candidate") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="网页 URL" value={event.url} wide />
          <Field label="证据查询上下文" value={event.evidence_query} wide />
          <Field label="Chunk" value={event.chunk_id} mono />
          <Field label="Prompt 字符数" value={event.prompt_chars} mono />
          <Field label="模型输出字符数" value={event.model_output_chars} mono />
          <Field label="Finish reason" value={event.finish_reason} mono />
          <Field label="重试次数" value={event.retry_count} mono />
        </div>
        <div className="trace-candidate-box">
          <div className="trace-subtitle">解析后的候选</div>
          <div className="trace-fields">
            <Field label="supported" value={String(candidate.supported)} mono />
            <Field label="事实条数" value={candidate.facts?.length || 0} mono />
            <Field label="事实" value={candidate.facts} wide />
            <Field label="原文引用" value={candidate.quote} wide />
          </div>
        </div>
        {jsonBlock("发送给候选模型的完整 Prompt", event.prompt, { open: true, className: "trace-long-text" })}
        {jsonBlock("RWKV Chunk 原始输出", event.model_output, { open: true })}
        {jsonBlock("候选完整 JSON", candidate)}
      </div>
    );
  }

  if (type === "page_candidate_merge") {
    const parallel = pageData?.parallel_candidate || {};
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="网页 URL" value={event.url || pageData?.url} wide />
          <Field label="正文字符数" value={pageData?.page_chars} mono />
          <Field label="Chunk 数" value={pageData?.chunk_count} mono />
          <Field label="Chunk 窗口" value={pageData?.chunk_window_tokens ? `${pageData.chunk_window_tokens} tokens` : undefined} mono />
          <Field label="聚合候选数" value={pageData?.candidate_count} mono />
          <Field label="并发 Worker" value={parallel.worker_count} mono />
          <Field label="调用次数" value={parallel.attempted_calls} mono />
          <Field label="重试次数" value={parallel.retry_calls} mono />
          <Field label="并行耗时" value={parallel.wall_time_ms ? `${parallel.wall_time_ms} ms` : undefined} mono />
        </div>
        <div className="trace-facts-box">
          <div className="trace-subtitle">进入下一轮 planner 的紧凑证据</div>
          <pre>{event.compact_facts || "没有支持性事实"}</pre>
        </div>
        {jsonBlock("并行候选元数据", parallel)}
        {jsonBlock("聚合后的候选 JSON", event.candidates)}
      </div>
    );
  }

  if (type === "ranking") {
    const ranking = data || {};
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="排序方法" value={ranking.method} mono />
          <Field label="排序阶段" value={ranking.stage} />
          <Field label="输出数量" value={ranking.output_count} mono />
        </div>
        {jsonBlock("排序后的候选来源", ranking.results, { open: true })}
        {jsonBlock("排序事件完整 JSON", event)}
      </div>
    );
  }

  if (type === "context_build") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="上下文 Token" value={data.context_stats?.context_tokens} mono />
          <Field label="上下文字符" value={data.context_stats?.context_chars} mono />
          <Field label="选中证据字符" value={data.context_stats?.selected_chars} mono />
          <Field label="来源字符" value={data.context_stats?.source_chars} mono />
          <Field label="上下文 Chunk 数" value={data.context_stats?.chunk_count} mono />
          <Field label="是否截断" value={String(data.context_stats?.context_truncated)} mono />
        </div>
        {jsonBlock("最终送入总结模型的上下文", data.context_text, { open: true, className: "trace-long-text" })}
        {jsonBlock("被选中的证据", data.selected_evidence)}
        {jsonBlock("上下文统计", data.context_stats)}
      </div>
    );
  }

  if (type === "synthesis_start" || type === "synthesis") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="模式" value={event.mode} mono />
          <Field label="证据数" value={event.evidence_count} mono />
          <Field label="耗时" value={event.duration_ms ? `${event.duration_ms} ms` : undefined} mono />
        </div>
        {event.content ? <div className="trace-answer-box"><div className="trace-subtitle">RWKV 当前总结</div><p>{event.content}</p></div> : null}
        {jsonBlock("总结模型 Prompt", event.prompt)}
        {jsonBlock("RWKV 总结原始输出", event.model_output, { open: true })}
        {jsonBlock("修复输出", event.repair_output)}
        {jsonBlock("总结使用的证据", event.selected_evidence)}
        {jsonBlock("总结上下文", event.context_text)}
      </div>
    );
  }

  if (type === "completion_judgement" || type === "completion_pending") {
    const judgement = type === "completion_judgement" ? data : data;
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="判断状态" value={<StatusPill value={judgement.status} />} />
          <Field label="缺少任务点" value={judgement.missing_point_ids} mono />
          <Field label="下一步关注" value={judgement.next_focus} wide />
          <Field label="判断原因" value={judgement.reason} wide />
        </div>
        {jsonBlock("完成性判断 JSON", judgement, { open: true })}
      </div>
    );
  }

  if (type === "citation_validation" || type === "risk_validation") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="校验器" value={data.validator_version} mono />
          <Field label="是否通过" value={<StatusPill value={data.valid === false ? "invalid" : "valid"} />} />
          <Field label="高风险" value={data.high_risk === undefined ? undefined : String(data.high_risk)} mono />
          <Field label="校验数量" value={data.total ?? data.checked ?? data.rows?.length} mono />
          <Field label="支持数量" value={data.supported} mono />
        </div>
        {jsonBlock("校验结果完整 JSON", data, { open: true })}
      </div>
    );
  }

  if (type === "final") {
    return (
      <div className="trace-detail-stack">
        <div className="trace-fields">
          <Field label="最终状态" value={<StatusPill value={event.status} />} />
          <Field label="模式" value={event.mode} mono />
          <Field label="动作" value={event.action} mono />
        </div>
        <div className="trace-answer-box"><div className="trace-subtitle">最终答案</div><p>{event.content || "没有最终答案"}</p></div>
        {jsonBlock("最终事件完整 JSON", event)}
      </div>
    );
  }

  return jsonBlock("事件完整 JSON", event, { open: type === "error" || type === "provider_error" });
}

function summaryFor(event) {
  const result = parseMaybeJson(event.result);
  if (event.type === "model_tool_decision") return `${event.action || "未选择工具"}${event.task_point_id ? ` · ${event.task_point_id}` : ""}`;
  if (event.type === "tool_call") return `${event.action || "工具"} · ${compact(event.args, 140)}`;
  if (event.type === "tool_result") return `${result?.provider || event.action || "工具"} · ${result?.status || event.execution_status || "返回"} · ${result?.count ?? result?.results?.length ?? 0} 条`;
  if (event.type === "page_chunk") return `${event.chunk_id || "chunk"} · ${event.chunk_tokens || 0} tokens · ${event.chunk_chars || 0} chars`;
  if (event.type === "page_chunk_candidate") return `${event.chunk_id || "chunk"} · supported=${String(event.candidate?.supported)} · ${event.finish_reason || "unknown"}`;
  if (event.type === "page_candidate_merge") return `${event.url || "网页"} · ${event.data?.candidate_count || 0} 个候选 · ${event.data?.chunk_count || 0} chunks`;
  if (event.type === "context_build") return `${event.data?.context_stats?.context_tokens || 0} tokens · ${event.data?.selected_evidence?.length || 0} 个证据`;
  if (event.type === "completion_judgement") return `${event.data?.status || "unknown"} · 缺少 ${(event.data?.missing_point_ids || []).join("、") || "无"}`;
  if (event.type === "final") return `${event.status || "结束"} · ${compact(event.content, 180)}`;
  if (event.type === "error" || event.type === "provider_error") return event.error || event.message || event.error_class || "执行错误";
  if (event.type === "task_plan" || event.type === "task_replan") return `${event.data?.atomic_points?.length || 0} 个原子任务点`;
  return compact(event.content || event.message || event.data || event.result || "状态已更新", 180);
}

function isImportant(event) {
  return ["task_plan", "task_replan", "model_tool_decision", "page_candidate_merge", "context_build", "synthesis", "completion_judgement", "final"].includes(event.type);
}

function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className="trace-metric">
      <span>{label}</span>
      <strong className={tone === "error" ? "text-rose-700" : tone === "success" ? "text-emerald-700" : ""}>{value}</strong>
    </div>
  );
}

export default function ExecutionEventFeed({ events = [] }) {
  const [query, setQuery] = useState("");
  const [group, setGroup] = useState("全部");
  const [expanded, setExpanded] = useState(() => new Set(events.filter(isImportant).map((event) => event.seq)));

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return events.filter((event) => {
      const definition = definitionFor(event);
      const matchesGroup = group === "全部" || definition.group === group;
      const matchesQuery = !normalized || JSON.stringify(event).toLowerCase().includes(normalized);
      return matchesGroup && matchesQuery;
    });
  }, [events, group, query]);

  const counts = useMemo(() => {
    const result = { pages: 0, chunks: 0, candidates: 0, contexts: 0, judgements: 0, errors: 0 };
    events.forEach((event) => {
      if (event.type === "page_candidate_merge") result.pages += 1;
      if (event.type === "page_chunk") result.chunks += 1;
      if (event.type === "page_chunk_candidate") result.candidates += 1;
      if (event.type === "context_build") result.contexts += 1;
      if (event.type === "completion_judgement") result.judgements += 1;
      if (event.type === "error" || event.type === "provider_error") result.errors += 1;
    });
    return result;
  }, [events]);

  function toggle(seq) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(seq)) next.delete(seq); else next.add(seq);
      return next;
    });
  }

  function expandAll() {
    setExpanded(new Set(filtered.map((event) => event.seq)));
  }

  function collapseAll() {
    setExpanded(new Set());
  }

  return (
    <section className="trace-inspector" aria-label="详细可审计执行上下文">
      <header className="trace-inspector-header">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Globe2 className="size-4 text-primary" aria-hidden="true" />
            <h2 className="text-sm font-semibold">详细执行上下文</h2>
            <span className="trace-event-count">{events.length} events</span>
          </div>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
            从任务计划、工具选择、网页正文、Chunk 候选、证据聚合、模型上下文到最终判断的完整链路。点击任意事件查看本阶段实际输入和输出。
          </p>
        </div>
        <div className="trace-inspector-actions">
          <button type="button" onClick={expandAll}>展开当前</button>
          <button type="button" onClick={collapseAll}>全部收起</button>
        </div>
      </header>

      <div className="trace-metric-grid">
        <Metric label="网页正文" value={counts.pages} />
        <Metric label="Chunk" value={counts.chunks} />
        <Metric label="候选" value={counts.candidates} />
        <Metric label="上下文构建" value={counts.contexts} />
        <Metric label="完成判断" value={counts.judgements} />
        <Metric label="错误" value={counts.errors} tone={counts.errors ? "error" : "success"} />
      </div>

      <div className="trace-toolbar">
        <div className="trace-filter-row" role="tablist" aria-label="事件阶段过滤">
          {GROUPS.map((item) => (
            <button key={item} type="button" className={cn(group === item && "is-active")} onClick={() => setGroup(item)}>{item}</button>
          ))}
        </div>
        <input
          className="execution-event-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索 URL、工具、chunk、错误或模型输出"
          aria-label="搜索详细执行上下文"
        />
      </div>

      <div className="trace-event-list">
        {filtered.length ? filtered.map((event) => {
          const definition = definitionFor(event);
          const Icon = definition.icon;
          const open = expanded.has(event.seq);
          const isError = definition.tone === "error";
          return (
            <article className={cn("trace-event", `is-${definition.tone}`, isError && "is-error")} key={`${event.task_id}-${event.seq}`}>
              <div className="trace-event-rail">
                <div className="trace-event-marker"><Icon className="size-3.5" aria-hidden="true" /></div>
              </div>
              <div className="min-w-0 flex-1">
                <button type="button" className="trace-event-summary" onClick={() => toggle(event.seq)} aria-expanded={open}>
                  <span className="trace-event-seq">#{event.seq}</span>
                  <span className="trace-event-label">{definition.label}</span>
                  {event.phase ? <span className="trace-event-phase">{event.phase}</span> : null}
                  {event.action ? <span className="trace-event-action">{event.action}</span> : null}
                  <span className="trace-event-time">{event.timestamp?.slice(11, 19) || ""}</span>
                  <ChevronDown className={cn("size-4 shrink-0 transition-transform", open && "rotate-180")} aria-hidden="true" />
                </button>
                <p className={cn("trace-event-summary-text", isError && "text-rose-700")}>{summaryFor(event)}</p>
                {open ? <div className="trace-event-detail"><EventDetail event={event} /></div> : null}
              </div>
            </article>
          );
        }) : (
          <div className="execution-event-empty"><FileSearch className="size-4" aria-hidden="true" />没有匹配的执行内容</div>
        )}
      </div>
    </section>
  );
}
