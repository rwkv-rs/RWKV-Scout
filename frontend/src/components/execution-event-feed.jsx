import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Check,
  ChevronDown,
  CircleDot,
  FileSearch,
  Globe2,
  MessageSquare,
  Route,
  Wrench,
} from "lucide-react";

import { cn } from "@/lib/utils";

const EVENT_LABELS = {
  user_input: "用户输入",
  analysis: "范围与状态分析",
  plan: "下一步计划",
  tool_call: "工具调用",
  tool_result: "工具返回",
  progress: "执行进度",
  error: "错误",
  final: "最终结果",
};

const PHASE_LABELS = {
  DISCOVERY: "探测与范围划定",
  EXTRACTION: "资料提取",
  SYNTHESIS: "综合与报告",
};

function iconFor(type) {
  if (type === "user_input") return MessageSquare;
  if (type === "analysis") return Route;
  if (type === "tool_call") return Wrench;
  if (type === "tool_result") return Check;
  if (type === "error") return AlertTriangle;
  if (type === "final") return Check;
  if (type === "plan") return CircleDot;
  return Activity;
}

function pretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? "", null, 2);
}

function detailFor(event) {
  if (event.type === "analysis") {
    return JSON.stringify(
      { phase: event.phase, analysis: event.data, context_snapshot: event.context_snapshot },
      null,
      2,
    );
  }
  if (event.type === "plan") return JSON.stringify({ action: event.action, args: event.args }, null, 2);
  if (event.type === "tool_call") return JSON.stringify({ action: event.action, args: event.args }, null, 2);
  if (event.type === "tool_result") return pretty(event.result);
  if (event.type === "synthesis") return pretty({ content: event.content, mode: event.mode, evidence_count: event.evidence_count, duration_ms: event.duration_ms, citation_refs: event.citation_refs });
  if (event.type === "final") return pretty(event.content);
  return pretty(event.content ?? event.message ?? event);
}

function summaryFor(event) {
  if (event.type === "user_input") return event.content;
  if (event.type === "analysis") return event.data?.missing_information || "已完成当前环境和缺口分析";
  if (event.type === "plan") return `${event.action || "未指定动作"}${event.args ? ` · ${pretty(event.args)}` : ""}`;
  if (event.type === "tool_call") return `${event.action || "工具"} 已发起`;
  if (event.type === "tool_result") return String(event.result || "工具已返回").split("\n")[0];
  if (event.type === "final") return String(event.content || "任务结束").split("\n")[0];
  return event.message || event.content || "状态已更新";
}

export default function ExecutionEventFeed({ events = [] }) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return events;
    return events.filter((event) => detailFor(event).toLowerCase().includes(normalized));
  }, [events, query]);

  return (
    <section className="execution-event-workspace" aria-label="可审计执行轨迹">
      <div className="execution-event-header">
        <div>
          <div className="flex items-center gap-2">
            <Globe2 className="size-4 text-primary" aria-hidden="true" />
            <h2 className="text-sm font-semibold">可审计执行轨迹</h2>
            <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              {events.length} events
            </span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            展示输入、范围、计划、工具调用和结果；不展示模型私有隐藏思维链。
          </p>
        </div>
        <input
          className="execution-event-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索步骤、文件、天气或结果"
          aria-label="搜索执行轨迹"
        />
      </div>

      <div className="execution-event-list">
        {filtered.length ? (
          filtered.map((event, index) => {
            const Icon = iconFor(event.type);
            const isError = event.type === "error";
            const isDone = event.type === "tool_result" || event.type === "synthesis" || event.type === "final";
            const phase = PHASE_LABELS[event.phase] || event.phase;
            return (
              <article className="execution-event-row" key={`${event.task_id}-${event.seq}`}>
                <div className={cn("execution-event-marker", isError && "is-error", isDone && "is-done")}>
                  <Icon className="size-3.5" aria-hidden="true" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="font-mono text-[10px] text-muted-foreground">#{event.seq}</span>
                    <span className="text-xs font-medium">{EVENT_LABELS[event.type] || event.type}</span>
                    {phase ? <span className="execution-event-phase">{phase}</span> : null}
                    {event.action ? <span className="font-mono text-[10px] text-muted-foreground">{event.action}</span> : null}
                    <time className="ml-auto text-[10px] text-muted-foreground" dateTime={event.timestamp}>
                      {event.timestamp?.slice(11, 19) || ""}
                    </time>
                  </div>
                  <p className={cn("mt-1 text-sm leading-6", isError && "text-rose-700")}>{summaryFor(event)}</p>
                  <details className="execution-event-details" open={index === filtered.length - 1}>
                    <summary>
                      <ChevronDown className="size-3.5" aria-hidden="true" />
                      查看完整内容
                    </summary>
                    <pre>{detailFor(event)}</pre>
                  </details>
                </div>
              </article>
            );
          })
        ) : (
          <div className="execution-event-empty">
            <FileSearch className="size-4" aria-hidden="true" />
            {events.length ? "没有匹配的执行内容" : "等待第一条执行事件"}
          </div>
        )}
      </div>
    </section>
  );
}
