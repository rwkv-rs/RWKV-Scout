import { useMemo, useState } from "react";
import { CheckCircle2, ChevronDown, ExternalLink, FlaskConical, Play, RotateCcw, Search, Timer, XCircle } from "lucide-react";
import { ACCEPTANCE_GROUPS } from "@/acceptance-cases";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

function linkFor(test, links, history) {
  const effectivePrompt = test.prompt.replace("{{as_of_date}}", new Date().toISOString().slice(0, 10));
  const matched = history.find((item) => item.acceptance_case_id === test.id || item.query === test.prompt || item.query === effectivePrompt);
  return links[test.id] || (matched ? { taskId: matched.id || matched.task_id } : null);
}

function statusFor(test, links, history) {
  const link = linkFor(test, links, history);
  if (!link?.taskId) return { key: "pending", label: "未运行", icon: Timer };
  const task = history.find((item) => item.id === link.taskId || item.task_id === link.taskId);
  if (!task || task.status === "running" || task.status === "queued") return { key: "running", label: "运行中", icon: RotateCcw };
  if (task.status === "completed" || task.status === "ready") return { key: "completed", label: "已完成，待判定", icon: CheckCircle2 };
  return { key: "failed", label: "执行失败", icon: XCircle };
}

function StatusIcon({ status }) {
  const Icon = status.icon;
  return <Icon className={cn("size-3.5", status.key === "running" && "animate-spin", status.key === "completed" && "text-emerald-600", status.key === "failed" && "text-rose-600")} />;
}

export default function AcceptancePanel({ history = [], links = {}, onRun, onOpen }) {
  const [expanded, setExpanded] = useState(true);
  const [activeGroup, setActiveGroup] = useState("all");
  const [query, setQuery] = useState("");
  const filteredGroups = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return ACCEPTANCE_GROUPS
      .filter((group) => activeGroup === "all" || group.id === activeGroup)
      .map((group) => ({
        ...group,
        cases: group.cases.filter((test) => !normalized || `${test.id} ${test.prompt} ${test.focus}`.toLowerCase().includes(normalized)),
      }))
      .filter((group) => group.cases.length);
  }, [activeGroup, query]);

  const allCases = ACCEPTANCE_GROUPS.flatMap((group) => group.cases);
  const stats = allCases.reduce((acc, test) => {
    acc[statusFor(test, links, history).key] += 1;
    return acc;
  }, { pending: 0, running: 0, completed: 0, failed: 0 });

  return (
    <Card className="acceptance-panel">
      <CardHeader className="acceptance-panel-header">
        <div className="flex min-w-0 items-start gap-3">
          <div className="acceptance-icon"><FlaskConical className="size-4" /></div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-base">编号边界验收</CardTitle>
              <Badge variant="outline" className="font-mono text-[10px]">{allCases.length} cases</Badge>
            </div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">逐条运行真实任务；状态只表示执行结果，是否“可做”需要结合事件、来源和阻塞原因判定。</p>
          </div>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={() => setExpanded((value) => !value)} aria-label="展开或收起编号验收">
          <ChevronDown className={cn("size-4 transition-transform", !expanded && "-rotate-90")} />
        </Button>
      </CardHeader>
      {expanded ? (
        <CardContent className="pt-0">
          <div className="acceptance-summary">
            <div><span className="acceptance-summary-number">{stats.completed}</span><span>已完成</span></div>
            <div><span className="acceptance-summary-number text-amber-600">{stats.running}</span><span>运行中</span></div>
            <div><span className="acceptance-summary-number text-muted-foreground">{stats.pending}</span><span>未运行</span></div>
            <div><span className="acceptance-summary-number text-rose-600">{stats.failed}</span><span>失败</span></div>
          </div>
          <div className="acceptance-toolbar">
            <div className="acceptance-search-wrap">
              <Search className="size-3.5 text-muted-foreground" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="按编号、测试点或问题搜索" aria-label="搜索编号测试" />
            </div>
            <div className="acceptance-filters" role="tablist" aria-label="编号测试分组">
              <button className={cn(activeGroup === "all" && "is-active")} onClick={() => setActiveGroup("all")}>全部</button>
              {ACCEPTANCE_GROUPS.map((group) => <button key={group.id} className={cn(activeGroup === group.id && "is-active")} onClick={() => setActiveGroup(group.id)}>{group.label}</button>)}
            </div>
          </div>
          <div className="acceptance-groups">
            {filteredGroups.map((group) => (
              <section key={group.id} className="acceptance-group">
                <div className="acceptance-group-heading">
                  <div className="flex items-center gap-2"><span className="acceptance-group-label">{group.label}</span><span className="text-sm font-medium">{group.title}</span></div>
                  <span className="text-[11px] text-muted-foreground">{group.description}</span>
                </div>
                <div className="acceptance-case-list">
                  {group.cases.map((test) => {
                    const status = statusFor(test, links, history);
    const link = linkFor(test, links, history);
                    return (
                      <div className="acceptance-case" key={test.id}>
                        <div className="acceptance-case-id">{test.id}</div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm leading-5 text-foreground">{test.prompt}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground"><span>{test.focus}</span>{link?.taskId ? <span className="font-mono">{link.taskId}</span> : null}</div>
                        </div>
                        <div className={cn("acceptance-case-status", `is-${status.key}`)}><StatusIcon status={status} /><span>{status.label}</span></div>
                        <div className="acceptance-case-actions">
                          {link?.taskId ? <Button variant="ghost" size="icon-sm" title="查看任务事件" onClick={() => onOpen(link.taskId)}><ExternalLink className="size-3.5" /></Button> : null}
                          <Button variant={status.key === "pending" ? "default" : "outline"} size="sm" onClick={() => onRun({ ...test, prompt: test.prompt.replace("{{as_of_date}}", new Date().toISOString().slice(0, 10)) })} disabled={status.key === "running"}>
                            <Play className="size-3.5" />{status.key === "pending" ? "运行" : "重跑"}
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        </CardContent>
      ) : null}
    </Card>
  );
}
