import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  MessageCircle,
  Search,
  SquarePen,
  StopCircle,
  Trash2,
} from "lucide-react";

import {
  getTaskStatusLabel,
  TaskStatusIndicator,
} from "@/components/task-status-badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

function getTaskLabel(item) {
  if (item.query?.trim()) {
    return item.query.trim();
  }

  return item.title || item.id;
}

function parseDate(value) {
  if (!value) return null;
  if (value instanceof Date) return value;

  const normalized = String(value).trim().replace(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$/,
    "$1-$2-$3T$4:$5:$6",
  );
  const date = new Date(normalized);

  return Number.isNaN(date.getTime()) ? null : date;
}

function formatRelativeTime(value, now) {
  const date = parseDate(value);
  if (!date) return "-";

  const seconds = Math.max(0, Math.floor((now.getTime() - date.getTime()) / 1000));
  if (seconds < 60) return "刚刚";

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}分钟前`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}小时前`;

  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}天前`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months}个月前`;

  return `${Math.floor(months / 12)}年前`;
}

function getDateGroup(value, now) {
  const date = parseDate(value);
  if (!date) return "更早";

  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterdayStart = new Date(todayStart);
  yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - 7);

  if (date >= todayStart) return "今天";
  if (date >= yesterdayStart) return "昨天";
  if (date >= weekStart) return "近7天";
  return "更早";
}

function groupByDate(items, now) {
  const order = ["今天", "昨天", "近7天", "更早"];
  const groups = new Map();

  for (const item of items) {
    const group = getDateGroup(item.updated_at, now);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(item);
  }

  return order.filter((key) => groups.has(key)).map((key) => ({
    label: key,
    items: groups.get(key),
  }));
}

export function AppSidebar({
  history,
  activeId,
  keyword,
  onKeywordChange,
  onSelect,
  onNewRun,
  onOpenChat,
  onStop,
  onDelete,
  publicMode = false,
  ...props
}) {
  const deferredKeyword = useDeferredValue(keyword);
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const filtered = useMemo(() => {
    const needle = deferredKeyword.trim().toLowerCase();

    return history.filter((item) => {
      const haystack = [
        item.id,
        item.title,
        item.query,
        item.status,
        getTaskStatusLabel(item.status),
        item.updated_at,
      ]
        .join(" ")
        .toLowerCase();

      return !needle || haystack.includes(needle);
    });
  }, [history, deferredKeyword]);

  const isSearching = deferredKeyword.trim().length > 0;
  const groups = useMemo(() => groupByDate(filtered, now), [filtered, now]);

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader className="gap-0 border-b border-sidebar-border p-0">
        <div className="flex h-11 items-center gap-2 px-3">
          <div
            className="flex size-6 shrink-0 items-center justify-center rounded bg-sidebar-foreground text-sidebar"
            aria-hidden="true"
          >
            <span className="text-[10px] font-bold tracking-[-0.04em]">R</span>
          </div>
          <div className="min-w-0 flex-1 truncate text-[13px] font-semibold tracking-tight text-sidebar-foreground">
            RWKV-ECRA
          </div>
        </div>

        <div className="px-2.5 pb-2.5 space-y-2">
          <div className="grid grid-cols-2 gap-2">
          <Button className="h-8 w-full justify-start rounded-md px-2.5 text-[13px]" onClick={onNewRun}>
            <SquarePen className="size-3.5" />
            新建任务
          </Button>
          <Button
            variant="outline"
            className="h-8 justify-start rounded-md px-2.5 text-[12px]"
            onClick={onOpenChat}
          >
            <MessageCircle className="size-3.5" />
            检索对话
          </Button>
          </div>
          {!publicMode ? <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3 -translate-y-1/2 text-sidebar-foreground/30" />
            <Input
              value={keyword}
              onChange={(event) => onKeywordChange(event.target.value)}
              placeholder="搜索…"
              aria-label="搜索历史任务"
              className="h-7 rounded border-transparent bg-sidebar-foreground/[0.04] pl-7 text-[11px] shadow-none placeholder:text-sidebar-foreground/30 focus:border-sidebar-border"
              type="search"
            />
          </div> : (
            <div className="px-1 text-[11px] text-sidebar-foreground/45">
              公网模式 · 仅显示本浏览器提交的任务
            </div>
          )}
        </div>
      </SidebarHeader>

      <SidebarContent className="overflow-y-auto">
        <SidebarGroup className="min-h-0 gap-0 p-0">
          {filtered.length ? (
            isSearching ? (
              <div className="px-1.5 py-1">
                <div className="px-2 py-1.5 text-[11px] font-medium text-sidebar-foreground/45">
                  {filtered.length} 个结果
                </div>
                {filtered.map((item) => (
                  <TaskItem
                    key={item.id}
                    item={item}
                    isActive={item.id === activeId}
                    now={now}
                    onSelect={onSelect}
                    onStop={onStop}
                    onDelete={onDelete}
                    allowDelete={!publicMode}
                  />
                ))}
              </div>
            ) : (
              groups.map((group) => (
                <div key={group.label} className="px-1.5 py-1">
                  <div className="px-2 py-1.5 text-[11px] font-medium text-sidebar-foreground/45">
                    {group.label}
                  </div>
                  {group.items.map((item) => (
                    <TaskItem
                      key={item.id}
                      item={item}
                      isActive={item.id === activeId}
                      now={now}
                      onSelect={onSelect}
                      onStop={onStop}
                      onDelete={onDelete}
                      allowDelete={!publicMode}
                    />
                  ))}
                </div>
              ))
            )
          ) : (
            <div className="px-4 py-10 text-center">
              <div className="text-sm text-sidebar-foreground/55">
                {keyword.trim() ? "没有匹配的任务" : publicMode ? "本浏览器还没有任务" : "还没有历史任务"}
              </div>
              <div className="mt-1.5 text-[11px] text-sidebar-foreground/35">
                {keyword.trim() ? "尝试修改搜索词" : "点击右上角新建"}
              </div>
            </div>
          )}
        </SidebarGroup>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}

function TaskItem({ item, isActive, now, onSelect, onStop, onDelete, allowDelete = true }) {
  const label = getTaskLabel(item);
  const relativeTime = formatRelativeTime(item.updated_at, now);

  return (
    <div
      className={cn(
        "group/task-item relative rounded-md transition-colors duration-100",
        isActive
          ? "bg-sidebar-foreground/[0.08]"
          : "hover:bg-sidebar-foreground/[0.04]",
      )}
    >
      <button
        type="button"
        aria-current={isActive ? "page" : undefined}
        onClick={() => onSelect(item.id)}
        className="flex w-full items-start gap-2 rounded-md px-2.5 py-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
      >
        <TaskStatusIndicator status={item.status} className="mt-[5px] shrink-0" />
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              "truncate text-[13px] leading-snug text-sidebar-foreground",
              isActive ? "font-medium" : "font-normal",
            )}
            title={label}
          >
            {label}
          </div>
          <div className="mt-0.5 text-[11px] text-sidebar-foreground/40">
            {relativeTime}
          </div>
        </div>
      </button>

      <div className="pointer-events-none absolute top-1.5 right-1 flex items-center gap-0.5 opacity-0 transition-opacity group-focus-within/task-item:pointer-events-auto group-focus-within/task-item:opacity-100 group-hover/task-item:pointer-events-auto group-hover/task-item:opacity-100">
        {item.status === "running" ? (
          <Button
            variant="ghost"
            size="icon-xs"
            title="停止任务"
            aria-label={`停止任务 ${label}`}
            onClick={(event) => {
              event.stopPropagation();
              onStop(item.id);
            }}
          >
            <StopCircle className="size-3.5" />
          </Button>
        ) : null}
        {allowDelete ? <Button
          variant="ghost"
          size="icon-xs"
          title="删除任务"
          aria-label={`删除任务 ${label}`}
          onClick={(event) => {
            event.stopPropagation();
            onDelete(item.id);
          }}
        >
          <Trash2 className="size-3.5" />
        </Button> : null}
      </div>
    </div>
  );
}
