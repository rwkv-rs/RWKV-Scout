import {
  Check,
  Clock3,
  LoaderCircle,
  Minus,
  Square,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const STATUS_STYLES = {
  ready: "border-emerald-200 bg-emerald-50 text-emerald-700",
  running: "border-amber-200 bg-amber-50 text-amber-700",
  network_error: "border-rose-200 bg-rose-50 text-rose-700",
  queued: "border-stone-200 bg-stone-100 text-stone-700",
  stopped: "border-stone-200 bg-stone-50 text-stone-600",
};

const STATUS_LABELS = {
  ready: "已有回答",
  running: "运行中",
  network_error: "网络错误",
  queued: "排队中",
  stopped: "已停止",
};

const STATUS_INDICATORS = {
  ready: {
    className: "bg-emerald-500 text-white",
    icon: Check,
  },
  running: {
    className: "bg-amber-400 text-amber-950",
    icon: LoaderCircle,
  },
  network_error: {
    className: "bg-rose-500 text-white",
    icon: X,
  },
  queued: {
    className: "bg-stone-300 text-stone-700",
    icon: Clock3,
  },
  stopped: {
    className: "bg-stone-300 text-stone-700",
    icon: Square,
  },
};

export function getTaskStatusLabel(status) {
  const normalized = String(status || "ready").toLowerCase();
  return STATUS_LABELS[normalized] || normalized;
}

export function TaskStatusIndicator({ status, className }) {
  const normalized = String(status || "ready").toLowerCase();
  const config = STATUS_INDICATORS[normalized] || {
    className: "bg-stone-300 text-stone-700",
    icon: Minus,
  };
  const Icon = config.icon;
  const label = getTaskStatusLabel(normalized);

  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex size-3 shrink-0 items-center justify-center rounded-[4px]",
        config.className,
        className,
      )}
    >
      <Icon
        className={cn(
          "size-2 stroke-[2.75]",
          normalized === "running" && "animate-spin motion-reduce:animate-none",
        )}
        aria-hidden="true"
      />
    </span>
  );
}

export function TaskStatusBadge({ status, className }) {
  const normalized = String(status || "ready").toLowerCase();

  return (
    <Badge
      variant="outline"
      className={cn(
        "shrink-0 rounded-md px-2 py-0.5 text-[11px] font-medium",
        STATUS_STYLES[normalized] || "border-border bg-muted text-muted-foreground",
        className,
      )}
    >
      {getTaskStatusLabel(normalized)}
    </Badge>
  );
}
