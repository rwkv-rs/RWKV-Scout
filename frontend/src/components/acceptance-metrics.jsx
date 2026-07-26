import { Activity, Gauge } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const METRICS = [
  ["strict_pass_rate", "严格通过率", "%"],
  ["usable_body_rate", "可用正文率", "%"],
  ["body_marker_hit", "正文标记命中", "%"],
  ["navigation_leakage", "导航泄漏", "%"],
  ["author_hit", "作者命中", "%"],
  ["date_hit", "日期命中", "%"],
  ["average_duration_ms", "平均耗时", "ms"],
  ["p95_duration_ms", "P95", "ms"],
];

function display(value, unit = "") {
  if (value === null || value === undefined) return "—";
  const normalized = typeof value === "number" ? Number(value.toFixed(2)) : value;
  return `${normalized}${unit}`;
}

export default function AcceptanceMetrics({ data }) {
  const metrics = data?.metrics || {};
  const auditRows = (data?.rows || []).slice(0, 16);
  const auditCounts = data?.final_answer_audit?.counts || {};

  return (
    <Card className="acceptance-metrics">
      <CardHeader className="acceptance-metrics-header">
        <div className="flex items-center gap-2">
          <Gauge className="size-4 text-primary" />
          <CardTitle className="text-sm">本地 RWKV-ECRA 验收指标</CardTitle>
          <span className="text-[11px] text-muted-foreground">
            {data?.sample_count || 0} 条编号任务 · 严格判定 {data?.evaluated_strict_count || 0} 条
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Activity className="size-3.5" />实时计算
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="acceptance-metric-grid">
          {METRICS.map(([key, label, unit]) => (
            <div className="acceptance-metric" key={key}>
              <span>{label}</span>
              <strong>{display(metrics[key], unit)}</strong>
            </div>
          ))}
        </div>

        <div className="acceptance-audit">
          <div className="acceptance-comparison-title">最终答案审核（仅本地 ECRA 任务）</div>
          <div className="acceptance-audit-summary">
            <span className="audit-pass">通过 {auditCounts.pass || 0}</span>
            <span className="audit-review">复核 {auditCounts.review || 0}</span>
            <span className="audit-fail">失败 {auditCounts.fail || 0}</span>
          </div>
          <div className="acceptance-audit-table">
            <div className="acceptance-audit-row acceptance-audit-head">
              <span>编号</span><span>审核</span><span>回答模式</span><span>最终答案摘要</span>
            </div>
            {auditRows.map((row) => {
              const audit = row.final_audit || {};
              return (
                <div className="acceptance-audit-row" key={`${row.case_id}-${row.task_id}`}>
                  <span className="font-mono">{row.case_id}</span>
                  <span className={`audit-${audit.status || "review"}`}>{audit.status || "review"}</span>
                  <span>{row.answer_mode || "—"}</span>
                  <span title={(audit.issues || []).join(", ")}>{audit.answer_preview || "—"}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="acceptance-project-note">
          指标只统计本地 ECRA 的编号任务；作者命中和日期命中表示检索结构中是否有对应字段，不能替代最终答案正确率。
          检索查询由本地 RWKV 生成候选，网页正文仅作为不可信数据传给最终回答模型。
        </div>
      </CardContent>
    </Card>
  );
}
