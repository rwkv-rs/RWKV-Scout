import fs from "node:fs";
import path from "node:path";
import { ACCEPTANCE_CASES } from "../frontend/src/acceptance-cases.js";

const API = process.env.RWKV_ECRA_TEST_API || "http://127.0.0.1:5177";
const MODEL_KEY = process.env.RWKV_ECRA_TEST_MODEL || "local_7b";
const RUN_ID = new Date().toISOString().replace(/[:.]/g, "-");
const OUTPUT_DIR = path.resolve("data/output/acceptance_runs");
const OUTPUT_FILE = path.join(OUTPUT_DIR, `${RUN_ID}.json`);
const effectiveCases = ACCEPTANCE_CASES.map((test) => ({
  ...test,
  prompt: test.prompt.replace("{{as_of_date}}", new Date().toISOString().slice(0, 10)),
}));
const results = [];

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${response.status}: ${body.detail || body.message || response.statusText}`);
  }
  return body;
}

async function waitForTask(taskId, timeoutMs = 15 * 60 * 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const body = await jsonFetch(`${API}/frontend-api/history`);
    const task = (body.data || []).find((item) => item.task_id === taskId || item.id === taskId);
    if (task && ["completed", "failed", "ready"].includes(task.status)) return task;
    await new Promise((resolve) => setTimeout(resolve, 2500));
  }
  throw new Error(`timeout waiting for ${taskId}`);
}

async function summarizeReport(records, taskId) {
  const retrieval = records.find((item) => item.record_type === "retrieval_result");
  const final = [...records].reverse().find((item) => item.type === "final" || item.type === "synthesis" || item.record_type === "final_beautified_markdown");
  const eventsBody = await jsonFetch(`${API}/frontend-api/history/${encodeURIComponent(taskId)}/events`);
  const events = eventsBody.data?.events || [];
  const toolCalls = events.filter((item) => item.type === "tool_call");
  return {
    final_answer: final?.content || retrieval?.answer || "",
    answer_mode: retrieval?.answer_mode || final?.mode || "",
    real_network: Boolean(retrieval?.real_network || records.some((item) => item.real_network)),
    action: retrieval?.action || toolCalls.at(-1)?.action || "",
    tool_calls: toolCalls.map((item) => item.action).filter(Boolean),
    tool_call_count: toolCalls.length,
    round_count: retrieval?.data?.round_count ?? null,
    source_count: retrieval?.data?.count ?? retrieval?.data?.results?.length ?? 0,
    evidence_policy: retrieval?.data?.evidence_policy || null,
  };
}

function save() {
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify({
    run_id: RUN_ID,
    api: API,
    model_key: MODEL_KEY,
    total: effectiveCases.length,
    completed: results.length,
    results,
  }, null, 2));
}

console.log(JSON.stringify({ run_id: RUN_ID, total: effectiveCases.length, model_key: MODEL_KEY, api: API }));

for (let index = 0; index < effectiveCases.length; index += 1) {
  const test = effectiveCases[index];
  const started = Date.now();
  const row = { index: index + 1, id: test.id, prompt: test.prompt, focus: test.focus, model_key: MODEL_KEY };
  try {
    const submitted = await jsonFetch(`${API}/frontend-api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: test.prompt,
        model_key: MODEL_KEY,
        acceptance_case_id: test.id,
        slm_async_enabled: false,
        queued_at: new Date().toISOString(),
      }),
    });
    row.task_id = submitted.task_id;
    const task = await waitForTask(submitted.task_id);
    row.status = task.status;
    row.error = task.error || "";
    const reportBody = await jsonFetch(`${API}/frontend-api/history/${encodeURIComponent(submitted.task_id)}/report`);
    row.report = await summarizeReport(reportBody.data || [], submitted.task_id);
  } catch (error) {
    row.status = "runner_error";
    row.error = String(error?.message || error);
  }
  row.duration_ms = Date.now() - started;
  results.push(row);
  save();
  console.log(JSON.stringify({
    index: row.index,
    id: row.id,
    status: row.status,
    task_id: row.task_id || null,
    duration_ms: row.duration_ms,
    real_network: row.report?.real_network ?? false,
    tool_calls: row.report?.tool_call_count ?? 0,
    answer_preview: (row.report?.final_answer || "").slice(0, 120),
    error: row.error || "",
  }));
}

save();
console.log(JSON.stringify({ done: true, run_id: RUN_ID, completed: results.length, output: OUTPUT_FILE }));
