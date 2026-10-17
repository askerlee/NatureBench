import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";

const sourceUrl = "https://frontisai.github.io/NatureBench/naturebench-data.js";
const response = await fetch(sourceUrl);
if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
const source = (await response.text()).trim();
const prefix = "window.NATUREBENCH_DATA = ";
assert(source.startsWith(prefix), "Unexpected leaderboard data format");
const data = JSON.parse(source.slice(prefix.length).replace(/;$/, ""));
const configurations = data.leaderboard.filter(
  (row) => !Array.isArray(row.tracks) || row.tracks.includes("full"),
);
assert.equal(new Set(configurations.map((row) => row.name)).size, configurations.length);
assert.equal(data.cases.length, data.benchmark.taskCount);

const cases = data.cases.map((task) => {
  const scores = Object.fromEntries(configurations.map((configuration) => {
    const score = configuration.scores?.[task.caseId]
      ?? task.scores[configuration.name]
      ?? { state: "none", value: null };
    assert(["valid", "invalid", "none"].includes(score.state));
    if (score.state === "valid") assert(Number.isFinite(score.value));
    return [configuration.name, score];
  }));
  const valid = Object.entries(scores).filter(([, score]) => score.state === "valid");
  const best = valid.reduce(
    (current, entry) => !current || entry[1].value > current[1].value ? entry : current,
    null,
  );
  return { ...task, scores, bestModel: best?.[0] ?? null, bestScore: best?.[1].value ?? null };
});

const snapshot = {
  sourceUrl,
  pageUrl: "https://frontisai.github.io/NatureBench/#cases",
  fetchedAt: new Date().toISOString(),
  metric: "Task-level SOTA-normalized relative gap g, not a percentage",
  configurations,
  models: data.models,
  cases,
};
const csvCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
const headers = ["case_id", "task_name", "domain", "ml_task_type", ...configurations.map(
  (configuration) => {
    const model = data.models.find((entry) => entry.name === configuration.name);
    return `${configuration.name} (${configuration.agent ?? model?.agent ?? "unknown harness"})`;
  },
)];
const rows = cases.map((task) => [
  task.caseId, task.title, task.domain, task.mlTaskType,
  ...configurations.map((configuration) => {
    const score = task.scores[configuration.name];
    return score.state === "valid" ? score.value : score.state;
  }),
]);
const outputDir = new URL("../assets/leaderboard/", import.meta.url);
await mkdir(outputDir, { recursive: true });
await writeFile(new URL("per_case_scores.json", outputDir), `${JSON.stringify(snapshot, null, 2)}\n`);
await writeFile(new URL("per_case_scores.csv", outputDir),
  [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n");
console.log(`Exported ${cases.length} tasks x ${configurations.length} configurations to ${outputDir.pathname}`);
console.log("CSV preserves valid numeric gaps and the distinct invalid/none states.");