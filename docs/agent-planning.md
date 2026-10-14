# Staged Planning Agent

The `planning` agent is an original implementation inspired by
[Paper2Code/PaperCoder](https://github.com/going-doer/Paper2Code)'s staged planning
workflow. It reuses NatureBench's LLM backend, rather than Paper2Code's vLLM
runtime or paper-to-JSON preprocessing. `DummyAgent` remains unchanged as a
deterministic smoke-test agent.

## Workflow

1. Plan methodology, experiments, and uncertainties.
2. Define a small architecture and file interfaces.
3. Order implementation tasks and specify validation checks.
4. Specify documented hyperparameters, defaults, and compute requirements.
5. Generate a JSON file map containing `run.py` and any supporting files.
6. Validate source syntax and workspace-relative paths, then execute the solution.
7. Submit outputs to the evaluation service and repair/refine using observed
   errors and scores, up to the attempt limit and remaining solve budget.

Each model call retains the preceding conversation. The adapter installs the
existing backend dependencies (`openai` and `backoff`) and mounts the local
`agent/` package read-only inside the task container. This requires package-index
access, or a derived task/base image with those dependencies already installed.

## Running

Set `OPENAI_API_KEY` in your shell. Set `OPENAI_BASE_URL` when using an
OpenAI-compatible endpoint. The existing backend accepts model names beginning
with `gpt`, `deepseek`, or `claude`; a `claude` model requires an OpenAI-compatible
gateway, not the native Anthropic API.

With Docker, the dataset, and evaluation environments configured as described in
the main README, run:

```bash
python run_naturebench.py \
  --agent planning \
  --model gpt-4.1 \
  --tasks cpu \
  --out-dir ./results/planning_cpu \
  --start-eval-services \
  --eval-env-mapping ./eval_env_mapping.json \
  --ensure-base-image
```

Use `--mode reproduce` for faithful paper reproduction. This requires the task's
paper markdown, mounted as `/task/paper.md`; base mode uses only the task/data
descriptions and a bounded listing of available data files. The agent does not
read the original paper in base mode.

Defaults are three execution attempts and a 600-second cap per solution process.
For longer training jobs, set `NATUREBENCH_PLANNING_EXECUTION_TIMEOUT` on the host;
set `NATUREBENCH_PLANNING_MAX_ATTEMPTS` to change the attempt count. The adapter
passes both settings to its CLI. For example, in fish:

```fish
set -gx NATUREBENCH_PLANNING_EXECUTION_TIMEOUT 1800
set -gx NATUREBENCH_PLANNING_MAX_ATTEMPTS 5
```

Use the harness's normal resume options to preserve the conversation and retry
the saved solution. The evaluation service tracks the best score. Every execution
starts with an empty output directory so stale predictions cannot be submitted.

## Artifacts And Limits

Under each task's `workspace/.planning/`, `messages.json` preserves model context,
`trajectory.json` records planning, generation, repair, and evaluation feedback,
and `execution.log` contains the latest subprocess output. The adapter exposes
the trajectory to the post-hoc judge.

This is a bounded planning-and-repair agent, not a general tool-calling coding
CLI. It does not dynamically install model-requested dependencies, search the web,
or inspect datasets through interactive tools. Generated Python is syntax-checked,
not certified scientifically correct; execution, evaluation, and judging remain
necessary. Generated code must be run in the standard isolated task container.
Common API-key/token environment variables are removed from the solver subprocess,
but this is not a substitute for container isolation.

The agent checks remaining time between model calls and executions and excludes
evaluation duration from its local solve timer. The harness's container timeout
is still the hard limit, including backend request retries.

For programmatic generation without execution, import `PlanningAgent` from
`agent.planning` and call `solve_task()` with `workspace_path`, `task_description`,
and `data_description`; reproduce mode also requires `paper_content`. This still
generates implementation files, not just a plan.

### Plan Only

To generate plans for a folder of tasks, run the standalone script from the
repository root:

```bash
python plan_task.py \
  --tasks-dir /path/to/tasks \
  --model gpt-4.1 \
  --output ./plans.md
```

Tasks are discovered in sorted order as `<tasks-dir>/<task-id>/problem/`.
Each problem directory must contain `README.md` and `data_description.md`.
Tasks run sequentially by default. Use `--workers N` to plan up to N tasks in
parallel, for example `--workers 4 --stages 2`. Each task has its own planner
and API client; its stages still run sequentially, retaining context within
a task but not across tasks. Start with a small worker count to stay within
your provider's concurrent-request and token rate limits. Parallelism does not
reduce the total number of model calls or tokens requested.
The output is one Markdown table with columns for
task ID, status, methodology, architecture, dependencies, and configuration.
Use `--stages M` to run only the first M stages (1 through 4; default 4), in this
order: methodology, architecture, dependencies, configuration. For example,
`--stages 2` makes two calls per task and includes only methodology and
architecture columns alongside task ID and status in the table.
Plans include recommended scientific methods, model families, libraries,
hyperparameters, compute requirements, and validation checks. The script does not
generate solution files, execute code, inspect datasets, or contact the evaluation
service. Docker and evaluation services are not required; the existing backend's
`openai` and `backoff` dependencies and API credentials are required.

For batch paper reproduction, add `--mode reproduce`. By default each task's
paper is read from `<task-id>/paper.md`; use `--paper-name problem/paper.md` or
another task-relative path for a different layout. A missing or empty paper is
recorded as a failed task rather than silently falling back to base mode.
Single-task use remains available with `--problem-dir /path/to/task/problem`;
single-task reproduce mode requires `--paper-path /path/to/paper.md`.

The table is saved by the main thread after every completed task, keeping rows
in task discovery order even when requests complete out of order. Stage and
heartbeat logs include task IDs so concurrent requests can be distinguished.
Failures appear in the status column and
do not prevent subsequent tasks from being processed; the script exits with
status 1 if any task fails. Rerunning overwrites the table and replans all tasks.
`--time-limit-minutes` sets each proposed solution's compute budget,
not a timeout for plan generation. `--max-tokens` sets each stage's token budget.
For GPT models it is sent as `max_completion_tokens` and includes reasoning
tokens; Claude and DeepSeek gateways receive `max_tokens`.
`--output` defaults to `plans.md` and overwrites an existing file at that path.
Model names and API endpoint settings follow the backend rules above; `--model`
selects the planner LLM, not the scientific model recommended in the plan.

Timestamped progress is flushed to stderr: input sizes, stage starts/completions,
call duration, token usage, task duration, and table saves. While a call is
pending, a heartbeat reports elapsed time every 15 seconds (adjust with
`--heartbeat-seconds`). A heartbeat indicates a pending call, not proof of
server-side progress; it may include backend retry waits. Add `--debug` to print
tracebacks for failed tasks. Prompts and responses are not dumped to the logs.

Focused tests (no API credentials, Docker, or benchmark data required):

```bash
python -m pytest agent/planning_test.py agent/lumen_adapter_test.py -q
```

## Codex With Staged Planning

Use `--agent codex-planning` to combine Codex's interactive tool execution with
the four planning stages above. `CodexPlanningAgent` inherits `CodexAgent` and
adds prompt guidance for a time-boxed plan, facts versus assumptions, local
output-contract validation, feedback classification, and an experiment history.
It does not use the planning agent's API backend, JSON file-map generation, or
fixed attempt limit. Both agents share the research-budgeting and
scientific-validity instructions; `codex-planning` adds the staged workflow.

```bash
python run_naturebench.py \
  --agent codex-planning \
  --model gpt-5.4 \
  --tasks cpu \
  --out-dir ./results/codex_planning_cpu \
  --start-eval-services \
  --eval-env-mapping ./eval_env_mapping.json \
  --ensure-base-image
```

The same Codex CLI installation, authentication options, and device-auth resume
machinery apply. All Codex modes remain available, including `reference` and
`reproduce`; reproduction constraints take precedence over baseline exploration.
On resume, the full current mode's prompt is re-injected alongside Codex's resume
notice, including scientific-validity rules and, for `codex-planning`, planning
guidance. Existing evaluated solutions should be continued rather than restarted.

Base and reference modes target an early valid, task-appropriate baseline using
measured data-processing and training costs, not a fixed five-minute deadline.
Experiments should allow meaningful training and reserve time for inference,
output checks, and evaluation. Compact, reproducible implementations are preferred
over production infrastructure. Task-permitted general-purpose pretrained weights
may be used as pipeline components after verifying permission and provenance;
the submitted output must still come from a method fit or optimized from the
provided data, not answer retrieval or a pretrained answer generator.

Base mode forbids withheld-paper materials and task-specific derivatives, answer
recovery from any split or artifact, and score-feedback-based answer construction.
Reference hints derived from the withheld paper and reproduce-mode paper access
are calibration features, not valid inputs for the withheld-paper benchmark
rubric. Reproduce mode retains its faithful-methodology constraints.

Codex's built-in web search is explicitly disabled in both harness runs and
direct `CodexAgent.solve_task()` / `CodexPlanningAgent.solve_task()` calls.
This is a tool restriction, not a blanket block on outbound network access.

The prompt requests `workspace/PLAN.md`, `workspace/experiments.jsonl`, and
workspace-local snapshots of the best evaluated source/configuration. These are
model-written artifacts, not harness-enforced validation or guaranteed files.
The Codex transcript remains available to the judge. This integration has not
been benchmarked for score improvements.

```bash
python -m pytest agent/codex_planning_test.py -q
```