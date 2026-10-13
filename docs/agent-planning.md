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
and `data_description`; reproduce mode also requires `paper_content`.

Focused tests (no API credentials, Docker, or benchmark data required):

```bash
python -m pytest agent/planning_test.py agent/lumen_adapter_test.py -q
```