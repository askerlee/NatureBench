"""Generate a Markdown table of task plans without implementation or execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import html
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Dict, Iterator

from agent.planning import PlanningAgent, PLANNING_STAGES


PLAN_ONLY_GUIDANCE = """Produce planning prose only. Do not generate implementation
source files or claim to have executed tools, inspected data, or evaluated models.
Recommend scientific methods, model families, and libraries appropriate to the
supplied task, metric, and compute budget. Rank a small number of feasible
alternatives, explain the recommended choice, and specify hyperparameters and
validation checks. In reproduce mode, use the supplied paper's methodology and
model rather than proposing replacements. Distinguish documented facts from
defaults and assumptions; identify missing evidence and checks for a later run.
"""


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
          file=sys.stderr, flush=True)


@contextmanager
def report_wait(label: str, interval: float) -> Iterator[None]:
    started = time.monotonic()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(interval):
            progress(f"{label}: still waiting for model ({time.monotonic() - started:.1f}s; "
                     "request or backend retries in progress)")

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    try:
        yield
    except Exception as error:
        progress(f"{label}: failed after {time.monotonic() - started:.1f}s "
                 f"({type(error).__name__})")
        raise
    finally:
        finished.set()
        reporter.join()


def generate_plan(
    planner: PlanningAgent,
    task: Dict[str, Any],
    *,
    max_tokens: int = 4096,
    heartbeat_seconds: float = 15,
) -> Dict[str, str]:
    messages = [{
        "role": "system",
        "content": planner.build_system_prompt(task) + "\n\n" + PLAN_ONLY_GUIDANCE,
    }]
    sections = {}
    token_parameter = ("max_completion_tokens" if planner.model_name.startswith("gpt")
                       else "max_tokens")
    for index, (stage, instruction) in enumerate(PLANNING_STAGES, start=1):
        messages.append({"role": "user", "content": instruction})
        label = f"Stage {index}/{len(PLANNING_STAGES)} {stage}"
        prompt_chars = sum(len(message["content"]) for message in messages)
        progress(f"{label}: requesting model={planner.model_name}, "
                 f"messages={len(messages)}, prompt_chars={prompt_chars}, "
                 f"{token_parameter}={max_tokens}")
        with report_wait(label, heartbeat_seconds):
            response, prompt_tokens, completion_tokens, duration = planner.call_model(
                messages, **{token_parameter: max_tokens},
            )
            if not response.strip():
                raise ValueError(f"Empty model response for planning stage: {stage}")
        progress(f"{label}: completed in {duration:.1f}s, response_chars={len(response)}, "
                 f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
        messages.append({"role": "assistant", "content": response})
        sections[stage] = response
    return sections


def table_cell(value: str) -> str:
    return html.escape(value).replace("|", "&#124;").replace("\r\n", "\n").replace(
        "\r", "\n",
    ).replace("\n", "<br>")


def write_table(output: Path, rows: list[Dict[str, str]], model: str, mode: str) -> None:
    columns = ["task", "status", *(stage for stage, _ in PLANNING_STAGES)]
    lines = [
        "# Task Plans", "", f"Planner model: {table_cell(model)}<br>Mode: {mode}", "",
        "| " + " | ".join(column.title() for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(table_cell(row.get(column, ""))
                                      for column in columns) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tasks-dir", type=Path,
                        help="Folder containing <task-id>/problem/ directories.")
    inputs.add_argument("--problem-dir", type=Path,
                        help="Single directory containing README.md and data_description.md.")
    parser.add_argument("--model", default="gpt-6.1-sol",
                        help="Planner LLM name (default: %(default)s).")
    parser.add_argument("--mode", choices=["base", "reproduce"], default="base")
    parser.add_argument("--paper-path", type=Path,
                        help="Paper markdown/text for single-task reproduce mode.")
    parser.add_argument("--paper-name", type=Path, default=Path("paper.md"),
                        help="Paper path relative to each task folder in batch reproduce mode.")
    parser.add_argument("--output", type=Path, default=Path("plans.md"))
    parser.add_argument("--time-limit-minutes", type=float, default=60,
                        help="Compute budget for the proposed solution, not this script.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Token budget per stage (includes reasoning tokens for GPT models).")
    parser.add_argument("--heartbeat-seconds", type=float, default=15,
                        help="Interval for model-wait progress messages (default: %(default)s).")
    parser.add_argument("--debug", action="store_true",
                        help="Print exception tracebacks for failed tasks.")
    args = parser.parse_args()
    if args.time_limit_minutes <= 0 or args.max_tokens <= 0:
        parser.error("--time-limit-minutes and --max-tokens must be positive")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.tasks_dir is not None and args.paper_path is not None:
        parser.error("Use --paper-name for batch reproduction, not --paper-path")
    if args.paper_name.is_absolute() or ".." in args.paper_name.parts:
        parser.error("--paper-name must be relative to each task folder")
    if args.mode == "reproduce" and args.problem_dir is not None and args.paper_path is None:
        parser.error("--paper-path is required for single-task reproduce mode")
    if args.tasks_dir is not None:
        if not args.tasks_dir.is_dir():
            parser.error("--tasks-dir must be an existing folder")
        problems = sorted(path / "problem" for path in args.tasks_dir.iterdir()
                          if path.is_dir() and (path / "problem").is_dir())
        if not problems:
            parser.error("No <task-id>/problem/ directories found in --tasks-dir")
    else:
        problems = [args.problem_dir]
    progress(f"Discovered {len(problems)} tasks; model={args.model}, mode={args.mode}, "
             f"token_budget={args.max_tokens}, heartbeat={args.heartbeat_seconds}s, "
             f"output={args.output}")
    progress("Initializing LLM backend")
    planner = PlanningAgent(args.model, mode=args.mode, backend_kwargs={"max_retries": 0})
    rows = []
    failed = 0
    for index, problem in enumerate(problems, start=1):
        task_id = problem.parent.name if problem.name == "problem" else problem.name
        print(f"[{index}/{len(problems)}] Planning {task_id}", flush=True)
        task_started = time.monotonic()
        progress(f"{task_id}: reading task documentation from {problem}")
        row = {"task": task_id, "status": "OK"}
        try:
            task = {
                "task_description": (problem / "README.md").read_text(encoding="utf-8"),
                "data_description": (problem / "data_description.md").read_text(encoding="utf-8"),
                "time_limit_minutes": args.time_limit_minutes,
            }
            if args.mode == "reproduce":
                paper = (problem.parent / args.paper_name
                         if args.tasks_dir is not None else args.paper_path)
                task["paper_content"] = paper.read_text(encoding="utf-8")
                if not task["paper_content"].strip():
                    raise ValueError(f"Paper must contain nonempty text: {paper}")
            progress(f"{task_id}: loaded task_chars={len(task['task_description'])}, "
                     f"data_chars={len(task['data_description'])}, "
                     f"paper_chars={len(task.get('paper_content', ''))}")
            row.update(generate_plan(planner, task, max_tokens=args.max_tokens,
                                     heartbeat_seconds=args.heartbeat_seconds))
        except Exception as error:
            failed += 1
            row["status"] = f"ERROR: {error}"
            progress(f"{task_id}: {type(error).__name__}: {error}")
            if args.debug:
                traceback.print_exc(file=sys.stderr)
        rows.append(row)
        write_table(args.output, rows, args.model, args.mode)
        progress(f"{task_id}: elapsed={time.monotonic() - task_started:.1f}s; "
                 f"table saved to {args.output} ({len(rows)} rows)")
    print(f"Saved {len(rows) - failed} plans; {failed} failed. Table: {args.output}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()