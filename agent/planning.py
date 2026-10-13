"""Staged scientific solver inspired by Paper2Code's planning workflow."""
from __future__ import annotations

import ast
import argparse
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base import BaseAgent


PLANNING_STAGES = (
    ("methodology", "Identify the objective, methodology, data splits, preprocessing, "
     "metrics, and output contract. Make an actionable scientific implementation plan. "
     "Separate supplied facts from assumptions and explicitly list missing details."),
    ("architecture", "Design a small implementation using installed scientific libraries. "
     "Define files, functions, interfaces, and data shapes. The entry point MUST be run.py. "
     "Prefer few modules; do not design web services, frontends, or OpenAPI schemas."),
    ("dependencies", "Order implementation tasks by dependency. Specify imports, "
     "preprocessing, training or algorithm steps, inference, and output validation. "
     "Include checks for leakage and file names, shapes, dtypes, and value ranges."),
    ("configuration", "Specify hyperparameters, random seeds, compute requirements, "
     "and a validation strategy. Distinguish documented values from framework defaults "
     "and assumptions. Do not invent paper details. Fit the available time budget."),
)

IMPLEMENTATION_PROMPT = """Implement the preceding plan completely. Return ONLY a JSON object:
{"files": {"run.py": "complete Python source", "optional_module.py": "complete Python source"}}
Include every required source/configuration file, using workspace-relative paths.
run.py must use DATA_DIR and OUTPUT_DIR and produce the task's exact output format.
Keep code, configuration, checkpoints, and temporary files inside the workspace;
only read the supplied task/data files. Never use test labels or hidden evaluators.
Do not call LLM APIs from the generated solution or fabricate predictions.
Use installed packages. Do not include placeholders, prose, or Markdown fences.
"""


class PlanningAgent(BaseAgent):
    """Plan in four stages, then generate a validated multi-file solution."""

    def build_system_prompt(self, task: Dict[str, Any]) -> str:
        if self.mode not in {"base", "reproduce"}:
            raise ValueError("PlanningAgent supports only base and reproduce modes.")
        context = [
            "You are a scientific ML programmer. Follow the task's data, interface, "
            "scientific-validity, and time-budget constraints throughout every stage.",
            "Task description:\n" + task.get("task_description", ""),
            "Data description:\n" + task.get("data_description", ""),
            "Time budget (minutes): " + str(task.get("time_limit_minutes", 60)),
        ]
        if self.mode == "reproduce":
            paper = task.get("paper_content", "")
            if not paper.strip():
                raise ValueError("Reproduce mode requires paper_content (paper markdown/text).")
            context.extend([
                "Faithfully reproduce the supplied paper. Do not substitute methods, "
                "invent hyperparameters, or add improvements not warranted by the paper. "
                "Document unavoidable deviations in DEVIATIONS.md.",
                "Paper:\n" + paper,
            ])
        else:
            context.append("Solve the supplied task. No original paper is available; "
                           "do not claim to have read one. Start with a feasible baseline.")
        return "\n\n".join(context)

    def _record_call(self, stage: str, instruction: str, llm_kwargs: Dict[str, Any]) -> str:
        options = dict(llm_kwargs)
        remaining_seconds = getattr(self, "_remaining_seconds", None)
        if remaining_seconds is not None:
            available = remaining_seconds()
            if available <= 1:
                raise TimeoutError("No solve time remaining for another model call.")
            options["timeout"] = min(120.0, available)
        self.messages.append({"role": "user", "content": instruction})
        response, prompt_tokens, completion_tokens, duration = self.call_model(
            self.messages, **options,
        )
        self.messages.append({"role": "assistant", "content": response})
        self.trajectory.append({
            "stage": stage, "response": response,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "llm_call_duration": duration,
        })
        self._save_history()
        return response

    def _save_history(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "trajectory.json").write_text(
            json.dumps(self.trajectory, indent=2), encoding="utf-8",
        )
        (self.artifacts_dir / "messages.json").write_text(
            json.dumps(self.messages, indent=2), encoding="utf-8",
        )

    def _write_solution(self, response: str) -> None:
        text = response.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[len("```json\n"):-3].strip()
        payload = json.loads(text)
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict) or not files or "run.py" not in files:
            raise ValueError("Implementation must contain a files object with run.py.")
        validated = []
        for filename, source in files.items():
            if not isinstance(filename, str) or not isinstance(source, str):
                raise ValueError("File names and contents must be strings.")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe generated path: {filename}")
            destination = self.workspace_path / relative
            resolved = destination.resolve()
            if not resolved.is_relative_to(self.workspace_path):
                raise ValueError(f"Generated path escapes workspace: {filename}")
            resolved_relative = resolved.relative_to(self.workspace_path)
            if (not resolved_relative.parts
                    or resolved_relative.parts[0] in {".planning", "output"}):
                raise ValueError(f"Reserved generated path: {filename}")
            if relative.suffix == ".py":
                ast.parse(source, filename=filename)
            validated.append((destination, source))
        for destination, source in validated:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source, encoding="utf-8")

    def solve_task(
        self,
        task: Dict[str, Any],
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.workspace_path = Path(task["workspace_path"]).resolve()
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.workspace_path / ".planning"
        self.system_prompt = self.build_system_prompt(task)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.trajectory = []
        options = {"max_tokens": 8192, **(llm_kwargs or {})}
        for stage, instruction in PLANNING_STAGES:
            self._record_call(stage, instruction, options)
        implementation = self._record_call("implementation", IMPLEMENTATION_PROMPT, options)
        self._write_solution(implementation)
        return {
            "trajectory": self.trajectory,
            "mode": self.mode, "model": self.model_name,
            "prompt_tokens": sum(item["prompt_tokens"] for item in self.trajectory),
            "completion_tokens": sum(item["completion_tokens"] for item in self.trajectory),
            "llm_call_duration": sum(item["llm_call_duration"] for item in self.trajectory),
        }

    def _execute(self, data_dir: Path, output_dir: Path, timeout: float) -> Dict[str, Any]:
        resolved_output = output_dir.resolve()
        if (output_dir.is_symlink() or not resolved_output.is_relative_to(self.workspace_path)
                or resolved_output == self.workspace_path
                or resolved_output.is_relative_to(self.artifacts_dir)):
            raise ValueError("OUTPUT_DIR must be a dedicated directory inside the workspace.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, DATA_DIR=str(data_dir), OUTPUT_DIR=str(output_dir))
        for key in tuple(env):
            if key.endswith(("API_KEY", "TOKEN")):
                env.pop(key)
        log_path = self.artifacts_dir / "execution.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, str(self.workspace_path / "run.py")],
                cwd=self.workspace_path, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        with log_path.open("rb") as log:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - 16000))
            excerpt = log.read().decode("utf-8", errors="replace")
        return {"returncode": process.returncode, "timed_out": timed_out, "log": excerpt}

    def run_task(
        self,
        task: Dict[str, Any],
        *,
        data_dir: Path,
        output_dir: Path,
        eval_service_url: str,
        eval_token: str,
        max_attempts: int = 3,
        execution_timeout: float = 600,
        resume: bool = False,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if max_attempts < 1 or execution_timeout <= 0:
            raise ValueError("max_attempts and execution_timeout must be positive.")
        deadline = time.monotonic() + float(task.get("time_limit_minutes", 60)) * 60

        def remaining() -> float:
            available = deadline - time.monotonic()
            timing = _service_request(eval_service_url, "/time_remaining", eval_token)
            service_remaining = timing.get("remaining_seconds")
            if isinstance(service_remaining, (float, int)):
                available = min(available, service_remaining)
            return max(0.0, available)

        if remaining() <= 0:
            raise TimeoutError("No solve time remaining.")
        self._remaining_seconds = remaining
        self.workspace_path = Path(task["workspace_path"]).resolve()
        self.artifacts_dir = self.workspace_path / ".planning"
        options = {"max_tokens": 8192, **(llm_kwargs or {})}
        feedback = None
        if resume and (self.artifacts_dir / "messages.json").is_file():
            self.messages = json.loads((self.artifacts_dir / "messages.json").read_text())
            self.trajectory = json.loads((self.artifacts_dir / "trajectory.json").read_text())
            if not (self.workspace_path / "run.py").is_file():
                feedback = {"implementation_error": "Previous generation did not produce run.py."}
        else:
            try:
                self.solve_task(task, options)
            except (ValueError, SyntaxError) as error:
                if not hasattr(self, "messages"):
                    raise
                feedback = {"implementation_error": str(error)}

        best_result = None
        best_score = float("-inf")
        for attempt in range(max_attempts):
            available = remaining()
            if available <= 1:
                break
            if feedback is not None:
                repair_prompt = (
                    "Repair or refine the implementation using this observed feedback. "
                    "Preserve scientific validity and the planned methodology. "
                    f"Remaining solve time: {available:.0f} seconds.\n"
                    + json.dumps(feedback) + "\n" + IMPLEMENTATION_PROMPT
                )
                try:
                    response = self._record_call("repair", repair_prompt, options)
                except TimeoutError:
                    break
                try:
                    self._write_solution(response)
                except (ValueError, SyntaxError) as error:
                    feedback = {"implementation_error": str(error)}
                    continue
            available = remaining()
            if available <= 1:
                break
            feedback = self._execute(data_dir, output_dir, min(execution_timeout, available))
            if feedback["returncode"] == 0 and not feedback["timed_out"]:
                evaluation_start = time.monotonic()
                evaluation = _service_request(eval_service_url, "/evaluate", eval_token)
                deadline += time.monotonic() - evaluation_start
                feedback["evaluation"] = evaluation
                score = evaluation.get("aggregate_improvement")
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = score
                    best_result = evaluation
            self.trajectory.append({
                "stage": "execution", "attempt": attempt + 1,
                "timestamp": time.time(), "feedback": feedback,
            })
            self._save_history()
        if best_result is None:
            raise RuntimeError("No scored attempt completed; see .planning/trajectory.json.")
        return best_result


def _service_request(base_url: str, endpoint: str, token: str) -> Dict[str, Any]:
    if endpoint == "/evaluate":
        request = Request(
            base_url.rstrip("/") + endpoint,
            data=json.dumps({"eval_token": token}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
    else:
        request = Request(base_url.rstrip("/") + endpoint + "?" + urlencode({"eval_token": token}))
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except HTTPError as error:
        if endpoint != "/evaluate":
            raise
        payload = {"error": error.read().decode("utf-8", errors="replace")[:16000]}
    if not isinstance(payload, dict):
        raise ValueError("Evaluation service must return a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["base", "reproduce"], default="base")
    parser.add_argument("--problem-dir", type=Path, default=Path("/task/problem"))
    parser.add_argument("--paper-path", type=Path, default=Path("/task/paper.md"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--eval-service-url", default=os.environ.get("EVAL_SERVICE_URL"))
    parser.add_argument("--eval-token", required=True)
    parser.add_argument("--time-limit-minutes", type=float, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--execution-timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.eval_service_url:
        parser.error("--eval-service-url or EVAL_SERVICE_URL is required")
    task = {
        "workspace_path": str(args.workspace),
        "task_description": (args.problem_dir / "README.md").read_text(encoding="utf-8"),
        "data_description": (args.problem_dir / "data_description.md").read_text(encoding="utf-8"),
        "time_limit_minutes": args.time_limit_minutes,
    }
    if args.mode == "reproduce":
        task["paper_content"] = args.paper_path.read_text(encoding="utf-8")
    data_dir = Path(os.environ.get("DATA_DIR", str(args.problem_dir / "data"))).resolve()
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(args.workspace / "output"))).resolve()
    data_files = (path for path in data_dir.rglob("*") if path.is_file())
    task["data_description"] += "\n\nAvailable data files (first 100):\n" + "\n".join(
        str(path.relative_to(data_dir)) for path in islice(data_files, 100)
    )
    planner = PlanningAgent(args.model, mode=args.mode, backend_kwargs={"max_retries": 0})
    result = planner.run_task(
        task, data_dir=data_dir, output_dir=output_dir,
        eval_service_url=args.eval_service_url, eval_token=args.eval_token,
        max_attempts=args.max_attempts, execution_timeout=args.execution_timeout,
        resume=args.resume,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()