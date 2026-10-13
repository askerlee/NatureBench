"""Launch the staged planning agent in NatureBench's task container."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .adapter import REGISTRY, AgentAdapter, AgentRunContext
from .claude import ClaudeAgent


class PlanningAdapter(AgentAdapter):
    name = "planning"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        return ClaudeAgent(ctx.model, mode=ctx.mode).build_system_prompt({
            "eval_service_url": ctx.eval_service_url,
            "eval_token": ctx.eval_token,
            "time_limit_minutes": ctx.time_limit_minutes,
        })

    def docker_mounts(self, ctx: AgentRunContext) -> List[str]:
        package_path = Path(__file__).resolve().parent
        return ["-v", f"{package_path}:/opt/naturebench/agent:ro"]

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        command = [
            "bash", "-lc",
            'set -e; python -m pip install "openai==2.31.0" "backoff==2.2.1"; '
            'export PYTHONPATH="/opt/naturebench${PYTHONPATH:+:$PYTHONPATH}"; '
            'exec python -m agent.planning "$@"',
            "planning",
            "--model", ctx.model,
            "--mode", ctx.mode,
            "--eval-service-url", ctx.eval_service_url,
            "--eval-token", ctx.eval_token,
            "--time-limit-minutes", str(ctx.time_limit_minutes),
            "--max-attempts", os.environ.get("NATUREBENCH_PLANNING_MAX_ATTEMPTS", "3"),
            "--execution-timeout", os.environ.get("NATUREBENCH_PLANNING_EXECUTION_TIMEOUT", "600"),
        ]
        if ctx.is_resume:
            command.append("--resume")
        return command

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        return task_out_dir / "workspace" / ".planning" / "trajectory.json"

    def excerpt_transcript(
        self,
        log_path: Path,
        *,
        max_bytes: int = 2_000_000,
        focus_start: Optional[float] = None,
        focus_end: Optional[float] = None,
    ) -> Optional[str]:
        if not log_path.is_file():
            return None
        content = log_path.read_bytes()
        return content[:max_bytes].decode("utf-8", errors="ignore")


REGISTRY.register(PlanningAdapter())