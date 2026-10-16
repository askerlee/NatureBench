"""Built-in adapters for the Claude Code, Codex, Gemini, and Copilot CLIs.

Each adapter wraps the existing prompt builders (the ``*Agent`` classes) and the
canonical command builders (``agent.cli_commands``), plus the agent-specific
session-log location used by the post-hoc judge. Behaviour must stay identical
to the pre-adapter dispatch in ``solve.py``; the equivalence tests pin this.

Importing this module registers the built-in adapters with the global
``REGISTRY``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter import REGISTRY, AgentAdapter, AgentRunContext
from . import cli_commands as cc
from .claude import ClaudeAgent
from .codex import CodexAgent
from .codex_planning import CodexPlanningAgent
from .gemini import GeminiAgent


def _task_info(ctx: AgentRunContext) -> Dict[str, Any]:
    """Reconstruct the prompt-template fields the *Agent classes expect."""
    return {
        "eval_service_url": ctx.eval_service_url,
        "eval_token": ctx.eval_token,
        "time_limit_minutes": ctx.time_limit_minutes,
    }


def _newest(paths: List[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return None
    try:
        return max(existing, key=lambda p: p.stat().st_mtime)
    except OSError:
        return existing[0]


class ClaudeAdapter(AgentAdapter):
    name = "claude"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        if ctx.is_resume:
            return cc.RESUME_PROMPT_CLAUDE
        return ClaudeAgent(model_name=ctx.model, mode=ctx.mode).build_system_prompt(_task_info(ctx))

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        return cc.build_claude_cmd(
            ctx.system_prompt, ctx.model,
            session_id=ctx.session_id,
            resume_session=ctx.is_resume,
        )

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        # Internal session jsonl is the most complete record; fall back to the
        # streamed stdout log written by solve.py.
        proj_dir = task_out_dir / ".claude_state" / "projects"
        if proj_dir.is_dir():
            cands = list(proj_dir.rglob("*.jsonl"))
            sid_file = task_out_dir / "claude_session_id.txt"
            if sid_file.exists():
                sid = sid_file.read_text(encoding="utf-8").strip()
                for c in cands:
                    if c.stem == sid:
                        return c
            newest = _newest(cands)
            if newest is not None:
                return newest
        stream = task_out_dir / "claude.jsonl"
        return stream if stream.exists() else None


class CodexAdapter(AgentAdapter):
    name = "codex"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        prompt = CodexAgent(model_name=ctx.model, mode=ctx.mode).build_system_prompt(_task_info(ctx))
        if ctx.is_resume:
            return cc.RESUME_PROMPT_CODEX + "\n\n" + prompt
        return prompt

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        cmd = cc.build_codex_cmd(
            ctx.system_prompt, ctx.model,
            resume_session_id=ctx.codex_resume_sid,
            use_api_key=ctx.codex_use_api_key,
            api_base_url=ctx.codex_api_base_url,
        )
        if ctx.codex_state_active:
            cmd = cc.codex_exec_cmd(cmd)
        return cmd

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        sess_dir = task_out_dir / ".codex_state" / "sessions"
        if sess_dir.is_dir():
            cands = list(sess_dir.rglob("rollout-*.jsonl"))
            sid_file = task_out_dir / "codex_session_id.txt"
            if sid_file.exists():
                sid = sid_file.read_text(encoding="utf-8").strip()
                for c in cands:
                    if sid in c.name:
                        return c
            newest = _newest(cands)
            if newest is not None:
                return newest
        stream = task_out_dir / "codex.jsonl"
        return stream if stream.exists() else None


class CodexPlanningAdapter(CodexAdapter):
    name = "codex-planning"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        agent = CodexPlanningAgent(model_name=ctx.model, mode=ctx.mode)
        prompt = agent.build_system_prompt(_task_info(ctx))
        if ctx.is_resume:
            return cc.RESUME_PROMPT_CODEX + "\n\n" + prompt
        return prompt

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        parent_path = super().transcript_path(task_out_dir)
        if parent_path is not None:
            return parent_path
        stream = task_out_dir / f"{self.name}.jsonl"
        return stream if stream.exists() else None


class GeminiAdapter(AgentAdapter):
    name = "gemini"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        if ctx.is_resume:
            return cc.RESUME_PROMPT_GEMINI
        return GeminiAgent(model_name=ctx.model, mode=ctx.mode).build_system_prompt(_task_info(ctx))

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        return cc.build_gemini_cmd(
            ctx.system_prompt, ctx.model,
            session_id=ctx.session_id,
            resume_session=ctx.is_resume,
        )

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        tmp_dir = task_out_dir / ".gemini_state" / "tmp"
        if tmp_dir.is_dir():
            cands: List[Path] = []
            for pat in ("*/chats/*.json", "*/chats/*.jsonl"):
                cands.extend(tmp_dir.glob(pat))
            sid_file = task_out_dir / "gemini_session_id.txt"
            if sid_file.exists():
                sid = sid_file.read_text(encoding="utf-8").strip()
                short = sid[:8]
                for c in cands:
                    if c.stem == sid or sid in c.name or short in c.name:
                        return c
            newest = _newest(cands)
            if newest is not None:
                return newest
        stream = task_out_dir / "gemini.jsonl"
        return stream if stream.exists() else None


class CopilotAdapter(AgentAdapter):
    name = "copilot"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        return CodexAgent(model_name=ctx.model, mode=ctx.mode).build_system_prompt(_task_info(ctx))

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        if ctx.is_resume:
            raise ValueError("Copilot session resume is not supported by NatureBench")
        return cc.build_copilot_cmd(ctx.system_prompt, ctx.model)

    def extra_env(self, ctx: AgentRunContext) -> List[str]:
        for key in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            token = os.environ.get(key)
            if token:
                return [
                    "-e", f"COPILOT_GITHUB_TOKEN={token}",
                    "-e", "COPILOT_HOME=/tmp/naturebench-copilot",
                ]
        raise ValueError(
            "Copilot requires COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. "
            "Use a fine-grained PAT with Copilot Requests permission or a supported OAuth token."
        )

    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        stream = task_out_dir / f"{self.name}.jsonl"
        return stream if stream.is_file() else None

    def excerpt_transcript(
        self,
        log_path: Path,
        *,
        max_bytes: int = 2_000_000,
        focus_start: Optional[float] = None,
        focus_end: Optional[float] = None,
    ) -> Optional[str]:
        if max_bytes <= 0:
            return ""
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - max_bytes))
            return stream.read(max_bytes).decode("utf-8", errors="ignore")


class CopilotPlanningAdapter(CopilotAdapter):
    name = "copilot-planning"

    def system_prompt(self, ctx: AgentRunContext) -> str:
        return CodexPlanningAdapter().system_prompt(ctx)


REGISTRY.register(ClaudeAdapter())
REGISTRY.register(CodexAdapter())
REGISTRY.register(CodexPlanningAdapter())
REGISTRY.register(GeminiAdapter())
REGISTRY.register(CopilotAdapter())
REGISTRY.register(CopilotPlanningAdapter())
