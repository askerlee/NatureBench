"""Focused tests for planning guidance layered onto Codex execution."""
from __future__ import annotations

import subprocess

import pytest

from agent.codex import CodexAgent
from agent.adapter import REGISTRY, AgentRunContext
import agent.cli_adapters
from agent.codex_planning import PlanningCodexAgent
from agent.planning import PLANNING_STAGES


@pytest.mark.parametrize("mode", ["base", "reference", "reproduce", "legacy_base"])
def test_prompt_preserves_parent_and_adds_planning(mode):
    task = {"eval_service_url": "http://evaluation:8321", "eval_token": "test-token",
            "time_limit_minutes": 30}
    agent = PlanningCodexAgent("gpt-test", mode=mode)
    prompt = agent.build_system_prompt(task)
    assert prompt.startswith(CodexAgent("gpt-test", mode=mode).build_system_prompt(task))
    for stage, instruction in PLANNING_STAGES:
        assert f"**{stage.title()}**: {instruction}" in prompt
    assert "PLAN.md" in prompt
    assert "experiments.jsonl" in prompt
    assert "at most 10%" in prompt
    assert "output-contract checks" in prompt
    assert "rather than switching" in prompt
    assert "no fixed attempt limit" in prompt


def test_inherited_solve_uses_planning_prompt(tmp_path, monkeypatch):
    calls = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def run(command, **kwargs):
        calls.append(command)
        (workspace / "run.py").write_text("print('solution')\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent.codex.subprocess.run", run)
    agent = PlanningCodexAgent("gpt-test")
    result = agent.solve_task({"output_path": str(tmp_path), "workspace_path": str(workspace)})
    assert PlanningCodexAgent.solve_task is CodexAgent.solve_task
    assert calls[0][0] == "codex"
    assert calls[0][-1] == agent.system_prompt
    assert "# Staged Planning And Evidence" in calls[0][-1]
    assert result["codex_returncode"] == 0
    assert result["trajectory"][0]["code"] == "print('solution')\n"


@pytest.mark.parametrize("options", [
    {},
    {"codex_state_active": True},
    {"codex_use_api_key": True, "codex_api_base_url": "http://provider/v1"},
    {"is_resume": True, "codex_resume_sid": "saved-session", "codex_state_active": True},
])
def test_adapter_inherits_codex_command(options):
    ctx = AgentRunContext(system_prompt="hybrid-prompt", model="gpt-test", **options)
    hybrid = REGISTRY.get("codex-planning")
    assert hybrid.build_command(ctx) == REGISTRY.get("codex").build_command(ctx)
    assert hybrid.build_command(ctx)[-1] == "hybrid-prompt"


@pytest.mark.parametrize("is_resume", [False, True])
def test_adapter_prompt_keeps_planning_on_resume(is_resume):
    ctx = AgentRunContext(system_prompt="", model="gpt-test", is_resume=is_resume)
    prompt = REGISTRY.get("codex-planning").system_prompt(ctx)
    assert "# Staged Planning And Evidence" in prompt
    assert "On resume, read PLAN.md" in prompt
    assert ("RESUME NOTICE" in prompt) is is_resume


def test_adapter_transcript_prefers_codex_session(tmp_path):
    adapter = REGISTRY.get("codex-planning")
    assert adapter.transcript_path(tmp_path) is None
    stream = tmp_path / "codex-planning.jsonl"
    stream.touch()
    assert adapter.transcript_path(tmp_path) == stream
    sessions = tmp_path / ".codex_state" / "sessions"
    sessions.mkdir(parents=True)
    rollout = sessions / "rollout-saved-session.jsonl"
    rollout.touch()
    (tmp_path / "codex_session_id.txt").write_text("saved-session")
    assert adapter.transcript_path(tmp_path) == rollout


def test_harness_resume_uses_codex_state(tmp_path):
    from solve import _resume_eligible

    (tmp_path / "result.json").write_text("{}")
    assert _resume_eligible(tmp_path, "codex-planning") == (False, "no codex_session_id.txt")
    (tmp_path / "codex_session_id.txt").write_text("saved-session")
    (tmp_path / ".codex_state").mkdir()
    assert _resume_eligible(tmp_path, "codex-planning") == (True, "ok")