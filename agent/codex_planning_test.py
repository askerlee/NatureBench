"""Focused tests for planning guidance layered onto Codex execution."""
from __future__ import annotations

import subprocess

import pytest

from agent.codex import CodexAgent
from agent.adapter import REGISTRY, AgentRunContext
import agent.cli_adapters
from agent.codex_planning import CodexPlanningAgent
from agent.planning import PLANNING_STAGES


@pytest.mark.parametrize("mode", ["base", "reference", "reproduce", "legacy_base"])
def test_prompt_preserves_parent_and_adds_planning(mode):
    task = {"eval_service_url": "http://evaluation:8321", "eval_token": "test-token",
            "time_limit_minutes": 30}
    agent = CodexPlanningAgent("gpt-test", mode=mode)
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
    assert "additional self-supervised pretraining" in prompt
    assert "better-suited permitted pretrained model" in prompt
    assert "held-out data for pretraining unless explicitly allowed" in prompt
    assert "distinguish baselines from research candidates" in prompt
    assert "Explore multiple candidates as time permits" in prompt
    assert "one baseline and one upgrade" in prompt
    assert "small setup and throughput check" in prompt
    assert "Select the best validated approach or justified ensemble" in prompt
    assert "they do not authorize replacing it" in prompt
    assert "Deprioritize CPU-first method selection" in prompt
    assert "Inspect accelerator availability" in prompt
    assert "classical methods primarily as baselines, diagnostics, or fallbacks" in prompt
    assert "validated method regardless of hardware" in prompt


@pytest.mark.parametrize("agent_class", [CodexAgent, CodexPlanningAgent])
def test_direct_solve_disables_search_and_preserves_execution(tmp_path, monkeypatch, agent_class):
    calls = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def run(command, **kwargs):
        calls.append(command)
        (workspace / "run.py").write_text("print('solution')\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent.codex.subprocess.run", run)
    agent = agent_class("gpt-test")
    result = agent.solve_task({"output_path": str(tmp_path), "workspace_path": str(workspace)})
    assert calls == [[
        "codex", "exec",
        "-C", str(tmp_path),
        "--skip-git-repo-check",
        "--yolo",
        "--json",
        "-c", "web_search=disabled",
        "-m", "gpt-test",
        "-o", str(tmp_path / "last.txt"),
        agent.system_prompt,
    ]]
    assert ("# Staged Planning And Evidence" in calls[0][-1]) is (
        agent_class is CodexPlanningAgent
    )
    assert result["codex_returncode"] == 0
    assert result["trajectory"][0]["code"] == "print('solution')\n"


@pytest.mark.parametrize("mode", ["base", "reproduce"])
def test_plan_only_prompt_uses_codex_heuristics(tmp_path, monkeypatch, mode):
    from plan_task import generate_plan

    calls = []

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def respond(self, messages, **kwargs):
            calls.append([dict(message) for message in messages])
            return "Proposed methodology", 10, 20

    monkeypatch.setattr("agent.backend.base_backend.LLMBackend", FakeBackend)
    agent = CodexPlanningAgent("gpt-test", mode=mode, plan_only=True)
    task = {
        "task_description": "Predict held-out values",
        "data_description": "Separate training and test data",
        "paper_content": "Required paper methodology",
    }
    sections = generate_plan(agent, task, stages=1)
    prompt = calls[0][0]["content"]
    assert sections == {"methodology": "Proposed methodology"}
    assert "Predict held-out values" in prompt
    assert "Separate training and test data" in prompt
    assert "Deprioritize CPU-first method selection" in prompt
    assert "additional self-supervised pretraining" in prompt
    assert "# Planning-Only Scope" in prompt
    assert "# Evaluation Service" not in prompt
    assert "# Session Rules" not in prompt
    assert ("Faithfully reproduce the supplied paper" in prompt) is (mode == "reproduce")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="cannot execute"):
        agent.solve_task(task)


@pytest.mark.parametrize("options", [
    {},
    {"codex_state_active": True},
    {"codex_use_api_key": True, "codex_api_base_url": "http://provider/v1"},
    {"is_resume": True, "codex_resume_sid": "saved-session", "codex_state_active": True},
])
def test_adapter_inherits_codex_command(options):
    ctx = AgentRunContext(system_prompt="hybrid-prompt", model="gpt-test", **options)
    hybrid = REGISTRY.get("codex-planning")
    command = hybrid.build_command(ctx)
    assert command == REGISTRY.get("codex").build_command(ctx)
    assert command[-1] == "hybrid-prompt"
    assert "--search" not in command
    assert command[command.index("web_search=disabled") - 1] == "-c"


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