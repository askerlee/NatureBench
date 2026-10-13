"""Focused tests for staged planning and the generated-file contract."""
from __future__ import annotations

import json
import io
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from agent.adapter import REGISTRY, AgentRunContext
from agent.planning import PlanningAgent, _service_request


@pytest.fixture
def planner(monkeypatch):
    class FakeBackend:
        def __init__(self, *args, **kwargs):
            self.calls = []
            self.responses = ["method", "architecture", "dependencies", "configuration",
                              json.dumps({"files": {"run.py": "print('solution')\n"}})]

        def respond(self, messages, **kwargs):
            self.calls.append([dict(message) for message in messages])
            return self.responses.pop(0), 10, 20

    monkeypatch.setattr("agent.backend.base_backend.LLMBackend", FakeBackend)
    return PlanningAgent("gpt-test")


def task(tmp_path, **overrides):
    values = {"workspace_path": str(tmp_path), "task_description": "Predict held-out values",
              "data_description": "Training and test data are separate"}
    values.update(overrides)
    return values


def test_stages_keep_context_and_write_solution(planner, tmp_path):
    result = planner.solve_task(task(tmp_path))
    assert [item["stage"] for item in result["trajectory"]] == [
        "methodology", "architecture", "dependencies", "configuration", "implementation",
    ]
    assert len(planner.backend.calls) == 5
    assert {"role": "assistant", "content": "method"} in planner.backend.calls[-1]
    assert (tmp_path / "run.py").read_text() == "print('solution')\n"
    assert result["prompt_tokens"] == 50
    assert result["completion_tokens"] == 100
    assert len(json.loads((tmp_path / ".planning/trajectory.json").read_text())) == 5


def test_reproduce_requires_paper_and_preserves_fidelity(planner, tmp_path):
    planner.mode = "reproduce"
    with pytest.raises(ValueError, match="paper_content"):
        planner.solve_task(task(tmp_path))
    prompt = planner.build_system_prompt(task(tmp_path, paper_content="Use an exact global solve"))
    assert "Use an exact global solve" in prompt
    assert "Faithfully reproduce" in prompt
    assert "DEVIATIONS.md" in prompt


@pytest.mark.parametrize("files", [
    {"main.py": "pass"},
    {"run.py": "pass", "../escaped.py": "pass"},
    {"run.py": "pass", "/tmp/escaped.py": "pass"},
    {"run.py": "pass", ".planning/messages.json": "{}"},
    {"run.py": "pass", "output/result.txt": "fake"},
    {"run.py": "pass", "broken.py": "def broken(:"},
])
def test_invalid_implementation_writes_nothing(planner, tmp_path, files):
    planner.backend.responses[-1] = json.dumps({"files": files})
    with pytest.raises((ValueError, SyntaxError)):
        planner.solve_task(task(tmp_path))
    assert not (tmp_path / "run.py").exists()


def test_symlink_escape_is_rejected(planner, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "outside").symlink_to(tmp_path, target_is_directory=True)
    planner.backend.responses[-1] = json.dumps({
        "files": {"run.py": "pass", "outside/escaped.py": "pass"},
    })
    with pytest.raises(ValueError, match="escapes workspace"):
        planner.solve_task(task(workspace))
    assert not (tmp_path / "escaped.py").exists()


def run_options(tmp_path, **overrides):
    options = {
        "data_dir": tmp_path / "data", "output_dir": tmp_path / "output",
        "eval_service_url": "http://evaluation", "eval_token": "opaque-token",
        "max_attempts": 2, "execution_timeout": 5,
    }
    options.update(overrides)
    return options


def fake_service(monkeypatch, evaluations):
    requests = []

    def request(base_url, endpoint, token):
        requests.append((endpoint, token))
        if endpoint == "/time_remaining":
            return {"remaining_seconds": 300}
        return evaluations.pop(0)

    monkeypatch.setattr("agent.planning._service_request", request)
    return requests


def test_execution_error_is_repaired_and_evaluated(planner, tmp_path, monkeypatch):
    planner.backend.responses[-1] = json.dumps({"files": {"run.py": "raise RuntimeError('bad')"}})
    planner.backend.responses.append(json.dumps({"files": {"run.py": "print('fixed')"}}))
    requests = fake_service(monkeypatch, [{"aggregate_improvement": 0.2}])
    result = planner.run_task(task(tmp_path), **run_options(tmp_path))
    assert result["aggregate_improvement"] == 0.2
    assert sum(endpoint == "/evaluate" for endpoint, _ in requests) == 1
    repair = planner.backend.calls[-1][-1]["content"]
    assert "RuntimeError: bad" in repair
    assert planner.trajectory[-1]["feedback"]["returncode"] == 0


def test_invalid_generation_is_repaired(planner, tmp_path, monkeypatch):
    planner.backend.responses[-1] = "not JSON"
    planner.backend.responses.append(json.dumps({"files": {"run.py": "pass"}}))
    fake_service(monkeypatch, [{"aggregate_improvement": 0}])
    result = planner.run_task(task(tmp_path), **run_options(tmp_path, max_attempts=1))
    assert result["aggregate_improvement"] == 0
    assert "implementation_error" in planner.backend.calls[-1][-1]["content"]


def test_refinement_keeps_best_score(planner, tmp_path, monkeypatch):
    planner.backend.responses.append(json.dumps({"files": {"run.py": "pass"}}))
    fake_service(monkeypatch, [{"aggregate_improvement": 0.4}, {"aggregate_improvement": 0.1}])
    result = planner.run_task(task(tmp_path), **run_options(tmp_path))
    assert result["aggregate_improvement"] == 0.4
    assert "0.4" in planner.backend.calls[-1][-1]["content"]


def test_resume_uses_saved_context_without_replanning(planner, tmp_path, monkeypatch):
    planner.solve_task(task(tmp_path))
    planner.backend.calls.clear()
    fake_service(monkeypatch, [{"aggregate_improvement": 0.1}])
    planner.run_task(task(tmp_path), **run_options(tmp_path, max_attempts=1, resume=True))
    assert planner.backend.calls == []
    assert planner.trajectory[-1]["stage"] == "execution"


def test_expired_budget_does_not_call_model(planner, tmp_path, monkeypatch):
    monkeypatch.setattr("agent.planning._service_request", lambda *args: {"remaining_seconds": 0})
    with pytest.raises(TimeoutError):
        planner.run_task(task(tmp_path), **run_options(tmp_path))
    assert planner.backend.calls == []


def test_timeout_is_reported(planner, tmp_path):
    planner.backend.responses[-1] = json.dumps({
        "files": {"run.py": "while True:\n    pass\n"},
    })
    planner.solve_task(task(tmp_path))
    result = planner._execute(tmp_path / "data", tmp_path / "output", timeout=0.1)
    assert result["timed_out"]
    assert result["returncode"] != 0


def test_adapter_launch_and_mount_contract():
    adapter = REGISTRY.get("planning")
    ctx = AgentRunContext(
        system_prompt="", model="deepseek-chat", mode="reproduce", eval_token="opaque-token",
        eval_service_url="http://evaluation", time_limit_minutes=45, is_resume=True,
    )
    command = adapter.build_command(ctx)
    assert "agent.planning" in command[2]
    assert command[command.index("--model") + 1] == "deepseek-chat"
    assert command[command.index("--mode") + 1] == "reproduce"
    assert command[-1] == "--resume"
    mounts = adapter.docker_mounts(ctx)
    assert mounts[0] == "-v"
    assert mounts[1].endswith(":/opt/naturebench/agent:ro")
    assert Path(mounts[1].split(":")[0]).is_dir()
    assert adapter.transcript_path(Path("results/task")) == Path(
        "results/task/workspace/.planning/trajectory.json",
    )


def test_previous_outputs_are_removed_and_credentials_not_inherited(planner, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret-for-test")
    planner.backend.responses[-1] = json.dumps({"files": {"run.py": (
        "import os\nfrom pathlib import Path\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        "output = Path(os.environ['OUTPUT_DIR'])\n"
        "assert not (output / 'stale.txt').exists()\n"
        "(output / 'result.txt').write_text('new')\n"
    )}})
    planner.solve_task(task(tmp_path))
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("old")
    result = planner._execute(tmp_path / "data", output, timeout=5)
    assert result["returncode"] == 0
    assert (output / "result.txt").read_text() == "new"
    assert not (output / "stale.txt").exists()


def test_budget_is_checked_between_planning_calls(planner, tmp_path, monkeypatch):
    budgets = iter([300, 300, 0])
    monkeypatch.setattr("agent.planning._service_request", lambda *args: {
        "remaining_seconds": next(budgets),
    })
    with pytest.raises(TimeoutError):
        planner.run_task(task(tmp_path), **run_options(tmp_path))
    assert len(planner.backend.calls) == 1


def test_no_scored_attempt_reports_failure(planner, tmp_path, monkeypatch):
    fake_service(monkeypatch, [{"aggregate_improvement": None, "error": "bad outputs"}])
    with pytest.raises(RuntimeError, match="No scored attempt"):
        planner.run_task(task(tmp_path), **run_options(tmp_path, max_attempts=1))


@pytest.mark.parametrize("endpoint", ["/evaluate", "/time_remaining"])
def test_service_request_protocol(monkeypatch, endpoint):
    captured = []

    def urlopen(request, timeout):
        captured.append(request)
        return io.BytesIO(b'{"remaining_seconds": 50}')

    monkeypatch.setattr("agent.planning.urlopen", urlopen)
    assert _service_request("http://evaluation/", endpoint, "token +/?") == {"remaining_seconds": 50}
    request = captured[0]
    assert urlparse(request.full_url).path == endpoint
    if endpoint == "/evaluate":
        assert request.get_method() == "POST"
        assert json.loads(request.data) == {"eval_token": "token +/?"}
    else:
        assert request.get_method() == "GET"
        assert parse_qs(urlparse(request.full_url).query) == {"eval_token": ["token +/?"]}


def test_reserved_symlink_alias_is_rejected(planner, tmp_path):
    (tmp_path / ".planning").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / ".planning", target_is_directory=True)
    planner.backend.responses[-1] = json.dumps({
        "files": {"run.py": "pass", "alias/messages.json": "{}"},
    })
    with pytest.raises(ValueError, match="Reserved"):
        planner.solve_task(task(tmp_path))


def test_evaluator_failure_response_is_feedback(monkeypatch):
    def urlopen(request, timeout):
        raise HTTPError(request.full_url, 500, "evaluation failed", {},
                        io.BytesIO(b'{"error": "wrong output shape"}'))

    monkeypatch.setattr("agent.planning.urlopen", urlopen)
    result = _service_request("http://evaluation", "/evaluate", "token")
    assert "wrong output shape" in result["error"]
    with pytest.raises(HTTPError):
        _service_request("http://evaluation", "/time_remaining", "token")


def test_adapter_iteration_settings(monkeypatch):
    monkeypatch.setenv("NATUREBENCH_PLANNING_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("NATUREBENCH_PLANNING_EXECUTION_TIMEOUT", "1800")
    command = REGISTRY.get("planning").build_command(AgentRunContext(system_prompt="", model="gpt-test"))
    assert command[command.index("--max-attempts") + 1] == "5"
    assert command[command.index("--execution-timeout") + 1] == "1800"