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
            self.options = []
            self.responses = ["method", "architecture", "dependencies", "configuration",
                              json.dumps({"files": {"run.py": "print('solution')\n"}})]

        def respond(self, messages, **kwargs):
            self.calls.append([dict(message) for message in messages])
            self.options.append(dict(kwargs))
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


def test_plan_only_cli_writes_plan_without_implementation(planner, tmp_path, monkeypatch):
    import plan_task

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "README.md").write_text("Predict held-out values")
    (problem / "data_description.md").write_text("Separate training and test data")
    output = tmp_path / "plans" / "PLAN.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", lambda *args, **kwargs: planner)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--problem-dir", str(problem), "--output", str(output),
    ])
    plan_task.main()
    assert len(planner.backend.calls) == 4
    assert len(planner.backend.responses) == 1
    assert "Recommend scientific methods, model families" in planner.backend.calls[0][0]["content"]
    assert {"role": "assistant", "content": "method"} in planner.backend.calls[-1]
    plan = output.read_text()
    for stage in ("Methodology", "Architecture", "Dependencies", "Configuration"):
        assert f"| {stage} " in plan
    assert f"| {tmp_path.name} | OK | method | architecture | dependencies | configuration |" in plan
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
                  if path.is_file()) == [
        "plans/PLAN.md", "problem/README.md", "problem/data_description.md",
    ]


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_plan_only_cli_limits_stages_and_table_columns(planner, tmp_path, monkeypatch,
                                                     capsys, stages):
    import plan_task

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "README.md").write_text("Task")
    (problem / "data_description.md").write_text("Data")
    output = tmp_path / "plans.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", lambda *args, **kwargs: planner)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--problem-dir", str(problem), "--output", str(output),
        "--stages", str(stages),
    ])
    plan_task.main()
    assert len(planner.backend.calls) == stages
    table = output.read_text()
    for index, (stage, _) in enumerate(plan_task.PLANNING_STAGES):
        assert (f"| {stage.title()} " in table) == (index < stages)
    logs = capsys.readouterr().err
    assert f"Stage {stages}/{stages}" in logs
    assert f"Stage {stages + 1}/" not in logs
    if stages >= 2:
        assert {"role": "assistant", "content": "method"} in planner.backend.calls[1]


@pytest.mark.parametrize("stages", [0, -1, 5])
def test_plan_only_cli_rejects_invalid_stage_count(tmp_path, monkeypatch, stages):
    import plan_task

    def unexpected_backend(*args, **kwargs):
        pytest.fail("Invalid stage count must be rejected before backend initialization")

    monkeypatch.setattr(plan_task, "PlanningAgent", unexpected_backend)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--problem-dir", str(tmp_path), "--stages", str(stages),
    ])
    with pytest.raises(SystemExit) as error:
        plan_task.main()
    assert error.value.code == 2


@pytest.mark.parametrize("model, parameter", [
    ("gpt-6.1-sol", "max_completion_tokens"),
    ("gpt-4.1", "max_completion_tokens"),
    ("claude-test", "max_tokens"),
    ("deepseek-chat", "max_tokens"),
])
def test_plan_only_uses_supported_token_parameter(planner, tmp_path, model, parameter):
    from plan_task import generate_plan

    planner.model_name = model
    generate_plan(planner, task(tmp_path), max_tokens=2048)
    assert planner.backend.options == [{parameter: 2048}] * 4


def test_plan_only_stage_logs_include_usage_without_prompt(planner, tmp_path, capsys):
    from plan_task import generate_plan

    generate_plan(planner, task(tmp_path, task_description="private-task-content"))
    logs = capsys.readouterr().err
    assert "Stage 1/4 methodology: requesting model=gpt-test" in logs
    assert "Stage 4/4 configuration: completed in" in logs
    assert "prompt_tokens=10, completion_tokens=20" in logs
    assert "private-task-content" not in logs


def test_plan_only_heartbeat_reports_wait_and_stops(monkeypatch, capsys):
    import threading
    import plan_task

    reported = threading.Event()
    original_progress = plan_task.progress

    def capture_progress(message):
        original_progress(message)
        reported.set()

    monkeypatch.setattr(plan_task, "progress", capture_progress)
    with plan_task.report_wait("Stage 1/4 methodology", 0.01):
        assert reported.wait(timeout=2)
    logs = capsys.readouterr().err
    assert "still waiting for model" in logs
    assert "request or backend retries in progress" in logs


def test_plan_only_stage_failure_logs_elapsed_time(planner, tmp_path, monkeypatch, capsys):
    from plan_task import generate_plan

    def fail_call(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(planner, "call_model", fail_call)
    with pytest.raises(RuntimeError, match="Model unavailable"):
        generate_plan(planner, task(tmp_path))
    logs = capsys.readouterr().err
    assert "Stage 1/4 methodology: failed after" in logs
    assert "(RuntimeError)" in logs
    assert "Stage 2/4" not in logs


def test_plan_only_cli_requires_paper_before_model_call(tmp_path, monkeypatch):
    import plan_task

    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--problem-dir", str(tmp_path), "--mode", "reproduce",
    ])
    with pytest.raises(SystemExit) as error:
        plan_task.main()
    assert error.value.code == 2


def test_plan_only_reproduce_includes_paper(planner, tmp_path):
    from plan_task import generate_plan

    planner.mode = "reproduce"
    plan = generate_plan(planner, task(tmp_path, paper_content="Use an exact global solve"))
    prompt = planner.backend.calls[0][0]["content"]
    assert "Use an exact global solve" in prompt
    assert "rather than proposing replacements" in prompt
    assert plan == {"methodology": "method", "architecture": "architecture",
                    "dependencies": "dependencies", "configuration": "configuration"}
    assert len(planner.backend.calls) == 4
    assert not list(tmp_path.iterdir())


def test_plan_only_batch_table_is_sorted_and_context_is_isolated(planner, tmp_path, monkeypatch):
    import plan_task

    tasks = tmp_path / "tasks"
    for name in ("beta", "alpha"):
        problem = tasks / name / "problem"
        problem.mkdir(parents=True)
        (problem / "README.md").write_text(f"Task {name}")
        (problem / "data_description.md").write_text("Separate training and test data")
    planner.backend.responses = ["method | A\nnext <step>", "architecture", "dependencies",
                                 "configuration", "method B", "architecture", "dependencies",
                                 "configuration"]
    output = tmp_path / "plans.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", lambda *args, **kwargs: planner)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--tasks-dir", str(tasks), "--output", str(output),
    ])
    plan_task.main()
    assert len(planner.backend.calls) == 8
    assert len(planner.backend.calls[4]) == 2
    assert "Task beta" in planner.backend.calls[4][0]["content"]
    assert "Task alpha" not in planner.backend.calls[4][0]["content"]
    lines = output.read_text().splitlines()
    rows = [line for line in lines if line.startswith(("| alpha |", "| beta |"))]
    assert len(rows) == 2
    assert "method &#124; A<br>next &lt;step&gt;" in rows[0]
    assert all(row.count("|") == 7 for row in rows)
    assert not list(tasks.rglob("run.py"))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("failure", ["missing-data", "model-error"])
def test_plan_only_batch_records_failure_and_continues(planner, tmp_path, monkeypatch,
                                                     failure, workers):
    import plan_task

    tasks = tmp_path / "tasks"
    for name in ("bad", "good"):
        problem = tasks / name / "problem"
        problem.mkdir(parents=True)
        (problem / "README.md").write_text(name)
    (tasks / "good" / "problem" / "data_description.md").write_text("Data")
    if failure == "model-error":
        (tasks / "bad" / "problem" / "data_description.md").write_text("Data")
        original_generate = plan_task.generate_plan

        def generate_with_failure(agent, task_values, **kwargs):
            if task_values["task_description"] == "bad":
                raise RuntimeError("Model unavailable")
            return original_generate(agent, task_values, **kwargs)

        monkeypatch.setattr(plan_task, "generate_plan", generate_with_failure)
    output = tmp_path / "plans.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", lambda *args, **kwargs: planner)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--tasks-dir", str(tasks), "--output", str(output),
        "--workers", str(workers),
    ])
    with pytest.raises(SystemExit) as error:
        plan_task.main()
    assert error.value.code == 1
    table = output.read_text()
    assert "| bad | ERROR:" in table
    assert "| good | OK | method |" in table
    assert len(planner.backend.calls) == 4


def test_plan_only_parallel_tasks_are_bounded_and_table_has_one_writer(tmp_path, monkeypatch,
                                                                     capsys):
    import threading
    import plan_task

    tasks = tmp_path / "tasks"
    names = ["alpha", "beta", "delta", "gamma"]
    for name in names:
        problem = tasks / name / "problem"
        problem.mkdir(parents=True)
        (problem / "README.md").write_text(name)
        (problem / "data_description.md").write_text("Data")
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    agents = []
    active = 0
    peak = 0

    class ConcurrentPlanner:
        def __init__(self, model, **kwargs):
            self.model_name = model
            self.calls = []
            with lock:
                agents.append(self)

        def build_system_prompt(self, task_values):
            self.task_id = task_values["task_id"]
            return task_values["task_description"]

        def call_model(self, messages, **kwargs):
            nonlocal active, peak
            self.calls.append([dict(message) for message in messages])
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
            finally:
                with lock:
                    active -= 1
            return f"{self.task_id} stage {len(self.calls)}", 10, 20, 0.01

    main_thread = threading.get_ident()
    snapshots = []
    original_write = plan_task.write_table

    def record_write(output, rows, *args, **kwargs):
        assert threading.get_ident() == main_thread
        snapshots.append([row["task"] for row in rows])
        original_write(output, rows, *args, **kwargs)

    output = tmp_path / "plans.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", ConcurrentPlanner)
    monkeypatch.setattr(plan_task, "write_table", record_write)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--tasks-dir", str(tasks), "--workers", "2", "--stages", "2",
        "--output", str(output),
    ])
    plan_task.main()
    assert peak == 2
    assert len(agents) == 4
    assert sorted(agent.task_id for agent in agents) == names
    for agent in agents:
        assert len(agent.calls) == 2
        assert {"role": "assistant", "content": f"{agent.task_id} stage 1"} in agent.calls[1]
    assert [len(snapshot) for snapshot in snapshots] == [1, 2, 3, 4]
    assert all(snapshot == sorted(snapshot) for snapshot in snapshots)
    assert snapshots[-1] == names
    table = output.read_text()
    assert all(f"| {name} | OK | {name} stage 1 | {name} stage 2 |" in table for name in names)
    logs = capsys.readouterr().err
    assert "alpha: Stage 1/2 methodology" in logs
    assert "beta: Stage 1/2 methodology" in logs


@pytest.mark.parametrize("workers", [0, -1])
def test_plan_only_rejects_invalid_worker_count(tmp_path, monkeypatch, workers):
    import plan_task

    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--problem-dir", str(tmp_path), "--workers", str(workers),
    ])
    with pytest.raises(SystemExit) as error:
        plan_task.main()
    assert error.value.code == 2


def test_plan_only_batch_reproduce_uses_each_tasks_paper(planner, tmp_path, monkeypatch):
    import plan_task

    planner.mode = "reproduce"
    planner.backend.responses = ["method", "architecture", "dependencies", "configuration"] * 2
    tasks = tmp_path / "tasks"
    for name in ("alpha", "beta"):
        problem = tasks / name / "problem"
        problem.mkdir(parents=True)
        (problem / "README.md").write_text(name)
        (problem / "data_description.md").write_text("Data")
        (problem / "paper.md").write_text(f"Paper for {name}")
    output = tmp_path / "plans.md"
    monkeypatch.setattr(plan_task, "PlanningAgent", lambda *args, **kwargs: planner)
    monkeypatch.setattr("sys.argv", [
        "plan_task.py", "--tasks-dir", str(tasks), "--mode", "reproduce",
        "--paper-name", "problem/paper.md", "--output", str(output),
    ])
    plan_task.main()
    assert "Paper for alpha" in planner.backend.calls[0][0]["content"]
    assert "Paper for beta" in planner.backend.calls[4][0]["content"]
    assert "Paper for alpha" not in planner.backend.calls[4][0]["content"]
    assert "Mode: reproduce" in output.read_text()


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