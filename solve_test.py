"""Offline regression tests for cumulative selected-task coverage."""
from __future__ import annotations

import copy
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import solve


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def finished(out, name, status="success", returncode=0, **extra):
    result = dict(task_name=name, status=status, returncode=returncode, duration=12.5)
    result.update(extra)
    write_json(out / name / "result.json", result)
    return result


def attempt(number, score, kind="success"):
    return dict(type=kind, attempt=number, aggregate_improvement=score,
                per_instance_improvement={"instance": score} if score is not None else {},
                raw_scores={"instance": {"metric": number}} if score is not None else {})


def write_attempts(out, name, records):
    (out / name / "submissions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
    )


def invoke(monkeypatch, tmp_path, tasks, extra=()):
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("\n".join(tasks) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "solve.py", "--task-set", str(task_file), "--data-dir", str(tmp_path / "data"),
        "--out-dir", str(tmp_path / "out"), "--agent", "claude", "--model", "test",
        *extra,
    ])
    solve.main()
    return json.loads((tmp_path / "out" / "run_summary.json").read_text())


def forbid(monkeypatch, *names):
    for name in names:
        monkeypatch.setattr(solve, name, Mock(side_effect=AssertionError(name)))


@pytest.mark.parametrize("status,code", [
    ("error", None), ("agent_error", 1), ("killed", 137), ("eval_skip", -2),
    ("success", 1), ("timeout", 137), ("success", True), ("pending", None),
])
def test_incomplete_execution_requires_choice(tmp_path, status, code):
    finished(tmp_path, "task", status, code)
    with pytest.raises(ValueError, match="incomplete or ambiguous"):
        solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert solve._select_execution_tasks(["task"], tmp_path, set(), {"task"}) == (["task"], {})
    assert solve._select_execution_tasks(["task"], tmp_path, {"task"}, set()) == (["task"], {})


@pytest.mark.parametrize("payload", [
    "{", "[]", '{"task_name":"task","status":"success"}',
    '{"task_name":"other","status":"success","returncode":0,"duration":1}',
    '{"task_name":"task","status":"success","returncode":0,"duration":NaN}',
    '{"task_name":"task","status":"success","returncode":0,"duration":-1}',
    '{"task_name":"task","status":"success","returncode":0,"duration":true}',
    '{"task_name":"task","status":"success","returncode":0,"duration":1,"resume_history":{}}',
    '{"task_name":"task","status":"success","returncode":0.0,"duration":1}',
    '{"task_name":"task","status":"success","returncode":0,"duration":1,"resume_history":[{}]}',
])
def test_malformed_execution_not_silently_skipped(tmp_path, payload):
    (tmp_path / "task").mkdir()
    path = tmp_path / "task" / "result.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert path.read_text() == payload
    assert solve._select_execution_tasks(["task"], tmp_path, set(), {"task"}) == (["task"], {})


@pytest.mark.parametrize("artifact", ["workspace", ".gemini_state", "custom-agent.jsonl", "eval_token.txt"])
def test_artifact_without_result_requires_choice(tmp_path, artifact):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / artifact).touch()
    with pytest.raises(ValueError, match="incomplete or ambiguous"):
        solve._select_execution_tasks(["task"], tmp_path, set(), set())


def test_archive_is_not_active_state(tmp_path):
    finished(tmp_path / "task" / "_force_fresh_archive_1", "task")
    assert solve._select_execution_tasks(["task"], tmp_path, set(), set()) == (["task"], {})


def test_restore_execution_only_result_scores_attempts_and_judge(tmp_path):
    raw = finished(tmp_path, "task", "timeout", -1)
    write_attempts(tmp_path, "task", [
        attempt(1, -0.3), attempt(2, None, "failure"), attempt(3, 0.4),
        attempt(4, 0.4), attempt(5, None),
    ])
    verdict = {"is_valid": False, "reason": "invalid method"}
    write_json(tmp_path / "task" / "judge_verdict.json", verdict)
    execution, reused = solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert execution == []
    result = reused["task"]
    assert result["status"] == "timeout"
    assert result["best_attempt"] == 3
    assert result["best_aggregate_improvement"] == 0.4
    assert result["best_raw_scores"] == {"instance": {"metric": 3}}
    assert result["best_per_instance_improvement"] == {"instance": 0.4}
    assert result["total_attempts"] == 5
    assert result["judge"] == verdict
    assert result["effective_improvement"] is None
    assert json.loads((tmp_path / "task" / "result.json").read_text()) == raw


def test_saved_summary_survives_empty_tracker_and_lower_journal_score(tmp_path):
    raw = finished(tmp_path, "task", resume_history=[
        {"status": "success", "returncode": 0, "duration": 12.5},
    ])
    saved = dict(raw, best_attempt=8, best_aggregate_improvement=0.7,
                 best_raw_scores={"saved": 1}, total_attempts=9,
                 judge={"is_valid": True, "reason": "saved verdict"},
                 effective_improvement=0.7)
    write_json(tmp_path / "run_summary.json", {"results": [saved]})
    write_attempts(tmp_path, "task", [attempt(1, -0.1)])
    _, reused = solve._select_execution_tasks(["task"], tmp_path, set(), set())
    summary = solve._build_summary(list(reused.values()), solve.ScoreTracker(), 0.1)
    assert summary["results"][0] == dict(saved, best_per_instance_improvement={})
    assert summary["successes"] == summary["scored_tasks"] == 1
    assert summary["total_duration"] == 0.1
    assert summary["cumulative_task_duration"] == 12.5


def test_stale_summary_not_attached_to_new_execution(tmp_path):
    raw = finished(tmp_path, "task")
    write_json(tmp_path / "run_summary.json", {"results": [
        dict(raw, duration=500, best_aggregate_improvement=99, judge={"is_valid": False}),
    ]})
    _, reused = solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert "best_aggregate_improvement" not in reused["task"]
    assert "judge" not in reused["task"]


def test_old_judge_not_reused_after_unjudged_resume(tmp_path):
    raw = finished(tmp_path, "task", resume_history=[
        {"status": "success", "returncode": 0, "duration": 12.5},
    ])
    verdict_path = tmp_path / "task" / "judge_verdict.json"
    write_json(verdict_path, {"is_valid": False, "reason": "previous execution"})
    result_path = tmp_path / "task" / "result.json"
    os.utime(verdict_path, ns=(1, 1))
    write_json(tmp_path / "run_summary.json", {"results": [raw]})
    _, reused = solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert "judge" not in reused["task"]
    assert result_path.stat().st_mtime_ns > verdict_path.stat().st_mtime_ns
    saved = dict(raw, judge={"is_valid": True, "reason": "matching summary"})
    write_json(tmp_path / "run_summary.json", {"results": [saved]})
    _, reused = solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert reused["task"]["judge"] == saved["judge"]


@pytest.mark.parametrize("payload", ["{", '{"attempt":1}', '{"attempt":1,"aggregate_improvement":NaN}'])
def test_malformed_journal_preserved(tmp_path, payload):
    finished(tmp_path, "task")
    (tmp_path / "task" / "submissions.jsonl").write_text(payload)
    with pytest.raises(ValueError, match="saved attempts"):
        solve._select_execution_tasks(["task"], tmp_path, set(), set())


@pytest.mark.parametrize("field,value", [
    ("best_aggregate_improvement", float("nan")), ("total_attempts", "5"),
    ("best_attempt", True), ("best_raw_scores", []), ("judge", {}),
])
def test_malformed_cached_reporting_state_is_not_overwritten(tmp_path, field, value):
    raw = finished(tmp_path, "task")
    write_json(tmp_path / "run_summary.json", {"results": [dict(raw, **{field: value})]})
    before = (tmp_path / "run_summary.json").read_bytes()
    with pytest.raises(ValueError, match="malformed"):
        solve._select_execution_tasks(["task"], tmp_path, set(), set())
    assert (tmp_path / "run_summary.json").read_bytes() == before


def test_explicit_fresh_can_bypass_malformed_summary(tmp_path):
    finished(tmp_path, "task")
    (tmp_path / "run_summary.json").write_text("{")
    assert solve._select_execution_tasks(["task"], tmp_path, set(), {"task"}) == (["task"], {})


def test_all_finished_is_offline_and_preserves_raw_artifacts(monkeypatch, tmp_path):
    out = tmp_path / "out"
    a = finished(out, "a")
    finished(out, "b", "timeout", -1)
    saved = dict(a, best_attempt=1, best_aggregate_improvement=0.8, total_attempts=2,
                 judge={"is_valid": False, "reason": "saved"}, effective_improvement=None)
    write_json(out / "run_summary.json", {"results": [saved]})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in out.glob("*/result.json")}
    forbid(monkeypatch, "_run_single_task", "_probe_codex_login",
           "_build_gpu_pool", "_ensure_clash_proxy_container", "load_or_create_control_token",
           "_load_eval_env_mapping", "_load_or_create_eval_token",
           "_register_task_with_token_retry", "_fetch_score_from_service",
           "start_server_background")
    summary = invoke(monkeypatch, tmp_path, ["b", "a", "a"], [
        "--agent", "codex", "--eval-env-mapping", "/nonexistent/mapping.json",
        "--gpu-devices", "0", "--proxy-mode", "sidecar",
    ])
    assert [r["task_name"] for r in summary["results"]] == ["b", "a"]
    assert summary["total_tasks"] == 2
    assert summary["successes"] == 1
    assert summary["skipped_tasks"] == ["b", "a"]
    assert summary["results"][1]["judge"] == saved["judge"]
    assert summary["results"][1]["best_aggregate_improvement"] == 0.8
    assert summary["cumulative_task_duration"] == 25
    assert summary["total_duration"] < 10
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.fixture
def external_runner(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(solve, "load_or_create_control_token", lambda *_: None)
    monkeypatch.setattr(solve, "_load_eval_env_mapping", lambda *_: ({}, 8321))
    monkeypatch.setattr(solve.time, "sleep", lambda *_: None)
    forbid(monkeypatch, "start_server_background", "_probe_codex_login")
    registered = []
    launched = []
    fetched = []
    snapshots = []
    writer = solve._write_summary

    def register(port, name, *args, **kwargs):
        registered.append((name, kwargs["force"]))
        return "ok", {}, name

    def run(name, *args, **kwargs):
        launched.append((name, kwargs["is_resume"]))
        return finished(tmp_path / "out", name, duration=3)

    def fetch(port, token, name):
        assert token == name
        fetched.append(name)
        return {}

    def snapshot(path, results, *args):
        snapshots.append(copy.deepcopy(results))
        writer(path, results, *args)

    monkeypatch.setattr(solve, "_register_task_with_token_retry", register)
    monkeypatch.setattr(solve, "_load_or_create_eval_token", lambda path, _: path.name)
    monkeypatch.setattr(solve.ScoreTracker, "register_task", lambda *a, **k: ({}, []))
    monkeypatch.setattr(solve, "_run_single_task", run)
    monkeypatch.setattr(solve, "_fetch_score_from_service", fetch)
    monkeypatch.setattr(solve, "_write_summary", snapshot)
    return data, registered, launched, fetched, snapshots


@pytest.mark.parametrize("workers", [1, 3])
def test_two_finished_plus_23_fresh_full_track(
    monkeypatch, tmp_path, external_runner, workers,
):
    data, registered, launched, fetched, snapshots = external_runner
    tasks = solve._read_task_set(ROOT / "task-set" / "naturebench-25" / "all.txt")
    out = tmp_path / "out"
    saved = []
    for index, name in enumerate(tasks[:2]):
        raw = finished(out, name)
        saved.append(dict(raw, best_attempt=1, best_aggregate_improvement=0.2,
                          total_attempts=1,
                          judge={"is_valid": bool(index), "reason": "cached"},
                          effective_improvement=0.2 if index else None))
    write_json(out / "run_summary.json", {"results": saved})
    finished(out, "unrelated")
    finished(out / tasks[0] / "_force_fresh_archive_1", "archived")
    for name in tasks[2:]:
        (data / name).mkdir()
        finished(out, name, "agent_error", 1)
    before = {p: p.read_bytes() for name in tasks[:2] for p in (out / name).rglob("*") if p.is_file()}

    import judge
    judges = Mock(return_value={})
    monkeypatch.setattr(judge, "run_judges", judges)
    summary = invoke(monkeypatch, tmp_path, tasks + [tasks[0]], [
        "--eval-env-mapping", "unused", "--max-workers", str(workers),
        "--force-fresh", *tasks[2:],
    ])
    assert len(registered) == len(launched) == 23
    assert set(name for name, _ in registered) == set(tasks[2:])
    assert all(force for _, force in registered)
    assert all(not resume for _, resume in launched)
    assert set(fetched) == set(tasks[2:])
    judges.assert_not_called()
    assert summary["total_tasks"] == summary["successes"] == 25
    assert summary["scored_tasks"] == 2
    assert summary["average_best_aggregate_improvement"] == 0.2
    assert summary["skipped_tasks"] == tasks[:2]
    assert [r["task_name"] for r in summary["results"]] == tasks
    assert summary["results"][:2] == [dict(r, best_per_instance_improvement={}, best_raw_scores={}) for r in saved]
    assert snapshots[0][2]["status"] == "pending"
    for snapshot in snapshots:
        assert [r["task_name"] for r in snapshot] == tasks
        assert snapshot[0]["judge"] == saved[0]["judge"]
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(list((out / name).glob("_force_fresh_archive_*/result.json")) for name in tasks[2:])


def test_explicit_overrides_execute_finished_tasks(monkeypatch, tmp_path, external_runner):
    data, registered, launched, _, _ = external_runner
    for name in ("fresh", "resume", "skipped"):
        finished(tmp_path / "out", name)
        (data / name).mkdir()
    monkeypatch.setattr(solve, "_resume_eligible", lambda *_: (True, ""))
    monkeypatch.setattr(solve, "_load_resume_eval_token", lambda *_: "resume")
    check = Mock(return_value=(True, ""))
    monkeypatch.setattr(solve, "_check_resume_registration", check)
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("resume\n")
    summary = invoke(monkeypatch, tmp_path, ["skipped", "resume", "fresh"], [
        "--eval-env-mapping", "unused", "--resume-task-file", str(resume_file),
        "--force-fresh", "fresh", "--skip-judge",
    ])
    assert registered == [("resume", False), ("fresh", True)]
    assert launched == [("resume", True), ("fresh", False)]
    check.assert_called_once()
    assert summary["skipped_tasks"] == ["skipped"]


def test_internal_evaluator_only_registers_execution_tasks(monkeypatch, tmp_path):
    out = tmp_path / "out"
    finished(out, "old")
    (tmp_path / "data" / "new").mkdir(parents=True)
    monkeypatch.setattr(solve, "load_or_create_control_token", lambda *_: None)
    monkeypatch.setattr(solve, "_load_or_create_eval_token", lambda *_: None)
    registered = Mock(return_value=({}, []))
    monkeypatch.setattr(solve.ScoreTracker, "register_task", registered)
    server = Mock()
    monkeypatch.setattr(solve, "start_server_background", lambda *a, **k: (server, None))
    monkeypatch.setattr(solve, "_run_single_task", lambda name, *a, **k: finished(out, name))
    forbid(monkeypatch, "_fetch_score_from_service", "_register_task_with_token_retry")
    summary = invoke(monkeypatch, tmp_path, ["old", "new"], ["--skip-judge"])
    assert registered.call_count == 1
    assert registered.call_args.args[0] == "new"
    assert summary["total_tasks"] == 2
    server.shutdown.assert_called_once()


def test_overlap_and_resume_preflight_still_protect_state(monkeypatch, tmp_path):
    finished(tmp_path / "out", "task")
    forbid(monkeypatch, "_run_single_task", "_register_task_with_token_retry")
    with pytest.raises(ValueError, match="overlap"):
        invoke(monkeypatch, tmp_path, ["task"], ["--resume-tasks", "task", "--force-fresh", "task"])
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(solve, "load_or_create_control_token", lambda *_: None)
    with pytest.raises(RuntimeError, match="not eligible"):
        invoke(monkeypatch, tmp_path, ["task"], ["--resume-tasks", "task"])


def test_cumulative_outputs_validate_complete_submission(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "submit-results"))
    prepare = importlib.import_module("prepare_submission")
    validate = importlib.import_module("validate_submission")
    package = tmp_path / "submission"
    assert prepare.main(["--track", "naturebench-25", "--output-dir", str(package)]) == 0
    tasks = solve._read_task_set(package / "tasks.txt")
    out = package / "raw-results"
    for i, name in enumerate(tasks):
        finished(out, name, "success" if i < 2 else "timeout", 0 if i < 2 else -1)
        (out / name / "trajectory.jsonl").write_text('{"event":"done"}\n')
    _, reused = solve._select_execution_tasks(tasks, out, set(), set())
    with (package / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validate.RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(dict(case_id=r["task_name"], final_status=r["status"],
                              best_score="", judge_verdict="not_applicable",
                              judge_reason="not_applicable") for r in reused.values())
    report = validate.ValidationReport()
    cases = validate.load_track_case_metadata(validate.default_case_metadata_path(), "naturebench-25")
    records = validate.load_results_csv(package / "results.csv", cases, report)
    validate.validate_raw_results(out, records, cases, report)
    assert len(records) == 25
    assert report.ok, report.errors


@pytest.mark.parametrize("override", ["none", "forwarded", "file"])
def test_wrapper_startup_honors_execution_overrides(monkeypatch, tmp_path, override):
    import run_naturebench as wrapper
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("task\n")
    data = tmp_path / "data"
    data.mkdir()
    finished(tmp_path / "out", "task")
    monkeypatch.setattr(wrapper, "_resolve_tasks_dir", lambda *_: data)
    calls = []
    monkeypatch.setattr(wrapper, "_run", lambda command, **kw: calls.append(command))
    argv = [
        "run_naturebench.py", "--tasks", str(tasks), "--data-dir", str(data),
        "--out-dir", str(tmp_path / "out"), "--skip-download", "--ensure-base-image",
        "--start-eval-services", "--eval-env-mapping", "unused",
    ]
    if override == "forwarded":
        argv += ["--", "--force-fresh", "task"]
    elif override == "file":
        fresh_file = tmp_path / "fresh.txt"
        fresh_file.write_text("task\n")
        argv += ["--force-fresh-task-file", str(fresh_file)]
    monkeypatch.setattr(sys, "argv", argv)
    wrapper.main()
    assert len(calls) == (1 if override == "none" else 3)
    command = calls[-1]
    assert str(ROOT / "solve.py") in command
    assert command[command.index("--task-set") + 1] == str(tasks)
    if override != "none":
        assert command[command.index("--force-fresh") + 1] == "task"
