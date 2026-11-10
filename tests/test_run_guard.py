"""Resource/accounting tests use temporary projects and no GPU workloads."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_guard.py"


@pytest.fixture
def guard(tmp_path):
    spec = importlib.util.spec_from_file_location("bh_run_guard_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "runs").mkdir()
    return module


def test_proc_identity_detects_pid_reuse(guard):
    info = guard.proc_info(os.getpid())
    assert guard.same_process(os.getpid(), info["start_ticks"], guard.boot_id())
    assert not guard.same_process(os.getpid(), info["start_ticks"]+1, guard.boot_id())
    assert not guard.same_process(os.getpid(), info["start_ticks"], "different-boot")


def test_orphan_costs_are_upper_bounds_and_never_reset(guard):
    now = time.time()
    event = dict(event="started", attempt_id="attempt", run_id="job", pid=999999999,
                 process_start_ticks=1, boot_id=guard.boot_id(), launcher_pid=999999998,
                 launcher_start_ticks=1, start_unix=now-100, gpu=True, threads=2, stage="development")
    guard.append(event)
    guard.atomic_json(guard.heartbeat_path(event), dict(attempt_id="attempt", observed_unix=now-50,
                                                      cpu_seconds=10, wall_seconds=50))
    totals = guard.totals()
    assert totals["gpu_hours"] >= 100/3600
    assert totals["cpu_core_hours"] >= 110/3600
    ending = guard.ledger_records()[-1]
    assert ending["event"] == "finished" and ending["cost_quality"] == "upper_bound"
    assert ending["status"] == "interrupted"
    assert guard.totals() == totals
    # Later inaccurate/lower-cost events cannot erase already charged usage.
    guard.append(dict(ending, gpu_seconds=0, cpu_seconds=0))
    assert guard.totals() == totals


def test_active_attempt_costs_count_before_finish(guard, monkeypatch):
    event = dict(event="started", attempt_id="live", run_id="live", pid=123, process_start_ticks=10,
                 boot_id=guard.boot_id(), start_unix=time.time()-3600, gpu=True, threads=1, stage="feasibility")
    guard.append(event)
    monkeypatch.setattr(guard, "workload_snapshot", lambda event: (True, 90., []))
    monkeypatch.setattr(guard, "same_process", lambda *args: True)
    result = guard.totals()
    assert result["gpu_hours"] >= 1
    assert result["development_gpu_hours"] >= 1
    assert result["feasibility_gpu_hours"] >= 1
    assert result["cpu_core_hours"] == pytest.approx(90/3600)
    assert len(guard.ledger_records()) == 1


def test_caps_allow_cpu_report_after_gpu_exhaustion(guard):
    limits = dict(gpu_hours=160., cpu_core_hours=10., development_gpu_hours=50., feasibility_gpu_hours=12.)
    assert guard.budget_reason(limits, False, "review") is None
    assert "GPU" in guard.budget_reason(limits, True, "formal")
    limits["cpu_core_hours"] = 499.999
    assert "CPU" in guard.budget_reason(limits, False, "review", reserve_seconds=10, threads=1)


def test_identity_binds_code_spec_and_lock(guard):
    (guard.ROOT/"scripts").mkdir()
    script = guard.ROOT/"scripts/worker.py"
    script.write_text("print('v1')\n")
    package = guard.ROOT/"src/bh_transfer"
    package.mkdir(parents=True)
    implementation = package/"neural.py"
    implementation.write_text("VERSION = 1\n")
    spec = guard.ROOT/"job.json"
    spec.write_text('{"seed":17}')
    command = [sys.executable, "scripts/worker.py", "--spec", "job.json"]
    baseline, components = guard.execution_identity(command)
    assert "scripts/worker.py" in components["project_scripts"]
    assert "src/bh_transfer/neural.py" in components["implementation_modules"]
    assert "job.json" in components["spec_json"]
    assert guard.reusable_completion(dict(status="completed", execution_identity=baseline), baseline)
    for path, changed in [(script, "print('v2')\n"), (implementation, "VERSION = 2\n"), (spec, '{"seed":42}')]:
        original = path.read_text()
        path.write_text(changed)
        new_identity, _ = guard.execution_identity(command)
        assert new_identity != baseline
        with pytest.raises(ValueError, match="new run-id"):
            guard.reusable_completion(dict(status="completed", execution_identity=baseline), new_identity)
        path.write_text(original)
    (guard.ROOT/"FINAL_EVAL_LOCK.json").write_text('{"status":"FROZEN"}')
    assert guard.execution_identity(command)[0] != baseline
    with pytest.raises(ValueError, match="unaudited"):
        guard.reusable_completion(dict(status="completed", command_hash="legacy"), baseline)


def test_identity_supports_equals_spec_argument(guard):
    (guard.ROOT/"spec.json").write_text('{"seed":17}')
    _, components = guard.execution_identity([sys.executable, "-m", "bh_transfer.neural", "--spec=spec.json"])
    assert "spec.json" in components["spec_json"]


def test_grace_reserves_all_active_cpu_threads(guard):
    assert guard.TERM_GRACE_SECONDS == 120
    limits = dict(gpu_hours=0., cpu_core_hours=500.-300/3600,
                  development_gpu_hours=0., feasibility_gpu_hours=0., active_cpu_threads=4)
    assert "CPU" in guard.budget_reason(limits, False, "review", reserve_seconds=120, threads=1)
    limits["cpu_core_hours"] = 0.
    limits["gpu_hours"] = 160.-119/3600
    assert "GPU" in guard.budget_reason(limits, True, "formal", reserve_seconds=120, threads=1)


def test_code_changed_in_queue_is_rejected_before_launch(guard, monkeypatch):
    scripts = guard.ROOT/"scripts"
    scripts.mkdir()
    worker = scripts/"worker.py"
    worker.write_text("print('original')\n")
    original_acquire = guard.acquire_cpu_slots
    handles = []
    def mutate_during_queue(count):
        leases = original_acquire(count)
        handles.extend(handle for _, handle in leases)
        worker.write_text("print('changed during queue')\n")
        return leases
    monkeypatch.setattr(guard, "acquire_cpu_slots", mutate_during_queue)
    monkeypatch.setattr(sys, "argv", ["run_guard.py", "--run-id", "queued", "--stage", "review", "--", sys.executable, "scripts/worker.py"])
    try:
        with pytest.raises(SystemExit, match="changed while"):
            guard.main()
        assert guard.ledger_records() == []
    finally:
        for handle in handles:
            handle.close()


def test_cpu_leases_do_not_overlap(guard):
    leases = guard.acquire_cpu_slots(2)
    assert len(leases) == 2
    for slot, _ in leases:
        with (guard.ROOT/"runs"/f"cpu_{slot}.lock").open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    other_leases = guard.acquire_cpu_slots(2)
    assert {s for s, _ in leases}.isdisjoint({s for s, _ in other_leases})
    for _, handle in leases + other_leases:
        handle.close()


def test_wrapper_death_keeps_workload_leases_and_prevents_duplicate(tmp_path):
    scripts = tmp_path/"scripts"
    scripts.mkdir()
    target = scripts/"run_guard.py"
    shutil.copyfile(SCRIPT, target)
    command = [sys.executable, str(target), "--run-id", "orphan", "--stage", "review", "--threads", "1", "--",
               sys.executable, "-c", "import time; time.sleep(20)"]
    wrapper = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        event = json.loads(wrapper.stdout.readline())
        child_pid = event["pid"]
        wrapper.kill()
        wrapper.wait(timeout=5)
        duplicate = subprocess.run(command, capture_output=True, text=True, timeout=5)
        assert duplicate.returncode != 0
        assert "active" in duplicate.stderr
        for lock_path in [tmp_path/"runs/orphan/job.lock", tmp_path/"runs/cpu_0.lock"]:
            with lock_path.open("a+") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)
        if child_pid is not None:
            # This is the known synthetic child created by this test, never an
            # external process discovered via GPU/process enumeration.
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_completed_run_is_skipped_without_duplicate_ledger(tmp_path):
    scripts = tmp_path/"scripts"
    scripts.mkdir()
    target = scripts/"run_guard.py"
    shutil.copyfile(SCRIPT, target)
    command = [sys.executable, str(target), "--run-id", "done", "--stage", "review", "--", sys.executable, "-c", "print('synthetic')"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    before = (tmp_path/"runs/ledger.jsonl").read_text()
    second = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert second.returncode == 0 and "Already completed" in second.stdout
    assert (tmp_path/"runs/ledger.jsonl").read_text() == before
