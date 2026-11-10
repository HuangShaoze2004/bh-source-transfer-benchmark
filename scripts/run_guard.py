#!/usr/bin/env python3
"""Linux job launcher with inherited resource leases and durable cost recovery.

Children inherit job/GPU/CPU flock descriptors, so wrapper death cannot free a
live workload's resources. Unknown orphan endings are charged through discovery.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 2.0
TERM_GRACE_SECONDS = 120.0
CPU_SLOTS = 4


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_identity(command):
    """Bind cached success to code, explicit job specification and evaluation lock.

    This deliberately invalidates conservatively: every bh_transfer module is a
    dependency even if a particular worker only imports a subset. The worker
    remains responsible for identity checks on its own data/model artifacts.
    """
    root = ROOT.resolve()
    scripts, specifications = {}, {}
    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else root/path).resolve()
    for argument in command:
        if argument.endswith(".py"):
            path = resolve(argument)
            if path.is_file() and path.is_relative_to(root):
                scripts[str(path.relative_to(root))] = sha256_file(path)
    for index, argument in enumerate(command):
        if argument == "--spec":
            if index+1 == len(command):
                raise ValueError("--spec requires a JSON path")
            path = resolve(command[index+1])
        elif argument.startswith("--spec="):
            path = resolve(argument.split("=", 1)[1])
        else:
            continue
        if not path.is_file():
            raise ValueError(f"Job specification is absent: {path}")
        json.loads(path.read_text())  # Reject invalid JSON before launching.
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        specifications[key] = sha256_file(path)
    dependencies = {str(path.relative_to(root)): sha256_file(path)
                    for path in sorted((root/"src/bh_transfer").glob("*.py"))}
    lock = root/"FINAL_EVAL_LOCK.json"
    components = dict(schema_version=1, command=command, project_scripts=scripts,
                      spec_json=specifications, implementation_modules=dependencies,
                      final_eval_lock_sha256=sha256_file(lock) if lock.exists() else None)
    digest = hashlib.sha256(json.dumps(components, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, components


def reusable_completion(previous, identity):
    if previous.get("status") != "completed":
        return False
    if previous.get("execution_identity") != identity:
        raise ValueError("Completed run has a different or unaudited execution identity; use a new run-id. Old artifacts are preserved.")
    return True


def append(event):
    event = dict(event, utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "runs/ledger.jsonl").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def accounting_lock():
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "runs/accounting.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def ledger_records():
    path = ROOT / "runs/ledger.jsonl"
    if not path.exists():
        return []
    with path.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return [json.loads(line) for line in handle if line.strip()]


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def proc_info(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        values = text[text.rfind(")") + 2:].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return {"pid": int(pid), "state": values[0], "ppid": int(values[1]),
                "pgrp": int(values[2]), "session": int(values[3]),
                "cpu_seconds": sum(float(values[i]) for i in (11, 12, 13, 14)) / ticks,
                "start_ticks": int(values[19])}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def same_process(pid, start_ticks, expected_boot):
    if expected_boot != boot_id():
        return False
    info = proc_info(pid)
    return bool(info and info["state"] != "Z" and info["start_ticks"] == start_ticks)


def workload_snapshot(event):
    """Activity and cumulative process-tree CPU for the recorded session."""
    if event.get("boot_id") not in (None, boot_id()):
        return False, 0.0, []
    expected_start = event.get("process_start_ticks")
    if expected_start is None:
        # Legacy records lack provable PID identity: refuse a relaunch while
        # that PID is live; never signal the unverified process.
        info = proc_info(event["pid"])
        return bool(info and info["state"] != "Z"), 0.0, []
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        info = proc_info(directory.name)
        if info and info["session"] == event["pid"] and info["start_ticks"] >= expected_start and info["state"] != "Z":
            members.append(info)
    root = proc_info(event["pid"])
    root_matches = bool(root and root["start_ticks"] == expected_start)
    if root and not root_matches:
        members = []
    cpu = sum(info["cpu_seconds"] for info in members)
    if root_matches and root["state"] == "Z":
        cpu += root["cpu_seconds"]
    return bool(members), cpu, members


def heartbeat_path(event):
    return ROOT / "runs" / event["run_id"] / (event["attempt_id"] + ".heartbeat.json")


def _started_unix(event):
    if "start_unix" in event:
        return float(event["start_unix"])
    if "utc" in event:
        return dt.datetime.fromisoformat(event["utc"]).timestamp()
    raise ValueError("Unfinished attempt lacks start timestamp; manual accounting required")


def read_heartbeat(event):
    path = heartbeat_path(event)
    if path.exists():
        state = json.loads(path.read_text())
        if state["attempt_id"] != event["attempt_id"]:
            raise ValueError("Heartbeat attempt identity mismatch")
        return state
    return {"attempt_id": event["attempt_id"], "observed_unix": _started_unix(event),
            "cpu_seconds": 0.0, "wall_seconds": 0.0}


def totals():
    """Include finished, live and conservatively reconciled orphan attempts."""
    with accounting_lock():
        records, starts = {}, {}
        for entry in ledger_records():
            attempt = entry.get("attempt_id")
            if entry.get("event") == "started":
                starts[attempt] = entry
            elif entry.get("event") == "finished":
                previous = records.get(attempt, {})
                records[attempt] = dict(entry,
                    gpu_seconds=max(previous.get("gpu_seconds", 0), entry.get("gpu_seconds", 0)),
                    cpu_seconds=max(previous.get("cpu_seconds", 0), entry.get("cpu_seconds", 0)))
        now = time.time()
        for attempt, event in starts.items():
            if attempt in records:
                continue
            active, current_cpu, _ = workload_snapshot(event)
            heartbeat = read_heartbeat(event)
            elapsed = max(float(heartbeat.get("wall_seconds", 0)), now - _started_unix(event), 0.0)
            previous_cpu = float(heartbeat.get("cpu_seconds", 0))
            leader_alive = same_process(event["pid"], event.get("process_start_ticks"), event.get("boot_id"))
            cpu = max(previous_cpu, current_cpu)
            if not leader_alive:
                cpu = max(cpu, previous_cpu + max(0, now-float(heartbeat["observed_unix"])) * int(event.get("threads", CPU_SLOTS)))
            accounted = dict(event, wall_seconds=elapsed, gpu_seconds=elapsed if event.get("gpu") else 0,
                             cpu_seconds=cpu, cost_quality="measured_live" if leader_alive else "upper_bound")
            # A live launcher can still collect exact wait/rusage totals after
            # child exit. Only reconcile after BOTH workload and launcher died.
            launcher_alive = same_process(event.get("launcher_pid", -1), event.get("launcher_start_ticks"), event.get("boot_id"))
            if not active and not launcher_alive:
                accounted.update(event="finished", status="interrupted", exit_code=None,
                                 reason="Unobserved workload ending; conservatively charged through discovery", discovered_unix=now)
                append(accounted)
                atomic_json(ROOT / "runs" / event["run_id"] / "job_status.json", accounted)
            records[attempt] = accounted
        return {"gpu_hours": sum(x.get("gpu_seconds", 0) for x in records.values())/3600,
                "cpu_core_hours": sum(x.get("cpu_seconds", 0) for x in records.values())/3600,
                "active_cpu_threads": sum(int(x.get("threads", CPU_SLOTS)) for x in records.values() if x.get("event") == "started"),
                "development_gpu_hours": sum(x.get("gpu_seconds", 0) for x in records.values() if x.get("stage") in ("development", "feasibility"))/3600,
                "feasibility_gpu_hours": sum(x.get("gpu_seconds", 0) for x in records.values() if x.get("stage") == "feasibility")/3600}


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]


def acquire_cpu_slots(count):
    """Acquire count leases atomically without holding a partial allocation."""
    if not 1 <= count <= CPU_SLOTS:
        raise ValueError("CPU lease count outside 1..4")
    allocator = ROOT / "runs/cpu_allocator.lock"
    while True:
        acquired = []
        with allocator.open("a+") as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            for slot in range(CPU_SLOTS):
                handle = (ROOT / "runs" / f"cpu_{slot}.lock").open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired.append((slot, handle))
                except BlockingIOError:
                    handle.close()
                if len(acquired) == count:
                    return acquired
            for _, handle in acquired:
                handle.close()
        time.sleep(POLL_SECONDS)


def budget_reason(limits, gpu, stage, reserve_seconds=0, threads=1):
    # Reserve the grace window for all simultaneously active CPU leases, so
    # several jobs cannot each spend the same remaining global CPU budget.
    reserved_threads = max(threads, limits.get("active_cpu_threads", 0))
    if limits["cpu_core_hours"] + reserve_seconds*reserved_threads/3600 >= 500:
        return "CPU core-hour budget exhausted"
    if gpu and limits["gpu_hours"] + reserve_seconds/3600 >= 160:
        return "Total GPU budget exhausted"
    if gpu and stage in ("development", "feasibility"):
        if limits["development_gpu_hours"] + reserve_seconds/3600 >= 50:
            return "Development GPU budget exhausted"
        if stage == "feasibility" and limits["feasibility_gpu_hours"] + reserve_seconds/3600 >= 12:
            return "Feasibility GPU budget exhausted"
    return None


def write_heartbeat(event, elapsed, cpu, state="running"):
    heartbeat = dict(attempt_id=event["attempt_id"], run_id=event["run_id"], observed_unix=time.time(),
                     wall_seconds=elapsed, gpu_seconds=elapsed if event["gpu"] else 0,
                     cpu_seconds=cpu, pid=event["pid"], process_start_ticks=event["process_start_ticks"],
                     boot_id=event["boot_id"], status=state)
    atomic_json(heartbeat_path(event), heartbeat)
    return heartbeat


def signal_workload(event, sig):
    # Only current identities in our recorded session, never arbitrary GPU PIDs.
    _, _, members = workload_snapshot(event)
    for info in members:
        if same_process(info["pid"], info["start_ticks"], event["boot_id"]):
            try:
                os.kill(info["pid"], sig)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=["setup", "feasibility", "development", "formal", "sensitivity", "review"], required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=1, choices=range(1, CPU_SLOTS+1))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a single path component")
    work = ROOT / "runs" / args.run_id
    work.mkdir(parents=True, exist_ok=True)
    job_lock = (work / "job.lock").open("a+")
    try:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This run or its surviving workload is active; no duplicate launched.")
    command_hash = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    identity, identity_components = execution_identity(command)
    status_path = work / "job_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if reusable_completion(old, identity):
            print("Already completed:", args.run_id, flush=True)
            return
        if old.get("execution_identity") not in (None, identity):
            raise SystemExit("Existing interrupted/failed run has changed execution identity; use a new run-id and preserve old artifacts.")
        if old.get("status") == "running" and workload_snapshot(old)[0]:
            raise SystemExit("Previous workload PID/start identity remains active; no duplicate launched.")
    limits = totals()
    reason = budget_reason(limits, args.gpu, args.stage)
    if reason:
        raise SystemExit(reason)
    gpu_lock = None
    if args.gpu:
        gpu_lock = (ROOT / "runs/gpu.lock").open("a+")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        while gpu_pids():
            print("GPU occupied; queued without touching other processes.", flush=True)
            time.sleep(15)
    cpu_leases = acquire_cpu_slots(args.threads)
    limits = totals()  # Refresh AFTER every potentially long resource queue.
    reason = budget_reason(limits, args.gpu, args.stage, POLL_SECONDS+TERM_GRACE_SECONDS, args.threads)
    if reason:
        raise SystemExit(reason)
    if execution_identity(command)[0] != identity:
        raise SystemExit("Code/spec/evaluation lock changed while this job was queued; relaunch with the intended frozen inputs.")
    env = os.environ.copy()
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[key] = "1"  # Prevent worker-count times BLAS-thread oversubscription.
    env.update(PYTHONPATH=str(ROOT / "src"), TOKENIZERS_PARALLELISM="false", HF_HOME=str(ROOT / "cache/huggingface"),
               MPLCONFIGDIR=str(ROOT / "cache/matplotlib"), PYTHONUNBUFFERED="1", BH_PROJECT_THREADS=str(args.threads))
    allowed_cpus = sorted(os.sched_getaffinity(0))
    slot_cpus = allowed_cpus[:CPU_SLOTS]
    if len(slot_cpus) < CPU_SLOTS:
        slot_cpus = [allowed_cpus[i % len(allowed_cpus)] for i in range(CPU_SLOTS)]
    assigned_cpus = {slot_cpus[slot] for slot, _ in cpu_leases}
    attempt_id = args.run_id + "__" + str(time.time_ns())
    log_path = work / (attempt_id + ".log")
    inherited_fds = [job_lock.fileno()] + [handle.fileno() for _, handle in cpu_leases]
    if gpu_lock is not None:
        inherited_fds.append(gpu_lock.fileno())
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    start, start_unix = time.monotonic(), time.time()
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=tuple(inherited_fds),
                                 preexec_fn=lambda: os.sched_setaffinity(0, assigned_cpus))
        info = proc_info(child.pid)
        if info is None:
            raise RuntimeError("Cannot establish child process identity")
        event = dict(event="started", run_id=args.run_id, attempt_id=attempt_id, stage=args.stage,
                     gpu=args.gpu, threads=args.threads, cpu_slots=[slot for slot, _ in cpu_leases], cpus=sorted(assigned_cpus),
                     pid=child.pid, process_start_ticks=info["start_ticks"], boot_id=boot_id(),
                     launcher_pid=os.getpid(), launcher_start_ticks=proc_info(os.getpid())["start_ticks"],
                     start_unix=start_unix, command=command, command_hash=command_hash,
                     execution_identity=identity, identity_components=identity_components,
                     log=str(log_path.relative_to(ROOT)), status="running")
        append(event)
        atomic_json(status_path, event)
        write_heartbeat(event, 0., 0.)
        print(json.dumps(event), flush=True)
        reason, termination_started, measured_cpu = None, None, 0.0
        while True:
            active, snapshot_cpu, _ = workload_snapshot(event)
            elapsed = time.monotonic()-start
            own = resource.getrusage(resource.RUSAGE_SELF)
            own_cpu = own.ru_utime+own.ru_stime-self_before.ru_utime-self_before.ru_stime
            measured_cpu = max(measured_cpu, snapshot_cpu+own_cpu)
            rc = child.poll()
            if rc is not None and not active:
                break
            write_heartbeat(event, elapsed, measured_cpu)
            current = totals()
            cap = budget_reason(current, args.gpu, args.stage, POLL_SECONDS+TERM_GRACE_SECONDS, args.threads)
            if cap and termination_started is None:
                reason, termination_started = cap, time.monotonic()
                signal_workload(event, signal.SIGTERM)
            elif termination_started is not None and time.monotonic()-termination_started >= TERM_GRACE_SECONDS:
                signal_workload(event, signal.SIGKILL)
            time.sleep(POLL_SECONDS)
        rc = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    own = resource.getrusage(resource.RUSAGE_SELF)
    cpu = max(measured_cpu, after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
              + own.ru_utime+own.ru_stime-self_before.ru_utime-self_before.ru_stime)
    elapsed = time.monotonic()-start
    write_heartbeat(event, elapsed, cpu, "completed" if rc == 0 else "failed")
    done = dict(event, event="finished", exit_code=rc, status="completed" if rc == 0 and reason is None else "failed",
                wall_seconds=elapsed, gpu_seconds=elapsed if args.gpu else 0, cpu_seconds=cpu,
                reason=reason, cost_quality="measured")
    with accounting_lock():
        append(done)
        atomic_json(status_path, done)
    atomic_json(ROOT / "runs/budget_totals.json", totals())
    print(json.dumps(done), flush=True)
    raise SystemExit(rc if rc else (1 if reason else 0))


if __name__ == "__main__":
    main()
