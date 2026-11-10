#!/usr/bin/env python3
"""Read-only progress inspection without accessing external predictions/metrics."""
import collections
import datetime as dt
import json
from pathlib import Path

import psutil

import run_guard as guard


def main():
    root = Path(__file__).resolve().parents[1]
    counts = collections.defaultdict(collections.Counter)
    active, unreadable = [], []
    for path in sorted((root / "runs").glob("*/job_status.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            unreadable.append({"path": str(path.relative_to(root)), "error": str(exc)})
            continue
        run_id = job["run_id"]
        family = "other"
        if run_id.startswith(("dev_", "formal_")):
            phase = "inner" if run_id.startswith("dev_") else "formal"
            model = "tree" if any(x in run_id for x in ("R0_RF", "R0_HGB", "R1_substitute")) else "neural"
            family = f"{phase}_{model}"
        counts[family][job["status"]] += 1
        if job["status"] != "running":
            continue
        alive = guard.same_process(job["pid"], job.get("process_start_ticks"), job.get("boot_id"))
        item = {"run_id": run_id, "pid": job["pid"], "process_identity_alive": alive}
        if alive:
            try:
                process = psutil.Process(job["pid"])
                item.update(process_state=process.status(), cpu_seconds=sum(process.cpu_times()[:2]))
            except psutil.NoSuchProcess:
                item["observation"] = "exited_during_inspection; recheck before any action"
        else:
            item["current_job_status"] = json.loads(path.read_text())["status"]
        history = root / "runs/trajectories" / run_id / "validation_history.csv"
        if history.exists():
            item["epochs_completed"] = max(0, len(history.read_text().splitlines()) - 1)
        active.append(item)
    controller_path = root / "runs/controller_status.json"
    controller = json.loads(controller_path.read_text()) if controller_path.exists() else {}
    if controller:
        try:
            process = psutil.Process(controller["controller_pid"])
            controller["process_identity_alive"] = (
                process.create_time() == controller["controller_start_time"]
                and process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            controller["process_identity_alive"] = False
    lock_path = root / "FINAL_EVAL_LOCK.json"
    mismatches = []
    if lock_path.exists():
        for name, expected in json.loads(lock_path.read_text())["artifact_hashes"].items():
            path = root / name
            if not path.exists() or guard.sha256_file(path) != expected:
                mismatches.append(name)
    print(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "counts": dict(counts), "active": active,
                      "controller": controller, "budget": guard.totals(),
                      "frozen_artifact_mismatches": mismatches,
                      "unreadable_status_files": unreadable}, indent=2))


if __name__ == "__main__":
    main()
