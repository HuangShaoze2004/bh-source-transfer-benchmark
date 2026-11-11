#!/usr/bin/env python3
"""Inspect a complete frozen sensitivity training matrix using saved states."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from audit_training import inspect_run


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--kind", choices=("author_labels", "dd_original"), required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    alternative = root / "sensitivity" / args.kind
    lock_path = alternative / "FINAL_EVAL_LOCK.json"
    lock = json.loads(lock_path.read_text())
    assert lock["status"] == "FROZEN"
    assert lock["main_lock_sha256"] == sha256(root / "FINAL_EVAL_LOCK.json")
    for name, expected in lock["artifact_hashes"].items():
        assert sha256(alternative / name) == expected, name
    for name, expected in lock["source_code_hashes"].items():
        assert sha256(name) == expected, name
    splits = json.loads((alternative / "data/processed/splits.json").read_text())
    rows = []
    budgets = {}
    for fold, part in splits["folds"].items():
        if not part["test_ids"]:
            continue
        for seed in lock["seeds"]:
            for objective in ("erm", "dro"):
                name = f"{fold}_{objective}_s{seed}"
                run_id = f"sens_{args.kind}_{name}"
                spec_path = alternative / "runs/specs" / f"{run_id}.json"
                spec = json.loads(spec_path.read_text())
                selected = lock["inherited_selections"][fold]["choices"][objective]
                config = dict(selected["config"], seed=seed, objective=objective)
                directory = alternative / "runs/trajectories" / name
                assert Path(spec["project"]).resolve() == alternative
                assert Path(spec["output_dir"]).resolve() == directory
                assert spec["train_ids"] == part["train_ids"] and spec["val_ids"] == []
                assert spec["config"] == config
                status_path = root / "runs" / run_id / "job_status.json"
                status = json.loads(status_path.read_text())
                assert status["status"] == "completed" and status["exit_code"] == 0, run_id
                row, _ = inspect_run(directory, part["train_ids"], config)
                budget_key = (fold, seed)
                assert budgets.setdefault(budget_key, row["updates"]) == row["updates"]
                result = json.loads((directory / "result.json").read_text())
                artifacts = result["output_artifact_hashes"]
                assert artifacts
                for filename, expected in artifacts.items():
                    path = (directory / filename).resolve()
                    assert path.is_relative_to(directory) and sha256(path) == expected
                rows.append(dict(row, fold=fold, run_id=run_id,
                                 spec_sha256=sha256(spec_path),
                                 guard_status_sha256=sha256(status_path),
                                 checkpoint_artifact_hashes=artifacts))
    expected_runs = sum(bool(part["test_ids"]) for part in splits["folds"].values()) * len(lock["seeds"]) * 2
    assert len(rows) == expected_runs == 30
    report = {"status": "PASS", "kind": args.kind,
              "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "formal_trajectories": len(rows), "matched_update_budgets": True,
              "sensitivity_lock_sha256": sha256(lock_path),
              "auditor_sha256": sha256(__file__),
              "state_auditor_sha256": sha256(root / "scripts/audit_training.py"),
              "trajectories": rows, "external_predictions_read": False}
    output = root / "reports" / f"{args.kind.upper()}_NEURAL_RUNTIME_AUDIT.json"
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": "PASS", "kind": args.kind,
                      "formal_trajectories": len(rows), "report": str(output)}))


if __name__ == "__main__":
    main()
