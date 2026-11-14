#!/usr/bin/env python3
"""Verify, reconstruct metrics, restore models, or replay saved experiment jobs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def manifest():
    return read(ROOT / "release/MANIFEST.json")


def local_source_path(recorded, metadata):
    path = Path(recorded)
    if path.is_absolute():
        path = path.relative_to(metadata["recorded_source_root"])
    result = ROOT / path
    if not result.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path escapes this checkout: {recorded}")
    return result


def verify(full=False):
    metadata = manifest()
    count = 0
    for entry in metadata["files"]:
        if entry["storage"] == "release" and not full:
            continue
        path = ROOT / entry["path"]
        if entry["kind"] == "symlink":
            if not path.is_symlink() or str(path.readlink()) != entry["target"]:
                raise ValueError(f"Symlink differs: {entry['path']}")
        elif not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact missing or changed: {entry['path']}")
        else:
            path.chmod(entry["mode"])
        count += 1
    print(json.dumps(dict(status="PASS", checked_files=count, checkpoints_included=full)), flush=True)


def load_auditor():
    spec = importlib.util.spec_from_file_location("original_independent_auditor", ROOT / "scripts/finalize_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_release(project, metadata):
    lock = read(project / "FINAL_EVAL_LOCK.json")
    completeness = read(project / "artifacts/PREDICTION_COMPLETENESS.json")
    evaluation = read(project / "reports/final/evaluation_manifest.json")
    if lock["status"] != "FROZEN" or not lock["evaluation_allowed"]:
        raise ValueError("Protocol does not permit evaluation")
    if not completeness["complete"] or evaluation["status"] != "EVALUATION_COMPLETE":
        raise ValueError("Prediction matrix or evaluation is incomplete")
    for relative, expected in lock.get("artifact_hashes", {}).items():
        if sha256(project / relative) != expected:
            raise ValueError(f"Frozen artifact differs: {relative}")
    for recorded, expected in lock.get("source_code_hashes", {}).items():
        if sha256(local_source_path(recorded, metadata)) != expected:
            raise ValueError(f"Frozen implementation differs: {recorded}")
    lock_hash = sha256(project / "FINAL_EVAL_LOCK.json")
    prediction_hash = sha256(project / "artifacts/predictions.parquet")
    for item in (completeness, evaluation):
        if item["lock_sha256"] != lock_hash:
            raise ValueError("Release/lock identity mismatch")
        if item.get("predictions_sha256", item.get("prediction_sha256")) != prediction_hash:
            raise ValueError("Prediction content differs from its release identity")
    return lock, completeness, evaluation


def metrics():
    import pandas as pd
    auditor = load_auditor()
    metadata = manifest()
    result = {}
    for name, directory in [("primary", ROOT), ("author_labels", ROOT / "sensitivity/author_labels"),
                            ("dd_original", ROOT / "sensitivity/dd_original")]:
        lock, completeness, evaluation = check_release(directory, metadata)
        records = pd.read_parquet(directory / "data/processed/records.parquet")
        splits = read(directory / "data/processed/splits.json")
        auditor.independent_split_audit(records, splits, lock)
        predictions = pd.read_parquet(directory / "artifacts/predictions.parquet")
        if len(predictions) != completeness["rows"]:
            raise ValueError("Prediction coverage differs")
        auditor.independent_prediction_audit(predictions, records, lock)
        by_seed, table = auditor.reconstruct_metrics(predictions, lock)
        saved = pd.read_csv(directory / "reports/final/main_table.csv")
        auditor.compare_saved_table(table, saved, ["model"], [*auditor.METRICS, "n_sources", "worst_source_auroc"], name)
        saved_seed = pd.read_csv(directory / "reports/final/metrics_by_source_seed.csv")
        auditor.compare_saved_table(by_seed, saved_seed, ["source", "model", "seed"],
                                   [*auditor.METRICS, "n_rows", "n_pairs", "positive_rate"], name)
        bundle = dict(project=directory, lock=lock, evaluation=evaluation, macro=table, by_seed=by_seed)
        auditor.validate_statistics(bundle)
        output = ROOT / "reproduction" / name
        output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "main_table.csv", index=False)
        by_seed.to_csv(output / "metrics_by_source_seed.csv", index=False)
        result[name] = dict(status="PASS", checked_prediction_rows=len(predictions),
                            macro_auroc=table.set_index("model").auroc.to_dict())
        print(json.dumps(dict(branch=name, **result[name])), flush=True)
    write(ROOT / "reproduction/METRIC_RECONSTRUCTION.json", result)


def download(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "bh-source-transfer-reproduction"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with temporary.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
    temporary.replace(path)


def extract_archive(path, archive_metadata, entries, zstd):
    if sha256(path) != archive_metadata["sha256"]:
        raise ValueError(f"Archive checksum differs: {path.name}")
    expected = dict(archive_metadata["members"])
    seen = set()
    process = subprocess.Popen([zstd, "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError("Unexpected or duplicate archive member")
                destination = ROOT / member.name
                if not destination.resolve().is_relative_to(ROOT):
                    raise ValueError("Archive member escapes checkout")
                if member.size != entries[member.name]["bytes"]:
                    raise ValueError("Archive member size differs")
                seen.add(member.name)
                if destination.exists():
                    if sha256(destination) != expected[member.name]:
                        raise ValueError(f"Preserve the differing local file before restoring: {member.name}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".restoring")
                with archive.extractfile(member) as source, temporary.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=4 * 1024 * 1024)
                if sha256(temporary) != expected[member.name]:
                    raise ValueError(f"Restored checksum differs: {member.name}")
                temporary.chmod(entries[member.name]["mode"])
                temporary.replace(destination)
        process.stdout.read()
        if process.wait() != 0 or seen != set(expected):
            raise ValueError("Incomplete checkpoint archive")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def restore(args):
    metadata = manifest()
    entries = {item["path"]: item for item in metadata["files"]}
    destination = ROOT / "release/downloads"
    destination.mkdir(parents=True, exist_ok=True)
    selected = [item for item in metadata["archives"] if not args.archive or item["name"] == args.archive]
    if not selected:
        raise ValueError("Archive name is absent from the manifest")
    for item in selected:
        if all((ROOT / name).is_file() and sha256(ROOT / name) == checksum for name, checksum in item["members"].items()):
            print(json.dumps(dict(archive=item["name"], status="already_restored")), flush=True)
            continue
        path = destination / item["name"]
        url = f"https://github.com/{metadata['repository']}/releases/download/{metadata['release_tag']}/{item['name']}"
        download(url, path)
        extract_archive(path, item, entries, args.zstd)
        print(json.dumps(dict(archive=item["name"], status="restored_and_verified")), flush=True)


def replay(args):
    matches = [path for directory in [ROOT / "runs/specs", ROOT / "sensitivity/author_labels/runs/specs",
                                      ROOT / "sensitivity/dd_original/runs/specs"]
               for path in directory.glob(args.job + ".json")]
    if len(matches) != 1:
        raise ValueError("Specify one exact saved job ID from runs/specs")
    original = read(matches[0])
    metadata = manifest()
    old_root = metadata["recorded_source_root"]

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and (value == old_root or value.startswith(old_root + "/")):
            return str(ROOT) + value[len(old_root):]
        return value

    spec = relocate(original)
    for field in ("output_dir", "output"):
        if field in spec:
            relative = Path(spec[field]).relative_to(ROOT)
            spec[field] = str(ROOT / "reproduction/work" / relative)
    if args.replayed_checkpoint and "checkpoint" in spec:
        spec["checkpoint"] = str(ROOT / "reproduction/work" / Path(spec["checkpoint"]).relative_to(ROOT))
    script = "predict_job.py" if "checkpoint" in spec else "run_neural_job.py" if "config" in spec else "run_tabular_job.py"
    gpu = "config" in spec and spec["config"]["device"] == "cuda"
    gpu |= "checkpoint" in spec and not spec["model"].startswith("R")
    spec_path = ROOT / "reproduction/specs" / (args.job + ".json")
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("A different replication specification already exists")
    write(spec_path, spec)
    runtime = ROOT / "reproduction/runtime"
    (runtime / "scripts").mkdir(parents=True, exist_ok=True)
    guard = runtime / "scripts/run_guard.py"
    shutil.copyfile(ROOT / "scripts/run_guard.py", guard)
    for name in ("src", "FINAL_EVAL_LOCK.json"):
        target = runtime / name
        if not target.exists():
            target.symlink_to(ROOT / name, target_is_directory=name == "src")
    command = [sys.executable, str(guard), "--run-id", "reproduce_" + args.job,
               "--stage", "formal", "--threads", "1"]
    if gpu:
        command.append("--gpu")
    command += ["--", sys.executable, str(ROOT / "scripts" / script), "--spec", str(spec_path)]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--full", action="store_true")
    sub.add_parser("metrics")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--archive")
    restore_parser.add_argument("--zstd", default="zstd")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--job", required=True)
    replay_parser.add_argument("--replayed-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.full)
    elif args.action == "metrics":
        metrics()
    elif args.action == "restore":
        restore(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
