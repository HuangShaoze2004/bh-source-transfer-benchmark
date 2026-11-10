#!/usr/bin/env python3
"""Prepare declared sensitivity datasets without modifying primary artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from bh_transfer.data import (CSV_NAME, CSV_SHA256, COMMIT, EXCLUDED_SOURCE,
    prepare_records, apply_tokenizer_serialization, build_features, make_splits,
    assert_split_integrity, sha256_file, summarize_rows)
from bh_transfer.tokenizer import StrictSmilesTokenizer


def artifact_hashes(directory):
    return {name: sha256_file(directory / name) for name in
            ("records.parquet", "features.npz", "splits.json", "feature_manifest.json")}


def independent_split_audit(records, splits):
    """Second implementation: verify saved membership and all entity intersections."""
    rows = records.set_index("row_id", drop=False)
    all_ids = set(map(int, rows.index))
    checks = []
    for slug, fold in splits["folds"].items():
        train_ids, test_ids = set(fold["train_ids"]), set(fold["test_ids"])
        expected_development = set(records.loc[records.source != fold["source"], "row_id"])
        assert train_ids == expected_development, f"{slug}: outer development membership"
        assert train_ids <= all_ids and test_ids <= all_ids
        partitions = [("outer", train_ids, test_ids)]
        for inner in fold["inner"]:
            left, right = set(inner["train_ids"]), set(inner["val_ids"])
            expected_train = set(records.loc[records.row_id.isin(train_ids) & (records.source != inner["source"]), "row_id"])
            assert left == expected_train
            assert left | right <= train_ids
            partitions.append(("inner:" + inner["source"], left, right))
        for name, left, right in partitions:
            a, b = rows.loc[sorted(left)], rows.loc[sorted(right)]
            a_entities = set(a.aryl_id) | set(a.amine_id)
            b_entities = set(b.aryl_id) | set(b.amine_id)
            overlaps = {"rows": len(left & right), "entities": len(a_entities & b_entities),
                        "pairs": len(set(a.pair_id) & set(b.pair_id)),
                        "duplicate_inputs": len(set(a.input_key) & set(b.input_key))}
            assert not any(overlaps.values()), (slug, name, overlaps)
            checks.append({"fold": slug, "partition": name, "left_rows": len(a), "right_rows": len(b), **overlaps})
    return {"passed": True, "checks": checks, "implementation": "independent set-based membership and overlap checks"}


def make_links(alt):
    for name in ("weights", "src"):
        link = alt / name
        target = Path("../..") / name
        if link.is_symlink():
            assert link.resolve() == (ROOT / name).resolve()
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


def run(kind):
    main = ROOT / "data/processed"
    before = artifact_hashes(main)
    alt = ROOT / "sensitivity" / kind
    output = alt / "data/processed"
    audit = alt / "data/audit"
    output.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    make_links(alt)
    parent_records = pd.read_parquet(main / "records.parquet")
    parent_splits = json.loads((main / "splits.json").read_text())
    primary_check = independent_split_audit(parent_records, parent_splits)
    manifest = json.loads((main / "feature_manifest.json").read_text())
    tokenizer = StrictSmilesTokenizer(ROOT / "weights/rxnfp/vocab.txt")
    if kind == "author_labels":
        records = parent_records.copy()
        records["y"] = records.author_label.astype(int)
        records.to_parquet(output / "records.parquet", index=False)
        shutil.copy2(main / "features.npz", output / "features.npz")
        shutil.copy2(main / "splits.json", output / "splits.json")
        splits = parent_splits
        pd.testing.assert_frame_equal(records.drop(columns="y"), parent_records.drop(columns="y"))
        assert sha256_file(output / "features.npz") == before["features.npz"]
        assert sha256_file(output / "splits.json") == before["splits.json"]
        changed = records.loc[records.y != parent_records.y, ["row_id", "source", "y", "author_label", "yield_raw"]]
        changed.to_csv(audit / "changed_labels.csv", index=False)
        manifest.update(label_policy="author-provided labels", changed_label_rows=len(changed),
            split_policy="exact byte copy of primary splits; eligibility and embedded summaries retain primary-label basis",
            embedded_split_records_sha256_is_parent=True,
            warning="Sensitivity only: 28 known AZ labels disagree with Yield>=10. No re-selection of splits or models is authorized by this alternative.")
    else:
        raw_path = ROOT / "data/raw" / CSV_NAME
        assert sha256_file(raw_path) == CSV_SHA256
        raw = pd.read_csv(raw_path, keep_default_na=False)
        # Local copy only: bypass the primary module's deliberate source exclusion,
        # then restore the original source immediately. No global constant changes.
        pending_source = "__dd_original_sensitivity_pending__"
        assert pending_source not in set(raw.Source)
        local = raw.copy()
        local.loc[local.Source == EXCLUDED_SOURCE, "Source"] = pending_source
        records, mapping, exclusions = prepare_records(local)
        records.loc[records.source == pending_source, "source"] = EXCLUDED_SOURCE
        assert len(records) == len(raw) == 27497 and exclusions.empty
        records, token_mapping = apply_tokenizer_serialization(records, tokenizer)
        token_audit = tokenizer.audit(records.reaction_smiles.tolist(), records.row_id.tolist())
        (audit / "token_audit.json").write_text(json.dumps(token_audit, indent=2) + "\n")
        assert token_audit["n_invalid"] == 0
        mapping.to_parquet(audit / "structure_mapping.parquet", index=False)
        token_mapping.to_parquet(audit / "token_serialization_mapping.parquet", index=False)
        records.to_parquet(output / "records.parquet", index=False)
        features = build_features(records)
        np.savez_compressed(output / "features.npz", **features)
        splits = make_splits(records)
        splits["records_sha256"] = sha256_file(output / "records.parquet")
        (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
        assert_split_integrity(records, splits)
        dd = records.loc[records.source == EXCLUDED_SOURCE].copy()
        duplicate = dd.loc[dd.input_key.duplicated(keep=False)].copy()
        duplicate["conflicting_labels"] = duplicate.groupby("input_key").y.transform("nunique") > 1
        duplicate.to_parquet(audit / "dd_duplicate_inputs.parquet", index=False)
        manifest.update(label_policy="Yield >= 10, including exactly 10", source_inclusion="all original eight sources; 27497 original rows",
            split_policy="recomputed strict LOSO with identical normalization/qualification/inner-source rules",
            unresolved_additive_rows=int((records.source == EXCLUDED_SOURCE).sum()),
            additive_identity_restored=False, additive_flags_used_as_features=False,
            warning="Sensitivity only: Dreher Doyle additive identities remain missing; no additive was inferred or fabricated. These incomplete inputs cannot validate a chemically complete additive-aware predictor.")
    assert np.array_equal(np.load(output / "features.npz")["row_ids"], records.row_id.to_numpy())
    assert np.isfinite(np.load(output / "features.npz")["descriptors"]).all()
    labels_bytes = records[["row_id", "y"]].to_csv(index=False).encode()
    author_bytes = records[["row_id", "author_label"]].to_csv(index=False).encode()
    manifest.update(sensitivity_kind=kind, primary_artifact_hashes=before, parent_project=str(ROOT),
        source_commit=COMMIT, source_csv_sha256=CSV_SHA256, row_count=len(records),
        labels_sha256=hashlib.sha256(labels_bytes).hexdigest(), author_labels_sha256=hashlib.sha256(author_bytes).hexdigest(),
        records_sha256=sha256_file(output / "records.parquet"), features_sha256=sha256_file(output / "features.npz"),
        splits_sha256=sha256_file(output / "splits.json"), tokenizer_version=tokenizer.version,
        effective_vocab_sha256=hashlib.sha256(("\n".join(tokenizer.tokens) + "\n").encode()).hexdigest())
    (output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    label_summary = {slug: {"source": fold["source"], "eligibility_frozen_in_split": fold["eligible"],
        "current_labels": summarize_rows(records.loc[records.row_id.isin(fold["test_ids"])])}
        for slug, fold in splits["folds"].items()}
    (audit / "label_summary.json").write_text(json.dumps(label_summary, indent=2) + "\n")
    alternative_check = independent_split_audit(records, splits)
    after = artifact_hashes(main)
    assert before == after, "primary artifacts changed during sensitivity preparation"
    report = {"kind": kind, "main_artifacts_unchanged": True, "primary_hashes_before": before,
        "primary_hashes_after": after, "primary_independent_split_audit": primary_check,
        "sensitivity_independent_split_audit": alternative_check, "row_count": len(records),
        "feature_manifest": manifest}
    (audit / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (alt / "README.md").write_text(f"# {kind} sensitivity data\n\n{manifest['warning']}\n\n"
        "Preparation only; no model was trained or evaluated. Use the main project's .venv and scripts with this directory as the project data root. weights and src are relative symlinks; no environment or pretrained weights are duplicated.\n\n"
        f"Label policy: {manifest['label_policy']}. Split policy: {manifest['split_policy']}.\n")
    print(json.dumps({"kind": kind, "root": str(alt), "rows": len(records), "eligible_sources": [f["source"] for f in splits["folds"].values() if f["eligible"]], "main_artifacts_unchanged": True}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("author_labels", "dd_original"))
    args = parser.parse_args()
    started, cpu = time.time(), time.process_time()
    status = "failed"
    try:
        run(args.kind)
        status = "complete"
    finally:
        directory = ROOT / "sensitivity" / args.kind
        directory.mkdir(parents=True, exist_ok=True)
        record = {"pid": os.getpid(), "command": sys.argv, "start_unix": started, "status": status,
                  "elapsed_seconds": time.time() - started, "cpu_seconds": time.process_time() - cpu,
                  "gpu_seconds": 0, "source_commit": COMMIT}
        with (directory / "preparation_runs.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
