#!/usr/bin/env python3
"""Download an immutable source snapshot, audit it, and prepare fixed inputs/splits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COMMIT = "3a44e398d9fed3f08cf396f1153be2960193f581"
CSV = "BH_HTE_Curated_Dataset_v20260219.csv"
EXPECTED_SHA = "247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333"
BASE = f"https://raw.githubusercontent.com/schwallergroup/bh-hte-ood/{COMMIT}/"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(relative, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected is None or digest(destination) == expected):
        print(f"Reusing {destination.relative_to(ROOT)}", flush=True)
    else:
        temp = destination.with_suffix(destination.suffix + ".partial")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE + urllib.parse.quote(relative), timeout=120) as response, temp.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                if expected and digest(temp) != expected:
                    raise ValueError("download hash does not match frozen snapshot")
                temp.replace(destination)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        print(f"Downloaded {destination.relative_to(ROOT)}", flush=True)
    destination.chmod(0o444)
    return {"path": str(destination.relative_to(ROOT)), "url": BASE + urllib.parse.quote(relative),
            "bytes": destination.stat().st_size, "sha256": digest(destination)}


def acquire():
    manifest = [download("data/" + CSV, ROOT / "data/raw" / CSV, EXPECTED_SHA)]
    for relative in ("README.md", "LICENSE.txt", "Production Model.ipynb", "Reagent Recommender.ipynb",
                     "Data Sources Benchmark.ipynb", "utils/drfp_cluster.py", "utils/ml_train.py", "data/reagent_dictionary.csv"):
        manifest.append(download(relative, ROOT / "third_party/bh-hte-ood" / relative))
    payload = {"commit": COMMIT, "code_license": "Apache-2.0", "data_license": "CC-BY-4.0",
               "data_archive": "https://zenodo.org/records/19636649", "files": manifest}
    manifest_path = ROOT / "data/raw/SOURCE_MANIFEST.json"
    serialized = json.dumps(payload, indent=2) + "\n"
    if manifest_path.exists():
        assert manifest_path.read_text() == serialized, "source snapshot manifest changed"
    else:
        manifest_path.write_text(serialized)
        manifest_path.chmod(0o444)


def prepare():
    import numpy as np
    import pandas as pd
    from rdkit import rdBase
    from bh_transfer.data import prepare_records, make_splits, assert_split_integrity, build_features, DESCRIPTOR_NAMES, ROLES, RAW_COLUMNS, stable_key, apply_tokenizer_serialization
    raw = pd.read_csv(ROOT / "data/raw" / CSV, keep_default_na=False)
    audit = ROOT / "data/audit"
    audit.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(raw.Yield.astype(float)).all(), "nonfinite yield needs explicit audit disposition"
    assert set(raw.labels.astype(int).unique()) <= {0, 1}, "unexpected author label"
    raw_summary = raw[["Source", "Yield", "labels", *RAW_COLUMNS.values(), "Product SMILES"]].copy()
    raw_summary.insert(0, "row_id", np.arange(len(raw)))
    raw_summary["primary_label"] = (raw.Yield.astype(float) >= 10).astype(int)
    raw_summary["raw_pair_id"] = [stable_key(a, b) for a, b in zip(raw["Aryl SMILES"], raw["Amine SMILES"])]
    raw_summary["raw_input_key"] = [stable_key(*values) for values in raw[list(RAW_COLUMNS.values())].itertuples(index=False, name=None)]
    raw_duplicates = raw_summary.loc[raw_summary.raw_input_key.duplicated(keep=False)].copy()
    raw_duplicates["conflicting_labels"] = raw_duplicates.groupby("raw_input_key").primary_label.transform("nunique") > 1
    raw_duplicates.to_parquet(audit / "raw_duplicate_inputs.parquet", index=False)
    raw_summary.groupby("Source").agg(rows=("row_id", "size"), raw_pairs=("raw_pair_id", "nunique"),
        positive_rows=("primary_label", "sum"), positive_rate=("primary_label", "mean"),
        minimum_yield=("Yield", "min"), maximum_yield=("Yield", "max")).to_csv(audit / "raw_source_summary.csv")
    original_qm = np.stack([np.fromstring(str(v).strip().strip("[]"), sep=" ") for v in raw.QM])
    assert original_qm.shape == (len(raw), 50), original_qm.shape
    np.savez_compressed(audit / "original_qm.npz", row_ids=np.arange(len(raw)), qm=original_qm)
    finite = np.isfinite(original_qm)
    additive_columns = [col for col in raw.columns if col.startswith("Additive_")]
    additive_tail_mismatch = int(np.any(original_qm[:, 42:50] != raw[additive_columns].to_numpy(dtype=float), axis=1).sum())
    mismatch = raw.loc[(raw.Yield.astype(float) >= 10).astype(int) != raw.labels.astype(int)].copy()
    mismatch.insert(0, "row_id", mismatch.index)
    mismatch["primary_label"] = (mismatch.Yield.astype(float) >= 10).astype(int)
    mismatch.to_csv(audit / "label_disagreements.csv", index=False)
    boundary = raw.loc[raw.Yield.astype(float) == 10, ["Source", "Yield", "labels"]].copy()
    boundary.insert(0, "row_id", boundary.index)
    boundary.to_csv(audit / "yield_exactly_10.csv", index=False)
    fields = [{"field": col, "dtype": str(raw[col].dtype), "missing_or_empty": int(raw[col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"]).sum()),
               "unique_values": int(raw[col].nunique())} for col in raw.columns]
    pd.DataFrame(fields).to_csv(audit / "fields.csv", index=False)
    records, mapping, excluded = prepare_records(raw)
    from bh_transfer.tokenizer import StrictSmilesTokenizer
    tokenizer = StrictSmilesTokenizer(ROOT / "weights/rxnfp/vocab.txt", max_length=512)
    records, serialization_mapping = apply_tokenizer_serialization(records, tokenizer)
    serialization_mapping.to_parquet(audit / "token_serialization_mapping.parquet", index=False)
    mapping.to_parquet(audit / "structure_mapping.parquet", index=False)
    excluded.to_csv(audit / "excluded_rows.csv", index=False)
    records.to_parquet(processed / "records.parquet", index=False)
    token_audit = {"full_reaction": tokenizer.audit(records.reaction_smiles.tolist(), records.row_id.tolist()), "roles": {}}
    for role in ROLES:
        role_mapping = serialization_mapping.loc[serialization_mapping.role == role].reset_index(drop=True)
        values = role_mapping.serialized_smiles
        nonempty = values != ""
        role_audit = tokenizer.audit(values.loc[nonempty].tolist(), role_mapping.loc[nonempty, "row_id"].tolist())
        role_audit["known_empty_rows"] = int((~nonempty).sum())
        token_audit["roles"][role] = role_audit
    lengths = np.array([len(tokenizer.regex.findall(v)) + 2 for v in records.reaction_smiles], dtype=int)
    token_audit["full_reaction"]["length_distribution_including_unrepresentable"] = {str(q): float(np.quantile(lengths, q / 100)) for q in (0, 25, 50, 75, 90, 95, 99, 100)}
    token_audit["full_reaction"]["overlength_row_ids"] = records.loc[lengths > tokenizer.max_length, "row_id"].tolist()
    token_audit["original_vocab_sha256"] = digest(ROOT / "weights/rxnfp/vocab.txt")
    token_audit["effective_vocab_sha256"] = hashlib.sha256(("\n".join(tokenizer.tokens) + "\n").encode()).hexdigest()
    token_audit["effective_vocab_size"] = len(tokenizer.tokens)
    token_audit["representation_version"] = "rxnfp-original-plus-two-dative-bond-tokens-v1"
    token_audit["serialization_rules"] = serialization_mapping.rule.value_counts().to_dict()
    (audit / "token_audit.json").write_text(json.dumps(token_audit, indent=2) + "\n")
    if token_audit["full_reaction"]["n_invalid"]:
        raise RuntimeError(f"{token_audit['full_reaction']['n_invalid']} tokenization failures; inspect data/audit/token_audit.json and resolve before features/split freeze; no rows silently dropped")
    duplicate = records.loc[records.input_key.duplicated(keep=False)].copy()
    duplicate["input_group_rows"] = duplicate.groupby("input_key").row_id.transform("size")
    duplicate["conflicting_labels"] = duplicate.groupby("input_key").y.transform("nunique") > 1
    duplicate.to_parquet(audit / "duplicate_inputs.parquet", index=False)
    source_summary = records.groupby("source").agg(rows=("row_id", "size"), pairs=("pair_id", "nunique"), positive_rows=("y", "sum"), positive_rate=("y", "mean")).reset_index()
    source_summary.to_csv(audit / "source_summary.csv", index=False)
    qm_consistency = []
    for offset, role in enumerate((*ROLES, "product")):
        column = role.capitalize() + " SMILES"
        seen = {}
        for smi, vector in zip(raw[column], original_qm[:, offset * 7:(offset + 1) * 7]):
            seen.setdefault(smi, set()).add(tuple(vector))
        qm_consistency.append({"role": role, "start": offset * 7, "end": (offset + 1) * 7,
                               "entities": len(seen), "inconsistent_entities": sum(len(v) != 1 for v in seen.values())})
    pd.DataFrame(qm_consistency).to_csv(audit / "qm_role_consistency.csv", index=False)
    splits = make_splits(records)
    assert_split_integrity(records, splits)
    grouping_sensitivity = []
    for fold in splits["folds"].values():
        development = records.loc[records.source != fold["source"]]
        candidates = records.loc[records.source == fold["source"]]
        same_role = candidates.loc[~candidates.aryl_id.isin(development.aryl_id) & ~candidates.amine_id.isin(development.amine_id)]
        grouping_sensitivity.append({"source": fold["source"], "same_role_unseen_rows": len(same_role),
            "role_union_unseen_rows": fold["test_summary"]["rows"], "same_role_pairs": int(same_role.pair_id.nunique()),
            "role_union_pairs": fold["test_summary"]["pairs"]})
    pd.DataFrame(grouping_sensitivity).to_csv(audit / "entity_rule_sensitivity.csv", index=False)
    splits["records_sha256"] = digest(processed / "records.parquet")
    (processed / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    print("Prepared records and verified all source/entity intersections", flush=True)
    features = build_features(records)
    np.savez_compressed(processed / "features.npz", **features)
    (processed / "feature_manifest.json").write_text(json.dumps({
        "product_input_allowed": False, "original_qm_used": False, "descriptor_names": [f"{role}.{name}" for role in ("catalyst", "base", "solvent") for name in DESCRIPTOR_NAMES],
        "r0": "standard DRFP, radius=3, 2048 bits, all five roles, empty products", "r1_substitute": "author Morgan radius=3 OR substrate bits, empty product side, plus public RDKit21 reagent descriptors",
        "neural_descriptors": "same RDKit21 for every neural comparison", "rdkit_version": rdBase.rdkitVersion,
        "records_sha256": digest(processed / "records.parquet"), "features_sha256": digest(processed / "features.npz"),
        "splits_sha256": digest(processed / "splits.json")}, indent=2) + "\n")
    summary = {"raw_rows": len(raw), "main_rows": len(records), "excluded_counts": excluded.reason.value_counts().to_dict(),
        "raw_sources": raw.Source.value_counts().to_dict(), "label_disagreements": len(mismatch), "at_exactly_10": len(boundary),
        "qm_shape": list(original_qm.shape), "qm_nonfinite": int((~finite).sum()), "qm_role_consistency": qm_consistency,
        "qm_42_50_additive_columns": additive_columns, "qm_additive_tail_mismatch_rows": additive_tail_mismatch,
        "truncated_drfp": {c: int(raw[c].str.contains("...", regex=False).sum()) for c in ("DRFP", "DRFP_subst_prod")},
        "duplicate_groups": int(duplicate.input_key.nunique()), "extra_duplicate_rows": int(len(duplicate) - duplicate.input_key.nunique()),
        "conflicting_groups": int(duplicate.loc[duplicate.conflicting_labels, "input_key"].nunique()),
        "raw_duplicate_groups": int(raw_duplicates.raw_input_key.nunique()),
        "raw_conflicting_groups": int(raw_duplicates.loc[raw_duplicates.conflicting_labels, "raw_input_key"].nunique()),
        "eligible_sources": [f["source"] for f in splits["folds"].values() if f["eligible"]],
        "splits": {k: {"source": f["source"], "eligible": f["eligible"], **f["test_summary"]} for k, f in splits["folds"].items()}}
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Data audit", "", f"Frozen author commit: `{COMMIT}`. CSV SHA256 `{EXPECTED_SHA}`.", "",
        "Code: Apache-2.0; Zenodo dataset: CC-BY-4.0. Attribution: Neves, Hao, Aikonen, Diccianni, Wegner, Schwaller and Strambeanu, Robust out-of-distribution prediction of Buchwald–Hartwig reactions, DOI 10.1038/s43588-026-01017-6. Cite each original source when distributing derived results.", "",
        f"Raw rows: {len(raw)}. Main usable rows: {len(records)}. Exclusions are enumerated in data/audit/excluded_rows.csv; no model residuals or label conflicts are used for exclusion.", "",
        "## Outcomes and measurement", "",
        f"Primary label is Yield >= 10 (percentage scale); exactly 10 is positive. {len(boundary)} boundary rows; {len(mismatch)} disagreements with author labels, enumerated in label_disagreements.csv. Author labels remain available only as metadata/sensitivity targets. Values above 100 are retained, not clipped.", "",
        "The paper reports heterogeneous CAD assay yields, isolated yields, MisER and LCAP; the JNJ assay uses calibrated UPLC-MS/CAD. These quantities are not interchangeable precise isolated yields. The shared 10% threshold is a harmonized feasibility endpoint. Temperature, time, concentrations and detailed loading are absent from this CSV, limiting causal/condition-specific interpretation.", "",
        "## Input permissions and normalization", "",
        "No product enters any prediction feature. Public code demonstrates product-based recommendation but does not establish every CSV product as a prospective target. Product SMILES remains metadata. R0 is therefore a declared input-only DRFP substitute; R1 is a declared substitute for the author feature representation, not an exact reproduction or a claim to outperform the published model.", "",
        "Model input canonicalization retains salts, formal charges, isotopes, stereo and coordination bonds; it removes atom map indices and standardizes equivalent SMILES. Each role remains explicit in metadata and sequence serialization has fixed aryl/amine/catalyst/base/solvent order. Grouping is deliberately more conservative: largest organic fragment(s), uncharged canonical tautomer, stereo/isotope removal. This grouping only establishes identity and does not change the chemical model input. Both substrate roles are pooled when checking cross-boundary entity overlap.", "",
        "Token audit found that the original RXNFP vocabulary cannot represent dative bond arrows. Before any model performance was observed, the original vocabulary was extended by exactly two complete bond tokens ('->', '<-'), preserving all original token IDs. This applies equally to every neural comparison and its effective hash is recorded in token_audit.json. Canonical dative SMILES remains intact. Three [NaH+] entries are represented as graph-identical explicit-H SMILES; RemoveHs/canonical-isomeric equality and unchanged charge are verified. No original chemical bond or charge is repaired by assumption. All full reactions must pass strict unknown-token, unmatched-character and 512-token checks; no silent truncation occurs.", "",
        "All repeated inputs and conflicting outcomes are retained. They remain attached to their substrate entities/pair. Conflicts can reflect repeated experiments or unreported conditions and are not classified as wrong labels. Source names, row IDs, labels and author clusters never enter features.", "",
        "Dreher Doyle 2018 is excluded from both main training and testing because additive identities are unavailable in the curated CSV. Eight coarse additive flags are not identities; no verified original-row mapping has been recovered. Excluded rows remain in the immutable original data for a separate declared sensitivity experiment. No claim of restored identity is made.", "",
        "## Descriptors and fingerprints", "",
        f"Both stored DRFP columns are truncated in every row and are never parsed. Original QM is shape {original_qm.shape}, with {int((~finite).sum())} nonfinite entries. Role slices 0:7 aryl, 7:14 amine, 14:21 catalyst, 21:28 base, 28:35 solvent and 35:42 product are checked against structure identities. QM[42:50] matches the eight additive flags in CSV column order, with {additive_tail_mismatch} mismatching rows. The 7 descriptor identities/provenance are not established by available code, so original QM is excluded from every primary predictor.", "",
        "Legal replacement: each catalyst, base and solvent receives MolWt, MolLogP, TPSA, HBD, HBA, RotatableBonds, FormalCharge (21 public RDKit values total). All neural variants receive the identical descriptor vector. Any fitted scaling/imputation is deferred to training partitions; none is fitted here. Known empty reagent components receive seven zeros; required substrate parsing failures are explicitly excluded. Author-style substrate fingerprints reproduce Morgan OR construction at radius 3/2048 bits, with an explicitly empty product side. Standard DRFP is recalculated separately. Its upstream reaction-string parser splits on every '>', which would mistake dative arrows for reaction delimiters and silently omit input components. We therefore call the same official molecular shingling/hash/fold functions directly; for an empty product side, the reaction fingerprint is exactly the union of all input shingles. Unit tests verify equality to the original API on ordinary reactions and contributions from all five roles in a dative reaction. This is an input parsing adaptation, not a new fingerprint algorithm.", "",
        "## Fixed split verification", "",
        "Outer leave-one-source-out development uses all other main sources. A held-out row is scored only if neither substrate entity appears anywhere in development, including inner validation. Inner source holdouts are filtered by the same rule; at most two valid sources with the greatest remaining pair counts are selected. A source needs at least 30 pairs, positive/negative coverage of at least 10 pairs each, and 20 positive/negative rows each for the primary macro average. Other nonempty sources are descriptive; empty folds do not score.", "",
        "| Source | Rows | Pairs | Positive rows | Negative rows | Eligible |", "|---|---:|---:|---:|---:|---|"]
    for f in splits["folds"].values():
        s = f["test_summary"]
        lines.append(f"| {f['source']} | {s['rows']} | {s['pairs']} | {s['positive_rows']} | {s['negative_rows']} | {f['eligible']} |")
    lines.extend(["", "All outer and inner row, entity, pair and input-key intersections are asserted empty. Full mappings, field/missingness tables, conflicts, exclusions and hashes live under data/audit and data/processed. These are data eligibility audits, not model test performance. Same-role versus pooled-role grouping sensitivity is saved separately; the current snapshot gives identical retained rows under these two checks. Merck's retained test pairs share only eight amine entities, so pair counts must not be mistaken for independent coverage of thousands of amines; shared-substrate bootstrap sensitivity remains necessary.", "", "## Provenance", "", "- https://github.com/schwallergroup/bh-hte-ood/tree/" + COMMIT, "- https://zenodo.org/records/19636649", "- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13593570/fullTextXML", ""])
    (ROOT / "DATA_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    started, cpu_started = time.time(), time.process_time()
    status = "failed"
    try:
        acquire()
        if not args.download_only:
            prepare()
        status = "complete"
    finally:
        out = ROOT / "data/prepare_run.json"
        run = {"pid": os.getpid(), "command": sys.argv, "status": status,
            "start_unix": started, "elapsed_seconds": time.time() - started,
            "cpu_seconds": time.process_time() - cpu_started, "commit": COMMIT}
        history = ROOT / "data/preparation_runs.jsonl"
        if not history.exists() and out.exists():
            with history.open("a") as handle:
                handle.write(json.dumps(json.loads(out.read_text())) + "\n")
        with history.open("a") as handle:
            handle.write(json.dumps(run) + "\n")
        out.write_text(json.dumps(run, indent=2) + "\n")
