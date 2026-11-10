#!/usr/bin/env python3
"""Fixed-split chemistry characterization; predictions are opt-in and lock-gated."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import lru_cache

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
SIMILARITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.0]
SIMILARITY_LABELS = ["[0,0.3]", "(0.3,0.5]", "(0.5,0.7]", "(0.7,1]"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def molecule(smiles):
    result = Chem.MolFromSmiles(smiles)
    if result is None:
        raise ValueError("Invalid previously audited SMILES")
    return result


@lru_cache(maxsize=None)
def fingerprint(smiles):
    return GENERATOR.GetFingerprint(molecule(smiles))


@lru_cache(maxsize=None)
def aryl_type(smiles):
    mol = molecule(smiles)
    hetero = any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                 and any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6) for i in ring)
                 for ring in mol.GetRingInfo().AtomRings())
    halogens = sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() in (9, 17, 35, 53)
                       and any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())})
    return bool(hetero), "+".join(halogens) if halogens else "none"


@lru_cache(maxsize=None)
def amine_type(smiles):
    """Coarse sites, not an asserted reacting atom or chemical mechanism."""
    aromatic = primary = secondary = 0
    mol = molecule(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if atom.GetIsAromatic():
            aromatic += 1
            continue
        neighbors = list(atom.GetNeighbors())
        # Exclude straightforward amide/thioamide/sulfonamide nitrogens from
        # the primary/secondary amine flags; other unusual N chemistry is other.
        acyl_or_sulfonyl = any(n.GetAtomicNum() in (6, 16) and
            any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetAtomicNum() in (8, 16)
                for b in n.GetBonds()) for n in neighbors)
        if acyl_or_sulfonyl:
            continue
        carbon = sum(n.GetAtomicNum() == 6 for n in neighbors)
        heavy = sum(n.GetAtomicNum() > 1 for n in neighbors)
        hydrogens = atom.GetTotalNumHs()
        if carbon == heavy == 1 and hydrogens >= 2:
            primary += 1
        elif carbon == heavy == 2 and hydrogens >= 1:
            secondary += 1
    if primary and not secondary:
        coarse = "primary"
    elif secondary and not primary:
        coarse = "secondary"
    elif not primary and not secondary and aromatic:
        coarse = "aromatic_N"
    else:
        coarse = "other"
    return {"amine_type": coarse, "amine_aromatic_n_count": aromatic,
            "amine_primary_n_count": primary, "amine_secondary_n_count": secondary,
            "amine_multiple_aliphatic_types": bool(primary and secondary)}


def nearest_similarity(test_smiles, training_smiles):
    train = [fingerprint(s) for s in sorted(set(training_smiles))]
    if not train:
        raise ValueError("No training structures for similarity calculation")
    return {s: float(max(DataStructs.BulkTanimotoSimilarity(fingerprint(s), train)))
            for s in sorted(set(test_smiles))}


def distribution(values):
    if not len(values):
        return None
    return {str(q): float(np.quantile(values, q / 100)) for q in (0, 25, 50, 75, 90, 95, 100)}


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(safe_json(obj), indent=2, allow_nan=False) + "\n")


def characterize(output):
    data = ROOT / "data/processed"
    names = ("records.parquet", "splits.json", "features.npz", "feature_manifest.json")
    before = {name: sha(data / name) for name in names}
    records = pd.read_parquet(data / "records.parquet")
    indexed = records.set_index("row_id", drop=False)
    splits = json.loads((data / "splits.json").read_text())
    source_rates = records.groupby("source").y.mean().to_dict()
    pieces, summaries = [], {}
    for fold_id, fold in splits["folds"].items():
        train, test = indexed.loc[fold["train_ids"]], indexed.loc[fold["test_ids"]]
        train_entities = set(train.aryl_id) | set(train.amine_id)
        assert not train_entities & (set(test.aryl_id) | set(test.amine_id))
        base = {"source": fold["source"], "eligible": bool(fold["eligible"]), "train_rows": len(train),
            "test_rows": len(test), "test_pairs": int(test.pair_id.nunique()),
            "train_positive_rate": float(train.y.mean()),
            "train_source_balanced_positive_rate": float(train.groupby("source").y.mean().mean()),
            "source_all_main_positive_rate": float(source_rates[fold["source"]]),
            "source_test_positive_rate": float(test.y.mean()) if len(test) else None,
            "training_unique_aryl_structures": int(train.aryl_smiles.nunique()),
            "training_unique_amine_structures": int(train.amine_smiles.nunique())}
        if test.empty:
            summaries[fold_id] = base
            continue
        aryl_similarity = nearest_similarity(test.aryl_smiles, train.aryl_smiles)
        amine_similarity = nearest_similarity(test.amine_smiles, train.amine_smiles)
        part = test[["row_id", "source", "pair_id", "aryl_id", "amine_id", "y"]].copy().rename(columns={"y": "y_true"})
        part["fold"] = fold_id
        part["eligible_source"] = fold["eligible"]
        part["aryl_max_train_tanimoto"] = test.aryl_smiles.map(aryl_similarity).to_numpy()
        part["amine_max_train_tanimoto"] = test.amine_smiles.map(amine_similarity).to_numpy()
        part["aryl_has_heteroaromatic_ring"] = [aryl_type(s)[0] for s in test.aryl_smiles]
        part["aryl_bound_halogen_class"] = [aryl_type(s)[1] for s in test.aryl_smiles]
        nitrogen = [amine_type(s) for s in test.amine_smiles]
        for key in nitrogen[0]:
            part[key] = [v[key] for v in nitrogen]
        for role in ("catalyst", "base", "solvent"):
            # Exact canonical, charge/stereo/salt-preserving structure identity;
            # no pooled-source frequency or label influences this membership.
            seen = set(train[role + "_smiles"])
            part[role + "_seen_in_train"] = test[role + "_smiles"].isin(seen).to_numpy()
            base[role + "_test_row_coverage"] = float(part[role + "_seen_in_train"].mean())
            base[role + "_test_unique_structure_coverage"] = float(np.mean([s in seen for s in set(test[role + "_smiles"])]))
        seen_count = part[[r + "_seen_in_train" for r in ("catalyst", "base", "solvent")]].sum(axis=1)
        part["reagent_coverage"] = np.select([seen_count == 3, seen_count == 0], ["all_seen", "none_seen"], default="partly_seen")
        for col in ("train_positive_rate", "train_source_balanced_positive_rate", "source_all_main_positive_rate", "source_test_positive_rate"):
            part[col] = base[col]
        for role in ("aryl", "amine"):
            part[role + "_similarity_bin"] = pd.cut(part[role + "_max_train_tanimoto"], SIMILARITY_BINS,
                labels=SIMILARITY_LABELS, include_lowest=True).astype(str)
            base[role + "_max_similarity_quantiles"] = distribution(part[role + "_max_train_tanimoto"].to_numpy())
        base["amine_type_rows"] = part.amine_type.value_counts().to_dict()
        base["aryl_halogen_class_rows"] = part.aryl_bound_halogen_class.value_counts().to_dict()
        base["heteroaromatic_rows"] = int(part.aryl_has_heteroaromatic_ring.sum())
        base["all_reagents_seen_rows"] = int((seen_count == 3).sum())
        summaries[fold_id] = base
        pieces.append(part)
    covariates = pd.concat(pieces, ignore_index=True).sort_values(["fold", "row_id"])
    assert not covariates.duplicated(["fold", "row_id"]).any()
    assert len(covariates) == sum(len(f["test_ids"]) for f in splits["folds"].values())
    covariates.to_parquet(output / "row_covariates.parquet", index=False)
    write_json(output / "coverage_summary.json", summaries)
    pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, dict)} | {"fold": fold}
                  for fold, summary in summaries.items()]).to_csv(output / "coverage_summary.csv", index=False)
    dd_splits_path = ROOT / "sensitivity/dd_original/data/processed/splits.json"
    dd_splits = json.loads(dd_splits_path.read_text())
    common, common_rows = {}, []
    for fold_id, fold in splits["folds"].items():
        other = dd_splits["folds"].get(fold_id)
        if other is None or other["source"] != fold["source"]:
            raise ValueError("DD source fold mismatch")
        a, b = set(fold["test_ids"]), set(other["test_ids"])
        ids = sorted(a & b)
        common[fold_id] = {"source": fold["source"], "main_eligible": fold["eligible"],
            "dd_eligible": other["eligible"], "main_test_rows": len(a), "dd_test_rows": len(b),
            "common_row_ids": ids, "main_only_row_ids": sorted(a - b), "dd_only_row_ids": sorted(b - a),
            "common_pairs": int(indexed.loc[ids].pair_id.nunique()),
            "common_positive_rows": int(indexed.loc[ids].y.sum())}
        common_rows.extend({"fold": fold_id, "source": fold["source"], "row_id": row_id} for row_id in ids)
    write_json(output / "dd_common_test_ids.json", {"main_splits_sha256": before["splits.json"],
        "dd_splits_sha256": sha(dd_splits_path), "folds": common})
    pd.DataFrame(common_rows).to_parquet(output / "dd_common_test_ids.parquet", index=False)
    after = {name: sha(data / name) for name in names}
    assert before == after, "Primary data changed during characterization"
    manifest = {"purpose": "chemistry and domain covariates only, no model performance",
        "predictions_read": False, "source_artifacts": before, "main_artifacts_unchanged": True,
        "script_sha256": sha(__file__), "rdkit_version": rdBase.rdkitVersion,
        "fingerprint": {"method": "Morgan", "radius": 2, "bits": 2048, "includeChirality": False},
        "similarity_reference": "outer train_ids only, including its internal validation rows; separate aryl and amine roles",
        "test_row_count": len(covariates), "unique_substrate_structures_cached": fingerprint.cache_info().currsize,
        "similarity_bins": SIMILARITY_BINS, "output_sha256": sha(output / "row_covariates.parquet")}
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text("# Fixed-split domain characterization\n\n"
        "No outer prediction files were read. These outputs describe data coverage, not predictive performance.\n\n"
        "Similarity is the maximum Morgan radius-2, 2048-bit Tanimoto to unique structures in the corresponding outer development set, separately by substrate role. Chirality is excluded from fingerprints; a similarity of 1 can occur without entity identity because of fingerprint collisions or unrepresented distinctions. Group-based leakage checks remain separate.\n\n"
        "The aryl heteroaromatic flag means any fully aromatic ring containing a non-carbon aromatic atom. Halogen classes list F/Cl/Br/I attached directly to aromatic atoms; this does not identify the reacting leaving group. Amine flags count nonaromatic primary/secondary N-H sites, excluding simple amide/thioamide/sulfonamide connections, and aromatic N atoms. Primary or secondary class takes precedence over incidental aromatic N; substrates with both aliphatic classes are 'other' with a mixed flag. These are coarse structural categories, not mechanistic assignments.\n\n"
        "Reagent coverage is exact canonical charge/stereo/salt-preserving structure membership in outer training rows. Source class rates distinguish the full source from the retained strict-OOD test subset. The DD common-row files provide fixed shared test IDs for fair paired sensitivity comparisons.\n\n"
        "Optional future stratified metrics require explicit --predictions paths and a FROZEN, evaluation_allowed=true lock; the full saved prediction panel must pass the project's independent completeness validation first. No such option was used for characterization.\n")
    return covariates, manifest


def add_stratified_metrics(covariates, paths, output):
    # Check access stage BEFORE opening any prediction file.
    lock_path = ROOT / "FINAL_EVAL_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "FROZEN" or lock.get("evaluation_allowed") is not True:
        raise ValueError("Stratified evaluation remains sealed")
    from bh_transfer.evaluate import validate_predictions, WeightedMetrics
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    predictions, _ = validate_predictions(predictions, lock, lock_path)
    overlap = set(predictions.columns) & set(covariates.columns) - {"row_id", "fold"}
    merged = predictions.merge(covariates.drop(columns=sorted(overlap)), on=["row_id", "fold"], how="left", validate="many_to_one")
    assert merged.aryl_max_train_tanimoto.notna().all()
    dimensions = ("aryl_similarity_bin", "amine_similarity_bin", "aryl_has_heteroaromatic_ring",
                  "aryl_bound_halogen_class", "amine_type", "reagent_coverage")
    metrics = []
    for dimension in dimensions:
        for (fold, model, seed, stratum), part in merged.groupby(["fold", "model", "seed", dimension], observed=True):
            auc, ap, brier = WeightedMetrics(part.y_true.to_numpy(), part.score.to_numpy())()
            metrics.append({"fold": fold, "model": model, "seed": seed, "dimension": dimension, "stratum": str(stratum),
                "rows": len(part), "pairs": part.pair_id.nunique(), "positive_rate": part.y_true.mean(),
                "auroc": auc, "auprc": ap, "brier": brier,
                "interpretation": "descriptive stratum metrics; rows and seeds are not independent chemical samples"})
    pd.DataFrame(metrics).to_parquet(output / "stratified_metrics.parquet", index=False)
    pd.DataFrame(metrics).to_csv(output / "stratified_metrics.csv", index=False)
    write_json(output / "stratified_metric_inputs.json", {"evaluation_lock_sha256": sha(lock_path),
        "prediction_sha256": {str(path): sha(path) for path in paths}, "covariates_sha256": sha(output / "row_covariates.parquet")})


def self_check():
    assert aryl_type("Clc1ccncc1") == (True, "Cl")
    assert aryl_type("FC(F)(F)c1ccccc1") == (False, "none")
    assert amine_type("Nc1ccncc1")["amine_type"] == "primary"
    assert amine_type("CN(C)C")["amine_type"] == "other"
    assert amine_type("CNCC")["amine_type"] == "secondary"
    assert amine_type("CC(=O)N")["amine_type"] == "other"
    assert amine_type("c1cc[nH]c1")["amine_type"] == "aromatic_N"
    assert nearest_similarity(["CCO"], ["CCO"])["CCO"] == 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", help="Explicit future unsealed complete prediction panel only")
    args = parser.parse_args()
    output = ROOT / "reports/domain"
    output.mkdir(parents=True, exist_ok=True)
    self_check()
    covariates, manifest = characterize(output)
    if args.predictions:
        add_stratified_metrics(covariates, args.predictions, output)
    print(json.dumps({"rows": len(covariates), "folds": int(covariates.fold.nunique()),
        "predictions_read": bool(args.predictions), "output": str(output), "main_artifacts_unchanged": True}), flush=True)
