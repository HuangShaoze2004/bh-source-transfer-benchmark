#!/usr/bin/env python3
"""Independent post-release metric/leakage audit and experiment report.

Never reads predictions until completeness and evaluation manifests both permit
release. Does not import bh_transfer.evaluate or any training implementation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CORE_MODELS = {"R0_RF", "R0_HGB", "R1_substitute", "B", "B_S", "B_bal", "B_bal_S", "B_A", "B_A_S"}
METRICS = ("auroc", "auprc", "brier")


class PendingEvaluation(RuntimeError):
    pass


class AuditFailure(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def release_gate(project):
    project = Path(project)
    complete = read_json(project/"artifacts/PREDICTION_COMPLETENESS.json", {})
    evaluation = read_json(project/"reports/final/evaluation_manifest.json", {})
    if complete.get("complete") is not True or evaluation.get("status") != "EVALUATION_COMPLETE":
        raise PendingEvaluation("Complete predictions and completed evaluation are both required; predictions remain unread")
    lock_path, predictions_path = project/"FINAL_EVAL_LOCK.json", project/"artifacts/predictions.parquet"
    lock = read_json(lock_path, {})
    if lock.get("status") != "FROZEN" or lock.get("evaluation_allowed") is not True:
        raise AuditFailure("Evaluation lock does not permit release")
    for artifact, expected in lock.get("artifact_hashes", {}).items():
        path = project/artifact
        if not path.is_file() or sha256(path) != expected:
            raise AuditFailure("Frozen artifact differs: " + artifact)
    for artifact, expected in lock.get("source_code_hashes", {}).items():
        if not Path(artifact).is_file() or sha256(artifact) != expected:
            raise AuditFailure("Sensitivity code dependency differs: " + artifact)
    # Only now access any bytes of the complete prediction artifact.
    lock_hash, prediction_hash = sha256(lock_path), sha256(predictions_path)
    for manifest in (complete, evaluation):
        if manifest.get("lock_sha256") != lock_hash or manifest.get("predictions_sha256", manifest.get("prediction_sha256")) != prediction_hash:
            raise AuditFailure("Release manifest has mismatched prediction/lock hashes")
    return lock, complete, evaluation


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def independent_split_audit(records, splits, lock):
    require(not records.row_id.duplicated().any(), "Duplicate source record row IDs")
    require(records.y.isin([0, 1]).all(), "Frozen target is not binary")
    require(not (records.groupby("pair_id")[["aryl_id", "amine_id"]].nunique() > 1).any().any(), "A substrate-pair identifier maps to inconsistent entities")
    if not lock.get("sensitivity_kind"):
        require("Dreher Doyle 2018" not in set(records.source), "Unresolved-additive source entered the main dataset")
    indexed = records.set_index("row_id", drop=False)
    fold_manifest = {str(f["fold"]): f for f in lock["folds"]}
    require(set(fold_manifest) == set(splits["folds"]), "Lock/split fold identity mismatch")
    details = []
    for name, part in splits["folds"].items():
        frozen = fold_manifest[name]
        require(str(part["source"]) == str(frozen["source"]) and bool(part["eligible"]) == bool(frozen["eligible"]), "Locked source/eligibility mismatch")
        train_ids, test_ids = list(part["train_ids"]), list(part["test_ids"])
        require(len(set(train_ids)) == len(train_ids) and len(set(test_ids)) == len(test_ids), "Duplicate split row IDs")
        require(set(test_ids) == set(frozen["test_row_ids"]), "Lock test row coverage differs from splits")
        development = records[records.source != part["source"]]
        require(set(train_ids) == set(development.row_id), "Outer development is not all other sources")
        seen = set(development.aryl_id) | set(development.amine_id)
        expected = records[(records.source == part["source"]) & ~records.aryl_id.isin(seen) & ~records.amine_id.isin(seen)]
        require(set(test_ids) == set(expected.row_id), "Outer test is not the exact both-unseen source subset")
        partitions = [(train_ids, test_ids, "outer")]
        for inner in part.get("inner", []):
            itrain, ival = list(inner["train_ids"]), list(inner["val_ids"])
            require(set(itrain) <= set(train_ids) and set(ival) <= set(train_ids), "Inner selection reads outside outer development")
            if "source" in inner:
                expected_train = development[development.source != inner["source"]]
                require(set(itrain) == set(expected_train.row_id), "Inner training source holdout is inconsistent")
                inner_seen = set(expected_train.aryl_id) | set(expected_train.amine_id)
                expected_val = development[(development.source == inner["source"]) & ~development.aryl_id.isin(inner_seen) & ~development.amine_id.isin(inner_seen)]
                require(set(ival) == set(expected_val.row_id), "Inner validation violates both-unseen rule")
            partitions.append((itrain, ival, "inner"))
        for left, right, kind in partitions:
            require(not set(left) & set(right), f"{kind} row leakage")
            a, b = indexed.loc[left], indexed.loc[right]
            require(not (set(a.aryl_id) | set(a.amine_id)) & (set(b.aryl_id) | set(b.amine_id)), f"{kind} substrate-entity leakage")
            require(not set(a.pair_id) & set(b.pair_id), f"{kind} substrate-pair leakage")
            if "input_key" in records:
                require(not set(a.input_key) & set(b.input_key), f"{kind} repeated-input leakage")
        details.append(dict(fold=name, source=part["source"], eligible=bool(part["eligible"]),
                            test_rows=len(expected), test_pairs=expected.pair_id.nunique(),
                            test_aryls=expected.aryl_id.nunique(), test_amines=expected.amine_id.nunique(),
                            inner_splits=len(part.get("inner", []))))
    return details


def independent_prediction_audit(predictions, records, lock):
    columns = {"row_id", "source", "pair_id", "aryl_id", "amine_id", "y_true", "model", "seed", "fold", "score"}
    require(columns <= set(predictions), "Prediction schema is incomplete")
    require(not predictions[list(columns)].isna().any().any(), "Missing prediction values")
    require(pd.api.types.is_integer_dtype(predictions.row_id), "row_id must be an integer")
    require(not predictions.duplicated(["fold", "row_id", "model", "seed"]).any(), "Duplicate method/seed/row predictions")
    require(set(predictions.model) == set(lock["models"]) and set(predictions.seed) == set(lock["seeds"]), "Incomplete model/seed matrix")
    require(len(lock["seeds"]) >= lock.get("minimum_seeds", 3), "Too few registered seeds")
    require(lock.get("score_type", "probability") == "probability", "Independent audit currently requires probability scores")
    for col in ("score", "cal_score"):
        if col in predictions:
            require(np.isfinite(predictions[col]).all() and predictions[col].between(0, 1).all(), f"Invalid {col} probability")
    truth = records.set_index("row_id")
    require(set(predictions.row_id) <= set(truth.index), "Prediction row absent from frozen records")
    expected = truth.loc[predictions.row_id]
    for column, original in (("source", "source"), ("pair_id", "pair_id"), ("aryl_id", "aryl_id"), ("amine_id", "amine_id"), ("y_true", "y")):
        require(np.array_equal(predictions[column].to_numpy(), expected[original].to_numpy()), "Ground-truth/chemistry mismatch: " + column)
    if "condition_id" in predictions and "input_key" in records:
        require(np.array_equal(predictions.condition_id.to_numpy(), expected.input_key.to_numpy()), "Condition ID differs from frozen input key")
    allowed_folds = set()
    for fold in lock["folds"]:
        part = predictions[predictions.fold.astype(str) == str(fold["fold"])]
        ids = set(fold["test_row_ids"])
        if ids:
            allowed_folds.add(str(fold["fold"]))
        require(set(part.source) == ({fold["source"]} if ids else set()), "Source identity differs from fold lock")
        for model in lock["models"]:
            for seed in lock["seeds"]:
                selected = part[(part.model == model) & (part.seed == seed)]
                require(set(selected.row_id) == ids and len(selected) == len(ids), "Incomplete locked row coverage")
    require(set(predictions.fold.astype(str)) == allowed_folds, "Unexpected prediction folds")
    if "author" in lock.get("primary_label", "").lower():
        require(np.array_equal(records.y.to_numpy(), records.author_label.to_numpy()), "Author-label sensitivity target mismatch")
    elif "yield_raw" in records:
        require(np.array_equal(records.y.to_numpy(), (records.yield_raw >= 10).astype(int).to_numpy()), "Main label is not Yield>=10")


def reconstruct_metrics(predictions, lock):
    """Independent sklearn calculation, not the production metric code."""
    eligible = {f["source"] for f in lock["folds"] if f["eligible"]}
    rows = []
    for (source, model, seed), part in predictions.groupby(["source", "model", "seed"], sort=True):
        binary = part.y_true.nunique() == 2
        rows.append(dict(source=source, model=model, seed=int(seed), eligible=source in eligible,
                         n_rows=len(part), n_pairs=part.pair_id.nunique(), positive_rate=float(part.y_true.mean()),
                         auroc=roc_auc_score(part.y_true, part.score) if binary else np.nan,
                         auprc=average_precision_score(part.y_true, part.score) if binary else np.nan,
                         brier=brier_score_loss(part.y_true, part.score),
                         cal_brier=brier_score_loss(part.y_true, part.cal_score) if "cal_score" in part else np.nan))
    by_seed = pd.DataFrame(rows)
    macro = []
    for model in lock["models"]:
        by_source = by_seed[(by_seed.model == model) & by_seed.eligible].groupby("source")[[*METRICS, "cal_brier"]].mean()
        require(len(by_source) == len(eligible), "Missing eligible source in macro calculation")
        require(np.isfinite(by_source[list(METRICS)]).all().all(), "Undefined metric in eligible source")
        macro.append(dict(model=model, n_sources=len(by_source), **by_source.mean().to_dict(), worst_source_auroc=by_source.auroc.min()))
    return by_seed, pd.DataFrame(macro)


def compare_saved_table(actual, saved, keys, values, name):
    require(not saved.duplicated(keys).any(), "Duplicate saved table keys: " + name)
    a, b = actual.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    require(a.index.equals(b.index), "Saved table row keys differ: " + name)
    for col in values:
        require(col in b, "Missing saved metric column: " + col)
        require(np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True), f"Independent metric differs: {name}/{col}")


def audit_project(project):
    project = Path(project)
    lock, completeness, evaluation = release_gate(project)
    records = pd.read_parquet(project/"data/processed/records.parquet")
    splits = read_json(project/"data/processed/splits.json")
    split_summary = independent_split_audit(records, splits, lock)
    predictions = pd.read_parquet(project/"artifacts/predictions.parquet")
    require(len(predictions) == completeness["rows"], "Completeness row count mismatch")
    independent_prediction_audit(predictions, records, lock)
    by_seed, macro = reconstruct_metrics(predictions, lock)
    saved_seed = pd.read_csv(project/"reports/final/metrics_by_source_seed.csv")
    compare_saved_table(by_seed, saved_seed, ["source", "model", "seed"], [*METRICS, "n_rows", "n_pairs", "positive_rate"] + (["cal_brier"] if "cal_score" in predictions else []), "per-source/per-seed")
    compare_saved_table(macro, pd.read_csv(project/"reports/final/main_table.csv"), ["model"], [*METRICS, "n_sources", "worst_source_auroc"], "macro")
    return dict(project=project, lock=lock, evaluation=evaluation, records=records, predictions=predictions,
                by_seed=by_seed, macro=macro, split_summary=split_summary,
                audit=dict(status="PASS", predictions_sha256=sha256(project/"artifacts/predictions.parquet"),
                           lock_sha256=sha256(project/"FINAL_EVAL_LOCK.json"), auditor_sha256=sha256(__file__),
                           checked_rows=len(predictions), metric_backend="sklearn; no bh_transfer.evaluate import",
                           split_summary=split_summary))


def validate_statistics(bundle):
    out, lock = bundle["project"]/"reports/final", bundle["lock"]
    manifest = bundle["evaluation"]
    for scheme, count in (("pair", lock.get("bootstrap_replicates", 5000)), ("pigeonhole", lock.get("sensitivity_replicates", 1000))):
        metadata = read_json(out/f"bootstrap_{scheme}.json", {})
        require(metadata.get("completed") == count and metadata.get("requested") == count, "Incomplete " + scheme + " bootstrap")
        require(metadata.get("signature") == manifest["signature"], "Bootstrap signature differs from evaluation")
        require(metadata.get("models") == lock["models"], "Bootstrap method order mismatch")
        with np.load(out/f"bootstrap_{scheme}.npz", allow_pickle=False) as values:
            require(int(values["completed"]) == count and str(values["signature"].item()) == manifest["signature"], "Bootstrap checkpoint identity mismatch")
    acceptance = read_json(out/"acceptance.json", {})
    require(acceptance == manifest.get("acceptance"), "Acceptance file/manifest disagreement")
    contrasts = read_json(out/"comparisons.json", [])
    registered = {(x["candidate"], x["comparator"]) for x in lock.get("comparisons", [])}
    require({(x["candidate"], x["comparator"]) for x in contrasts} == registered and len(contrasts) == len(registered), "Registered comparison family incomplete")
    if contrasts:
        p = np.array([x["p_bootstrap_centered"] for x in contrasts])
        order = np.argsort(p, kind="stable")
        corrected = np.empty_like(p)
        corrected[order] = np.minimum(1., np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
        require(np.allclose(corrected, [x["p_holm"] for x in contrasts]), "Holm correction mismatch")
    candidate, baselines = lock.get("acceptance_candidate"), lock.get("acceptance_comparators", [])
    if candidate and baselines:
        models = lock["models"]
        point = bundle["macro"].set_index("model").auroc
        source = bundle["by_seed"][bundle["by_seed"].eligible].groupby(["source", "model"]).auroc.mean().unstack()
        delta = float(point[candidate]-point[baselines].max())
        source_delta = float((source[candidate]-source[baselines].max(axis=1)).min())
        with np.load(out/"bootstrap_pair.npz", allow_pickle=False) as checkpoint:
            draws = checkpoint["macro"].copy()
        diffs = draws[:, models.index(candidate), 0]-np.max(draws[:, [models.index(m) for m in baselines], 0], axis=1)
        finite = diffs[np.isfinite(diffs)]
        supported = len(finite) >= .95*len(diffs) and len(finite) > 0 and np.quantile(finite, .025) > 0
        combination = True
        for comparator in lock.get("combination_comparators", []):
            differences = draws[:, models.index(candidate), 0]-draws[:, models.index(comparator), 0]
            valid = differences[np.isfinite(differences)]
            combination &= bool(point[candidate] > point[comparator] and len(valid) and np.quantile(valid, .025) > 0)
        passed = bool(delta >= .02 and source_delta >= -.03 and supported and combination)
        require((acceptance.get("status") == "INCREMENT_CRITERION_MET") == passed, "Independent primary acceptance reconstruction disagrees")
        require(np.isclose(delta, acceptance["delta_auroc"]) and np.isclose(source_delta, acceptance["min_source_delta_vs_max_comparator"]), "Acceptance effect sizes disagree with independent metrics")
    return acceptance, contrasts


def secondary_assessment(bundle, contrasts):
    lock = bundle["lock"]
    source = bundle["by_seed"][bundle["by_seed"].eligible].groupby(["source", "model"]).auroc.mean().unstack()
    baselines = lock.get("acceptance_comparators", [])
    results = []
    for candidate in lock.get("secondary_candidates", []):
        chosen = [c for c in contrasts if c["candidate"] == candidate and c["comparator"] in baselines]
        complete = len(chosen) == len(baselines) and bool(chosen)
        minimum_source = float((source[candidate]-source[baselines].max(axis=1)).min()) if complete else np.nan
        points = bundle["macro"].set_index("model").auroc
        balanced_swa_difference = float(points[candidate]-points["B_bal_S"]) if "B_bal_S" in points else np.nan
        passed = bool(complete and minimum_source >= -.03 and np.isfinite(balanced_swa_difference) and balanced_swa_difference >= 0
                      and all(c["delta_auroc"] >= .02 and c.get("low") is not None and c["low"] > 0 and c["p_holm"] <= .05 and c.get("invalid_fraction", 1) <= .05 for c in chosen))
        results.append(dict(candidate=candidate, preregistered=True, family="all frozen contrasts, Holm corrected",
                            meets_secondary_rule=passed, minimum_source_delta=minimum_source,
                            delta_vs_balanced_swa=balanced_swa_difference,
                            interpretation="Secondary evidence only; no post-test winner declared as independent primary confirmation"))
    return results


def common_row_comparison(main, alternative, kind, common_ids=None):
    """Pair models/seeds on identical rows and the fixed main truth."""
    left, right = main["predictions"], alternative["predictions"]
    model_set = sorted(set(left.model) & set(right.model))
    if common_ids is None:
        common_ids = set(left.row_id) & set(right.row_id)
    main_truth = main["records"].set_index("row_id").y
    results = []
    for model in model_set:
        for seed in sorted(set(left.seed) & set(right.seed)):
            a = left[(left.model == model) & (left.seed == seed) & left.row_id.isin(common_ids)].set_index("row_id")
            b = right[(right.model == model) & (right.seed == seed) & right.row_id.isin(common_ids)].set_index("row_id")
            require(set(a.index) == set(b.index) == set(common_ids), "Sensitivity common-row coverage is not paired")
            b = b.loc[a.index]
            require(np.array_equal(a.source.to_numpy(), b.source.to_numpy()), "Sensitivity common-row source mismatch")
            for source, part in a.groupby("source"):
                ids = part.index
                y = main_truth.loc[ids].to_numpy()
                native = b.loc[ids].y_true.to_numpy()
                for regime, truth in (("fixed_primary_truth", y), ("native_sensitivity_truth", native)):
                    binary = len(np.unique(truth)) == 2
                    for metric, function in (("auroc", roc_auc_score), ("auprc", average_precision_score), ("brier", brier_score_loss)):
                        before = float(function(truth, part.score)) if binary or metric == "brier" else np.nan
                        after = float(function(truth, b.loc[ids].score)) if binary or metric == "brier" else np.nan
                        results.append(dict(kind=kind, source=source, model=model, seed=int(seed), truth_regime=regime,
                                            n_rows=len(ids), n_pairs=part.pair_id.nunique(), metric=metric,
                                            primary_model=before, sensitivity_model=after, paired_delta=after-before,
                                            changed_labels=int(np.sum(y != native))))
    return pd.DataFrame(results)


def domain_analysis(main, out):
    path = main["project"]/"reports/domain/row_covariates.parquet"
    if not path.exists():
        return dict(status="MISSING", reason="Domain covariates have not been produced")
    covariates = pd.read_parquet(path)
    require(not covariates.duplicated(["fold", "row_id"]).any(), "Duplicate domain row covariates")
    for role in ("aryl", "amine"):
        covariates[role+"_fixed_bin"] = pd.cut(covariates[role+"_max_train_tanimoto"], [0, .3, .5, .7, 1], include_lowest=True,
                                              labels=["[0,0.3]", "(0.3,0.5]", "(0.5,0.7]", "(0.7,1]"]).astype(str)
    fields = ["aryl_fixed_bin", "amine_fixed_bin", "aryl_has_heteroaromatic_ring", "aryl_bound_halogen_class", "amine_type", "reagent_coverage"]
    merged = main["predictions"].merge(covariates[["fold", "row_id", *fields]], on=["fold", "row_id"], validate="many_to_one", how="left")
    require(not merged[fields].isna().any().any(), "Missing frozen test covariates")
    rows = []
    for field in fields:
        for (source, model, seed, value), part in merged.groupby(["source", "model", "seed", field], observed=True):
            binary = part.y_true.nunique() == 2
            rows.append(dict(analysis="exploratory", stratum=field, value=str(value), source=source, model=model, seed=int(seed),
                             n_rows=len(part), n_pairs=part.pair_id.nunique(), n_sources=1, positive_rate=part.y_true.mean(),
                             auroc=roc_auc_score(part.y_true, part.score) if binary else np.nan,
                             auprc=average_precision_score(part.y_true, part.score) if binary else np.nan,
                             brier=brier_score_loss(part.y_true, part.score)))
    pd.DataFrame(rows).to_csv(out/"exploratory_domain_strata.csv", index=False)
    summary = covariates.groupby("source").agg(n_rows=("row_id", "size"), n_pairs=("pair_id", "nunique"),
        n_aryls=("aryl_id", "nunique"), n_amines=("amine_id", "nunique"),
        aryl_similarity_mean=("aryl_max_train_tanimoto", "mean"), amine_similarity_mean=("amine_max_train_tanimoto", "mean"),
        catalyst_exact_coverage=("catalyst_seen_in_train", "mean"), base_exact_coverage=("base_seen_in_train", "mean"),
        solvent_exact_coverage=("solvent_seen_in_train", "mean"), positive_rate=("y_true", "mean")).reset_index()
    summary.to_csv(out/"domain_coverage_summary.csv", index=False)
    return dict(status="COMPLETE", summary=summary.to_dict("records"),
                interpretation="Exploratory descriptive strata; exact chemical forms preserve charge/salts/coordination, not catalyst-class novelty; no mechanism claims")


def scientific_figures(bundle, intervals, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    macro = bundle["macro"].set_index("model").loc[bundle["lock"]["models"]]
    ci = intervals[(intervals.metric == "auroc") & (intervals.scheme == "pair")].set_index("model").loc[macro.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(macro))
    ax.scatter(macro.auroc, y, color="#1768ac")
    for i, model in enumerate(macro.index):
        ax.plot([ci.loc[model, "low"], ci.loc[model, "high"]], [i, i], color="#1768ac")
    ax.set(yticks=y, yticklabels=macro.index, xlabel="Equal-source macro AUROC (95% pair bootstrap interval)", xlim=(0, 1))
    ax.invert_yaxis()
    ax.axvline(.5, color="gray", linestyle="--", linewidth=.7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"macro_auroc.{suffix}", dpi=200)
    plt.close(fig)
    source = bundle["by_seed"][bundle["by_seed"].eligible].groupby(["source", "model"]).auroc.mean().unstack().reindex(columns=bundle["lock"]["models"])
    fig, ax = plt.subplots(figsize=(11, max(3, len(source)*.7)))
    heat = ax.imshow(source.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=np.arange(len(source.columns)), xticklabels=source.columns, yticks=np.arange(len(source)), yticklabels=source.index)
    for i in range(len(source)):
        for j in range(len(source.columns)):
            ax.text(j, i, f"{source.iloc[i,j]:.3f}", ha="center", va="center", color="white" if source.iloc[i,j] < .65 else "black", fontsize=8)
    fig.colorbar(heat, ax=ax, label="Mean AUROC across seeds")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"source_auroc_heatmap.{suffix}", dpi=200)
    plt.close(fig)


def ledger_inventory(project, out):
    path = project/"runs/ledger.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    attempts = {}
    for entry in entries:
        key = entry.get("attempt_id")
        if not key:
            continue
        previous = attempts.get(key, {})
        attempts[key] = dict(previous, **entry)
        for metric in ("cpu_seconds", "gpu_seconds"):
            attempts[key][metric] = max(previous.get(metric, 0), entry.get(metric, 0))
    rows = []
    for key, entry in attempts.items():
        rows.append(dict(attempt_id=key, run_id=entry.get("run_id"), stage=entry.get("stage"), status=entry.get("status"),
                         exit_code=entry.get("exit_code"), gpu_hours=entry.get("gpu_seconds", 0)/3600,
                         cpu_core_hours=entry.get("cpu_seconds", 0)/3600, cost_quality=entry.get("cost_quality"),
                         command=" ".join(entry.get("command", [])), log=entry.get("log"), reason=entry.get("reason")))
    table = pd.DataFrame(rows)
    table.to_csv(out/"run_inventory.csv", index=False)
    saved = read_json(project/"runs/budget_totals.json", {})
    calculated = dict(gpu_hours=sum(x["gpu_hours"] for x in rows), cpu_core_hours=sum(x["cpu_core_hours"] for x in rows))
    budget = dict(saved)
    for key, value in calculated.items():
        budget[key] = max(float(saved.get(key, 0)), value)
    return budget, table


def choose_status(acceptance, core_complete, audit_passed, sensitivity_states, budget, secondary=None):
    remaining = budget.get("gpu_hours", 0) < 160 and budget.get("cpu_core_hours", 0) < 500
    pending = [s for s in sensitivity_states.values() if s not in ("COMPLETE", "INCOMPLETE_BUDGET", "NOT_STARTED_BUDGET")]
    if not audit_passed or not core_complete or pending:
        return "CONTINUE" if remaining else "NOT_READY"
    if any(s != "COMPLETE" for s in sensitivity_states.values()):
        return "NOT_READY"
    if acceptance.get("status") == "INCREMENT_CRITERION_MET" or any(s.get("meets_secondary_rule") for s in (secondary or [])):
        return "READY_FOR_AUTHOR_REVIEW"
    return "NOT_READY"


def markdown_table(frame):
    if frame.empty:
        return "无可评估记录。"
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"]*len(frame.columns)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def update_state(project, status, reason, audit=None):
    state = read_json(project/"STATE.json", {})
    state.update(status=status, phase="experiment_report" if status != "CONTINUE" else "awaiting_remaining_experiments_or_review",
                 finalization_reason=reason, last_report_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    if audit:
        state["independent_audit"] = audit
        state["outer_metrics_released"] = audit.get("status") == "PASS"
    dump(project/"STATE.json", state)


def finalize(project):
    project = Path(project).resolve()
    bundle = audit_project(project)
    training_audit = {"status": "NOT_APPLICABLE", "reason": "Synthetic fixture without a neural training protocol"}
    if "neural_base_config" in bundle["lock"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("saved_training_auditor", project/"scripts/audit_training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        training_audit = module.audit_project(project)
    out = project/"reports/final"
    acceptance, contrasts = validate_statistics(bundle)
    secondary = secondary_assessment(bundle, contrasts)
    bundle["by_seed"].to_csv(out/"independent_metrics_by_source_seed.csv", index=False)
    bundle["macro"].to_csv(out/"independent_main_table.csv", index=False)
    dump(out/"independent_audit.json", bundle["audit"])
    dump(out/"secondary_assessment.json", secondary)
    sensitivity_states, sensitivity_text = {}, []
    for kind in ("author_labels", "dd_original"):
        alternative_root = project/"sensitivity"/kind
        try:
            alternative = audit_project(alternative_root)
            plan = bundle["lock"].get("sensitivity_plan", {})
            require(alternative["lock"].get("main_lock_sha256") == sha256(project/"FINAL_EVAL_LOCK.json"), "Sensitivity was not bound to this main lock")
            require(set(alternative["lock"]["models"]) == set(plan.get("models", [])) and set(alternative["lock"]["seeds"]) == set(plan.get("seeds", [])), "Sensitivity matrix differs from the preregistered model/seed plan")
            validate_statistics(alternative)
            dump(alternative_root/"reports/final/independent_audit.json", alternative["audit"])
            alternative["macro"].to_csv(out/f"{kind}_native_main_table.csv", index=False)
            common = None
            if kind == "dd_original":
                path = project/"reports/domain/dd_common_test_ids.json"
                require(path.exists(), "DD common-row manifest is missing")
                manifest = read_json(path)
                require(manifest["main_splits_sha256"] == sha256(project/"data/processed/splits.json") and manifest["dd_splits_sha256"] == sha256(alternative_root/"data/processed/splits.json"), "DD common-row manifest uses different splits")
                common = {int(row) for part in manifest["folds"].values() for row in part["common_row_ids"]}
                expected = set(bundle["predictions"].row_id) & set(alternative["predictions"].row_id)
                require(common == expected, "DD registered common-row intersection mismatch")
            comparison = common_row_comparison(bundle, alternative, kind, common)
            comparison.to_csv(out/f"{kind}_paired_common_rows.csv", index=False)
            sensitivity_states[kind] = "COMPLETE"
            compact = comparison.groupby(["model", "truth_regime", "metric"]).paired_delta.agg(lambda values: values.mean(skipna=False)).unstack("metric").reset_index()
            sensitivity_text.extend([f"### {kind}", "", markdown_table(alternative["macro"]), "", "共同测试行的来源等权、seed平均点差（敏感性减主模型；原始概率；来源集合由共同测试行确定，不替代主宏指标；任一来源单类时保留NA而不省略该来源）：", "", markdown_table(compact), ""])
        except PendingEvaluation:
            previous = read_json(alternative_root/"STATE.json", {})
            state = previous.get("status", "PENDING")
            sensitivity_states[kind] = state if state in ("INCOMPLETE_BUDGET", "NOT_STARTED_BUDGET") else "PENDING"
            sensitivity_text.extend([f"### {kind}", "", f"状态：{sensitivity_states[kind]}。未读取未完成的预测，也不将部分矩阵作为完成证据。", ""])
    budget, inventory = ledger_inventory(project, out)
    domain = domain_analysis(bundle, out)
    dump(out/"domain_analysis.json", domain)
    intervals = pd.read_csv(out/"intervals.csv")
    scientific_figures(bundle, intervals, out)
    core_complete = CORE_MODELS <= set(bundle["lock"]["models"]) and len(bundle["lock"]["seeds"]) >= 3 and domain["status"] == "COMPLETE" and training_audit['status'] in ('PASS','NOT_APPLICABLE')
    status = choose_status(acceptance, core_complete, True, sensitivity_states, budget, secondary)
    evidence_basis = ("primary_confirmed" if acceptance.get("status") == "INCREMENT_CRITERION_MET"
                      else "preregistered_secondary_holm" if any(s["meets_secondary_rule"] for s in secondary) else "no_increment_criterion_met")
    main = bundle["macro"].merge(intervals[(intervals.metric == "auroc") & (intervals.scheme == "pair")][["model", "low", "high", "invalid_fraction"]], on="model", validate="one_to_one")
    per_source = bundle["by_seed"].groupby(["source", "model", "eligible"])[["auroc", "auprc", "brier", "cal_brier", "positive_rate", "n_rows", "n_pairs"]].mean().reset_index()
    spread = bundle["by_seed"].groupby(["source", "model"]).auroc.std().rename("seed_auroc_std").reset_index()
    per_source = per_source.merge(spread, on=["source", "model"], validate="one_to_one")
    source_intervals = pd.read_csv(out/"source_auroc_intervals.csv")
    per_source = per_source.merge(source_intervals[["source", "model", "low", "high", "invalid_fraction"]], on=["source", "model"], how="left", validate="one_to_one")
    failures = inventory[inventory.status.isin(["failed", "interrupted"])] if len(inventory) else inventory
    tokenizer = read_json(project/"weights/rxnfp/tokenizer_extension_manifest.json", {})
    data_audit = read_json(project/"data/audit/summary.json", {})
    lines = ["# 跨来源 C–N 偶联可行性预测：实验报告", "", "本文件是计算实验记录与证据核验，不是论文；未投稿、不承诺录用。", "",
             f"当前状态：**{status}**。主候选固定为 `{bundle['lock'].get('acceptance_candidate')}`；接受规则结果 `{acceptance.get('status')}`；证据类别 `{evidence_basis}`。", "",
             *(["这是阶段报告；仍有实验或核验待完成。", ""] if status == "CONTINUE" else []),
             "若仅secondary达标，明确表示主B_A_S确认未成功、预注册简化方案获得多重校正支持；这不改写主假设，不从外层结果追认新的独立确认。简化方案还必须在宏AUROC上不劣于B_bal_S。", "",
             "## 数据、问题与固定协议", "", "目标为未见来源且芳基、胺各自均未见时，判断记录的 Yield 是否至少为10；恰好10为正例。不是精确产率、高产率或未实测条件推荐的验证。",
             f"原始行数 `{data_audit.get('raw_rows', '见DATA_AUDIT')}`；主数据行数 `{len(bundle['records'])}`。合格主来源数 `{sum(f['eligible'] for f in bundle['lock']['folds'])}`；所有来源统计与逐行处置见 [DATA_AUDIT](../DATA_AUDIT.md)。", "",
             f"机器可核验冻结记录见 [FINAL_EVAL_LOCK](../FINAL_EVAL_LOCK.json)，SHA256 `{sha256(project/'FINAL_EVAL_LOCK.json')}`。EXPERIMENT_PROTOCOL.md保留冻结前的working标题；实际冻结状态、训练预算、候选方法、seed及内层选择规则由该只读锁和其绑定制品给出，外层评价按该锁执行。", "",
             markdown_table(pd.DataFrame(bundle["split_summary"])), "",
             "外层LOSO的整个开发集包含验证/校准，测试两类底物与其实体集合均无交集；同对条件和重复实验不随机跨边界。独立审计从冻结records与splits重新核验，而非复用训练检查函数。",
             "原始存储指纹不可解析，已重算。R0为不使用产品的DRFP；R1_substitute为作者启发的底物Morgan表示＋公开RDKit21试剂描述符，是声明的强替代对照，未准确重建作者底物–产品DRFP＋原QM生产模型，不能声称超过原作模型。全部神经方法使用同样的RDKit21输入。", "",
             f"Tokenizer显式调整：`{json.dumps(tokenizer, ensure_ascii=False)}`。两个配位箭头token扩充原词表，原ID保留，新embedding复制原减号embedding；所有神经对照一致。无静默截断或删除关键试剂。", "",
             "## 主表、逐来源结果与消融", "", "每个seed独立计算AUROC/AP/Brier，再对seed平均，最后对合格来源等权平均。AP使用非插值average precision；不同来源不混合计算主AUC。cal_brier是内部温度校准后的辅助概率误差，不能解释为排序改善。", "",
             markdown_table(main), "", "![主AUROC及配对区间](final/macro_auroc.png)", "", "![来源AUROC热图](final/source_auroc_heatmap.png)", "",
             markdown_table(per_source), "", "固定比较与消融（所有预注册对比统一Holm校正）：", "", markdown_table(pd.DataFrame(contrasts)), "",
             "预注册简化方案逐一判定，不挑选外层赢家后冒充新的独立确认：", "", markdown_table(pd.DataFrame(secondary)), "",
             "## 不确定性、校准与覆盖", "", markdown_table(intervals), "",
             "主区间使用来源内底物对分组配对bootstrap，所有方法和seed共享抽样；seed不是独立化学实验。双向芳基/胺pigeonhole为共享底物敏感性。只有4个主来源，区间条件于这些来源及已拟合轨迹，不能覆盖所有新实验室或训练集不确定性。Merck严格测试虽含许多底物对，却仅含8个胺实体，双向bootstrap不能据此创造更丰富的独立化学覆盖。", "",
             "逐来源原始/校准可靠性数据与图见final/reliability_by_source_seed.csv及reliability_source_*.png。Top-k仅来自同对多个实测条件，重复条件对整体排除并报告覆盖；见final/topk_measured_conditions.csv或topk_status.json。", "",
             "结构相似度固定bins为[0,.3],(.3,.5],(.5,.7],(.7,1]；粗类型/试剂覆盖结果均为探索性描述，不进行选择性分层机理宣称：", "",
             markdown_table(pd.DataFrame(domain.get("summary", []))), "",
             "精确试剂覆盖保留电荷、盐和配位形态。AZ/Merck/Roche催化剂精确结构训练覆盖可能为0，JNJ约1.95%；这不等于催化剂化学类别全新。需同时考虑底物新颖性、相似度、试剂形态覆盖和来源正类率。全部分层含行/底物对/来源覆盖见final/exploratory_domain_strata.csv。", "",
             *(["主实验逐来源退步、消融、校准和实测条件覆盖的解释见 [主实验失败分析](PRIMARY_FAILURE_ANALYSIS.md)。", ""] if (project/"reports/PRIMARY_FAILURE_ANALYSIS.md").exists() else []),
             "## 预注册敏感性分析", "", *sensitivity_text,
             "Dreher Doyle原版本缺少具体添加剂身份，仅作为条件不完整敏感性，不纳入主结论。其加入会改变严格测试集合；共同row_id配对结果使用固定主真值，不能把集合变化解释成训练改善。作者标签敏感性同时报告原生标签与固定主真值，标签标准变化和模型变化不能混为一谈。", "",
             "## 运行成本、失败与独立核验", "", "实际训练审计：`"+json.dumps(training_audit,ensure_ascii=False)+"`。逐轨迹更新数、样本曝光、SWA快照数及GroupDRO权重/有效样本量见final/training_budget_audit.csv与final/groupdro_trajectories.csv；权重图见final/groupdro_weight_trajectories.png。权重集中仅描述优化行为，不能证明未见来源泛化或反应机理。", "", "预算记录：`" + json.dumps(budget, ensure_ascii=False) + "`。失败与重试累计；upper_bound标记保守收费，包括结束时刻不明的孤儿任务及协调器自身CPU用量预留；子训练任务单独计费。完整成功/失败账本见final/run_inventory.csv及../runs/ledger.jsonl。报告生成任务若仍在运行，其最终小额成本以随后完成的guard账本为准。", "",
             ("账本无失败或中断的任务记录；环境下载中断见 [DECISIONS](../DECISIONS.md)。" if failures.empty else
              markdown_table(failures[[c for c in ("run_id", "stage", "status", "exit_code", "gpu_hours", "cpu_core_hours", "reason") if c in failures]])), "",
             "CPU对照实际拟合证据：[30次主内层拟合审计](ALL_TREE_INNER_RUNTIME_AUDIT.json)、[45次主正式拟合审计](ALL_FORMAL_TREE_RUNTIME_AUDIT.json)、[30次敏感性RF拟合审计](ALL_SENSITIVITY_TREE_RUNTIME_AUDIT.json)。具体检查范围、训练分区、固定参数和已保存文件的核验记录见各审计报告；审计过程未读取外层预测或指标。", "",
             "独立审计PASS：逐条标签/来源/底物实体/底物对/条件ID与冻结records一致；锁定模型×seed×行矩阵完整；sklearn独立重建主表及逐来源逐seed指标，与原评价表在1e-10容差内一致。审计清单和源码hash见final/independent_audit.json。", "",
             "## 局限、文献差异与复现", "", "CAD、isolated yield、MisER与LCAP混合测量，统一10%阈值不消除测量差异；温度、时间、浓度与部分载量缺失。无标签预训练与测试结构的重叠不能排除。来源均衡可能解释部分收益，因此保留B_bal及B_bal_S。GroupDRO/SWA为成熟方法，注意力或特征归因不能证明机理。负结果如实交付，未达到门槛不强行宣称增量。", "",
             "原文与直接近例已包含多来源或双底物OOD研究，差异来自受控实验而非首创这些任务。详见 [文献差异](LITERATURE_DIFFERENCES.md) 与 [期刊范围及subscription非OA路径](JOURNAL_SCOPE.md)。Reaction Chemistry & Engineering与Chemical Papers是否接受由编辑和评审判断，本项目不投稿。", "",
             "```bash", "# 已冻结项目；先恢复/完成协调器，避免重复启动已运行任务", ".venv/bin/python scripts/pipeline.py --phase all",
             ".venv/bin/python -m bh_transfer.evaluate --predictions artifacts/predictions.parquet --lock FINAL_EVAL_LOCK.json --out reports/final",
             ".venv/bin/python scripts/sensitivity_pipeline.py --kind all --phase all",
             "# 每个完整敏感性目录使用其自身predictions/lock运行同一evaluate模块",
             ".venv/bin/python scripts/finalize_report.py --project .", "```", "",
             "请使用scripts/run_guard.py为实际CPU/GPU任务登记run_id、线程槽、成本与日志；上列命令展示入口，不授权并行GPU运行或改动冻结配置。"]
    (project/"reports/EXPERIMENT_REPORT.md").write_text("\n".join(lines)+"\n")
    dump(out/"finalization.json", dict(status=status, independent_audit=bundle["audit"], sensitivity_states=sensitivity_states,
                                      core_complete=core_complete, acceptance=acceptance, secondary=secondary, evidence_basis=evidence_basis, budget=budget))
    update_state(project, status, "Independent reconstruction and report complete; status reflects frozen acceptance and required experiment coverage", bundle["audit"])
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    try:
        status = finalize(project)
        print(json.dumps(dict(status=status, report=str(project/"reports/EXPERIMENT_REPORT.md")), ensure_ascii=False))
    except PendingEvaluation as exc:
        update_state(project, "CONTINUE", str(exc))
        print(json.dumps(dict(status="CONTINUE", reason=str(exc), predictions_read=False), ensure_ascii=False))
        raise SystemExit(2)
    except Exception as exc:
        failure = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}", auditor_sha256=sha256(__file__))
        dump(project/"reports/final/independent_audit.json", failure)
        budget = read_json(project/"runs/budget_totals.json", {})
        status = "NOT_READY" if budget.get("gpu_hours", 0) >= 160 or budget.get("cpu_core_hours", 0) >= 500 else "CONTINUE"
        update_state(project, status, "Independent audit failed; READY is forbidden until resolved", failure)
        raise


if __name__ == "__main__":
    main()
