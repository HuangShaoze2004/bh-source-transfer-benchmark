"""Locked evaluation of saved, complete LOSO predictions.

All chemistry uncertainty is clustered; seeds are averaged, never resampled as
independent reactions. No prediction metrics are computed before lock validation.
The bootstrap conditions on the observed sources and fitted training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = ("auroc", "auprc", "brier")
REQUIRED = {"row_id", "source", "pair_id", "aryl_id", "amine_id", "y_true", "model", "seed", "fold", "score"}


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_folds(lock: dict, lock_path: Path) -> list[dict]:
    folds = lock.get("folds")
    if folds is None:
        path = Path(lock["splits_path"])
        if not path.is_absolute():
            path = lock_path.parent / path
        splits = json.loads(path.read_text())
        folds = splits.get("folds", splits)
    if isinstance(folds, dict):
        folds = [dict(value, fold=value.get("fold", key)) for key, value in folds.items()]
    if not isinstance(folds, list) or not folds:
        raise ValueError("Lock must specify a nonempty folds manifest")
    for fold in folds:
        if not {"fold", "source", "eligible", "test_row_ids"}.issubset(fold):
            raise ValueError("Each fold needs fold/source/eligible/test_row_ids")
    return folds


def validate_predictions(frame: pd.DataFrame, lock: dict, lock_path: Path = Path("FINAL_EVAL_LOCK.json")) -> tuple[pd.DataFrame, list[dict]]:
    """Only integrity checks, no outcome-performance summaries."""
    if lock.get("status") != "FROZEN" or lock.get("evaluation_allowed") is not True:
        raise ValueError("Evaluation is sealed: require FROZEN lock and evaluation_allowed=true")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Required prediction metadata/scores contain missing values")
    frame = frame.copy()
    if not pd.api.types.is_integer_dtype(frame.row_id) or not pd.api.types.is_integer_dtype(frame.seed):
        raise ValueError("row_id and seed must be integers")
    frame["fold"] = frame.fold.astype(str)
    for key in ("source", "pair_id", "aryl_id", "amine_id", "model"):
        frame[key] = frame[key].astype(str)
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError("y_true must be binary")
    if not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        raise ValueError("Scores must be finite")
    score_type = lock.get("score_type", "probability")
    if score_type == "logit":
        x = frame.score.to_numpy(dtype=float)
        frame["score"] = np.exp(-np.logaddexp(0.0, -x))
    elif score_type != "probability":
        raise ValueError("score_type must be probability or logit")
    if not frame.score.between(0, 1).all():
        raise ValueError("Probability scores outside [0,1]")
    if "cal_score" in frame and (frame.cal_score.isna().any() or not frame.cal_score.between(0, 1).all()):
        raise ValueError("If supplied, cal_score must be complete probabilities")
    models, seeds = lock.get("models"), lock.get("seeds")
    if not models or not seeds or len(models) != len(set(models)) or len(seeds) != len(set(seeds)):
        raise ValueError("Lock must specify unique nonempty models and seeds")
    if len(seeds) < int(lock.get("minimum_seeds", 3)):
        raise ValueError("Fewer than the predeclared minimum model seeds")
    if set(frame.model) != set(models) or set(frame.seed) != set(seeds):
        raise ValueError("Observed model/seed set differs from evaluation lock")
    if frame.duplicated(["fold", "row_id", "model", "seed"]).any():
        raise ValueError("Duplicate predictions for fold/row/model/seed")
    # An identifier must carry identical chemistry and ground truth everywhere.
    identity_columns = ["source", "pair_id", "aryl_id", "amine_id", "y_true", "fold"]
    if "condition_id" in frame:
        if frame.condition_id.isna().any():
            raise ValueError("If supplied, condition_id must be complete")
        frame["condition_id"] = frame.condition_id.astype(str)
        identity_columns.append("condition_id")
    identities = frame.groupby("row_id")[identity_columns].nunique()
    if (identities > 1).any().any():
        raise ValueError("Inconsistent row metadata across methods or seeds")
    pairs = frame.groupby("pair_id")[["aryl_id", "amine_id"]].nunique()
    if (pairs > 1).any().any():
        raise ValueError("pair_id maps to inconsistent substrate entities")
    folds = load_folds(lock, lock_path)
    if len({str(f["fold"]) for f in folds}) != len(folds):
        raise ValueError("Repeated fold IDs in lock")
    nonempty_folds = {str(f["fold"]) for f in folds if f["test_row_ids"]}
    if set(frame.fold) != nonempty_folds:
        raise ValueError("Prediction folds differ from locked nonempty folds")
    for fold in folds:
        expected = set(fold["test_row_ids"])
        if len(expected) != len(fold["test_row_ids"]):
            raise ValueError("Duplicate locked test row IDs")
        part = frame[frame.fold == str(fold["fold"])]
        if set(part.source) != ({str(fold["source"])} if expected else set()):
            raise ValueError("Prediction source differs from locked fold source")
        for model in models:
            for seed in seeds:
                actual = set(part.loc[(part.model == model) & (part.seed == seed), "row_id"])
                if actual != expected:
                    raise ValueError(f"Incomplete row coverage: {fold['fold']}/{model}/{seed}; missing={len(expected-actual)}, extra={len(actual-expected)}")
        if fold["eligible"]:
            base = part.drop_duplicates("row_id")
            if base.empty or base.y_true.nunique() != 2:
                raise ValueError("Eligible fold is empty or single-class")
    eligible_sources = [str(f["source"]) for f in folds if f["eligible"]]
    if len(set(eligible_sources)) != len(eligible_sources):
        raise ValueError("Eligible sources must be distinct")
    return frame, folds


class WeightedMetrics:
    """Pre-sort once, then evaluate exact weighted rank metrics including ties."""

    def __init__(self, y: np.ndarray, score: np.ndarray):
        self.y = np.asarray(y, dtype=float)
        self.score = np.asarray(score, dtype=float)
        _, self.group = np.unique(self.score, return_inverse=True)
        self.n_groups = int(self.group.max()) + 1
        self.squared_error = (self.score - self.y) ** 2

    def __call__(self, weight: np.ndarray | None = None) -> np.ndarray:
        w = np.ones_like(self.y) if weight is None else np.asarray(weight, dtype=float)
        if w.shape != self.y.shape or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("Invalid metric weights")
        p = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        n = np.bincount(self.group, weights=w * (1 - self.y), minlength=self.n_groups)
        total_p, total_n = p.sum(), n.sum()
        if total_p == 0 or total_n == 0:
            return np.array([np.nan, np.nan, np.dot(w, self.squared_error) / w.sum() if w.sum() else np.nan])
        auc = np.dot(p, np.cumsum(n) - 0.5 * n) / (total_p * total_n)
        tp, denom = np.cumsum(p[::-1]), np.cumsum((p + n)[::-1])
        precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        ap = np.dot(p[::-1], precision) / total_p
        return np.array([auc, ap, np.dot(w, self.squared_error) / w.sum()])


def metric_tables(frame: pd.DataFrame, folds: list[dict], models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eligible = {str(f["source"]) for f in folds if f["eligible"]}
    for (source, model, seed), p in frame.groupby(["source", "model", "seed"], sort=True):
        values = WeightedMetrics(p.y_true.to_numpy(), p.score.to_numpy())()
        record = dict(source=source, model=model, seed=int(seed), eligible=source in eligible,
                      n_rows=len(p), n_pairs=p.pair_id.nunique(), positive_rate=float(p.y_true.mean()), **dict(zip(METRICS, values)))
        if "cal_score" in p:
            record["cal_brier"] = float(np.mean((p.cal_score-p.y_true)**2))
        rows.append(record)
    per_seed = pd.DataFrame(rows)
    summary = []
    for model in models:
        per_source = per_seed[(per_seed.model == model) & per_seed.eligible].groupby("source")[list(METRICS)].mean()
        summary.append(dict(model=model, n_sources=len(per_source), **per_source.mean().to_dict(), worst_source_auroc=float(per_source.auroc.min())))
    return per_seed, pd.DataFrame(summary)


def prepare_bootstrap(frame: pd.DataFrame, folds: list[dict], models: list[str], seeds: list[int]) -> list[dict]:
    result = []
    for fold in folds:
        if not fold["eligible"]:
            continue
        part = frame[frame.fold == str(fold["fold"])]
        base = part[(part.model == models[0]) & (part.seed == seeds[0])].sort_values("row_id")
        groups = {}
        for col in ("pair_id", "aryl_id", "amine_id"):
            codes, labels = pd.factorize(base[col], sort=True)
            groups[col] = (codes, len(labels))
        evaluators = []
        for model in models:
            evaluators.append([WeightedMetrics(p.y_true.to_numpy(), p.score.to_numpy()) for seed in seeds
                               for p in [part[(part.model == model) & (part.seed == seed)].sort_values("row_id")]])
        result.append(dict(source=str(fold["source"]), groups=groups, evaluators=evaluators))
    return result


def bootstrap(prepared: list[dict], models: list[str], n_replicates: int, seed: int, scheme: str,
              output: Path, signature: str, checkpoint_every: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Resume deterministic replicate-index RNG; no resampling of model seeds."""
    if scheme not in ("pair", "pigeonhole") or n_replicates <= 0:
        raise ValueError("Invalid bootstrap scheme or count")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / f"bootstrap_{scheme}.npz"
    metadata_path = output / f"bootstrap_{scheme}.json"
    macro = np.full((n_replicates, len(models), len(METRICS)), np.nan)
    source_auc = np.full((n_replicates, len(prepared), len(models)), np.nan)
    start = 0
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            if str(state["signature"].item()) != signature or int(state["seed"]) != seed or str(state["scheme"].item()) != scheme:
                raise ValueError("Bootstrap checkpoint belongs to different inputs/configuration")
            start = int(state["completed"])
            if state["macro"].shape != macro.shape or state["source_auc"].shape != source_auc.shape:
                raise ValueError("Bootstrap checkpoint dimensions differ; use a new output path")
            macro[:] = state["macro"]
            source_auc[:] = state["source_auc"]
    def save(completed: int) -> None:
        tmp = state_path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, macro=macro, source_auc=source_auc, completed=np.array(completed),
                                signature=np.array(signature), seed=np.array(seed), scheme=np.array(scheme))
        os.replace(tmp, state_path)
        atomic_json(metadata_path, dict(signature=signature, scheme=scheme, completed=completed, requested=n_replicates,
                                       bootstrap_seed=seed, models=models, sources=[p["source"] for p in prepared],
                                       rng="SeedSequence([seed, scheme_index, replicate_index, source_index])"))
    for r in range(start, n_replicates):
        all_sources = []
        for i, part in enumerate(prepared):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if scheme == "pair" else 1, r, i]))
            cols = ("pair_id",) if scheme == "pair" else ("aryl_id", "amine_id")
            weight = None
            for col in cols:
                codes, count = part["groups"][col]
                draws = rng.multinomial(count, np.full(count, 1.0 / count))
                w = draws[codes].astype(float)
                weight = w if weight is None else weight * w
            # The exact same chemistry weights are used for every method/seed.
            values = np.array([np.mean([ev(weight) for ev in group], axis=0) for group in part["evaluators"]])
            source_auc[r, i] = values[:, 0]
            all_sources.append(values)
        # np.mean deliberately propagates undefined sources; never omit a source.
        macro[r] = np.mean(all_sources, axis=0)
        if (r + 1) % checkpoint_every == 0 or r + 1 == n_replicates:
            save(r + 1)
            print(f"{scheme} bootstrap {r + 1}/{n_replicates}", flush=True)
    return macro, source_auc


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm requires finite probabilities")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def interval(values: np.ndarray) -> dict:
    valid = np.asarray(values)[np.isfinite(values)]
    return dict(low=float(np.quantile(valid, .025)) if len(valid) else None,
                high=float(np.quantile(valid, .975)) if len(valid) else None,
                valid_replicates=len(valid), invalid_fraction=1-len(valid)/len(values))


def comparison_table(macro: pd.DataFrame, bootstrap_values: np.ndarray, models: list[str], comparisons: list[dict]) -> list[dict]:
    observed = macro.set_index("model").auroc.to_dict()
    rows = []
    for contrast in comparisons:
        candidate, comparator = contrast["candidate"], contrast["comparator"]
        if candidate not in models or comparator not in models:
            raise ValueError("Unknown model in locked comparison")
        delta = observed[candidate] - observed[comparator]
        draws = bootstrap_values[:, models.index(candidate), 0] - bootstrap_values[:, models.index(comparator), 0]
        valid = draws[np.isfinite(draws)]
        p = float((1 + np.sum(np.abs(valid-delta) >= abs(delta))) / (len(valid)+1)) if len(valid) else 1.0
        rows.append(dict(candidate=candidate, comparator=comparator, delta_auroc=float(delta), **interval(draws),
                         p_bootstrap_centered=p, p_method="two-sided centered paired bootstrap; approximate"))
    for row, adjusted in zip(rows, holm_adjust([r["p_bootstrap_centered"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def acceptance_report(per_seed: pd.DataFrame, macro: pd.DataFrame, draws: np.ndarray, models: list[str], lock: dict) -> dict:
    candidate = lock.get("acceptance_candidate")
    comparators = lock.get("acceptance_comparators", [])
    if not comparators and lock.get("acceptance_comparator"):
        comparators = [lock["acceptance_comparator"]]
    if not candidate or not comparators:
        return dict(status="NOT_ASSESSED", reason="No primary candidate/comparator set frozen in lock")
    if candidate not in models or any(c not in models or c == candidate for c in comparators):
        raise ValueError("Invalid frozen acceptance candidate/comparators")
    observed = macro.set_index("model").auroc.to_dict()
    strongest = max(comparators, key=lambda c: observed[c])
    delta = observed[candidate]-observed[strongest]
    diff = draws[:, models.index(candidate), 0]-np.max(draws[:, [models.index(c) for c in comparators], 0], axis=1)
    ci = interval(diff)
    source = per_seed[per_seed.eligible].groupby(["source", "model"]).auroc.mean().unstack()
    # Guard against losing >0.03 against ANY registered comparator on any source.
    source_differences = source[candidate]-source[comparators].max(axis=1)
    minimum = float(source_differences.min())
    combo_comparators = lock.get("combination_comparators", ["B_A", "B_S", "B_bal_S"] if candidate == "B_A_S" else [])
    combo_results = []
    for comparator in combo_comparators:
        if comparator not in models:
            raise ValueError("Missing combination comparator in locked matrix")
        d = draws[:, models.index(candidate), 0]-draws[:, models.index(comparator), 0]
        combo_ci = interval(d)
        combo_results.append(dict(comparator=comparator, delta=float(observed[candidate]-observed[comparator]), **combo_ci))
    passed = bool(delta >= .02 and ci["low"] is not None and ci["low"] > 0 and ci["invalid_fraction"] <= .05
                  and minimum >= -.03 and all(r["delta"] > 0 and r["low"] is not None and r["low"] > 0 for r in combo_results))
    return dict(status="INCREMENT_CRITERION_MET" if passed else "INCREMENT_CRITERION_NOT_MET", candidate=candidate,
                comparator_set=comparators, strongest_observed_comparator=strongest, delta_auroc=float(delta),
                bootstrap_vs_max_comparator=ci, min_source_delta_vs_max_comparator=minimum,
                source_deltas=source_differences.to_dict(), combination_contrasts=combo_results,
                criteria=dict(min_macro_delta=.02, max_source_regression=.03, ci_low_above=0, max_invalid_bootstrap_fraction=.05),
                limitation="Evidence conditional on these sources and fitted seeds; not a publication-readiness verdict")


def reliability_tables(frame: pd.DataFrame, out: Path, bins: int = 10, plots: bool = True) -> pd.DataFrame:
    records = []
    for score_col in [c for c in ("score", "cal_score") if c in frame]:
        for (source, model, seed), part in frame.groupby(["source", "model", "seed"]):
            bin_index = np.minimum((part[score_col].to_numpy()*bins).astype(int), bins-1)
            for b in range(bins):
                chosen = part[bin_index == b]
                records.append(dict(source=source, model=model, seed=int(seed), score_type=score_col, bin=b,
                                    lower=b/bins, upper=(b+1)/bins, n=len(chosen),
                                    mean_probability=float(chosen[score_col].mean()), observed_rate=float(chosen.y_true.mean())))
    table = pd.DataFrame(records)
    table.to_csv(out / "reliability_by_source_seed.csv", index=False)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for number, (source, part) in enumerate(table.groupby("source")):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], "k--", linewidth=.8)
            for (model, score_type), p in part.groupby(["model", "score_type"]):
                # Each seed contributes equally; these are plot summaries only.
                points = p[p.n > 0].groupby("bin")[["mean_probability", "observed_rate"]].mean()
                ax.plot(points.mean_probability, points.observed_rate, marker=".", label=f"{model}/{score_type}")
            ax.set(xlabel="Predicted probability", ylabel="Observed positive fraction", title=str(source), xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=5, loc="best")
            fig.tight_layout()
            fig.savefig(out / f"reliability_source_{number:02d}.png", dpi=160)
            plt.close(fig)
    return table


def topk_table(frame: pd.DataFrame, out: Path, ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Only observed multi-condition pairs; require explicit condition identity.

    row_id alone is NOT a condition ID because repeated observations may exist.
    Pairs with repeated observations of any condition are excluded wholesale
    from this optional diagnostic; no label is averaged or selected.
    """
    if "condition_id" not in frame:
        atomic_json(out / "topk_status.json", dict(status="NOT_ASSESSED", reason="condition_id absent: repeated rows cannot establish multiple measured conditions"))
        return pd.DataFrame()
    records = []
    for (source, model, seed), part in frame.groupby(["source", "model", "seed"]):
        if part.condition_id.isna().any():
            raise ValueError("Top-k condition identifiers must be complete")
        all_pairs = part.pair_id.nunique()
        repeated = set(part.loc[part.duplicated(["pair_id", "condition_id"], keep=False), "pair_id"])
        conditions = part[~part.pair_id.isin(repeated)].copy()
        sizes = conditions.groupby("pair_id").size()
        for k in ks:
            usable = sizes[sizes >= max(2, k)].index
            selected = conditions[conditions.pair_id.isin(usable)].sort_values(["pair_id", "score", "condition_id"], ascending=[True, False, True]).groupby("pair_id").head(k)
            pair_success = selected.groupby("pair_id").y_true.mean()
            hit_at_k = selected.groupby("pair_id").y_true.max()
            random_precision = conditions[conditions.pair_id.isin(usable)].groupby("pair_id").y_true.mean()
            records.append(dict(source=source, model=model, seed=int(seed), k=k, eligible_pairs=len(usable), all_pairs=all_pairs,
                                coverage=len(usable)/all_pairs if all_pairs else 0, repeated_condition_pairs_excluded=len(repeated),
                                mean_measured_condition_success=float(pair_success.mean()),
                                hit_at_k=float(hit_at_k.mean()), random_precision_baseline=float(random_precision.mean())))
    result = pd.DataFrame(records)
    result.to_csv(out / "topk_measured_conditions.csv", index=False)
    atomic_json(out / "topk_status.json", dict(status="DESCRIPTIVE_ONLY", limitation="Ranking of measured conditions only; no claims about unmeasured candidates", tie_policy="condition_id lexical order", repeat_policy="Exclude entire pair if any measured condition is repeated; report coverage loss"))
    return result


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_evaluation(prediction_path: Path, lock_path: Path, out: Path, plots: bool = True) -> dict:
    lock = json.loads(lock_path.read_text())
    # Gate before reading predictions, including when callers invoke this function.
    if lock.get("status") != "FROZEN" or lock.get("evaluation_allowed") is not True:
        raise ValueError("Final evaluation remains sealed")
    frame = pd.read_parquet(prediction_path)
    frame, folds = validate_predictions(frame, lock, lock_path)
    out.mkdir(parents=True, exist_ok=True)
    evaluator_hash = file_hash(Path(__file__))
    signature = hashlib.sha256((file_hash(prediction_path)+file_hash(lock_path)+evaluator_hash+json.dumps(folds, sort_keys=True)).encode()).hexdigest()
    models, seeds = lock["models"], lock["seeds"]
    per_seed, macro = metric_tables(frame, folds, models)
    per_seed.to_csv(out / "metrics_by_source_seed.csv", index=False)
    per_seed.groupby(["source", "model", "eligible"])[list(METRICS)].agg(["mean", "std"]).to_csv(out / "metrics_by_source.csv")
    macro.to_csv(out / "main_table.csv", index=False)
    prepared = prepare_bootstrap(frame, folds, models, seeds)
    if not prepared:
        reliability_tables(frame, out, plots=plots)
        topk_table(frame, out)
        result = dict(status="NO_ELIGIBLE_SOURCES", signature=signature, models=models, seeds=seeds,
                      eligible_sources=[], acceptance=dict(status="NOT_ASSESSED", reason="No eligible source; descriptive results only"))
        atomic_json(out / "evaluation_manifest.json", result)
        return result
    primary, source_auc = bootstrap(prepared, models, int(lock.get("bootstrap_replicates", 5000)), int(lock.get("bootstrap_seed", 1729)), "pair", out, signature)
    sensitivity, _ = bootstrap(prepared, models, int(lock.get("sensitivity_replicates", 1000)), int(lock.get("bootstrap_seed", 1729)), "pigeonhole", out, signature)
    comparisons = lock.get("comparisons", [])
    contrasts = comparison_table(macro, primary, models, comparisons)
    atomic_json(out / "comparisons.json", clean_json(contrasts))
    intervals = []
    for m, model in enumerate(models):
        for j, metric in enumerate(METRICS):
            intervals.append(dict(model=model, metric=metric, scheme="pair", **interval(primary[:, m, j])))
            intervals.append(dict(model=model, metric=metric, scheme="pigeonhole", **interval(sensitivity[:, m, j])))
        intervals.append(dict(model=model, metric="worst_source_auroc", scheme="pair", **interval(np.min(source_auc[:, :, m], axis=1))))
    pd.DataFrame(intervals).to_csv(out / "intervals.csv", index=False)
    source_intervals = [dict(source=p["source"], model=model, metric="auroc", **interval(source_auc[:, i, m]))
                        for i, p in enumerate(prepared) for m, model in enumerate(models)]
    pd.DataFrame(source_intervals).to_csv(out / "source_auroc_intervals.csv", index=False)
    acceptance = acceptance_report(per_seed, macro, primary, models, lock)
    atomic_json(out / "acceptance.json", clean_json(acceptance))
    reliability_tables(frame, out, plots=plots)
    topk_table(frame, out)
    result = dict(status="EVALUATION_COMPLETE", signature=signature, prediction_sha256=file_hash(prediction_path),
                  lock_sha256=file_hash(lock_path), evaluator_sha256=evaluator_hash, models=models, seeds=seeds,
                  eligible_sources=[p["source"] for p in prepared], acceptance=acceptance,
                  inference="Equal fixed-source macro, seedwise metrics averaged; pair-cluster paired bootstrap conditional on fitted runs",
                  auprc_definition="Non-interpolated average precision, weighted for bootstrap",
                  calibration="Ranking uses raw probabilities; calibration is auxiliary Brier/reliability only")
    atomic_json(out / "evaluation_manifest.json", clean_json(result))
    return result


def synthetic_self_test() -> None:
    from tempfile import TemporaryDirectory
    records = []
    models, seeds = ["B", "B_A"], [11, 22, 33]
    for source_index, source in enumerate(("synthetic_a", "synthetic_b")):
        for i in range(60):
            y = i % 2
            for model in models:
                for seed in seeds:
                    records.append(dict(row_id=source_index*1000+i, source=source, pair_id=f"{source}_p{i//2}",
                                        aryl_id=f"{source}_a{i//6}", amine_id=f"{source}_n{(i//2)%7}", y_true=y,
                                        model=model, seed=seed, fold=source, score=.2+.6*y if model == "B_A" else .5))
    frame = pd.DataFrame(records)
    lock = dict(status="FROZEN", evaluation_allowed=True, models=models, seeds=seeds,
                folds=[dict(fold=s, source=s, eligible=True, test_row_ids=frame.loc[frame.source == s, "row_id"].unique().tolist()) for s in frame.source.unique()],
                bootstrap_replicates=25, sensitivity_replicates=25,
                comparisons=[dict(candidate="B_A", comparator="B")], acceptance_candidate="B_A", acceptance_comparator="B")
    with TemporaryDirectory(prefix="bh-evaluation-synthetic-") as temp:
        root = Path(temp)
        frame.to_parquet(root/"predictions.parquet", index=False)
        atomic_json(root/"lock.json", lock)
        result = run_evaluation(root/"predictions.parquet", root/"lock.json", root/"report", plots=False)
        assert result["acceptance"]["status"] == "INCREMENT_CRITERION_MET"
        run_evaluation(root/"predictions.parquet", root/"lock.json", root/"report", plots=False)
    print("Synthetic lock/coverage/metrics/bootstrap/resume self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-test", nargs="?", const="synthetic", choices=["synthetic"])
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if not all((args.predictions, args.lock, args.out)):
        parser.error("--predictions, --lock and --out are required outside synthetic self-test")
    result = run_evaluation(args.predictions, args.lock, args.out, plots=not args.no_plots)
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
