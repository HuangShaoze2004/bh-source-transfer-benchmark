"""Post-release auditor tests use fabricated complete projects only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def auditor():
    path = Path(__file__).resolve().parents[1]/"scripts/finalize_report.py"
    spec = importlib.util.spec_from_file_location("independent_report_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_project(tmp_path, auditor):
    project = tmp_path/"synthetic_project"
    for name in ("data/processed", "artifacts", "reports/final"):
        (project/name).mkdir(parents=True)
    models, seeds = sorted(auditor.CORE_MODELS), [17, 42, 2026]
    records = []
    for source, n, offset in (("source_a", 4, 0), ("source_b", 8, 1000)):
        for i in range(n):
            records.append(dict(row_id=offset+i, source=source, y=i % 2, author_label=i % 2,
                                yield_raw=20*(i % 2), aryl_id=f"{source}_aryl{i//2}",
                                amine_id=f"{source}_amine{i//2}", pair_id=f"{source}_pair{i//2}",
                                input_key=f"{source}_condition{i}"))
    frame = pd.DataFrame(records)
    frame.to_parquet(project/"data/processed/records.parquet", index=False)
    folds = {source: dict(source=source, eligible=True,
                         train_ids=frame.loc[frame.source != source, "row_id"].tolist(),
                         test_ids=frame.loc[frame.source == source, "row_id"].tolist(), inner=[])
             for source in frame.source.unique()}
    auditor.dump(project/"data/processed/splits.json", dict(folds=folds))
    lock = dict(status="FROZEN", evaluation_allowed=True, models=models, seeds=seeds, minimum_seeds=3,
                primary_label="Yield>=10", score_type="probability",
                folds=[dict(fold=s, source=s, eligible=True, test_row_ids=f["test_ids"]) for s, f in folds.items()],
                artifact_hashes={p: auditor.sha256(project/p) for p in ("data/processed/records.parquet", "data/processed/splits.json")})
    auditor.dump(project/"FINAL_EVAL_LOCK.json", lock)
    predictions, per_seed = [], []
    for source, part in frame.groupby("source"):
        for model in models:
            reversed_score = model == "B" and source == "source_a"
            for seed in seeds:
                for row in part.itertuples():
                    score = (.8 if row.y else .2)
                    if reversed_score:
                        score = 1-score
                    predictions.append(dict(row_id=row.row_id, source=source, pair_id=row.pair_id,
                                            aryl_id=row.aryl_id, amine_id=row.amine_id, condition_id=row.input_key,
                                            y_true=row.y, model=model, seed=seed, fold=source, score=score, cal_score=.5))
                # Exact analytic metrics: perfect separation or reverse ordering
                # with balanced binary outcomes, independent of the auditor.
                per_seed.append(dict(source=source, model=model, seed=seed, eligible=True, n_rows=len(part),
                                     n_pairs=part.pair_id.nunique(), positive_rate=.5,
                                     auroc=0. if reversed_score else 1., auprc=.5 if reversed_score else 1.,
                                     brier=.64 if reversed_score else .04, cal_brier=.25))
    predictions = pd.DataFrame(predictions)
    predictions.to_parquet(project/"artifacts/predictions.parquet", index=False)
    pd.DataFrame(per_seed).to_csv(project/"reports/final/metrics_by_source_seed.csv", index=False)
    macro = [dict(model=model, n_sources=2, auroc=.5 if model == "B" else 1.,
                  auprc=.75 if model == "B" else 1., brier=.34 if model == "B" else .04,
                  worst_source_auroc=0. if model == "B" else 1.) for model in models]
    pd.DataFrame(macro).to_csv(project/"reports/final/main_table.csv", index=False)
    refresh_release_manifests(project, auditor)
    return project


def refresh_release_manifests(project, auditor):
    n = len(pd.read_parquet(project/"artifacts/predictions.parquet"))
    identities = dict(lock_sha256=auditor.sha256(project/"FINAL_EVAL_LOCK.json"),
                      predictions_sha256=auditor.sha256(project/"artifacts/predictions.parquet"))
    auditor.dump(project/"artifacts/PREDICTION_COMPLETENESS.json", dict(complete=True, rows=n, **identities))
    auditor.dump(project/"reports/final/evaluation_manifest.json", dict(status="EVALUATION_COMPLETE", **identities))


def test_independent_reconstruction_and_equal_source_macro(auditor, synthetic_project):
    result = auditor.audit_project(synthetic_project)
    assert result["audit"]["status"] == "PASS"
    b = result["macro"].set_index("model").loc["B"]
    assert b.auroc == pytest.approx(.5)
    assert b.auprc == pytest.approx(.75)
    assert b.brier == pytest.approx(.34)
    assert b.cal_brier == pytest.approx(.25)
    assert result["audit"]["checked_rows"] == 12*9*3


def test_release_gate_does_not_read_partial_predictions(auditor, tmp_path, monkeypatch):
    auditor.dump(tmp_path/"artifacts/PREDICTION_COMPLETENESS.json", dict(complete=True))
    def forbidden(*args, **kwargs):
        raise AssertionError("A partial prediction artifact was read")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(auditor.PendingEvaluation, match="predictions remain unread"):
        auditor.audit_project(tmp_path)


def test_saved_metric_corruption_is_detected(auditor, synthetic_project):
    path = synthetic_project/"reports/final/main_table.csv"
    table = pd.read_csv(path)
    table.loc[0, "auroc"] += .01
    table.to_csv(path, index=False)
    with pytest.raises(auditor.AuditFailure, match="Independent metric differs"):
        auditor.audit_project(synthetic_project)


def test_consistently_wrong_prediction_label_is_detected(auditor, synthetic_project):
    path = synthetic_project/"artifacts/predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[predictions.row_id == 0, "y_true"] = 1
    predictions.to_parquet(path, index=False)
    refresh_release_manifests(synthetic_project, auditor)
    with pytest.raises(auditor.AuditFailure, match="Ground-truth/chemistry mismatch: y_true"):
        auditor.audit_project(synthetic_project)


def test_chemistry_identity_mismatch_is_detected(auditor, synthetic_project):
    path = synthetic_project/"artifacts/predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "amine_id"] = "fabricated_amine"
    predictions.to_parquet(path, index=False)
    refresh_release_manifests(synthetic_project, auditor)
    with pytest.raises(auditor.AuditFailure, match="Ground-truth/chemistry mismatch: amine_id"):
        auditor.audit_project(synthetic_project)


def test_incomplete_method_row_matrix_is_detected(auditor, synthetic_project):
    path = synthetic_project/"artifacts/predictions.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    refresh_release_manifests(synthetic_project, auditor)
    with pytest.raises(auditor.AuditFailure, match="Incomplete locked row coverage"):
        auditor.audit_project(synthetic_project)


def test_entity_overlap_not_just_pair_overlap_is_detected(auditor, synthetic_project):
    records = pd.read_parquet(synthetic_project/"data/processed/records.parquet")
    splits = auditor.read_json(synthetic_project/"data/processed/splits.json")
    lock = auditor.read_json(synthetic_project/"FINAL_EVAL_LOCK.json")
    # Pair IDs remain distinct, but one aryl entity crosses sources.
    records.loc[records.row_id.isin([1000, 1001]), "aryl_id"] = records.loc[records.row_id == 0, "aryl_id"].iloc[0]
    with pytest.raises(auditor.AuditFailure, match="both-unseen"):
        auditor.independent_split_audit(records, splits, lock)


def test_frozen_data_changes_cannot_be_hidden_by_prediction_metadata(auditor, synthetic_project):
    path = synthetic_project/"data/processed/records.parquet"
    records = pd.read_parquet(path)
    records.loc[0, "yield_raw"] = 80
    records.to_parquet(path, index=False)
    with pytest.raises(auditor.AuditFailure, match="Frozen artifact differs"):
        auditor.audit_project(synthetic_project)


def test_sensitivity_fixed_primary_truth_is_distinct_from_native(auditor, synthetic_project):
    main = auditor.audit_project(synthetic_project)
    alternative = dict(main, predictions=main["predictions"].copy())
    alternative["predictions"]["y_true"] = 1-alternative["predictions"].y_true
    paired = auditor.common_row_comparison(main, alternative, "author_labels")
    fixed = paired[(paired.model == "B_A") & (paired.metric == "auroc") & (paired.truth_regime == "fixed_primary_truth")]
    native = paired[(paired.model == "B_A") & (paired.metric == "auroc") & (paired.truth_regime == "native_sensitivity_truth")]
    assert (fixed.primary_model == 1).all()
    assert (native.primary_model == 0).all()
    assert (paired.paired_delta == 0).all()
    assert (paired.changed_labels > 0).all()


def test_secondary_requires_all_strong_controls_and_balanced_swa(auditor):
    models = ["B_A", "B", "B_bal", "R0_RF", "R0_HGB", "R1_substitute", "B_bal_S"]
    baselines = models[1:-1]
    points = {m: .7 for m in models}
    points["B_A"] = .75
    bundle = dict(lock=dict(secondary_candidates=["B_A"], acceptance_comparators=baselines),
                  macro=pd.DataFrame([dict(model=m, auroc=v) for m, v in points.items()]),
                  by_seed=pd.DataFrame([dict(source=s, model=m, eligible=True, auroc=v) for s in ("a", "b") for m, v in points.items()]))
    contrasts = [dict(candidate="B_A", comparator=m, delta_auroc=.05, low=.01, p_holm=.03, invalid_fraction=0.) for m in baselines]
    assert auditor.secondary_assessment(bundle, contrasts)[0]["meets_secondary_rule"]
    assert not auditor.secondary_assessment(bundle, contrasts[:-1])[0]["meets_secondary_rule"]
    bundle["macro"].loc[bundle["macro"].model == "B_bal_S", "auroc"] = .76
    assert not auditor.secondary_assessment(bundle, contrasts)[0]["meets_secondary_rule"]


def test_readiness_cannot_hide_incomplete_or_failed_audits(auditor):
    positive = dict(status="INCREMENT_CRITERION_MET")
    complete = dict(author_labels="COMPLETE", dd_original="COMPLETE")
    assert auditor.choose_status(positive, True, True, complete, {}) == "READY_FOR_AUTHOR_REVIEW"
    assert auditor.choose_status(positive, True, False, complete, {}) == "CONTINUE"
    assert auditor.choose_status(positive, True, True, dict(author_labels="COMPLETE", dd_original="PENDING"), {}) == "CONTINUE"
    assert auditor.choose_status(positive, True, True, dict(author_labels="COMPLETE", dd_original="INCOMPLETE_BUDGET"), {}) == "NOT_READY"
    negative = dict(status="INCREMENT_CRITERION_NOT_MET")
    assert auditor.choose_status(negative, True, True, complete, {}) == "NOT_READY"
    assert auditor.choose_status(negative, True, True, complete, {}, [dict(meets_secondary_rule=True)]) == "READY_FOR_AUTHOR_REVIEW"


def test_synthetic_end_to_end_report_and_standalone_figures(auditor, synthetic_project):
    # The production evaluator is used only to produce an independent fixture;
    # every file and prediction below lives in pytest's synthetic temp project.
    from bh_transfer.evaluate import run_evaluation
    lock_path = synthetic_project/"FINAL_EVAL_LOCK.json"
    lock = auditor.read_json(lock_path)
    baselines = ["B", "B_bal", "R0_RF", "R0_HGB", "R1_substitute"]
    lock.update(bootstrap_replicates=20, sensitivity_replicates=20,
                acceptance_candidate="B_A_S", acceptance_comparators=baselines,
                combination_comparators=["B_A", "B_S", "B_bal_S"], secondary_candidates=[],
                comparisons=[dict(candidate="B_A_S", comparator=m) for m in baselines+["B_A", "B_S", "B_bal_S"]])
    auditor.dump(lock_path, lock)
    refresh_release_manifests(synthetic_project, auditor)
    run_evaluation(synthetic_project/"artifacts/predictions.parquet", lock_path, synthetic_project/"reports/final", plots=False)
    records = pd.read_parquet(synthetic_project/"data/processed/records.parquet")
    covariates = records.rename(columns={"y": "y_true"})
    covariates["fold"] = covariates.source
    covariates["aryl_max_train_tanimoto"] = .2
    covariates["amine_max_train_tanimoto"] = .4
    covariates["aryl_has_heteroaromatic_ring"] = False
    covariates["aryl_bound_halogen_class"] = "Br"
    covariates["amine_type"] = "secondary"
    covariates["reagent_coverage"] = "partly_seen"
    covariates["catalyst_seen_in_train"] = False
    covariates["base_seen_in_train"] = True
    covariates["solvent_seen_in_train"] = True
    (synthetic_project/"reports/domain").mkdir()
    covariates.to_parquet(synthetic_project/"reports/domain/row_covariates.parquet", index=False)
    assert auditor.finalize(synthetic_project) == "CONTINUE"  # sensitivities absent
    assert (synthetic_project/"reports/EXPERIMENT_REPORT.md").is_file()
    for filename in ("macro_auroc.png", "macro_auroc.pdf", "source_auroc_heatmap.png", "source_auroc_heatmap.pdf"):
        assert (synthetic_project/"reports/final"/filename).stat().st_size > 0
    assert auditor.read_json(synthetic_project/"reports/final/independent_audit.json")["status"] == "PASS"
