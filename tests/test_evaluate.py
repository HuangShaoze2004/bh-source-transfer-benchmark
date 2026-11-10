"""Synthetic-only evaluation tests: never load the reaction test set."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bh_transfer.evaluate import (
    WeightedMetrics, acceptance_report, bootstrap, holm_adjust, metric_tables,
    prepare_bootstrap, run_evaluation, topk_table, validate_predictions,
)


@pytest.fixture
def synthetic():
    rows = []
    for s, n in (("small", 20), ("large", 80)):
        for i in range(n):
            for model in ("B", "B_A"):
                for seed in (11, 22, 33):
                    rows.append(dict(row_id=(0 if s == "small" else 1000)+i, source=s,
                                     pair_id=f"{s}_{i//2}", aryl_id=f"{s}_a{i//4}",
                                     amine_id=f"{s}_n{(i//2)%5}", y_true=i % 2,
                                     model=model, seed=seed, fold=s,
                                     score=.5 if model == "B" else (.8 if i % 2 else .2)))
    frame = pd.DataFrame(rows)
    lock = dict(status="FROZEN", evaluation_allowed=True, models=["B", "B_A"], seeds=[11, 22, 33],
                folds=[dict(fold=s, source=s, eligible=True, test_row_ids=frame.loc[frame.source == s, "row_id"].unique().tolist()) for s in frame.source.unique()],
                acceptance_candidate="B_A", acceptance_comparator="B")
    return frame, lock


def test_weighted_ties_match_independent_sklearn():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    score = np.array([.1, .1, .3, .4, .4, .8, .8, .9])
    w = np.array([2, 0, 4, 1, 2, 3, 0, 1])
    actual = WeightedMetrics(y, score)(w)
    expected = [roc_auc_score(y, score, sample_weight=w),
                average_precision_score(y, score, sample_weight=w),
                brier_score_loss(y, score, sample_weight=w)]
    np.testing.assert_allclose(actual, expected)
    degenerate = WeightedMetrics(y, score)(1-y)
    assert np.isnan(degenerate[0]) and np.isnan(degenerate[1])


def test_macro_sources_not_rows_and_seeds_not_ensembled(synthetic):
    frame, lock = synthetic
    # In small source, B_A seed 11 is perfect and the other two are reversed.
    chosen = (frame.source == "small") & (frame.model == "B_A") & (frame.seed != 11)
    frame.loc[chosen, "score"] = 1-frame.loc[chosen, "score"]
    per_seed, macro = metric_tables(frame, lock["folds"], lock["models"])
    actual = macro.set_index("model").loc["B_A", "auroc"]
    assert actual == pytest.approx((1/3+1)/2)
    # Ensemble in the small source is perfectly reversed; it would yield .5.
    assert actual != pytest.approx(.5)
    assert len(per_seed) == 2*2*3


def test_lock_and_coverage_reject_partial_results(synthetic):
    frame, lock = synthetic
    validate_predictions(frame, lock)
    with pytest.raises(ValueError, match="Incomplete row coverage"):
        validate_predictions(frame.iloc[1:], lock)
    with pytest.raises(ValueError, match="Duplicate predictions"):
        validate_predictions(pd.concat([frame, frame.iloc[:1]]), lock)
    with pytest.raises(ValueError, match="sealed"):
        validate_predictions(frame, dict(lock, evaluation_allowed=False))
    with pytest.raises(ValueError, match="model/seed"):
        validate_predictions(frame[frame.seed != 33], lock)


def test_sealed_gate_precedes_prediction_read(tmp_path):
    lock_path = tmp_path/"lock.json"
    lock_path.write_text('{"status":"FROZEN","evaluation_allowed":false}')
    with pytest.raises(ValueError, match="sealed"):
        run_evaluation(tmp_path/"DOES_NOT_EXIST.parquet", lock_path, tmp_path/"report")
    assert not (tmp_path/"report").exists()


def test_inconsistent_ground_truth_rejected(synthetic):
    frame, lock = synthetic
    frame.loc[0, "y_true"] = 1-frame.loc[0, "y_true"]
    with pytest.raises(ValueError, match="Inconsistent row"):
        validate_predictions(frame, lock)


def test_probability_and_pair_identity_validation(synthetic):
    frame, lock = synthetic
    invalid = frame.copy()
    invalid.loc[0, "score"] = 1.1
    with pytest.raises(ValueError, match="outside"):
        validate_predictions(invalid, lock)
    invalid = frame.copy()
    invalid.loc[invalid.row_id == 1, "amine_id"] = "different_entity"
    with pytest.raises(ValueError, match="pair_id"):
        validate_predictions(invalid, lock)


def test_pair_and_pigeonhole_share_chemistry_draws_and_resume(synthetic, tmp_path):
    frame, lock = synthetic
    frame.loc[frame.model == "B", "score"] = frame.loc[frame.model == "B_A", "score"].to_numpy()
    prepared = prepare_bootstrap(frame, lock["folds"], lock["models"], lock["seeds"])
    for scheme in ("pair", "pigeonhole"):
        first, first_sources = bootstrap(prepared, lock["models"], 17, 42, scheme, tmp_path, "synthetic_signature", checkpoint_every=5)
        second, second_sources = bootstrap(prepared, lock["models"], 17, 42, scheme, tmp_path, "synthetic_signature", checkpoint_every=5)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_sources, second_sources)
        np.testing.assert_allclose(first[:, 0], first[:, 1], equal_nan=True)
        with pytest.raises(ValueError, match="different inputs"):
            bootstrap(prepared, lock["models"], 17, 42, scheme, tmp_path, "wrong_signature")


def test_holm_known_values():
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert holm_adjust([]) == []


def test_acceptance_and_source_regression_guard(synthetic):
    frame, lock = synthetic
    per_seed, macro = metric_tables(frame, lock["folds"], lock["models"])
    draws = np.tile(macro[["auroc", "auprc", "brier"]].to_numpy(), (100, 1, 1))
    assert acceptance_report(per_seed, macro, draws, lock["models"], lock)["status"] == "INCREMENT_CRITERION_MET"
    per_seed.loc[(per_seed.model == "B_A") & (per_seed.source == "small"), "auroc"] = .46
    result = acceptance_report(per_seed, macro, draws, lock["models"], lock)
    assert result["status"] == "INCREMENT_CRITERION_NOT_MET"
    assert result["min_source_delta_vs_max_comparator"] == pytest.approx(-.04)


def test_topk_requires_actual_condition_identifiers(synthetic, tmp_path):
    frame, _ = synthetic
    assert topk_table(frame, tmp_path).empty
    assert "NOT_ASSESSED" in (tmp_path/"topk_status.json").read_text()


def test_topk_does_not_treat_repeats_as_more_conditions(synthetic, tmp_path):
    frame, _ = synthetic
    frame["condition_id"] = frame.row_id % 2
    # Whole first pair has a repeated condition; it must be omitted without
    # choosing the label or averaging the repeat observations.
    repeats = frame[frame.row_id == 0].copy()
    repeats["row_id"] = 99999
    repeats["y_true"] = 1
    frame = pd.concat([frame, repeats], ignore_index=True)
    result = topk_table(frame, tmp_path, ks=(1,))
    row = result[(result.source == "small") & (result.model == "B_A")].iloc[0]
    assert row.repeated_condition_pairs_excluded == 1
    assert row.coverage == pytest.approx(.9)
    assert row.hit_at_k == pytest.approx(1)
