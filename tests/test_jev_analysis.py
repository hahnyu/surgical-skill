"""Tests for src/jev_analysis.py, on small hand-built cases and on the real folds."""

import numpy as np
import pandas as pd
import pytest

from baseline import load_config, load_dataset
from jev_analysis import (
    calibrated_predictions,
    composite_of_elements,
    headline_for_scores,
    rank_agreement,
    repeatability,
    score_vs_features,
    scores_for,
)
from loaders import OSATS_ELEMENTS, load_splits


@pytest.fixture(scope="module")
def data():
    return load_dataset(load_config()["features_file"])


BOOT = {"n_resamples": 100, "confidence": 0.95, "seed": 0}


def linear_scores(data, task, slope=1.0):
    """A score that is exactly linear in GRS (slope may be negative)."""
    sub = data[data["task"] == task]
    return pd.Series(slope * sub["grs"].to_numpy(float), index=sub.index)


# --- repeatability -------------------------------------------------------------------


def make_table(run_scores):
    rows = []
    for run, values in enumerate(run_scores):
        for i, v in enumerate(values):
            rows.append({"trial": f"T{i}", "run": run, "overall_skill__score": v})
    return pd.DataFrame(rows)


def test_identical_runs_are_fully_repeatable():
    table = make_table([[1.0, 2.0, 3.0]] * 3)
    r = repeatability(table, "overall_skill")["0v1"]
    assert r["spearman"] == pytest.approx(1.0) and r["mean_abs_diff"] == 0 and r["identical_share"] == 1.0


def test_repeatability_on_a_hand_worked_case():
    # Run 0: 1.0, 2.0, 3.0. Run 1: 1.5, 2.0, 2.0. Differences 0.5, 0, 1.0.
    table = make_table([[1.0, 2.0, 3.0], [1.5, 2.0, 2.0], [1.0, 2.0, 3.0]])
    r = repeatability(table, "overall_skill")["0v1"]
    assert r["mean_abs_diff"] == pytest.approx(0.5)
    assert r["max_abs_diff"] == pytest.approx(1.0)
    assert r["identical_share"] == pytest.approx(1 / 3)
    assert repeatability(table, "overall_skill")["0v2"]["mean_abs_diff"] == 0


def test_the_primary_score_is_run_zero_and_the_sensitivity_is_the_mean():
    table = make_table([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert list(scores_for(table, "overall_skill", "run0")) == [1.0, 2.0]
    assert list(scores_for(table, "overall_skill", "mean_of_runs")) == [3.0, 4.0]
    with pytest.raises(ValueError):
        scores_for(table, "overall_skill", "median")


def test_the_composite_is_the_mean_of_the_six_element_scores_in_run_zero():
    rows = []
    for run in (0, 1):
        row = {"trial": "T0", "run": run}
        row.update({f"{e}__score": float(i + 1) + (100 if run else 0) for i, e in enumerate(OSATS_ELEMENTS)})
        rows.append(row)
    composite = composite_of_elements(pd.DataFrame(rows))
    assert composite["T0"] == pytest.approx(3.5)        # mean of 1..6, ignoring run 1


# --- calibration and the direction guard ----------------------------------------------


def test_a_perfectly_linear_score_is_recovered_exactly(data):
    scores = linear_scores(data, "Suturing", slope=0.5)
    oof = calibrated_predictions(scores, data, "Suturing", "LOUO")
    assert np.allclose(oof["y_pred"], oof["y_true"])
    assert (oof["slope"] > 0).all()


def test_an_inverted_score_is_flagged_by_a_negative_slope_in_every_fold(data):
    """The direction guard. A score that ranks skill backwards is fitted by a line with a
    NEGATIVE slope, which makes its predictions look excellent. Only the slope sign
    reveals the inversion."""
    scores = linear_scores(data, "Suturing", slope=-1.0)
    oof = calibrated_predictions(scores, data, "Suturing", "LOSO")
    assert np.allclose(oof["y_pred"], oof["y_true"]), "the line hides the inversion..."
    assert (oof["slope"] < 0).all(), "...so the slope sign must expose it"
    row = headline_for_scores(scores, data, "Suturing", "LOSO", BOOT)
    assert row["n_negative_slopes"] == row["n_folds"] == 5
    assert row["mean_slope"] < 0


def test_the_line_is_fitted_on_training_trials_only(data):
    """Changing a held-out trial's GRS must not change that fold's slope or its predictions."""
    scores = pd.Series(np.random.default_rng(0).normal(size=len(data)), index=data.index)
    first_test = load_splits("Suturing", "LOSO")[0].test
    altered = data.copy()
    altered.loc[first_test, "grs"] = 999.0
    before = calibrated_predictions(scores, data, "Suturing", "LOSO")
    after = calibrated_predictions(scores, altered, "Suturing", "LOSO")
    assert np.array_equal(before.loc[first_test, "y_pred"], after.loc[first_test, "y_pred"])
    assert np.array_equal(before.loc[first_test, "slope"], after.loc[first_test, "slope"])


def test_constant_training_scores_give_the_training_mean_and_zero_slope(data):
    sub = data[data["task"] == "Suturing"]
    constant = pd.Series(3.0, index=sub.index)
    oof = calibrated_predictions(constant, data, "Suturing", "LOUO")
    assert (oof["slope"] == 0).all()
    fold = load_splits("Suturing", "LOUO")[0]
    expected = sub.loc[fold.train, "grs"].mean()
    assert oof.loc[fold.test, "y_pred"].iloc[0] == pytest.approx(expected)


def test_every_trial_is_predicted_exactly_once(data):
    oof = calibrated_predictions(linear_scores(data, "Knot_Tying"), data, "Knot_Tying", "LOUO")
    assert len(oof) == 36 and oof.index.is_unique


# --- rank agreement --------------------------------------------------------------------


def test_rank_agreement_is_one_for_a_perfectly_ordered_score_and_minus_one_when_reversed(data):
    up = rank_agreement(linear_scores(data, "Suturing"), data, "Suturing", BOOT)
    down = rank_agreement(linear_scores(data, "Suturing", slope=-1.0), data, "Suturing", BOOT)
    assert up["rho"] == pytest.approx(1.0) and down["rho"] == pytest.approx(-1.0)
    assert up["n_subjects"] == 8


def test_a_useless_score_gets_no_credit_beyond_the_mean(data):
    """A score of pure noise should not beat the feature-free reference by a wide margin."""
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(size=len(data)), index=data.index)
    row = headline_for_scores(noise, data, "Suturing", "LOUO", BOOT)
    assert row["improvement_low"] <= 0 or row["improvement"] < 1.0


# --- post-hoc helpers -----------------------------------------------------------------------


def test_score_vs_features_finds_a_feature_the_score_copies_and_one_it_reverses(data):
    columns = ["task_time_s", "PSM1_path_length"]
    score = data["task_time_s"].copy()
    out = score_vs_features(score, data, columns).set_index(["task", "feature"])["rho"]
    assert out[("Suturing", "task_time_s")] == pytest.approx(1.0)
    reversed_score = -data["task_time_s"]
    out2 = score_vs_features(reversed_score, data, columns).set_index(["task", "feature"])["rho"]
    assert out2[("Knot_Tying", "task_time_s")] == pytest.approx(-1.0)


def test_a_score_equal_to_grs_would_beat_the_reference_by_the_whole_reference_error(data):
    """Sanity bound on the yardstick: an oracle score gives zero error, so the improvement
    equals the reference MAE."""
    row = headline_for_scores(linear_scores(data, "Knot_Tying"), data, "Knot_Tying", "LOUO", BOOT)
    assert row["model_mae"] == pytest.approx(0.0, abs=1e-9)
    assert row["improvement"] == pytest.approx(row["reference_mae"])
