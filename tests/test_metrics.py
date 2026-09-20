"""Tests for src/metrics.py.

The bootstrap is the part most likely to be quietly wrong: it must resample whole
surgeons, not individual trials, or every interval is too narrow.
"""

import numpy as np
import pytest

from metrics import (
    accuracy,
    balanced_accuracy,
    classification_report,
    kendall,
    mae,
    quadratic_kappa,
    regression_report,
    rmse,
    spearman,
    subject_bootstrap,
)


# --- point statistics -------------------------------------------------------


def test_spearman_of_a_perfect_monotone_relationship_is_one():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    # Monotone but not linear: rank correlation still 1.
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_of_a_reversed_relationship_is_minus_one():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_and_kendall_are_nan_when_an_input_is_constant():
    assert np.isnan(spearman([1, 2, 3], [5, 5, 5]))
    assert np.isnan(kendall([1, 1, 1], [1, 2, 3]))


def test_kendall_on_a_hand_worked_case():
    # Pairs of (x, y): (1,1) (2,3) (3,2). Of the 3 pairs, 2 agree in order and 1
    # disagrees, so tau = (2 - 1) / 3 = 1/3.
    assert kendall([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_mae_and_rmse_on_a_hand_worked_case():
    # Errors are 1, -2, 3: MAE = (1+2+3)/3 = 2; RMSE = sqrt((1+4+9)/3).
    assert mae([1, 2, 3], [2, 0, 6]) == pytest.approx(2.0)
    assert rmse([1, 2, 3], [2, 0, 6]) == pytest.approx(np.sqrt(14 / 3))


def test_balanced_accuracy_is_not_fooled_by_a_big_class():
    # Always guessing the majority class gets 8/10 right but only 1/2 balanced.
    y_true = ["N"] * 8 + ["E"] * 2
    y_pred = ["N"] * 10
    assert accuracy(y_true, y_pred) == pytest.approx(0.8)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_quadratic_kappa_perfect_and_reversed():
    y = ["N", "I", "E", "N", "I", "E"]
    assert quadratic_kappa(y, y) == pytest.approx(1.0)
    flipped = ["E", "I", "N", "E", "I", "N"]
    assert quadratic_kappa(y, flipped) < 0


def test_quadratic_kappa_penalises_a_far_miss_more_than_a_near_miss():
    y_true = ["N", "I", "E", "N", "I", "E", "N", "I", "E"]
    near = ["I", "I", "E", "N", "E", "E", "N", "I", "I"]   # errors are one step off
    far = ["E", "I", "N", "E", "I", "E", "N", "I", "E"]    # same count of errors (3), two steps off
    assert sum(a != b for a, b in zip(y_true, near)) == sum(a != b for a, b in zip(y_true, far))
    assert quadratic_kappa(y_true, near) > quadratic_kappa(y_true, far)


# --- the surgeon bootstrap --------------------------------------------------


def test_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds():
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    pred = y + rng.normal(scale=1.0, size=40)
    subjects = np.repeat(list("ABCDEFGH"), 5)
    a = subject_bootstrap(y, pred, subjects, spearman, n_resamples=300, seed=7)
    b = subject_bootstrap(y, pred, subjects, spearman, n_resamples=300, seed=7)
    c = subject_bootstrap(y, pred, subjects, spearman, n_resamples=300, seed=8)
    assert a == b
    assert (a["low"], a["high"]) != (c["low"], c["high"])


def test_bootstrap_resamples_whole_surgeons_never_individual_trials():
    """The load-bearing test. Encode each trial's surgeon in y_true, then check
    that in every resample each surgeon appears either not at all or in whole
    multiples of the number of trials that surgeon has. Resampling trials would
    break this immediately."""
    sizes = {"A": 2, "B": 3, "C": 5}
    subjects = np.concatenate([[s] * n for s, n in sizes.items()])
    y_true = np.array([{"A": 0, "B": 1, "C": 2}[s] for s in subjects])
    y_pred = np.zeros(len(subjects))
    seen_totals = set()

    def check(yt, yp):
        counts = {code: int((yt == code).sum()) for code in (0, 1, 2)}
        for code, name in enumerate(("A", "B", "C")):
            assert counts[code] % sizes[name] == 0, f"surgeon {name} was split up"
        seen_totals.add(len(yt))
        return 0.0

    subject_bootstrap(y_true, y_pred, subjects, check, n_resamples=300, seed=0)
    # A trial-level bootstrap would always give exactly 10 rows; whole-surgeon
    # resampling gives sample sizes that vary.
    assert len(seen_totals) > 1


def test_bootstrap_interval_is_degenerate_for_a_perfect_prediction():
    y = np.arange(40, dtype=float)
    subjects = np.repeat(list("ABCDEFGH"), 5)
    result = subject_bootstrap(y, y, subjects, spearman, n_resamples=200, seed=0)
    assert result["estimate"] == pytest.approx(1.0)
    assert result["low"] == pytest.approx(1.0) and result["high"] == pytest.approx(1.0)
    assert result["n_skipped"] == 0


def test_bootstrap_interval_contains_the_estimate_for_a_noisy_case():
    rng = np.random.default_rng(3)
    y = rng.normal(size=80)
    pred = 0.6 * y + rng.normal(scale=0.8, size=80)
    subjects = np.repeat(list("ABCDEFGHIJ"), 8)
    r = subject_bootstrap(y, pred, subjects, spearman, n_resamples=1000, seed=0)
    assert r["low"] < r["estimate"] < r["high"]
    assert r["high"] - r["low"] > 0.05


def test_bootstrap_counts_resamples_where_the_statistic_is_undefined():
    # Surgeon A always has y = 1, surgeon B always y = 2. A resample that draws
    # only A (or only B) has a constant y, so Spearman is undefined there.
    subjects = np.array(["A"] * 5 + ["B"] * 5)
    y = np.array([1.0] * 5 + [2.0] * 5)
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    r = subject_bootstrap(y, pred, subjects, spearman, n_resamples=400, seed=0)
    assert r["n_skipped"] > 0
    assert not np.isnan(r["estimate"])


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        subject_bootstrap([1, 2, 3], [1, 2], ["A", "B", "C"], spearman)


def test_reports_have_the_expected_keys():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    pred = y + rng.normal(size=40)
    subjects = np.repeat(list("ABCDEFGH"), 5)
    reg = regression_report(y, pred, subjects, n_resamples=50)
    for key in ("spearman", "spearman_low", "spearman_high", "kendall", "mae", "rmse", "n_trials", "n_subjects"):
        assert key in reg
    assert reg["n_trials"] == 40 and reg["n_subjects"] == 8

    labels = np.array(["N", "I", "E", "N", "I", "E", "N", "I"] * 5)
    cls = classification_report(labels, labels, subjects, n_resamples=50)
    assert cls["accuracy"] == 1.0 and cls["quadratic_kappa"] == pytest.approx(1.0)
