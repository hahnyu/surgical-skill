"""Tests for src/evaluation.py, on hand-worked cases first and the real data second."""

import numpy as np
import pandas as pd
import pytest

from evaluation import (
    N_BINS,
    aurc,
    assemble,
    baseline_proba,
    bland_altman,
    calibration_report,
    coverage_table,
    cumulative_risk,
    ece_mce,
    icc_2_1,
    jev_proba,
    multiclass_brier,
    netcal_ece_mce,
    oracle_aurc,
    reliability_table,
    selective_report,
    true_level,
    usefulness,
)

BOOT = {"n_resamples": 100, "confidence": 0.95, "seed": 0}


# --- the level Jev is judged against -----------------------------------------------------


def test_true_level_maps_grs_onto_the_zero_to_four_scale():
    # GRS / 6 is the mean element score on the 1-5 scale; nearest whole score, halves up.
    assert list(true_level([6, 8, 9, 14, 15, 20, 21, 26, 27, 30])) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # 8 -> 1.33 -> 1 (level 0);  9 -> 1.5 -> 2 (level 1);  27 -> 4.5 -> 5 (level 4).


def test_true_level_stays_inside_zero_to_four():
    assert set(true_level(np.arange(6, 31))) == {0, 1, 2, 3, 4}
    assert true_level([-5])[0] == 0 and true_level([99])[0] == 4     # clipped, not an error


# --- Brier and reliability -----------------------------------------------------------------


def test_multiclass_brier_on_hand_worked_cases():
    p = [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]
    # Trial 1 (true class 0): 0.3^2 + 0.2^2 + 0.1^2 = 0.14. Trial 2 (true 2): 0.1^2 + 0.1^2 + 0.2^2 = 0.06.
    assert multiclass_brier(p, [0, 2]) == pytest.approx(0.10)
    assert multiclass_brier([[1, 0, 0], [0, 1, 0]], [0, 1]) == 0.0                # perfect
    assert multiclass_brier([[1 / 3] * 3] * 4, [0, 1, 2, 0]) == pytest.approx(2 / 3)   # uniform: (k-1)/k


def test_reliability_table_on_a_hand_worked_case():
    # Four predictions, all 0.9 confident; two are right. One bin holds all four: the
    # accuracy is 0.5 against a confidence of 0.9, a gap of 0.4.
    proba = np.array([[0.9, 0.1]] * 4)
    truth = [0, 0, 1, 1]
    t = reliability_table(proba, truth, n_bins=5)
    filled = t[t["n"] > 0].iloc[0]
    assert filled["n"] == 4 and filled["mean_confidence"] == pytest.approx(0.9) and filled["accuracy"] == pytest.approx(0.5)
    ece, mce = ece_mce(proba, truth, n_bins=5)
    assert ece == pytest.approx(0.4) and mce == pytest.approx(0.4)


def test_a_perfectly_calibrated_case_has_zero_ece():
    # In the 0.9 bin, 9 of 10 are right.
    proba = np.array([[0.9, 0.1]] * 10)
    truth = [0] * 9 + [1]
    assert ece_mce(proba, truth)[0] == pytest.approx(0.0)


def test_confidence_of_exactly_one_lands_in_the_last_bin_not_off_the_end():
    t = reliability_table(np.array([[1.0, 0.0]]), [0], n_bins=5)
    assert t.loc[4, "n"] == 1 and t["n"].sum() == 1


def test_my_ece_equals_netcals_on_random_data():
    """The pre-declared implementation is netcal's; mine is used inside the bootstrap, so
    they must agree exactly."""
    rng = np.random.default_rng(0)
    for k in (3, 5):
        raw = rng.gamma(1.0, size=(200, k))
        proba = raw / raw.sum(axis=1, keepdims=True)
        truth = rng.integers(0, k, size=200)
        mine, theirs = ece_mce(proba, truth), netcal_ece_mce(proba, truth)
        assert mine[0] == pytest.approx(theirs[0]) and mine[1] == pytest.approx(theirs[1])


def test_my_ece_equals_netcals_on_the_real_predictions():
    for scheme in ("LOUO", "LOSO"):
        f = assemble(scheme)
        truth = f["base_class_true"].map({"N": 0, "I": 1, "E": 2}).to_numpy()
        for proba, y in ((baseline_proba(f), truth), (jev_proba(f), f["jev_level_true"].to_numpy())):
            mine, theirs = ece_mce(proba, y), netcal_ece_mce(proba, y)
            assert mine[0] == pytest.approx(theirs[0], abs=1e-9) and mine[1] == pytest.approx(theirs[1], abs=1e-9)


def test_calibration_report_skill_is_zero_for_the_constant_and_positive_for_a_better_forecast():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 3, size=60)
    subjects = np.repeat(list("ABCDEF"), 10)
    freq = np.bincount(truth, minlength=3) / 60
    constant = np.tile(freq, (60, 1))
    same = calibration_report(constant, truth, subjects, BOOT, constant)
    assert same["brier_skill"] == pytest.approx(0.0)
    sharp = np.eye(3)[truth] * 0.8 + 0.1                     # mostly right, and sums to 1
    better = calibration_report(sharp, truth, subjects, BOOT, constant)
    assert better["brier_skill"] > 0.5 and better["accuracy"] == 1.0


# --- selective prediction -----------------------------------------------------------------------


def test_cumulative_risk_and_aurc_on_a_hand_worked_case():
    errors = [1, 2, 3, 4]
    confidence = [0.9, 0.8, 0.7, 0.6]                        # most confident = smallest error
    assert list(cumulative_risk(errors, confidence)) == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert aurc(errors, confidence) == pytest.approx(1.75)   # mean of 1, 1.5, 2, 2.5
    assert oracle_aurc(errors) == pytest.approx(1.75)        # already the best order
    assert usefulness(errors, confidence) == pytest.approx(2.5 - 1.75)


def test_confidence_that_points_the_wrong_way_has_negative_usefulness():
    errors = [1, 2, 3, 4]
    backwards = [0.6, 0.7, 0.8, 0.9]                         # most confident = biggest error
    assert list(cumulative_risk(errors, backwards)) == pytest.approx([4.0, 3.5, 3.0, 2.5])
    assert usefulness(errors, backwards) == pytest.approx(2.5 - 3.25)


def test_uninformative_confidence_gives_about_zero_usefulness_on_average():
    rng = np.random.default_rng(3)
    errors = rng.exponential(3.0, size=2000)
    assert abs(usefulness(errors, rng.random(2000))) < 0.15


def test_oracle_is_never_worse_than_any_ordering():
    rng = np.random.default_rng(4)
    errors = rng.exponential(3.0, size=100)
    assert oracle_aurc(errors) <= aurc(errors, rng.random(100))
    assert oracle_aurc(errors) <= aurc(errors, -errors + rng.normal(0, 0.1, 100)) + 1e-9


def test_coverage_table_on_a_hand_worked_case():
    errors = [1, 2, 3, 4]
    reference = [2, 2, 2, 2]
    t = coverage_table(errors, reference, [0.9, 0.8, 0.7, 0.6], coverages=(1.0, 0.5))
    full, half = t.iloc[0], t.iloc[1]
    assert full["mae"] == pytest.approx(2.5) and full["improvement"] == pytest.approx(2.0 - 2.5)
    # Keeping the 2 most confident (errors 1, 2): MAE 1.5. Reference errors there: 2, 2 -> 2.0. Gain 0.5.
    assert half["n_kept"] == 2 and half["mae"] == pytest.approx(1.5) and half["improvement"] == pytest.approx(0.5)


def test_selective_report_flags_informative_and_uninformative_confidence():
    rng = np.random.default_rng(5)
    subjects = np.repeat(list("ABCDEFGH"), 12)
    errors = rng.exponential(3.0, size=96)
    ref = np.full(96, 3.0)
    good = selective_report(errors, ref, -errors, subjects, BOOT)                # confidence = negative error
    junk = selective_report(errors, ref, rng.random(96), subjects, BOOT)
    assert good["informative"] and good["usefulness_low"] > 0
    assert not junk["informative"]
    assert good["e_aurc"] == pytest.approx(0.0, abs=1e-9) and good["aurc"] == pytest.approx(good["oracle_aurc"])


# --- agreement ---------------------------------------------------------------------------------------


def test_icc_matches_the_formula_worked_out_independently():
    """ICC(2,1) = (MSR - MSE) / (MSR + (k-1) MSE + k (MSC - MSE) / n), from the two-way
    ANOVA mean squares, computed here without pingouin."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])
    n, k = 5, 2
    data = np.column_stack([truth, pred])
    grand = data.mean()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((data - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_error / ((n - 1) * (k - 1))
    expected = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    assert icc_2_1(truth, pred)["icc"] == pytest.approx(expected)


def test_icc_is_one_for_identical_ratings_and_penalises_a_constant_offset():
    x = np.array([10.0, 12.0, 15.0, 20.0, 25.0])
    assert icc_2_1(x, x)["icc"] == pytest.approx(1.0)
    assert icc_2_1(x, x + 5)["icc"] < 0.95        # absolute agreement punishes the bias


def test_bland_altman_on_a_hand_worked_case():
    r = bland_altman([10, 10, 10, 10], [11, 9, 11, 9])          # differences +1, -1, +1, -1
    assert r["bias"] == pytest.approx(0.0)
    assert r["sd"] == pytest.approx(np.sqrt(4 / 3))             # sample SD of +-1 with n = 4
    assert r["upper"] == pytest.approx(1.96 * np.sqrt(4 / 3)) and r["lower"] == pytest.approx(-r["upper"])


# --- the assembled real data ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["LOUO", "LOSO"])
def test_assemble_gives_one_row_per_trial_with_consistent_columns(scheme):
    f = assemble(scheme)
    assert len(f) == 103 and f.index.is_unique and list(f.index) == sorted(f.index)
    assert np.allclose(baseline_proba(f).sum(axis=1), 1.0)
    # Jev's API returns probabilities rounded to two decimals, so a row can sum to 0.99.
    assert np.abs(jev_proba(f).sum(axis=1) - 1.0).max() <= 0.0101
    assert f["jev_conf"].between(0, 1).all() and f["base_conf"].between(0, 1).all()
    assert set(f["jev_level_true"]) <= {0, 1, 2, 3, 4}
    assert f[["grs", "base_pred", "jev_pred", "free_pred"]].notna().all().all()


def test_jev_probabilities_and_gpr_predictions_are_scheme_specific_only_where_they_should_be():
    a, b = assemble("LOUO"), assemble("LOSO")
    assert np.allclose(a[[f"jev_p{k}" for k in range(5)]], b[[f"jev_p{k}" for k in range(5)]])   # Jev is never trained
    assert (a["jev_conf"] == b["jev_conf"]).all()
    assert not np.allclose(a["base_conf"], b["base_conf"])                                        # the classifier is refit per scheme
    assert not np.allclose(a["jev_pred"], b["jev_pred"])                                          # the calibration line is refit per fold
