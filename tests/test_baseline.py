"""Tests for src/baseline.py.

The leakage tests come first: each one changes something the model must NOT be
able to see and confirms the predictions do not move. Leakage does not break
anything visibly; it just makes results look better than they are.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

import baseline
from baseline import (
    CLASS_ORDER,
    cross_validate,
    feature_columns,
    feature_free_reference,
    load_config,
    load_dataset,
    log_run,
    make_classifier,
    intercept_only_under_shuffles,
    make_regressor,
    permutation_sentinel,
    run_experiment,
)
from loaders import Fold, load_splits
from metrics import spearman


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    return load_dataset(cfg["features_file"])


def synthetic(data, signal):
    """Real trial names, subjects and folds, but invented features and labels.

    With signal=True, GRS depends on the first two invented features. With
    signal=False the invented features are unrelated to GRS.
    """
    rng = np.random.default_rng(0)
    fake = data.copy()
    columns = feature_columns(data)
    fake[columns] = rng.normal(size=(len(fake), len(columns)))
    if signal:
        fake["grs"] = (15 + 3 * fake[columns[0]] + 2 * fake[columns[1]]
                       + rng.normal(scale=1.0, size=len(fake)))
    return fake


# --- the data going in --------------------------------------------------------


def test_the_model_inputs_are_only_the_16_features_never_a_label_or_identifier(data):
    columns = feature_columns(data)
    assert len(columns) == 16
    for forbidden in ("task", "subject", "skill", "grs", "repetition"):
        assert forbidden not in columns


def test_the_dataset_has_103_labelled_trials(data):
    assert len(data) == 103
    assert data["task"].value_counts().to_dict() == {"Suturing": 39, "Knot_Tying": 36, "Needle_Passing": 28}


# --- leakage -----------------------------------------------------------------


def test_test_fold_labels_cannot_change_that_folds_predictions(data, cfg):
    """Replace the labels of the trials being tested with nonsense: predictions for
    that fold must be identical, because a model never sees its own test labels."""
    first_test = load_splits("Suturing", "LOSO")[0].test
    scrambled = data.copy()
    scrambled.loc[first_test, "grs"] = 999.0
    before = cross_validate("Suturing", "LOSO", "regression", data, cfg)
    after = cross_validate("Suturing", "LOSO", "regression", scrambled, cfg)
    assert np.array_equal(before.loc[first_test, "y_pred"], after.loc[first_test, "y_pred"])


def test_test_fold_features_cannot_change_other_test_rows_predictions(data, cfg):
    """If the scaler were fit on the test rows, altering some test rows would
    shift the scaling and so change the predictions for the untouched ones."""
    first_test = load_splits("Suturing", "LOSO")[0].test
    changed, untouched = first_test[: len(first_test) // 2], first_test[len(first_test) // 2:]
    altered = data.copy()
    altered.loc[changed, feature_columns(data)] = altered.loc[changed, feature_columns(data)] * 50 + 7
    before = cross_validate("Suturing", "LOSO", "regression", data, cfg)
    after = cross_validate("Suturing", "LOSO", "regression", altered, cfg)
    assert np.array_equal(before.loc[untouched, "y_pred"], after.loc[untouched, "y_pred"])


def test_the_scaler_learns_its_mean_from_the_training_rows_only(data, cfg):
    columns = feature_columns(data)
    fold = load_splits("Suturing", "LOUO")[0]
    train = data.loc[fold.train]
    model = make_regressor(cfg["regression"])
    model.fit(train[columns], train["grs"], groups=train["subject"])
    scaler = model.best_estimator_.named_steps["scale"]
    assert np.allclose(scaler.mean_, train[columns].mean().to_numpy())
    # And it is NOT the mean of everything (which would include the test trials).
    assert not np.allclose(scaler.mean_, data[data["task"] == "Suturing"][columns].mean().to_numpy())


def test_alpha_is_chosen_by_leave_one_subject_out_inside_the_training_set(data, cfg):
    model = make_regressor(cfg["regression"])
    assert isinstance(model.cv, LeaveOneGroupOut)
    columns = feature_columns(data)
    fold = load_splits("Suturing", "LOSO")[0]
    train = data.loc[fold.train]
    with pytest.raises(ValueError):                 # refuses to run without surgeon groups
        make_regressor(cfg["regression"]).fit(train[columns], train["grs"])


def test_the_classifier_scaler_is_inside_the_pipeline(cfg):
    model = make_classifier(cfg["classification"])
    assert list(model.named_steps) == ["scale", "model"]
    assert model.named_steps["model"].C == 1.0


def test_shuffled_labels_give_a_spearman_near_zero_on_invented_features(data, cfg):
    """The sentinel, exercised on data where there is provably nothing to find."""
    fake = synthetic(data, signal=False)
    small = {**cfg, "permutation_sentinel": {"n_permutations": 10, "max_abs_mean_rho": 0.15}}
    result = permutation_sentinel("Suturing", "LOUO", fake, small)
    assert result["passed"], result
    assert abs(result["mean_rho"]) < 0.15


# --- the pipeline can learn, and predicts each trial exactly once ---------------


def test_it_recovers_a_real_signal_under_leave_one_surgeon_out(data, cfg):
    fake = synthetic(data, signal=True)
    oof = cross_validate("Suturing", "LOUO", "regression", fake, cfg)
    assert spearman(oof["y_true"], oof["y_pred"]) > 0.8


@pytest.mark.parametrize("task, n", [("Suturing", 39), ("Knot_Tying", 36), ("Needle_Passing", 28)])
@pytest.mark.parametrize("scheme", ["LOSO", "LOUO"])
def test_every_trial_is_predicted_exactly_once(data, cfg, task, n, scheme):
    oof = cross_validate(task, scheme, "regression", data, cfg)
    assert len(oof) == n
    assert oof.index.is_unique
    assert set(oof.index) == set(data[data["task"] == task].index)
    assert (oof["subject"] == data.loc[oof.index, "subject"]).all()
    assert (oof["y_true"] == data.loc[oof.index, "grs"]).all()


def test_louo_never_trains_on_the_held_out_surgeon(data):
    for fold in load_splits("Suturing", "LOUO"):
        held_out = {t.split("_")[-1][0] for t in fold.test}
        in_training = {t.split("_")[-1][0] for t in fold.train}
        assert not held_out & in_training


def test_the_result_is_deterministic(data, cfg):
    a = cross_validate("Knot_Tying", "LOUO", "regression", data, cfg)
    b = cross_validate("Knot_Tying", "LOUO", "regression", data, cfg)
    assert a.equals(b)


def test_ridge_picks_alpha_from_the_declared_grid(data, cfg):
    oof = cross_validate("Suturing", "LOSO", "regression", data, cfg)
    assert set(oof["alpha"]) <= set(cfg["regression"]["alpha_grid"])


# --- classification ---------------------------------------------------------------


def test_class_probabilities_sum_to_one_and_match_the_prediction(data, cfg):
    oof = cross_validate("Suturing", "LOSO", "classification", data, cfg)
    proba = oof[[f"proba_{c}" for c in CLASS_ORDER]]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all().all()
    predicted = proba.idxmax(axis=1).str.replace("proba_", "")
    assert (predicted == oof["y_pred"]).all()


def test_a_class_missing_from_training_gets_probability_zero(data, cfg, monkeypatch):
    """Each of three folds trains without one class. The missing class must show up
    as a column of zeros, not as a missing column or an error."""
    sub = data[data["task"] == "Suturing"]
    by_class = {c: list(sub.index[sub["skill"] == c]) for c in CLASS_ORDER}
    folds = []
    for missing in CLASS_ORDER:
        train = [t for c in CLASS_ORDER if c != missing for t in by_class[c]]
        folds.append(Fold(name=f"no_{missing}", train=train, test=by_class[missing]))
    monkeypatch.setattr(baseline, "load_splits", lambda task, scheme: folds)

    oof = cross_validate("Suturing", "LOSO", "classification", data, cfg)
    for missing in CLASS_ORDER:
        rows = oof[oof["fold"] == f"no_{missing}"]
        assert (rows[f"proba_{missing}"] == 0).all()
        assert np.allclose(rows[[f"proba_{c}" for c in CLASS_ORDER]].sum(axis=1), 1.0)


def test_bad_kind_is_rejected(data, cfg):
    with pytest.raises(ValueError):
        cross_validate("Suturing", "LOSO", "ranking", data, cfg)


# --- the experiments log -------------------------------------------------------------


def test_log_run_creates_the_file_then_appends(tmp_path):
    path = tmp_path / "experiments.csv"
    log_run({"task": "Suturing", "spearman": 0.5}, path)
    log_run({"task": "Knot_Tying", "spearman": 0.4}, path)
    log = pd.read_csv(path)
    assert list(log["task"]) == ["Suturing", "Knot_Tying"]
    assert list(log["spearman"]) == [0.5, 0.4]


def test_log_run_accepts_a_new_column_without_losing_old_rows(tmp_path):
    path = tmp_path / "experiments.csv"
    log_run({"task": "Suturing", "spearman": 0.5}, path)
    log_run({"task": "Suturing", "accuracy": 0.7}, path)
    log = pd.read_csv(path)
    assert len(log) == 2
    assert log.loc[0, "spearman"] == 0.5 and pd.isna(log.loc[0, "accuracy"])
    assert log.loc[1, "accuracy"] == 0.7 and pd.isna(log.loc[1, "spearman"])


# --- the second config (2026-09-20) --------------------------------------------------


V2_PATH = baseline.PROJECT_ROOT / "configs" / "baseline_v2.yaml"


def test_v2_differs_from_the_original_only_where_intended(cfg):
    v2 = load_config(V2_PATH)
    old_grid, new_grid = cfg["regression"]["alpha_grid"], v2["regression"]["alpha_grid"]
    assert new_grid[: len(old_grid)] == old_grid          # the old grid is kept whole
    assert new_grid == [0.1, 1, 10, 100, 1000, 10000]     # and extended by exactly two steps
    # Everything else about the regression is unchanged.
    for key in ("target", "model", "inner_cv", "inner_scoring", "standardize"):
        assert v2["regression"][key] == cfg["regression"][key]
    assert v2["seed"] == cfg["seed"]
    assert v2["bootstrap"] == cfg["bootstrap"]
    assert v2["features_file"] == cfg["features_file"]
    assert v2["classification"] == cfg["classification"]
    assert v2["run_kinds"] == ["regression"]


def test_the_original_config_is_untouched(cfg):
    """Its results are reported alongside v2, so it must not have moved."""
    assert cfg["regression"]["alpha_grid"] == [0.1, 1, 10, 100]
    assert cfg["permutation_sentinel"] == {"n_permutations": 50, "max_abs_mean_rho": 0.10}
    assert "run_kinds" not in cfg and "run_name" not in cfg


def test_sentinel_one_sided_rule_ignores_a_negative_mean_but_flags_a_positive_one(data):
    fake = synthetic(data, signal=False)
    base = {"seed": 0, "regression": load_config()["regression"]}
    # Invented features carry no signal, so the shuffled mean is near zero or slightly negative.
    lenient = {**base, "permutation_sentinel": {"n_permutations": 5, "max_positive_mean_rho": 0.10}}
    result = permutation_sentinel("Suturing", "LOUO", fake, lenient)
    assert result["criterion"] == "one_sided"
    assert result["passed"] == (result["mean_rho"] <= 0.10)
    # A limit set far below the observed mean must flag it: proves the rule can fail.
    strict = {**base, "permutation_sentinel": {"n_permutations": 5, "max_positive_mean_rho": -0.99}}
    assert permutation_sentinel("Suturing", "LOUO", fake, strict)["passed"] is False


def test_sentinel_two_sided_rule_still_works_for_the_original_config(data):
    fake = synthetic(data, signal=False)
    cfg2 = {"seed": 0, "regression": load_config()["regression"],
            "permutation_sentinel": {"n_permutations": 5, "max_abs_mean_rho": 0.0001}}
    result = permutation_sentinel("Suturing", "LOUO", fake, cfg2)
    assert result["criterion"] == "two_sided"
    assert result["passed"] is False        # a mean of exactly ~0 is essentially impossible


def test_feature_free_predictor_is_biased_below_zero_under_shuffled_labels(data, cfg):
    """The reason the sentinel's rule was made one-sided."""
    small = {**cfg, "permutation_sentinel": {"n_permutations": 20, "max_abs_mean_rho": 0.1}}
    for scheme in ("LOSO", "LOUO"):
        assert intercept_only_under_shuffles("Suturing", scheme, data, small) < -0.15


def test_run_experiment_prefixes_prediction_files_and_records_the_config(data, cfg, tmp_path, monkeypatch):
    """A v2 run must never overwrite the original run's predictions."""
    logged = []
    monkeypatch.setattr(baseline, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(baseline, "log_run", lambda row, path=None: logged.append(row))
    quick = {**load_config(V2_PATH), "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 0}}
    state = {"git_commit": "test", "git_dirty": False}

    run_experiment("Suturing", "LOSO", "regression", "smoothed_5hz", data, quick, state,
                   config_file="configs/baseline_v2.yaml")
    assert (tmp_path / "v2_smoothed_5hz_Suturing_LOSO_regression.csv").exists()
    assert logged[0]["config_file"] == "configs/baseline_v2.yaml"

    run_experiment("Suturing", "LOSO", "regression", "smoothed_5hz", data,
                   {**cfg, "bootstrap": quick["bootstrap"]}, state)
    assert (tmp_path / "smoothed_5hz_Suturing_LOSO_regression.csv").exists()
    assert logged[1]["config_file"] == "configs/baseline.yaml"


def test_feature_free_reference_matches_a_hand_computation(data):
    """Suturing LOSO: predict every held-out trial with its fold's training mean,
    then correlate with the truth. Computed here directly, not through the module."""
    from scipy.stats import spearmanr
    sub = data[data["task"] == "Suturing"]
    fold_mean = {}
    for fold in load_splits("Suturing", "LOSO"):
        mean = sub.loc[fold.train, "grs"].mean()
        for trial in fold.test:
            fold_mean[trial] = mean
    predicted = pd.Series(fold_mean).loc[sub.index]
    expected = spearmanr(sub["grs"], predicted)[0]
    assert feature_free_reference("Suturing", "LOSO", data)["spearman"] == pytest.approx(expected)


def test_feature_free_reference_ignores_the_features_entirely(data):
    assert feature_free_reference("Suturing", "LOUO", data) == feature_free_reference(
        "Suturing", "LOUO", synthetic(data, signal=False))


def test_feature_free_mae_on_a_hand_computation(data):
    """Suturing LOSO: mean absolute gap between each trial's GRS and the mean GRS of
    the other folds' trials, worked out directly."""
    sub = data[data["task"] == "Suturing"]
    misses = []
    for fold in load_splits("Suturing", "LOSO"):
        mean = sub.loc[fold.train, "grs"].mean()
        misses += [abs(sub.loc[t, "grs"] - mean) for t in fold.test]
    assert feature_free_reference("Suturing", "LOSO", data)["mae"] == pytest.approx(np.mean(misses))
