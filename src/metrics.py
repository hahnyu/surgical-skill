"""Agreement metrics and their uncertainty.

Every statistic here is computed from out-of-fold predictions: each trial was
predicted once by a model that never saw it. Uncertainty comes from resampling
whole SURGEONS, not trials: the 4-5 trials one surgeon contributes to a task are
not independent of each other, so treating them as independent would give
intervals that are too narrow.

What the interval does and does not cover (state this in the paper): it reflects
which surgeons happened to be in the evaluation. It does not reflect how the
result would move if the model were refit on different training data, because the
predictions are held fixed and nothing is refit.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

# The ordering of the self-declared skill labels: novice < intermediate < expert.
SKILL_ORDER = {"N": 0, "I": 1, "E": 2}


def _undefined(a, b) -> bool:
    """A correlation is undefined when either input never varies."""
    return np.ptp(a) == 0 or np.ptp(b) == 0


def spearman(y_true, y_pred) -> float:
    """Rank correlation (Spearman rho). NaN if either input is constant."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.nan if _undefined(y_true, y_pred) else float(spearmanr(y_true, y_pred)[0])


def kendall(y_true, y_pred) -> float:
    """Rank correlation (Kendall tau-b). NaN if either input is constant."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.nan if _undefined(y_true, y_pred) else float(kendalltau(y_true, y_pred)[0])


def mae(y_true, y_pred) -> float:
    """Mean absolute error: the average size of the miss, in GRS points."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error: like MAE but punishes big misses more."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred) -> float:
    """Average of the per-class recalls, so a big class cannot dominate."""
    return float(balanced_accuracy_score(y_true, y_pred))


def quadratic_kappa(y_true, y_pred) -> float:
    """Quadratic-weighted Cohen's kappa on the ordered skill labels N < I < E.

    Chance-corrected agreement in which predicting N for an expert costs four
    times as much as predicting I for an expert.
    """
    a = [SKILL_ORDER[v] for v in y_true]
    b = [SKILL_ORDER[v] for v in y_pred]
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def subject_bootstrap(
    y_true,
    y_pred,
    subjects,
    statistic,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Estimate, and a percentile interval from resampling whole surgeons.

    Each resample draws surgeons WITH replacement (as many as there are surgeons),
    takes every trial of every drawn surgeon, and recomputes the statistic. The
    interval is the middle `confidence` share of those values.

    Resamples where the statistic is undefined (NaN) are skipped and counted, not
    silently dropped: a large skipped count means the interval is not reliable.
    """
    y_true, y_pred, subjects = np.asarray(y_true), np.asarray(y_pred), np.asarray(subjects)
    if not (len(y_true) == len(y_pred) == len(subjects)):
        raise ValueError("y_true, y_pred and subjects must be the same length")

    unique = np.unique(subjects)
    rows_of = {s: np.flatnonzero(subjects == s) for s in unique}
    rng = np.random.default_rng(seed)

    values = []
    skipped = 0
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_of[s] for s in drawn])
        value = statistic(y_true[rows], y_pred[rows])
        if np.isnan(value):
            skipped += 1
        else:
            values.append(value)

    tail = (1 - confidence) / 2 * 100
    if values:
        low, high = np.percentile(values, [tail, 100 - tail])
    else:
        low = high = np.nan
    return {
        "estimate": statistic(y_true, y_pred),
        "low": float(low),
        "high": float(high),
        "n_skipped": skipped,
        "n_subjects": len(unique),
    }


def _with_interval(prefix, result, out):
    out[prefix] = result["estimate"]
    out[f"{prefix}_low"] = result["low"]
    out[f"{prefix}_high"] = result["high"]
    out[f"{prefix}_boot_skipped"] = result["n_skipped"]


def regression_report(y_true, y_pred, subjects, n_resamples=2000, confidence=0.95, seed=0) -> dict:
    """Spearman and Kendall (with intervals), plus MAE and RMSE for context."""
    out = {"n_trials": len(y_true), "n_subjects": len(np.unique(subjects))}
    for name, fn in (("spearman", spearman), ("kendall", kendall)):
        result = subject_bootstrap(y_true, y_pred, subjects, fn, n_resamples, confidence, seed)
        _with_interval(name, result, out)
    out["mae"] = mae(y_true, y_pred)
    out["rmse"] = rmse(y_true, y_pred)
    return out


def classification_report(y_true, y_pred, subjects, n_resamples=2000, confidence=0.95, seed=0) -> dict:
    """Accuracy, balanced accuracy and quadratic kappa (with intervals)."""
    out = {"n_trials": len(y_true), "n_subjects": len(np.unique(subjects))}
    for name, fn in (
        ("accuracy", accuracy),
        ("balanced_accuracy", balanced_accuracy),
        ("quadratic_kappa", quadratic_kappa),
    ):
        result = subject_bootstrap(y_true, y_pred, subjects, fn, n_resamples, confidence, seed)
        _with_interval(name, result, out)
    return out
