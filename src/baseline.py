"""Phase 3 baseline: hand-crafted kinematic features + scikit-learn.

Every setting lives in configs/baseline.yaml and was fixed before the first model
ran. Nothing here selects features, tunes on the test fold, or pools across tasks.

How leakage is prevented (each is checked by a test in tests/test_baseline.py):
  * A fresh model is built for every fold and fit only on that fold's training
    trials. The StandardScaler sits inside the model (a scikit-learn Pipeline), so
    its mean and spread are computed from training rows only.
  * Ridge's alpha is chosen inside the training set, by leave-one-SUBJECT-out, so
    the search cannot be rewarded for recognising a surgeon's style.
  * Test-fold labels are never passed to anything that fits.

Run everything:  uv run python src/baseline.py
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loaders import TASKS, load_all_meta, load_splits
from metrics import classification_report, regression_report, spearman

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"
EXPERIMENTS_CSV = PROJECT_ROOT / "results" / "experiments.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

LABEL_COLUMNS = ["task", "subject", "skill", "grs"]
CLASS_ORDER = ["N", "I", "E"]      # novice < intermediate < expert
SCHEMES = ["LOSO", "LOUO"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(feature_file) -> pd.DataFrame:
    """Features joined to the labels: one row per trial, indexed by trial name.

    The feature columns are whatever the matrix holds (16 of them); task, subject,
    skill and grs are added alongside and are NEVER used as model inputs (see
    feature_columns).
    """
    features = pd.read_csv(PROJECT_ROOT / feature_file, index_col="trial")
    overlap = set(features.columns) & set(LABEL_COLUMNS)
    if overlap:
        raise ValueError(f"the feature file contains label columns: {sorted(overlap)}")
    meta = load_all_meta().set_index("trial")[LABEL_COLUMNS]
    data = features.join(meta, how="inner")
    if len(data) != len(features) or len(data) != 103:
        raise ValueError(f"expected 103 labelled trials, got {len(data)} of {len(features)}")
    return data


def feature_columns(data: pd.DataFrame) -> list:
    """The model inputs: every column except the labels and identifiers."""
    return [c for c in data.columns if c not in LABEL_COLUMNS]


def make_regressor(cfg: dict) -> GridSearchCV:
    """Ridge on standardized features, with alpha chosen by leave-one-subject-out.

    Ridge regression is linear regression with a penalty on large coefficients;
    alpha sets how heavy the penalty is. GridSearchCV tries each alpha in the grid,
    scores it by leaving one surgeon out at a time inside the training set (the
    `groups` passed to fit), and finally refits the winning alpha on the whole
    training set. The scaler is inside the pipeline, so it is refit within every
    one of those inner splits too.
    """
    if not cfg["standardize"]:
        raise NotImplementedError("the pre-declared design always standardizes")
    pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    return GridSearchCV(
        pipeline,
        param_grid={"model__alpha": cfg["alpha_grid"]},
        cv=LeaveOneGroupOut(),
        scoring=cfg["inner_scoring"],
    )


def make_classifier(cfg: dict) -> Pipeline:
    """Multinomial logistic regression on standardized features, C fixed."""
    if not cfg["standardize"]:
        raise NotImplementedError("the pre-declared design always standardizes")
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"],
                                     class_weight=cfg["class_weight"])),
    ])


def cross_validate(task: str, scheme: str, kind: str, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Out-of-fold predictions for one task under LOSO or LOUO.

    Every trial of the task is predicted exactly once, by a model that was fit
    without it (and without, under LOUO, any trial from the same surgeon).
    Returns one row per trial: fold, subject, the truth, the prediction, and for
    regression the alpha that fold chose, for classification the class probabilities.
    """
    if kind not in ("regression", "classification"):
        raise ValueError(f"kind must be 'regression' or 'classification', got {kind!r}")
    sub = data[data["task"] == task]
    columns = feature_columns(data)

    pieces = []
    for fold in load_splits(task, scheme):
        train, test = sub.loc[fold.train], sub.loc[fold.test]
        out = pd.DataFrame({"fold": fold.name, "subject": test["subject"]}, index=test.index)

        if kind == "regression":
            model = make_regressor(cfg["regression"])
            model.fit(train[columns], train["grs"], groups=train["subject"])
            out["y_true"] = test["grs"]
            out["y_pred"] = model.predict(test[columns])
            out["alpha"] = model.best_params_["model__alpha"]
        else:
            model = make_classifier(cfg["classification"])
            model.fit(train[columns], train["skill"])
            proba = pd.DataFrame(model.predict_proba(test[columns]),
                                 columns=model.classes_, index=test.index)
            # A class absent from a training fold gets probability 0, not a missing column.
            proba = proba.reindex(columns=CLASS_ORDER, fill_value=0.0)
            out["y_true"] = test["skill"]
            out["y_pred"] = model.predict(test[columns])
            for c in CLASS_ORDER:
                out[f"proba_{c}"] = proba[c]
        pieces.append(out)

    result = pd.concat(pieces)
    if len(result) != len(sub) or not result.index.is_unique:
        raise ValueError(f"{task} {scheme}: not every trial was predicted exactly once")
    return result.loc[sub.index]


def permutation_sentinel(task: str, scheme: str, data: pd.DataFrame, cfg: dict) -> dict:
    """Leakage check: refit under SHUFFLED labels.

    Shuffling GRS across a task's trials destroys any real link between features
    and skill. A leak-free pipeline then has nothing to find, so its pooled
    Spearman should average near zero. A clearly positive mean would mean
    something in the pipeline is seeing information it should not.
    """
    settings = cfg["permutation_sentinel"]
    mask = data["task"] == task
    rhos = []
    for i in range(settings["n_permutations"]):
        shuffled = data.copy()
        rng = np.random.default_rng(cfg["seed"] + i)
        shuffled.loc[mask, "grs"] = rng.permutation(data.loc[mask, "grs"].to_numpy())
        oof = cross_validate(task, scheme, "regression", shuffled, cfg)
        rhos.append(spearman(oof["y_true"], oof["y_pred"]))
    rhos = np.array(rhos)
    return {
        "mean_rho": float(np.nanmean(rhos)),
        "sd_rho": float(np.nanstd(rhos)),
        "n_permutations": len(rhos),
        "passed": bool(abs(np.nanmean(rhos)) <= settings["max_abs_mean_rho"]),
    }


def git_state() -> dict:
    """The commit the code was at, and whether uncommitted changes existed."""
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    return {"git_commit": run("rev-parse", "--short", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def log_run(row: dict, path: Path = EXPERIMENTS_CSV) -> None:
    """Append one row to the experiments log, creating the file if needed.

    New columns are allowed (regression and classification report different
    metrics); rows keep their original values and gain empty cells as needed.
    """
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def run_experiment(task, scheme, kind, feature_set, data, cfg, state) -> dict:
    """One task, one scheme, one model: predict, evaluate, save, log."""
    oof = cross_validate(task, scheme, kind, data, cfg)
    boot = cfg["bootstrap"]
    args = dict(n_resamples=boot["n_resamples"], confidence=boot["confidence"], seed=boot["seed"])

    if kind == "regression":
        metrics = regression_report(oof["y_true"], oof["y_pred"], oof["subject"], **args)
        counts = oof.groupby("fold")["alpha"].first().value_counts().sort_index()
        extra = {"model": "ridge", "alpha_counts": ";".join(f"{a:g}:{n}" for a, n in counts.items())}
    else:
        metrics = classification_report(oof["y_true"], oof["y_pred"], oof["subject"], **args)
        extra = {"model": "multinomial_logistic", "alpha_counts": ""}

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(PREDICTIONS_DIR / f"{feature_set}_{task}_{scheme}_{kind}.csv")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": "configs/baseline.yaml",
        **state,
        "feature_set": feature_set,
        "cv_scheme": scheme,
        "task": task,
        "kind": kind,
        "seed": cfg["seed"],
        "sklearn_version": sklearn.__version__,
        **extra,
        **metrics,
    }
    log_run(row)
    return row


def main() -> None:
    cfg = load_config()
    state = git_state()
    if state["git_dirty"]:
        print("WARNING: uncommitted changes; this run is not exactly reproducible from the commit.")

    feature_sets = {cfg["feature_set"]: cfg["features_file"]}
    feature_sets.update({name: path for name, path in cfg["sensitivity_feature_files"].items()})

    for feature_set, feature_file in feature_sets.items():
        data = load_dataset(feature_file)
        print(f"\n=== feature set: {feature_set} ({feature_file}) ===")
        for scheme in SCHEMES:
            for task in TASKS:
                reg = run_experiment(task, scheme, "regression", feature_set, data, cfg, state)
                cls = run_experiment(task, scheme, "classification", feature_set, data, cfg, state)
                print(f"{scheme} {task:15s} GRS rho={reg['spearman']:+.2f} "
                      f"[{reg['spearman_low']:+.2f}, {reg['spearman_high']:+.2f}]  "
                      f"MAE={reg['mae']:.2f} | N/I/E acc={cls['accuracy']:.2f} "
                      f"bal.acc={cls['balanced_accuracy']:.2f} kappa={cls['quadratic_kappa']:+.2f}")

    print("\n=== leakage sentinel (shuffled labels, primary features) ===")
    data = load_dataset(cfg["features_file"])
    for scheme in SCHEMES:
        for task in TASKS:
            result = permutation_sentinel(task, scheme, data, cfg)
            log_run({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config_file": "configs/baseline.yaml", **state,
                "feature_set": cfg["feature_set"], "cv_scheme": scheme, "task": task,
                "kind": "permutation_sentinel", "model": "ridge", "seed": cfg["seed"],
                "sklearn_version": sklearn.__version__,
                "sentinel_mean_rho": result["mean_rho"], "sentinel_sd_rho": result["sd_rho"],
                "sentinel_n_permutations": result["n_permutations"], "sentinel_passed": result["passed"],
            })
            print(f"{scheme} {task:15s} mean rho={result['mean_rho']:+.3f} (sd {result['sd_rho']:.3f}) "
                  f"over {result['n_permutations']} shuffles -> {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
