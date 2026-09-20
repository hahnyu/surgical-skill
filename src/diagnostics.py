"""Post-hoc diagnostics for the Phase 3 baseline.

These were NOT part of the pre-declared design in configs/baseline.yaml. They were
added after the first results, to interpret two of them: the leakage sentinel's
negative mean and the near-perfect LOSO skill classification. They change no
model, no setting and no result; they only explain. Log them as post-hoc.

Run:  uv run python src/diagnostics.py
"""

from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import (EXPERIMENTS_CSV, SCHEMES, feature_columns, git_state, intercept_only_under_shuffles,
                      load_config, load_dataset, log_run)
from loaders import TASKS, load_splits


def surgeon_identification_accuracy(task: str, data: pd.DataFrame, cfg: dict) -> dict:
    """Under LOSO, how often do the features alone name the right surgeon?

    The target is the surgeon's letter, not skill and not GRS. Each fold trains on
    the other repetitions of every surgeon and predicts the held-out repetition,
    exactly the situation the LOSO skill classifier is in. Same features, same
    standardization inside the pipeline, same C.
    """
    sub = data[data["task"] == task]
    columns = feature_columns(data)
    correct = 0
    for fold in load_splits(task, "LOSO"):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=cfg["classification"]["C"], max_iter=cfg["classification"]["max_iter"])),
        ])
        model.fit(sub.loc[fold.train, columns], sub.loc[fold.train, "subject"])
        correct += int((model.predict(sub.loc[fold.test, columns]) == sub.loc[fold.test, "subject"]).sum())
    return {
        "accuracy": correct / len(sub),
        "chance": 1 / sub["subject"].nunique(),
        "majority_class_skill_rate": float(sub["skill"].value_counts(normalize=True).max()),
        "n_subjects": int(sub["subject"].nunique()),
    }


def main() -> None:
    cfg = load_config()
    state = git_state()
    data = load_dataset(cfg["features_file"])
    log = pd.read_csv(EXPERIMENTS_CSV)
    primary = log[(log["feature_set"] == cfg["feature_set"]) & (log["config_file"] == "configs/baseline.yaml")]
    stamp = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"config_file": "configs/baseline.yaml", **state, "feature_set": cfg["feature_set"],
              "seed": cfg["seed"], "sklearn_version": sklearn.__version__}

    print("A) Shuffled labels: intercept-only predictor vs the ridge sentinel (mean pooled Spearman)")
    print(f"   {'scheme':5} {'task':15} {'intercept-only':>15} {'ridge sentinel':>15}")
    for scheme in SCHEMES:
        for task in TASKS:
            null = intercept_only_under_shuffles(task, scheme, data, cfg)
            row = primary[(primary["kind"] == "permutation_sentinel") & (primary["cv_scheme"] == scheme)
                          & (primary["task"] == task)].iloc[-1]
            print(f"   {scheme:5} {task:15} {null:>+15.3f} {row['sentinel_mean_rho']:>+15.3f}")
            log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": task,
                     "kind": "diagnostic_intercept_only", "model": "training_mean",
                     "sentinel_mean_rho": null, "sentinel_n_permutations": cfg["permutation_sentinel"]["n_permutations"]})

    print("\nB) Surgeon identification from the features, LOSO (target = surgeon, not skill)")
    print(f"   {'task':15} {'surgeon acc':>12} {'chance':>7} | {'N/I/E acc LOSO':>15} {'N/I/E acc LOUO':>15} {'majority rate':>14}")
    for task in TASKS:
        result = surgeon_identification_accuracy(task, data, cfg)
        get = lambda scheme: primary[(primary["kind"] == "classification") & (primary["cv_scheme"] == scheme)
                                     & (primary["task"] == task)].iloc[-1]["accuracy"]
        print(f"   {task:15} {result['accuracy']:>12.2f} {result['chance']:>7.2f} | "
              f"{get('LOSO'):>15.2f} {get('LOUO'):>15.2f} {result['majority_class_skill_rate']:>14.2f}")
        log_run({"timestamp": stamp(), **common, "cv_scheme": "LOSO", "task": task,
                 "kind": "diagnostic_surgeon_identity", "model": "multinomial_logistic",
                 "accuracy": result["accuracy"], "chance_accuracy": result["chance"],
                 "n_subjects": result["n_subjects"]})


if __name__ == "__main__":
    main()
