"""Two housekeeping jobs on results/experiments.csv, done once and recorded here.

1. TAG (not delete) the six duplicate `posthoc_time_only_headline` rows produced by a
   verification re-run on 2026-09-20. The re-run genuinely happened, so deleting the rows
   would make the log an inaccurate record of what was executed; but six undifferentiated
   duplicates could be read as independent evidence, which they are not. Tagging in a
   `run_note` column preserves both facts. The two batches carry different git commits
   (4785059 and cad5bfd) and identical values, which also documents that the result
   reproduces across commits.

2. LOG three numbers that were computed and printed but never written to the log, so that
   every number destined for the paper traces to a logged row rather than to prose in
   docs/NOTES.md:
     * Jev's score against each feature, per task (the "Jev tracks task time" claim). ALL
       16 features are logged, not just task time, so the claim cannot be read as
       cherry-picked.
     * Mean stated confidence per path and scheme (the "76% confident / 38% correct" claim).
     * Majority-class rate per task (the "new-surgeon accuracy is at or below always
       guessing the biggest class" claim).

These are addenda to runs that already happened, computed from the same cached
predictions and responses; they are marked as such in `kind` and `run_note` and are not
new experiments.

Run once:  uv run python src/log_addenda.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from baseline import EXPERIMENTS_CSV, feature_columns, git_state, load_config, load_dataset, log_run
from evaluation import assemble, baseline_proba, jev_proba
from jev_analysis import PRIMARY, load_scores, score_vs_features, scores_for
from jev_client import load_jev_config
from loaders import TASKS, load_all_meta

DUPLICATE_NOTE = ("verification re-run 2026-09-20 (identical values, later commit); "
                  "duplicates the earlier batch, NOT independent evidence")
ORIGINAL_NOTE = "original run"
ADDENDUM_NOTE = ("addendum logged 2026-09-20 from cached predictions/responses; "
                 "computed earlier and reported in docs/NOTES.md but not previously logged")


def tag_duplicate_runs(path=EXPERIMENTS_CSV) -> int:
    """Mark the later of two identical posthoc batches as a verification re-run."""
    log = pd.read_csv(path)
    if "run_note" not in log.columns:
        log["run_note"] = pd.NA

    rows = log[log["kind"] == "posthoc_time_only_headline"]
    if rows["timestamp"].nunique() < 2:
        print("no duplicate batch found; nothing tagged")
        return 0

    latest = rows["timestamp"].max()
    is_dup = (log["kind"] == "posthoc_time_only_headline") & (log["timestamp"] == latest)
    is_orig = (log["kind"] == "posthoc_time_only_headline") & (log["timestamp"] != latest)
    log.loc[is_dup, "run_note"] = DUPLICATE_NOTE
    log.loc[is_orig, "run_note"] = ORIGINAL_NOTE
    log.to_csv(path, index=False)
    print(f"tagged {int(is_dup.sum())} rows as a verification re-run, {int(is_orig.sum())} as original")
    return int(is_dup.sum())


def log_addenda() -> None:
    base_cfg, jev_cfg = load_config(), load_jev_config()
    data = load_dataset(base_cfg["features_file"])
    table = load_scores(jev_cfg)
    jev = scores_for(table, PRIMARY, "run0", jev_cfg["primary_run"])
    state = git_state()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"timestamp": stamp, **state, "seed": base_cfg["seed"], "run_note": ADDENDUM_NOTE}

    # 1. Jev's score against every feature, per task. No labels are used anywhere here.
    corr = score_vs_features(jev, data, feature_columns(data))
    for row in corr.itertuples():
        log_run({**common, "config_file": "configs/jev.yaml", "feature_set": "smoothed_5hz",
                 "cv_scheme": "none", "task": row.task, "kind": "addendum_jev_score_vs_feature",
                 "model": "jev-1.13.0", "predictor": f"{PRIMARY} vs {row.feature}", "spearman": row.rho})
    top = corr.loc[corr.groupby("task")["rho"].apply(lambda s: s.abs().idxmax())]
    print("Jev score vs features, strongest per task:")
    for r in top.itertuples():
        print(f"   {r.task:15s} {r.feature:20s} {r.rho:+.2f}")

    # 2. Mean stated confidence per path and scheme.
    for scheme in ("LOUO", "LOSO"):
        frame = assemble(scheme)
        proba = baseline_proba(frame)
        truth = frame["base_class_true"].map({c: i for i, c in enumerate(["N", "I", "E"])}).to_numpy()
        log_run({**common, "config_file": "configs/baseline.yaml", "feature_set": "smoothed_5hz",
                 "cv_scheme": scheme, "task": "pooled", "kind": "addendum_mean_confidence",
                 "model": "baseline", "mean_confidence": float(proba.max(axis=1).mean()),
                 "accuracy": float((proba.argmax(axis=1) == truth).mean()), "n": len(frame)})
        print(f"baseline {scheme}: mean confidence {proba.max(axis=1).mean():.3f}, "
              f"accuracy {(proba.argmax(axis=1) == truth).mean():.3f}")

    frame = assemble("LOUO")                       # Jev is never trained: identical either scheme
    proba = jev_proba(frame)
    truth = frame["jev_level_true"].to_numpy()
    log_run({**common, "config_file": "configs/jev.yaml", "feature_set": "jev_v1", "cv_scheme": "none",
             "task": "pooled", "kind": "addendum_mean_confidence", "model": "jev-1.13.0",
             "mean_confidence": float(proba.max(axis=1).mean()),
             "accuracy": float((proba.argmax(axis=1) == truth).mean()), "n": len(frame)})
    print(f"jev        : mean confidence {proba.max(axis=1).mean():.3f}, "
          f"accuracy {(proba.argmax(axis=1) == truth).mean():.3f}")

    # 3. Majority-class rate per task, beside the new-surgeon accuracy it is compared with.
    meta = load_all_meta()
    log = pd.read_csv(EXPERIMENTS_CSV)
    louo = log[(log["kind"] == "classification") & (log["cv_scheme"] == "LOUO")
               & (log["feature_set"] == "smoothed_5hz")].set_index("task")["accuracy"]
    print("majority-class rate vs new-surgeon accuracy:")
    for task in TASKS:
        sub = meta[meta["task"] == task]
        rate = float(sub["skill"].value_counts(normalize=True).max())
        accuracy = float(louo.loc[task])
        log_run({**common, "config_file": "configs/baseline.yaml", "feature_set": "smoothed_5hz",
                 "cv_scheme": "LOUO", "task": task, "kind": "addendum_majority_class",
                 "model": "always_predict_majority_class", "majority_class_rate": rate,
                 "accuracy": accuracy, "accuracy_minus_majority": accuracy - rate, "n": len(sub)})
        print(f"   {task:15s} majority {rate:.3f}  LOUO accuracy {accuracy:.3f}  "
              f"difference {accuracy - rate:+.3f}")


if __name__ == "__main__":
    tag_duplicate_runs()
    log_addenda()
