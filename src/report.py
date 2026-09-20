"""The headline table: how much better than predicting the average, on new surgeons.

Decided 2026-09-20 (docs/NOTES.md): the headline is LOUO, measured as the model's
average miss (MAE, in GRS points) relative to a feature-free reference that predicts
each held-out trial with the mean GRS of its fold's training trials. The pooled
Spearman is shown next to ITS reference, never alone, because under LOUO that
reference scores about -0.8 by construction.

Interpretation rule, fixed BEFORE the intervals were computed: a model beats the
reference across new surgeons only if the lower end of its 95% interval for the MAE
improvement is above zero. With 7-8 surgeons the interval is crude, so "excludes
zero" is suggestive, not proof, and nothing is selected by it.

Run:  uv run python src/report.py
"""

from datetime import datetime, timezone

import pandas as pd
import sklearn

from baseline import (PREDICTIONS_DIR, PROJECT_ROOT, SCHEMES, feature_free_predictions, git_state,
                      load_config, load_dataset, log_run)
from loaders import TASKS
from metrics import mae_improvement, spearman

RESULTS = PROJECT_ROOT / "results"
# (label, config file, prefix on the saved prediction files)
CONFIGS = [("v1", "configs/baseline.yaml", ""), ("v2", "configs/baseline_v2.yaml", "v2_")]
ORDER = ["LOUO", "LOSO"]      # LOUO first: it is the headline


def headline_row(oof: pd.DataFrame, data: pd.DataFrame, task: str, scheme: str, boot: dict) -> dict:
    """One row: the model against the feature-free reference, on the same trials."""
    reference = feature_free_predictions(task, scheme, data).loc[oof.index]
    improvement = mae_improvement(oof["y_true"], oof["y_pred"], reference, oof["subject"],
                                  n_resamples=boot["n_resamples"], confidence=boot["confidence"],
                                  seed=boot["seed"])
    return {
        "task": task,
        "cv_scheme": scheme,
        **improvement,
        "beats_reference": bool(improvement["improvement_low"] > 0),
        "model_rho": spearman(oof["y_true"], oof["y_pred"]),
        "reference_rho": spearman(oof["y_true"], reference),
    }


def build_table() -> pd.DataFrame:
    boot = load_config()["bootstrap"]
    rows = []
    for label, config_file, prefix in CONFIGS:
        cfg = load_config(PROJECT_ROOT / config_file)
        feature_sets = {cfg["feature_set"]: cfg["features_file"], **cfg["sensitivity_feature_files"]}
        for feature_set, feature_file in feature_sets.items():
            data = load_dataset(feature_file)
            for scheme in ORDER:
                for task in TASKS:
                    oof = pd.read_csv(PREDICTIONS_DIR / f"{prefix}{feature_set}_{task}_{scheme}_regression.csv",
                                      index_col=0)
                    rows.append({"config": label, "config_file": config_file, "feature_set": feature_set,
                                 **headline_row(oof, data, task, scheme, boot)})
    return pd.DataFrame(rows)


def main() -> None:
    table = build_table()
    table.to_csv(RESULTS / "headline_table.csv", index=False)

    primary = table[table["feature_set"] == "smoothed_5hz"]
    print("MAE improvement over predicting the training mean (GRS points; positive = model better)")
    print("LOUO is the headline. Interval: 95%, resampling surgeons.\n")
    print(f"{'scheme':5} {'task':15} {'cfg':3} {'ref MAE':>8} {'model MAE':>10} {'improvement [95% interval]':>32}  {'beats?':>6} | {'rho model':>9} {'rho ref':>8}")
    for _, r in primary.iterrows():
        print(f"{r['cv_scheme']:5} {r['task']:15} {r['config']:3} {r['reference_mae']:>8.2f} {r['model_mae']:>10.2f} "
              f"{r['improvement']:>+12.2f} [{r['improvement_low']:+.2f}, {r['improvement_high']:+.2f}]"
              f"{'':>4}{'YES' if r['beats_reference'] else 'no':>6} | {r['model_rho']:>+9.2f} {r['reference_rho']:>+8.2f}")

    state = git_state()
    for _, r in table.iterrows():
        log_run({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "config_file": r["config_file"], **state, "feature_set": r["feature_set"],
                 "cv_scheme": r["cv_scheme"], "task": r["task"], "kind": "headline_mae_vs_reference",
                 "model": "ridge", "sklearn_version": sklearn.__version__,
                 "mae": r["model_mae"], "reference_mae": r["reference_mae"],
                 "mae_improvement": r["improvement"], "mae_improvement_low": r["improvement_low"],
                 "mae_improvement_high": r["improvement_high"], "beats_reference": r["beats_reference"],
                 "spearman": r["model_rho"], "reference_rho": r["reference_rho"], "n_subjects": r["n_subjects"]})
    print(f"\nwrote {len(table)} rows to {RESULTS / 'headline_table.csv'} and logged them to experiments.csv")


if __name__ == "__main__":
    main()
