"""Phase 4 analysis: how well does Jev's score agree with the expert GRS?

The plan was fixed before any score existed (docs/NOTES.md, "Phase 4 -- Jev: design
PRE-DECLARED"). In that order:
  1. Repeatability: does Jev give the same answer when asked again?
  2. Rank agreement: Spearman of the score with GRS, per task. Jev is never trained, so
     this has none of the cross-validation bias that pulls the baseline's pooled LOUO
     correlation toward -0.8. Do NOT compare it with that number.
  3. The common yardstick: MAE improvement over predicting the mean, as for the baseline.
     Within each training fold a straight line GRS ~ a + b * score is fitted on the
     TRAINING trials only, then applied to the held-out trials.
  4. Direction guard: the sign of the fitted slope is reported for every fold. A negative
     slope means Jev ranks skill backwards. It is reported as that; the line must not be
     allowed to hide it.

Run:  uv run python src/jev_analysis.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from baseline import SCHEMES, git_state, load_config, load_dataset, log_run
from jev_client import SCORES_CSV, load_jev_config
from loaders import TASKS, OSATS_ELEMENTS, load_all_meta, load_splits
from metrics import spearman, subject_bootstrap
from report import headline_row

PRIMARY = "overall_skill"


def load_scores(cfg: dict) -> pd.DataFrame:
    """The parsed responses, after checking they are what the plan said they would be."""
    table = pd.read_csv(SCORES_CSV)
    if set(table["model"]) != {cfg["model"]}:
        raise ValueError(f"responses from unexpected models: {sorted(set(table['model']))}")
    if table["prompt_hash"].nunique() != 1:
        raise ValueError("responses come from more than one prompt")
    expected = 103 * cfg["repeats"]
    if len(table) != expected or table.duplicated(["trial", "run"]).any():
        raise ValueError(f"expected {expected} unique responses, got {len(table)}")
    return table


def scores_for(table: pd.DataFrame, question: str, how: str = "run0", primary_run: int = 0) -> pd.Series:
    """One score per trial: from the primary run, or the mean over all runs."""
    column = f"{question}__score"
    if how == "run0":
        return table[table["run"] == primary_run].set_index("trial")[column]
    if how == "mean_of_runs":
        return table.groupby("trial")[column].mean()
    raise ValueError(f"unknown way of combining runs: {how!r}")


def composite_of_elements(table: pd.DataFrame, primary_run: int = 0) -> pd.Series:
    """Mean of the six OSATS element scores, from the primary run."""
    run = table[table["run"] == primary_run].set_index("trial")
    return run[[f"{e}__score" for e in OSATS_ELEMENTS]].mean(axis=1)


def repeatability(table: pd.DataFrame, question: str) -> dict:
    """How much one trial's score changes between two asks of the same request."""
    wide = table.pivot(index="trial", columns="run", values=f"{question}__score")
    out = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        diff = (wide[a] - wide[b]).abs()
        out[f"{a}v{b}"] = {"spearman": spearman(wide[a], wide[b]),
                           "mean_abs_diff": float(diff.mean()),
                           "max_abs_diff": float(diff.max()),
                           "identical_share": float((diff == 0).mean())}
    return out


def calibrated_predictions(scores: pd.Series, data: pd.DataFrame, task: str, scheme: str) -> pd.DataFrame:
    """Turn Jev's scores into GRS points, fold by fold, using only training trials.

    In each fold an ordinary least-squares line GRS ~ a + b * score is fitted on the
    training trials and applied to the held-out ones. np.polyfit(x, y, 1) returns the
    slope b and intercept a. If the training scores never vary, no line exists and the
    prediction is the training mean, with slope 0. The slope is kept per row so its sign
    can be reported.
    """
    sub = data[data["task"] == task]
    pieces = []
    for fold in load_splits(task, scheme):
        train, test = sub.loc[fold.train], sub.loc[fold.test]
        x_train, y_train = scores.loc[fold.train].to_numpy(float), train["grs"].to_numpy(float)
        if np.ptp(x_train) == 0:
            slope, intercept = 0.0, float(y_train.mean())
        else:
            slope, intercept = np.polyfit(x_train, y_train, 1)
        pieces.append(pd.DataFrame({
            "fold": fold.name, "subject": test["subject"], "y_true": test["grs"],
            "y_pred": intercept + slope * scores.loc[fold.test].to_numpy(float), "slope": slope,
        }, index=test.index))
    result = pd.concat(pieces)
    return result.loc[sub.index]


def rank_agreement(scores: pd.Series, data: pd.DataFrame, task: str, boot: dict) -> dict:
    """Spearman of the score with GRS in one task, with a surgeon-bootstrap interval."""
    sub = data[data["task"] == task]
    r = subject_bootstrap(sub["grs"], scores.loc[sub.index], sub["subject"], spearman,
                          boot["n_resamples"], boot["confidence"], boot["seed"])
    return {"rho": r["estimate"], "low": r["low"], "high": r["high"], "n_subjects": r["n_subjects"]}


def headline_for_scores(scores: pd.Series, data: pd.DataFrame, task: str, scheme: str, boot: dict) -> dict:
    """MAE improvement over predicting the mean, plus the direction guard."""
    oof = calibrated_predictions(scores, data, task, scheme)
    row = headline_row(oof.drop(columns="slope"), data, task, scheme, boot)
    slopes = oof.groupby("fold")["slope"].first()
    row["n_folds"] = len(slopes)
    row["n_negative_slopes"] = int((slopes < 0).sum())
    row["mean_slope"] = float(slopes.mean())
    return row


def main() -> None:
    cfg = load_jev_config()
    base = load_config()
    boot = base["bootstrap"]
    table = load_scores(cfg)
    data = load_dataset(base["features_file"])
    meta = load_all_meta().set_index("trial")
    print(f"{len(table)} responses, model {cfg['model']}, prompt {table['prompt_version'].iloc[0]} "
          f"({table['prompt_hash'].iloc[0]}); input tokens in total: {int(table['input_tokens'].sum()):,}")

    print("\n1) REPEATABILITY of overall_skill (same request asked three times)")
    rep = repeatability(table, PRIMARY)
    for pair, r in rep.items():
        print(f"   runs {pair}: Spearman {r['spearman']:+.3f}  mean |diff| {r['mean_abs_diff']:.3f}  "
              f"max |diff| {r['max_abs_diff']:.3f}  identical {100 * r['identical_share']:.0f}%")
    print("   all seven questions, runs 0 vs 1: mean |score diff| ->",
          {q.replace('__score', ''): round(repeatability(table, q.replace('__score', ''))['0v1']['mean_abs_diff'], 3)
           for q in [c for c in table.columns if c.endswith('__score')]})

    predictors = {
        "overall_skill, run 0 (PRIMARY)": scores_for(table, PRIMARY, "run0", cfg["primary_run"]),
        "overall_skill, mean of 3 runs": scores_for(table, PRIMARY, "mean_of_runs"),
        "mean of six element scores": composite_of_elements(table, cfg["primary_run"]),
    }
    state = git_state()
    stamp = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"config_file": "configs/jev.yaml", **state, "feature_set": "jev_v1", "model": cfg["model"],
              "seed": base["seed"]}

    print("\n2) RANK AGREEMENT with GRS (Spearman, 95% interval resampling surgeons). Not comparable to the")
    print("   baseline's pooled LOUO correlation, which is biased toward -0.8 by construction.")
    for name, scores in predictors.items():
        cells = []
        for task in TASKS:
            r = rank_agreement(scores, data, task, boot)
            cells.append(f"{task} {r['rho']:+.2f} [{r['low']:+.2f},{r['high']:+.2f}]")
            log_run({"timestamp": stamp(), **common, "cv_scheme": "none", "task": task,
                     "kind": "jev_rank_agreement", "predictor": name, "spearman": r["rho"],
                     "spearman_low": r["low"], "spearman_high": r["high"], "n_subjects": r["n_subjects"]})
        print(f"   {name:34s} " + " | ".join(cells))

    print("\n   Element-level, run 0 (EXPLORATORY: the element order rests on the dataset readme's word)")
    print(f"   {'element':26s} " + "  ".join(f"{t:>14}" for t in TASKS))
    run0 = table[table["run"] == cfg["primary_run"]].set_index("trial")
    for element in OSATS_ELEMENTS:
        cells = []
        for task in TASKS:
            sub = data[data["task"] == task]
            rho = spearman(meta.loc[sub.index, element], run0.loc[sub.index, f"{element}__score"])
            cells.append(f"{rho:>+14.2f}")
        print(f"   {element:26s} " + "  ".join(cells))

    print("\n3+4) MAE IMPROVEMENT over predicting the mean, with the DIRECTION GUARD (LOUO is the headline)")
    print("   improvement in GRS points, positive = better; slope = fitted GRS points per score level")
    print(f"   {'predictor':34s} {'scheme':5} {'task':15} {'ref MAE':>7} {'model MAE':>9} {'improvement [95%]':>24} "
          f"{'beats':>5} | {'neg. slopes':>11} {'mean slope':>10}")
    for name, scores in predictors.items():
        for scheme in ("LOUO", "LOSO"):
            for task in TASKS:
                r = headline_for_scores(scores, data, task, scheme, boot)
                print(f"   {name:34s} {scheme:5} {task:15} {r['reference_mae']:>7.2f} {r['model_mae']:>9.2f} "
                      f"{r['improvement']:>+8.2f} [{r['improvement_low']:+.2f},{r['improvement_high']:+.2f}] "
                      f"{'YES' if r['beats_reference'] else 'no':>5} | {r['n_negative_slopes']}/{r['n_folds']:<9} "
                      f"{r['mean_slope']:>+10.2f}")
                log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": task,
                         "kind": "jev_headline_mae_vs_reference", "predictor": name,
                         "mae": r["model_mae"], "reference_mae": r["reference_mae"],
                         "mae_improvement": r["improvement"], "mae_improvement_low": r["improvement_low"],
                         "mae_improvement_high": r["improvement_high"], "beats_reference": r["beats_reference"],
                         "n_negative_slopes": r["n_negative_slopes"], "n_folds": r["n_folds"],
                         "mean_slope": r["mean_slope"], "n_subjects": r["n_subjects"]})


if __name__ == "__main__":
    main()
