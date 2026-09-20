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

from baseline import EXPERIMENTS_CSV, PREDICTIONS_DIR, feature_columns, git_state, load_config, load_dataset, log_run
from diagnostics import surgeon_identification_predictions
from evaluation import assemble, baseline_proba, jev_proba
from jev_analysis import PRIMARY, load_scores, score_vs_features, scores_for
from jev_client import load_jev_config
from loaders import TASKS, load_all_meta
from metrics import subject_bootstrap

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


def log_narrative_numbers() -> None:
    """Log every number that reached the Methods/Results draft with only a NOTES.md entry.

    Each was computed from committed code and reported in prose; this puts it in the log so
    no number in the paper is orphaned. Grouped by the claim it supports.
    """
    import numpy as np
    from features import (acceleration, count_movements, load_feature_config, smoothness, speed,
                          step_lengths)
    from evaluation import JEV_LEVELS, assemble, jev_proba, reliability_table, baseline_proba
    from jev_analysis import repeatability
    from loaders import load_all_meta, load_kinematics

    base_cfg, jev_cfg = load_config(), load_jev_config()
    fcfg = load_feature_config()
    state = git_state()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"timestamp": stamp, **state, "seed": base_cfg["seed"], "run_note": ADDENDUM_NOTE,
              "feature_set": "smoothed_5hz"}
    meta = load_all_meta()

    # (a) The self-declared label does not track GRS: per-surgeon mean GRS beside the label.
    print("per-surgeon mean GRS vs self-declared label:")
    for (subject, skill), group in meta.groupby(["subject", "skill"]):
        log_run({**common, "config_file": "configs/baseline.yaml", "cv_scheme": "none",
                 "task": "pooled", "kind": "addendum_label_vs_grs", "model": "none",
                 "subject": subject, "skill_label": skill, "mean_grs": float(group["grs"].mean()),
                 "n": len(group)})
        print(f"   surgeon {subject} ({skill}): mean GRS {group['grs'].mean():.1f} over {len(group)} trials")

    # (b) Speed from finite differences against the recorded velocity channels.
    trial = "Suturing_B001"
    kin = load_kinematics(trial, windowed=True)
    for arm in ("PSM1", "PSM2"):
        pos = kin[[f"{arm}_{a}" for a in "xyz"]].to_numpy()
        recorded = np.linalg.norm(kin[[f"{arm}_v{a}" for a in "xyz"]].to_numpy(), axis=1)[1:]
        fd = step_lengths(pos) * 30
        rho = float(np.corrcoef(fd, recorded)[0, 1])
        log_run({**common, "config_file": "configs/features.yaml", "cv_scheme": "none", "task": "Suturing",
                 "kind": "addendum_speed_validation", "model": arm, "trial": trial,
                 "pearson_fd_vs_recorded": rho, "ratio_of_means": float(fd.mean() / recorded.mean()),
                 "n": len(fd)})
        print(f"speed validation {arm}: r={rho:.3f}, mean ratio {fd.mean()/recorded.mean():.3f}")

    # (c) Why three features were removed before modelling.
    jumps, sparc_rows = [], []
    for _, row in meta.iterrows():
        df = load_kinematics(row.trial, windowed=True)
        for arm in ("PSM1", "PSM2"):
            pos = df[[f"{arm}_{a}" for a in "xyz"]].to_numpy()
            jumps.append({"task": row.task, "max_step_m": float(step_lengths(pos).max())})
            sparc_rows.append({"sparc": smoothness(pos, cutoff_hz=fcfg["sparc_cutoff_hz"],
                                                   amp_threshold=fcfg["sparc_amp_threshold"],
                                                   padlevel=fcfg["sparc_padlevel"]),
                               "n_movements": count_movements(pos, fcfg["moving_speed_threshold"],
                                                              fcfg["min_movement_duration_s"]),
                               "duration_s": len(df) / 30})
    jumps = pd.DataFrame(jumps)
    log_run({**common, "config_file": "configs/features.yaml", "cv_scheme": "none", "task": "pooled",
             "kind": "addendum_position_jumps", "model": "none",
             "max_step_mm": float(jumps["max_step_m"].max() * 1000),
             "n_arm_trials_above_5mm": int((jumps["max_step_m"] > 0.005).sum()),
             "n_arm_trials_above_10mm": int((jumps["max_step_m"] > 0.010).sum()),
             "n_arm_trials_above_20mm": int((jumps["max_step_m"] > 0.020).sum()), "n": len(jumps)})
    print(f"largest single-frame step: {jumps['max_step_m'].max()*1000:.1f} mm; "
          f"{(jumps['max_step_m'] > 0.005).sum()} of {len(jumps)} arm-trials above 5 mm")

    sp = pd.DataFrame(sparc_rows)
    from metrics import spearman as sp_rho
    log_run({**common, "config_file": "configs/features.yaml", "cv_scheme": "none", "task": "pooled",
             "kind": "addendum_sparc_confound", "model": "none",
             "spearman_sparc_vs_movements": sp_rho(sp["sparc"], sp["n_movements"]),
             "spearman_sparc_vs_duration": sp_rho(sp["sparc"], sp["duration_s"]), "n": len(sp)})
    print(f"SPARC vs movement count {sp_rho(sp['sparc'], sp['n_movements']):+.2f}, "
          f"vs duration {sp_rho(sp['sparc'], sp['duration_s']):+.2f}")

    # (d) Per-bin reliability, the detail behind the ECE figures.
    for label, scheme in (("baseline", "LOUO"), ("baseline", "LOSO"), ("jev", "LOUO")):
        frame = assemble(scheme)
        if label == "baseline":
            proba = baseline_proba(frame)
            truth = frame["base_class_true"].map({c: i for i, c in enumerate(["N", "I", "E"])}).to_numpy()
        else:
            proba, truth = jev_proba(frame), frame["jev_level_true"].to_numpy()
        for r in reliability_table(proba, truth).itertuples():
            if r.n:
                log_run({**common, "config_file": "configs/baseline.yaml + configs/jev.yaml",
                         "cv_scheme": scheme if label == "baseline" else "none", "task": "pooled",
                         "kind": "addendum_reliability_bin", "model": label, "bin": r.bin,
                         "bin_low": r.low, "bin_high": r.high, "n": r.n,
                         "mean_confidence": r.mean_confidence, "accuracy": r.accuracy})
        # (e) For Jev only: how close the misses are, and where its levels sit.
        if label == "jev":
            top = proba.argmax(axis=1)
            log_run({**common, "config_file": "configs/jev.yaml", "feature_set": "jev_v1",
                     "cv_scheme": "none", "task": "pooled", "kind": "addendum_jev_level_agreement",
                     "model": "jev-1.13.0", "accuracy": float((top == truth).mean()),
                     "within_one_level_accuracy": float((np.abs(top - truth) <= 1).mean()),
                     "mean_predicted_level": float(top.mean()), "mean_true_level": float(truth.mean()),
                     "n": len(truth)})
            for level in range(JEV_LEVELS):
                log_run({**common, "config_file": "configs/jev.yaml", "feature_set": "jev_v1",
                         "cv_scheme": "none", "task": "pooled", "kind": "addendum_jev_level_usage",
                         "model": "jev-1.13.0", "level": level,
                         "n_predicted": int((top == level).sum()), "n_true": int((truth == level).sum())})
            print(f"Jev within-one-level {float((np.abs(top - truth) <= 1).mean()):.2f}; "
                  f"mean predicted level {top.mean():.2f} vs true {truth.mean():.2f}")

    # (f) Repeatability of the zero-shot model across three identical requests.
    table = load_scores(jev_cfg)
    print("repeatability:")
    for question in [c.replace("__score", "") for c in table.columns if c.endswith("__score")]:
        for pair, r in repeatability(table, question).items():
            log_run({**common, "config_file": "configs/jev.yaml", "feature_set": "jev_v1",
                     "cv_scheme": "none", "task": "pooled", "kind": "addendum_jev_repeatability",
                     "model": "jev-1.13.0", "predictor": question, "run_pair": pair,
                     "spearman": r["spearman"], "mean_abs_diff": r["mean_abs_diff"],
                     "max_abs_diff": r["max_abs_diff"], "identical_share": r["identical_share"]})
        if question == "overall_skill":
            for pair, r in repeatability(table, question).items():
                print(f"   overall_skill {pair}: rho {r['spearman']:.3f}, mean |diff| "
                      f"{r['mean_abs_diff']:.3f}, identical {r['identical_share']:.0%}")


def already_logged(kind: str) -> bool:
    """True if this addendum is already in the log. Keeps the script safe to re-run:
    the duplicate-row problem this file exists to fix must not be recreated by it."""
    log = pd.read_csv(EXPERIMENTS_CSV)
    return kind in set(log["kind"])


def log_accuracy_intervals() -> None:
    """Surgeon-cluster intervals for every accuracy that appears in Figure 1.

    Panels A and B report accuracies; with 8 surgeons a bare percentage implies more
    precision than the sample supports. Three things are logged:
      * classification accuracy per task and scheme, with an interval;
      * surgeon-identification accuracy per task, with an interval;
      * the LOSO-minus-LOUO gap, bootstrapped PAIRED on the same resampled surgeons,
        which is the right way to put uncertainty on the paper's headline claim.
    Predictions are held fixed, so an interval reflects which surgeons were evaluated,
    not how the result would move if the model were refit.
    """
    base_cfg = load_config()
    boot = base_cfg["bootstrap"]
    data = load_dataset(base_cfg["features_file"])
    state = git_state()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"timestamp": stamp, **state, "seed": base_cfg["seed"], "run_note": ADDENDUM_NOTE,
              "config_file": "configs/baseline.yaml", "feature_set": "smoothed_5hz"}
    args = (boot["n_resamples"], boot["confidence"], boot["seed"])
    share_correct = lambda correct, _: float(correct.mean())

    print("classification accuracy with 95% surgeon-bootstrap intervals:")
    correct_by = {}
    for scheme in ("LOSO", "LOUO"):
        for task in TASKS:
            oof = pd.read_csv(PREDICTIONS_DIR / f"smoothed_5hz_{task}_{scheme}_classification.csv", index_col=0)
            correct = (oof["y_true"] == oof["y_pred"]).astype(float)
            correct_by[(scheme, task)] = (correct, oof["subject"])
            r = subject_bootstrap(correct, np.zeros(len(correct)), oof["subject"], share_correct, *args)
            log_run({**common, "cv_scheme": scheme, "task": task, "kind": "addendum_classification_ci",
                     "model": "multinomial_logistic", "accuracy": r["estimate"],
                     "accuracy_low": r["low"], "accuracy_high": r["high"], "n": len(oof)})
            print(f"   {scheme} {task:15s} {r['estimate']:.3f} [{r['low']:.3f}, {r['high']:.3f}]")

    print("surgeon identification accuracy with intervals:")
    for task in TASKS:
        per_trial = surgeon_identification_predictions(task, data, base_cfg)
        r = subject_bootstrap(per_trial["correct"], np.zeros(len(per_trial)), per_trial["subject"],
                              share_correct, *args)
        chance = 1 / per_trial["subject"].nunique()
        log_run({**common, "cv_scheme": "LOSO", "task": task, "kind": "addendum_surgeon_identity_ci",
                 "model": "multinomial_logistic", "accuracy": r["estimate"], "accuracy_low": r["low"],
                 "accuracy_high": r["high"], "chance_accuracy": chance, "n": len(per_trial)})
        print(f"   {task:15s} {r['estimate']:.3f} [{r['low']:.3f}, {r['high']:.3f}]  chance {chance:.3f}")

    print("the protocol gap (LOSO minus LOUO), paired on the same resampled surgeons:")
    for task in TASKS:
        loso, subjects = correct_by[("LOSO", task)]
        louo, _ = correct_by[("LOUO", task)]
        gap = (loso - louo.reindex(loso.index)).astype(float)
        r = subject_bootstrap(gap, np.zeros(len(gap)), subjects, lambda g, _: float(g.mean()), *args)
        log_run({**common, "cv_scheme": "LOSO_minus_LOUO", "task": task, "kind": "addendum_protocol_gap",
                 "model": "multinomial_logistic", "accuracy_gap": r["estimate"],
                 "accuracy_gap_low": r["low"], "accuracy_gap_high": r["high"], "n": len(gap)})
        print(f"   {task:15s} {r['estimate']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]")


if __name__ == "__main__":
    tag_duplicate_runs()
    if already_logged("addendum_jev_score_vs_feature"):
        print("addenda already logged; skipping (this script is safe to re-run)")
    else:
        log_addenda()
    if already_logged("addendum_classification_ci"):
        print("accuracy intervals already logged; skipping")
    else:
        log_accuracy_intervals()
    if already_logged("addendum_label_vs_grs"):
        print("narrative numbers already logged; skipping")
    else:
        log_narrative_numbers()
