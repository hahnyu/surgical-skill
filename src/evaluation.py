"""Phase 5: calibration, selective prediction and agreement, for both prediction paths.

The design (what is compared with what, every metric, the interpretation rule) was fixed
and committed in docs/NOTES.md, "Phase 5 -- evaluation: design PRE-DECLARED", before any
number here existed. Do not change bins, coverage levels or thresholds in response to
results.

Two paths, each with one confidence per trial:
  * baseline: ridge GRS prediction; the logistic classifier's top-class probability is
    its confidence (two baseline models combined, declared in advance).
  * jev: GRS points from the per-fold line on Jev's score; Jev's own `confidence`.

Run:  uv run python src/evaluation.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pingouin as pg
import sklearn
from netcal.metrics import ECE, MCE
from scipy.stats import kendalltau

from baseline import (CLASS_ORDER, PREDICTIONS_DIR, feature_free_predictions, git_state, load_config,
                      load_dataset, log_run)
from jev_analysis import PRIMARY, calibrated_predictions, load_scores, scores_for
from jev_client import load_jev_config
from loaders import TASKS
from metrics import mae, spearman, subject_bootstrap

N_BINS = 5                            # reliability bins, fixed in advance
COVERAGES = (1.0, 0.8, 0.6, 0.4)      # fixed in advance
JEV_LEVELS = 5


# --- truth for Jev's levels ---------------------------------------------------------------


def true_level(grs) -> np.ndarray:
    """The 0-4 level a GRS corresponds to on the 1-5 scale Jev's levels mirror.

    GRS is the sum of six elements scored 1-5, so GRS / 6 is the mean element score on
    that scale. Round to the nearest whole score (halves round up), clip to 1-5, and
    shift to 0-4 to match Jev's level numbering.
    """
    level = np.floor(np.asarray(grs, dtype=float) / 6 + 0.5)
    return (np.clip(level, 1, 5) - 1).astype(int)


# --- calibration ---------------------------------------------------------------------------


def multiclass_brier(proba, truth) -> float:
    """Mean over trials of the summed squared gap between the probabilities and the
    outcome (1 for the true class, 0 for the others). 0 is perfect; predicting the same
    probability for every class scores (k - 1) / k."""
    proba = np.asarray(proba, dtype=float)
    onehot = np.eye(proba.shape[1])[np.asarray(truth, dtype=int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_table(proba, truth, n_bins: int = N_BINS) -> pd.DataFrame:
    """Top-label reliability: in each confidence bin, the average confidence and the
    share of predictions that were right. A calibrated model has the two equal."""
    proba = np.asarray(proba, dtype=float)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(truth, dtype=int)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.minimum(np.digitize(confidence, edges[1:-1]), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = which == b
        rows.append({"bin": b, "low": edges[b], "high": edges[b + 1], "n": int(in_bin.sum()),
                     "mean_confidence": float(confidence[in_bin].mean()) if in_bin.any() else np.nan,
                     "accuracy": float(correct[in_bin].mean()) if in_bin.any() else np.nan})
    return pd.DataFrame(rows)


def ece_mce(proba, truth, n_bins: int = N_BINS) -> tuple:
    """Expected and maximum calibration error, computed directly from the bins.

    ECE = sum over bins of (share of trials in the bin) * |accuracy - confidence|;
    MCE = the largest |accuracy - confidence| over non-empty bins. Verified equal to
    netcal's values on the real data (tests), and used for the bootstrap resamples.
    """
    table = reliability_table(proba, truth, n_bins)
    table = table[table["n"] > 0]
    gap = (table["accuracy"] - table["mean_confidence"]).abs()
    return float((table["n"] / table["n"].sum() * gap).sum()), float(gap.max())


def netcal_ece_mce(proba, truth, n_bins: int = N_BINS) -> tuple:
    """The same two numbers from the netcal library (the pre-declared implementation)."""
    proba, truth = np.asarray(proba, dtype=float), np.asarray(truth, dtype=int)
    return float(ECE(bins=n_bins).measure(proba, truth)), float(MCE(bins=n_bins).measure(proba, truth))


def _cluster_bootstrap(n: int, subjects, statistic, boot: dict) -> dict:
    """Surgeon-cluster bootstrap for a statistic that needs several arrays.

    subject_bootstrap resamples whole surgeons and hands the statistic the trials it
    drew. We pass the ROW POSITIONS as the data, so the statistic can index into any
    number of arrays held outside.
    """
    positions = np.arange(n, dtype=float)
    return subject_bootstrap(positions, np.zeros(n), subjects, lambda p, _: statistic(p.astype(int)),
                             boot["n_resamples"], boot["confidence"], boot["seed"])


def calibration_report(proba, truth, subjects, boot: dict, reference_proba) -> dict:
    """Accuracy, ECE, MCE, Brier and Brier skill score, with surgeon-bootstrap intervals.

    reference_proba is a best-constant forecast (the class frequencies of all trials):
    strong, because it uses the test labels. The Brier skill score is
    1 - Brier / reference Brier; positive means better than the constant.
    """
    proba, truth = np.asarray(proba, dtype=float), np.asarray(truth, dtype=int)
    reference = np.asarray(reference_proba, dtype=float)
    ece, mce = netcal_ece_mce(proba, truth)
    brier, ref_brier = multiclass_brier(proba, truth), multiclass_brier(reference, truth)
    out = {"n": len(truth), "accuracy": float((proba.argmax(axis=1) == truth).mean()),
           "mean_confidence": float(proba.max(axis=1).mean()),
           "ece": ece, "mce": mce, "brier": brier, "brier_reference": ref_brier,
           "brier_skill": 1 - brier / ref_brier}
    stats = {
        "ece": lambda i: ece_mce(proba[i], truth[i])[0],
        "brier": lambda i: multiclass_brier(proba[i], truth[i]),
        "brier_skill": lambda i: 1 - multiclass_brier(proba[i], truth[i]) / multiclass_brier(reference[i], truth[i]),
        "accuracy": lambda i: float((proba[i].argmax(axis=1) == truth[i]).mean()),
    }
    for name, fn in stats.items():
        r = _cluster_bootstrap(len(truth), subjects, fn, boot)
        out[f"{name}_low"], out[f"{name}_high"] = r["low"], r["high"]
    return out


# --- selective prediction -------------------------------------------------------------------


def cumulative_risk(errors, confidence) -> np.ndarray:
    """Average error of the k most confident trials, for k = 1..n.

    Sorted by confidence, highest first. Ties keep their original order (the data are
    ordered by trial name), which is arbitrary but deterministic.
    """
    errors, confidence = np.asarray(errors, dtype=float), np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    return np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)


def aurc(errors, confidence) -> float:
    """Area under the risk-coverage curve: the mean of the cumulative risks."""
    return float(cumulative_risk(errors, confidence).mean())


def oracle_aurc(errors) -> float:
    """The best possible AURC: defer the largest errors first."""
    return aurc(errors, -np.asarray(errors, dtype=float))


def usefulness(errors, confidence) -> float:
    """Overall MAE minus AURC. Positive means deferring the least confident trials
    lowers the error of what remains, compared with deferring at random."""
    return float(np.mean(errors) - aurc(errors, confidence))


def coverage_table(errors, reference_errors, confidence, coverages=COVERAGES) -> pd.DataFrame:
    """For each coverage, the error of the retained trials and its gain over the reference.

    Coverage 0.8 keeps the 80% most confident trials. The reference errors are for the
    SAME retained trials, so the comparison is like with like.
    """
    errors, reference_errors = np.asarray(errors, dtype=float), np.asarray(reference_errors, dtype=float)
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    rows = []
    for c in coverages:
        keep = order[: int(round(c * len(errors)))]
        rows.append({"coverage": c, "n_kept": len(keep), "mae": float(errors[keep].mean()),
                     "reference_mae": float(reference_errors[keep].mean()),
                     "improvement": float(reference_errors[keep].mean() - errors[keep].mean())})
    return pd.DataFrame(rows)


def selective_report(errors, reference_errors, confidence, subjects, boot: dict) -> dict:
    """Usefulness of the confidence (with an interval), AURC, oracle, E-AURC, coverage table."""
    errors, confidence = np.asarray(errors, dtype=float), np.asarray(confidence, dtype=float)
    a, o = aurc(errors, confidence), oracle_aurc(errors)
    r = _cluster_bootstrap(len(errors), subjects, lambda i: usefulness(errors[i], confidence[i]), boot)
    return {"overall_mae": float(errors.mean()), "aurc": a, "oracle_aurc": o, "e_aurc": a - o,
            "usefulness": usefulness(errors, confidence),
            "usefulness_low": r["low"], "usefulness_high": r["high"],
            "informative": bool(r["low"] > 0),
            "coverage": coverage_table(errors, reference_errors, confidence)}


# --- agreement ------------------------------------------------------------------------------


def icc_2_1(truth, prediction) -> dict:
    """ICC(2,1): two-way random, absolute agreement, single rater. In pingouin 0.6.1 the
    row is named ICC(A,1). The interval treats trials as independent, so it is indicative."""
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    n = len(truth)
    long = pd.DataFrame({"target": np.tile(np.arange(n), 2), "rater": ["truth"] * n + ["model"] * n,
                         "rating": np.concatenate([truth, prediction])})
    row = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="rating").set_index("Type").loc["ICC(A,1)"]
    return {"icc": float(row["ICC"]), "icc_low": float(row["CI95"][0]), "icc_high": float(row["CI95"][1])}


def bland_altman(truth, prediction) -> dict:
    """Bias and 95% limits of agreement of (prediction - truth)."""
    diff = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return {"bias": bias, "sd": sd, "lower": bias - 1.96 * sd, "upper": bias + 1.96 * sd}


def agreement_row(truth, prediction) -> dict:
    return {"spearman": spearman(truth, prediction),
            "kendall": float(kendalltau(truth, prediction)[0]),
            "mae": mae(truth, prediction), **icc_2_1(truth, prediction), **bland_altman(truth, prediction)}


# --- assembling the two paths -----------------------------------------------------------------


def assemble(scheme: str) -> pd.DataFrame:
    """One row per trial (103), sorted by trial name, with everything both paths need."""
    base_cfg, jev_cfg = load_config(), load_jev_config()
    data = load_dataset(base_cfg["features_file"])
    jev = load_scores(jev_cfg)
    jev0 = jev[jev["run"] == jev_cfg["primary_run"]].set_index("trial")
    jev_score = scores_for(jev, PRIMARY, "run0", jev_cfg["primary_run"])

    pieces = []
    for task in TASKS:
        stem = f"smoothed_5hz_{task}_{scheme}"
        reg = pd.read_csv(PREDICTIONS_DIR / f"{stem}_regression.csv", index_col=0)
        cls = pd.read_csv(PREDICTIONS_DIR / f"{stem}_classification.csv", index_col=0)
        jev_pred = calibrated_predictions(jev_score, data, task, scheme)
        free = feature_free_predictions(task, scheme, data)
        piece = pd.DataFrame({
            "task": task, "subject": reg["subject"], "grs": reg["y_true"],
            "base_pred": reg["y_pred"], "free_pred": free.loc[reg.index],
            "jev_pred": jev_pred.loc[reg.index, "y_pred"],
            "base_class_true": cls["y_true"], "base_class_pred": cls["y_pred"],
            "base_conf": cls[[f"proba_{c}" for c in CLASS_ORDER]].max(axis=1),
        })
        for c in CLASS_ORDER:
            piece[f"base_proba_{c}"] = cls[f"proba_{c}"]
        pieces.append(piece)
    out = pd.concat(pieces).sort_index()
    out["jev_conf"] = jev0.loc[out.index, f"{PRIMARY}__confidence"]
    out["jev_score"] = jev0.loc[out.index, f"{PRIMARY}__score"]
    for k in range(JEV_LEVELS):
        out[f"jev_p{k}"] = jev0.loc[out.index, f"{PRIMARY}__p{k}"]
    out["jev_maxp"] = out[[f"jev_p{k}" for k in range(JEV_LEVELS)]].max(axis=1)
    out["jev_level_true"] = true_level(out["grs"])
    if len(out) != 103:
        raise ValueError(f"expected 103 trials, got {len(out)}")
    return out


def baseline_proba(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"base_proba_{c}" for c in CLASS_ORDER]].to_numpy()


def jev_proba(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"jev_p{k}" for k in range(JEV_LEVELS)]].to_numpy()


def best_constant(truth, n_classes: int, n_rows: int) -> np.ndarray:
    """The forecast that gives every trial the class frequencies of all the trials."""
    freq = np.bincount(np.asarray(truth, dtype=int), minlength=n_classes) / len(truth)
    return np.tile(freq, (n_rows, 1))


def main() -> None:
    base_cfg = load_config()
    boot = base_cfg["bootstrap"]
    state = git_state()
    stamp = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"config_file": "configs/baseline.yaml + configs/jev.yaml", **state, "seed": base_cfg["seed"],
              "sklearn_version": sklearn.__version__}
    frames = {scheme: assemble(scheme) for scheme in ("LOUO", "LOSO")}

    print("A) AGREEMENT per task (GRS points; ICC = ICC(2,1); Spearman/Kendall/MAE as before)")
    print(f"   {'scheme':5} {'task':15} {'path':9} {'Spearman':>8} {'Kendall':>8} {'ICC(2,1) [95%, indicative]':>28} {'MAE':>6} {'bias':>6} {'limits of agreement':>21}")
    for scheme in ("LOUO", "LOSO"):
        for task in TASKS:
            sub = frames[scheme][frames[scheme]["task"] == task]
            for path, col in (("baseline", "base_pred"), ("jev", "jev_pred"), ("mean-only", "free_pred")):
                r = agreement_row(sub["grs"], sub[col])
                print(f"   {scheme:5} {task:15} {path:9} {r['spearman']:>+8.2f} {r['kendall']:>+8.2f} "
                      f"{r['icc']:>+8.2f} [{r['icc_low']:+.2f},{r['icc_high']:+.2f}]{'':>7} {r['mae']:>6.2f} {r['bias']:>+6.2f} "
                      f"[{r['lower']:+.1f}, {r['upper']:+.1f}]")
                log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": task, "kind": "eval_agreement",
                         "model": path, **{k: v for k, v in r.items()}})

    print("\nB) CALIBRATION, each path on its own truth (5 bins; pooled 103 trials; surgeon-bootstrap 95%)")
    print(f"   {'path / scheme':22} {'acc':>5} {'ECE [95%]':>20} {'MCE':>5} {'Brier':>6} {'const.':>6} {'skill [95%]':>20}")
    calibrations = {}
    for scheme in ("LOUO", "LOSO"):
        f = frames[scheme]
        truth = f["base_class_true"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
        calibrations[("baseline", scheme)] = (baseline_proba(f), truth,
                                              calibration_report(baseline_proba(f), truth, f["subject"], boot,
                                                                 best_constant(truth, 3, len(f))))
    f = frames["LOUO"]                                         # Jev is never trained: same for both schemes
    truth = f["jev_level_true"].to_numpy()
    calibrations[("jev", "both")] = (jev_proba(f), truth,
                                     calibration_report(jev_proba(f), truth, f["subject"], boot,
                                                        best_constant(truth, JEV_LEVELS, len(f))))
    for (path, scheme), (_, _, r) in calibrations.items():
        print(f"   {path + ' / ' + scheme:22} {r['accuracy']:>5.2f} {r['ece']:>7.3f} [{r['ece_low']:.3f},{r['ece_high']:.3f}] "
              f"{r['mce']:>5.2f} {r['brier']:>6.3f} {r['brier_reference']:>6.3f} {r['brier_skill']:>+7.3f} [{r['brier_skill_low']:+.3f},{r['brier_skill_high']:+.3f}]")
        log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": "pooled", "kind": "eval_calibration",
                 "model": path, **r})
    jev_p, jev_truth, _ = calibrations[("jev", "both")]
    print(f"   Jev within-one-level accuracy: {float((np.abs(jev_p.argmax(axis=1) - jev_truth) <= 1).mean()):.2f}; "
          f"true levels present: {dict(pd.Series(jev_truth).value_counts().sort_index())}; "
          f"Jev top levels used: {dict(pd.Series(jev_p.argmax(axis=1)).value_counts().sort_index())}")

    print("\nC) SELECTIVE PREDICTION on GRS error (pooled 103; usefulness = overall MAE - AURC; positive = confidence helps)")
    print(f"   {'path / scheme':22} {'overall MAE':>11} {'AURC':>6} {'oracle':>7} {'E-AURC':>7} {'usefulness [95%]':>22} {'informative':>11}")
    selective = {}
    signals = (("baseline", "base_pred", "base_conf"), ("jev", "jev_pred", "jev_conf"), ("jev (max prob)", "jev_pred", "jev_maxp"))
    for scheme in ("LOUO", "LOSO"):
        f = frames[scheme]
        for name, pred_col, conf_col in signals:
            err = (f[pred_col] - f["grs"]).abs().to_numpy()
            ref_err = (f["free_pred"] - f["grs"]).abs().to_numpy()
            r = selective_report(err, ref_err, f[conf_col].to_numpy(), f["subject"], boot)
            selective[(name, scheme)] = r
            print(f"   {name + ' / ' + scheme:22} {r['overall_mae']:>11.2f} {r['aurc']:>6.2f} {r['oracle_aurc']:>7.2f} {r['e_aurc']:>7.2f} "
                  f"{r['usefulness']:>+8.2f} [{r['usefulness_low']:+.2f},{r['usefulness_high']:+.2f}] {'YES' if r['informative'] else 'no':>11}")
            log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": "pooled", "kind": "eval_selective",
                     "model": name, **{k: v for k, v in r.items() if k != "coverage"}})

    print("\n   Headline: error of the trials KEPT as the least confident are deferred (improvement over the mean-only reference)")
    print(f"   {'path / scheme':22} " + "  ".join(f"{int(c * 100):>3}% cover: MAE / gain" for c in COVERAGES))
    for (name, scheme), r in selective.items():
        if name == "jev (max prob)":
            continue
        cells = "  ".join(f"{row.mae:>5.2f} / {row.improvement:>+5.2f}    " for row in r["coverage"].itertuples())
        print(f"   {name + ' / ' + scheme:22} {cells}")
        for row in r["coverage"].itertuples():
            log_run({"timestamp": stamp(), **common, "cv_scheme": scheme, "task": "pooled", "kind": "eval_coverage",
                     "model": name, "coverage": row.coverage, "n_kept": row.n_kept, "mae": row.mae,
                     "reference_mae": row.reference_mae, "mae_improvement": row.improvement})
    return frames, calibrations, selective


if __name__ == "__main__":
    main()
