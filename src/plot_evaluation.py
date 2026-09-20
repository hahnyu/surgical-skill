"""Phase 5 figures: reliability diagrams, risk-coverage curves, Bland-Altman plots.

Colours follow the dataviz skill's reference palette: two categorical slots (blue for the
baseline, orange for Jev), validated for two series, with the reference and oracle lines in
neutral ink. Every series is also labelled directly, so identity never rests on colour.

Run:  uv run python src/plot_evaluation.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (CLASS_ORDER, JEV_LEVELS, assemble, baseline_proba, bland_altman, cumulative_risk,
                        ece_mce, jev_proba, reliability_table)
from loaders import TASKS
from plot_features import BASELINE as AXIS_LINE, GRID, INK, INK_2, MUTED, SURFACE, FIGURES

BLUE, ORANGE = "#2a78d6", "#eb6834"        # baseline, Jev
PATH_COLOUR = {"baseline": BLUE, "jev": ORANGE}
PATH_LABEL = {"baseline": "Baseline (ridge + logistic)", "jev": "Jev (jev-1.13.0)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def _fonts():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def reliability_figure(frames: dict, out) -> None:
    """Top-label confidence against accuracy, five bins, one panel per path and scheme."""
    _fonts()
    truth_b = {s: frames[s]["base_class_true"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy() for s in frames}
    panels = [
        ("baseline", "LOUO", baseline_proba(frames["LOUO"]), truth_b["LOUO"], "Baseline, new surgeon (LOUO)\nagainst self-declared skill N/I/E"),
        ("baseline", "LOSO", baseline_proba(frames["LOSO"]), truth_b["LOSO"], "Baseline, known surgeon (LOSO)\nagainst self-declared skill N/I/E"),
        ("jev", "both", jev_proba(frames["LOUO"]), frames["LOUO"]["jev_level_true"].to_numpy(), "Jev, never trained\nagainst the level implied by GRS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), facecolor=SURFACE)
    for ax, (path, _, proba, truth, title) in zip(axes, panels):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        table = reliability_table(proba, truth)
        filled = table[table["n"] > 0]
        ax.plot(filled["mean_confidence"], filled["accuracy"], color=PATH_COLOUR[path], linewidth=1.6, zorder=2)
        ax.scatter(filled["mean_confidence"], filled["accuracy"], s=filled["n"] * 9 + 25, color=PATH_COLOUR[path],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for r in filled.itertuples():
            offset = (9, -11) if r.accuracy > 0.1 else (9, 8)         # keep labels off the bottom axis
            ax.annotate(f"n={r.n}", (r.mean_confidence, r.accuracy), textcoords="offset points", xytext=offset,
                        fontsize=8, color=INK_2)
        ece, mce = ece_mce(proba, truth)
        overall = float((proba.argmax(axis=1) == truth).mean())
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        ax.text(0.02, 0.96, f"ECE {ece:.2f}   accuracy {overall:.2f}", transform=ax.transAxes, va="top", fontsize=9, color=INK_2)
        ax.text(0.98, 0.04, "perfectly calibrated", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED, rotation=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("stated confidence (top class)", color=INK_2, fontsize=9)
        ax.set_ylabel("share actually right" if path == "baseline" and title.startswith("Baseline, new") else "", color=INK_2, fontsize=9)
    fig.suptitle("When the model is confident, is it right?  Points below the dashed line are overconfident.",
                 x=0.01, ha="left", fontsize=13, color=INK, y=0.99)
    fig.text(0.01, 0.012, "Marker area shows how many trials fall in the bin (5 equal-width bins, 103 trials pooled over tasks). "
             "Different truths on the left and right, so the panels are not a head-to-head comparison.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def risk_coverage_figure(frames: dict, out) -> None:
    """Average error of the trials kept, as the least confident are deferred to a human."""
    _fonts()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), facecolor=SURFACE, sharey=True)
    n = 103
    coverage = np.arange(1, n + 1) / n
    for r, path in enumerate(("baseline", "jev")):
        for c, scheme in enumerate(("LOUO", "LOSO")):
            ax = axes[r, c]
            _style(ax)
            f = frames[scheme]
            pred, conf = (("base_pred", "base_conf") if path == "baseline" else ("jev_pred", "jev_conf"))
            err = (f[pred] - f["grs"]).abs().to_numpy()
            by_conf = cumulative_risk(err, f[conf].to_numpy())
            ideal = cumulative_risk(err, -err)
            ax.plot(coverage, ideal, color=MUTED, linestyle=(0, (5, 3)), linewidth=1.3)
            ax.axhline(err.mean(), color=INK_2, linestyle=(0, (1, 2)), linewidth=1.3)
            ax.plot(coverage, by_conf, color=PATH_COLOUR[path], linewidth=2)
            tag = dict(fontsize=8.5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=5)
            ax.text(0.66, err.mean() + 0.55, "random order (overall MAE)", ha="center", color=INK_2, **tag)
            ax.text(0.66, ideal[int(0.66 * n)] - 0.65, "ideal order (oracle)", ha="center", color=MUTED, **tag)
            ax.text(0.30, 5.4, "coloured line: ordered by the path's own confidence", ha="left", color=INK, **tag)
            if by_conf.max() > 6.2:
                ax.text(0.02, 6.0, f"first trials reach {by_conf.max():.1f}", ha="left", color=INK_2, **tag)
            ax.set_title(f"{PATH_LABEL[path]}, {'new surgeon (LOUO)' if scheme == 'LOUO' else 'known surgeon (LOSO)'}",
                         loc="left", fontsize=10, color=INK, pad=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 6.2)
            ax.set_xlabel("coverage: share of trials kept (least confident deferred first)" if r == 1 else "", color=INK_2, fontsize=9)
            ax.set_ylabel("average error of kept trials (GRS points)" if c == 0 else "", color=INK_2, fontsize=9)
    fig.suptitle("Does deferring the least confident trials lower the error of the rest?  Barely: the coloured line hugs the random reference.",
                 x=0.01, ha="left", fontsize=13, color=INK, y=0.995)
    fig.text(0.01, 0.005, "Absolute GRS error, 103 trials pooled over tasks. The coloured curve should fall below the dotted line as coverage drops if "
             "confidence is informative; the dashed line is the best any ordering could do.", fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def bland_altman_figure(frames: dict, out) -> None:
    """Prediction minus truth against their mean, per task, on new surgeons (LOUO)."""
    _fonts()
    f = frames["LOUO"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), facecolor=SURFACE, sharey=True)
    for r, path in enumerate(("baseline", "jev")):
        col = "base_pred" if path == "baseline" else "jev_pred"
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            _style(ax)
            sub = f[f["task"] == task]
            mean = (sub[col] + sub["grs"]) / 2
            diff = sub[col] - sub["grs"]
            ba = bland_altman(sub["grs"], sub[col])
            ax.scatter(mean, diff, s=34, color=PATH_COLOUR[path], edgecolor=SURFACE, linewidth=0.8, zorder=3, alpha=0.95)
            ax.axhline(0, color=AXIS_LINE, linewidth=1)
            ax.axhline(ba["bias"], color=INK, linewidth=1.6)
            for y in (ba["lower"], ba["upper"]):
                ax.axhline(y, color=INK_2, linestyle=(0, (4, 3)), linewidth=1.1)
            ax.text(0.99, 0.97, f"bias {ba['bias']:+.1f}   limits {ba['lower']:+.1f} to {ba['upper']:+.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
            ax.set_title(f"{PATH_LABEL[path]}: {task.replace('_', ' ')}", loc="left", fontsize=10, color=INK, pad=8)
            ax.set_xlabel("mean of prediction and truth (GRS)" if r == 1 else "", color=INK_2, fontsize=9)
            ax.set_ylabel("prediction minus truth (GRS points)" if c == 0 else "", color=INK_2, fontsize=9)
    fig.suptitle("Agreement on new surgeons: predictions are pulled toward the middle (the downward slope), and the limits are wide except in Knot-Tying.",
                 x=0.01, ha="left", fontsize=13, color=INK, y=0.995)
    fig.text(0.01, 0.005, "Solid line: mean difference (bias). Dashed: 95% limits of agreement (bias +/- 1.96 SD). One dot per trial.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    frames = {s: assemble(s) for s in ("LOUO", "LOSO")}
    FIGURES.mkdir(parents=True, exist_ok=True)
    reliability_figure(frames, FIGURES / "eval_reliability.png")
    risk_coverage_figure(frames, FIGURES / "eval_risk_coverage.png")
    bland_altman_figure(frames, FIGURES / "eval_bland_altman_LOUO.png")
    print("wrote 3 figures to", FIGURES)


if __name__ == "__main__":
    main()
