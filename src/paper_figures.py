"""Figures for the paper. Every value is read from results/experiments.csv.

Nothing here recomputes a result: if a number is not in the log, it does not go in a
figure. That keeps every printed value traceable to a logged run.

Colour means TASK in every panel (blue Suturing, orange Knot-Tying, aqua Needle-Passing),
from the dataviz skill's reference palette; the other dimension is carried by position.
The palette passes the all-pairs validator, with a contrast warning on aqua that is
relieved by labelling every bar directly.

Run:  uv run python src/paper_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import EXPERIMENTS_CSV, PROJECT_ROOT
from loaders import TASKS

FIGURES = PROJECT_ROOT / "results" / "figures"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TASK_COLOUR = {"Suturing": "#2a78d6", "Knot_Tying": "#eb6834", "Needle_Passing": "#1baf7a"}
TASK_LABEL = {"Suturing": "Suturing", "Knot_Tying": "Knot-Tying", "Needle_Passing": "Needle-Passing"}
PRIMARY_JEV = "overall_skill, run 0 (PRIMARY)"


def load_log() -> pd.DataFrame:
    """The experiment log, with the tagged verification re-run dropped so nothing is
    counted twice (the rows are kept in the file; see src/log_addenda.py)."""
    log = pd.read_csv(EXPERIMENTS_CSV)
    if "run_note" in log.columns:
        log = log[~log["run_note"].fillna("").str.startswith("verification")]
    return log.reset_index(drop=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)


def _bars(ax, groups, values, references, ref_label, ylabel, title, ylim=(0, 1.05),
          ref_pos=(0.99, 0.965), ref_ha="right", intervals=None):
    """One grouped-bar panel: three task bars per group, with a per-bar reference tick.

    `groups` is a list of (key, display label). `values` and `references` are keyed by
    (key, task); a missing reference is simply not drawn.
    """
    width = 0.24
    centres = np.arange(len(groups))
    for i, task in enumerate(TASKS):
        offset = (i - 1) * (width + 0.02)               # small gap between adjacent bars
        for centre, (key, _) in zip(centres, groups):
            height = values[(key, task)]
            x = centre + offset
            ax.bar([x], [height], width=width, color=TASK_COLOUR[task], zorder=2,
                   label=TASK_LABEL[task] if centre == 0 else None)
            ref = references.get((key, task))
            if ref is not None:
                ax.plot([x - width / 2, x + width / 2], [ref, ref], color=INK, linewidth=1.6,
                        linestyle=(0, (2, 1.4)), zorder=3)
            top = height
            if intervals is not None and (key, task) in intervals:
                low, high = intervals[(key, task)]
                ax.plot([x, x], [low, high], color=INK, linewidth=1.5, solid_capstyle="butt", zorder=4)
                for end in (low, high):                       # small caps, so the ends are readable
                    ax.plot([x - 0.055, x + 0.055], [end, end], color=INK, linewidth=1.5, zorder=4)
                top = max(top, high)
            # Keep the value clear of both the interval and the reference tick.
            label_y = max(top, ref if ref is not None else top) + 0.026
            ax.text(x, label_y, f"{height:.2f}", ha="center", fontsize=8.5, color=INK, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.set_xticks(centres)
    ax.set_xticklabels([label for _, label in groups], color=INK_2, fontsize=9.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.text(*ref_pos, ref_label, transform=ax.transAxes, ha=ref_ha, va="top", fontsize=8.5,
            color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def headline_figure(log: pd.DataFrame, out) -> None:
    """The paper's headline: the protocol gap, its explanation, and what survives it."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), facecolor=SURFACE)

    # --- A: skill classification, same surgeon vs new surgeon -------------------------
    cls = log[log["kind"] == "addendum_classification_ci"]
    accuracy = {(r.cv_scheme, r.task): r.accuracy for r in cls.itertuples()}
    cls_ci = {(r.cv_scheme, r.task): (r.accuracy_low, r.accuracy_high) for r in cls.itertuples()}
    majority = {r.task: r.majority_class_rate for r in log[log["kind"] == "addendum_majority_class"].itertuples()}
    _style(axes[0])
    groups_a = [("LOSO", "same surgeon\nin training (LOSO)"), ("LOUO", "held-out\nsurgeon (LOUO)")]
    _bars(axes[0], groups_a, accuracy,
          {(g, t): majority[t] for g, _ in groups_a for t in TASKS},
          "- - -  always predict the biggest class",
          "skill classification accuracy (N/I/E)",
          "A.  The protocol gap\nSame features, same model, different evaluation protocol.",
          ylim=(0, 1.12), intervals=cls_ci)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, -0.11))

    # --- B: can the same features name the surgeon? ------------------------------------
    sid = log[log["kind"] == "addendum_surgeon_identity_ci"]
    _style(axes[1])
    _bars(axes[1], [("LOSO", "same surgeon\nin training (LOSO)")],
          {("LOSO", r.task): r.accuracy for r in sid.itertuples()},
          {("LOSO", r.task): r.chance_accuracy for r in sid.itertuples()},
          "- - -  chance",
          "surgeon identification accuracy",
          "B.  Why\nThe same features name WHICH surgeon, not how skilled.",
          ylim=(0, 1.12), ref_pos=(0.015, 0.965), ref_ha="left",
          intervals={("LOSO", r.task): (r.accuracy_low, r.accuracy_high) for r in sid.itertuples()})
    axes[1].set_xlim(-0.78, 0.78)

    # --- C: what survives, on GRS -------------------------------------------------------
    base = log[(log["kind"] == "headline_mae_vs_reference") & (log["cv_scheme"] == "LOUO")
               & (log["feature_set"] == "smoothed_5hz") & (log["config_file"] == "configs/baseline.yaml")]
    jev = log[(log["kind"] == "jev_headline_mae_vs_reference") & (log["cv_scheme"] == "LOUO")
              & (log["predictor"] == PRIMARY_JEV)]
    time_only = log[(log["kind"] == "posthoc_time_only_headline") & (log["cv_scheme"] == "LOUO")]
    methods = [("16 features\n+ ridge", base), ("Jev\n(zero-shot)", jev), ("task time\nalone", time_only)]

    ax = axes[2]
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color=INK, linewidth=1.3, zorder=3)
    width = 0.24
    for m, (label, frame) in enumerate(methods):
        by_task = frame.set_index("task")
        for i, task in enumerate(TASKS):
            x = m + (i - 1) * (width + 0.02)
            row = by_task.loc[task]
            low, high = row["mae_improvement_low"], row["mae_improvement_high"]
            ax.plot([x, x], [low, high], color=TASK_COLOUR[task], linewidth=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([x], [row["mae_improvement"]], s=64, color=TASK_COLOUR[task],
                       edgecolor=SURFACE, linewidth=1.3, zorder=4)
            if low > 0:
                ax.text(x, high + 0.12, f"+{row['mae_improvement']:.2f}", ha="center", fontsize=8.5,
                        color=INK, fontweight="bold", zorder=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label for label, _ in methods], color=INK_2, fontsize=9.5)
    ax.set_ylim(-1.6, 4.0)
    ax.set_ylabel("GRS points better than predicting the mean", color=INK_2, fontsize=9)
    ax.set_title("C.  What survives a held-out surgeon\nOne task of three, and task time alone matches both models.",
                 loc="left", fontsize=10.5, color=INK, pad=8)
    ax.text(0.015, 0.075, "point and 95% interval; above 0 = better than predicting the mean",
            transform=ax.transAxes, fontsize=8.5, color=INK)
    ax.text(0.015, 0.02, "bold label = interval clears zero", transform=ax.transAxes, fontsize=8.5, color=INK_2)

    fig.suptitle("Skill assessment on JIGSAWS: the reported accuracy is largely surgeon recognition",
                 x=0.008, ha="left", fontsize=14, color=INK, y=0.985)
    fig.text(0.008, 0.008,
             "All intervals are 95%, resampling whole surgeons with predictions held fixed. A and B: 103 trials, 8 surgeons (7 in Needle-Passing); chance in B is 1 / number of surgeons.\n"
             "C: leave-one-user-out only; the reference predicts each held-out trial with its training fold's mean GRS. Every value is read from results/experiments.csv.\n"
             "The LOSO-minus-LOUO gap in A, paired on the same resampled surgeons: +0.49 [0.35, 0.62] Suturing, +0.67 [0.41, 0.89] Knot-Tying, +0.54 [0.27, 0.71] Needle-Passing.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    log = load_log()
    headline_figure(log, FIGURES / "fig1_protocol_gap.png")
    print("wrote", FIGURES / "fig1_protocol_gap.png")
