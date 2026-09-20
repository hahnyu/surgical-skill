"""Phase 2 figures: each feature against skill level, one task at a time.

The rules for these figures (GRS bands within each task, layout, expected
directions, what may not change afterwards) were fixed and committed in
docs/NOTES.md BEFORE this file existed. Do not change the bands, the features
shown, or any feature or threshold in response to how the figures look.

Run:  uv run python src/plot_features.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from loaders import TASKS, load_all_meta

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"

# Chart colours: the reference palette in the dataviz skill. The three band
# colours are steps 250, 450 and 650 of its blue ramp -- one hue, light to dark,
# because the bands are ordered. Validated with validate_palette.js --ordinal.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]

BAND_ORDER = ["low", "mid", "high"]
SKILL_ORDER = ["N", "I", "E"]
SKILL_NAMES = {"N": "novice", "I": "intermediate", "E": "expert"}


def grs_bands(grs: pd.Series) -> pd.Series:
    """Split one task's GRS scores into low / mid / high bands.

    Cut points are the 33.3rd and 66.7th percentiles of the scores passed in, so
    call this once PER TASK: GRS is distributed differently in each task and a
    pooled cut would mostly separate tasks. low = score <= the lower cut,
    high = score > the upper cut, mid = everything else. Tied scores always land
    in the same band (no arbitrary tie-breaking), so the bands can be unequal in
    size. "High" is a different absolute GRS in each task.
    """
    lower, upper = np.percentile(grs, [100 / 3, 200 / 3])
    bands = pd.Series("mid", index=grs.index)
    bands[grs <= lower] = "low"
    bands[grs > upper] = "high"
    return pd.Categorical(bands, categories=BAND_ORDER, ordered=True)


def load_plot_data() -> pd.DataFrame:
    """Features joined to the labels, one row per trial, with GRS bands."""
    features = pd.read_csv(RESULTS / "features.csv", index_col="trial")
    meta = load_all_meta().set_index("trial")
    data = features.join(meta[["task", "subject", "skill", "grs"]], how="inner")
    if len(data) != len(features):
        raise ValueError("some trials in the feature matrix have no label")

    data["grs_band"] = pd.Series(index=data.index, dtype="object")
    for task in TASKS:
        rows = data["task"] == task
        data.loc[rows, "grs_band"] = np.asarray(grs_bands(data.loc[rows, "grs"]))
    data["grs_band"] = pd.Categorical(data["grs_band"], categories=BAND_ORDER, ordered=True)
    return data


# (column, panel title, multiply by this to display). Arms sit side by side.
PANELS = [
    ("task_time_s", "Task time (s)", 1),
    ("bimanual_speed_corr", "Bimanual speed correlation", 1),
    ("PSM1_path_length", "PSM1 path length (m)", 1),
    ("PSM2_path_length", "PSM2 path length (m)", 1),
    ("PSM1_speed_mean", "PSM1 mean speed (m/s)", 1),
    ("PSM2_speed_mean", "PSM2 mean speed (m/s)", 1),
    ("PSM1_speed_std", "PSM1 speed SD (m/s)", 1),
    ("PSM2_speed_std", "PSM2 speed SD (m/s)", 1),
    ("PSM1_accel_mean", "PSM1 mean acceleration (m/s²)", 1),
    ("PSM2_accel_mean", "PSM2 mean acceleration (m/s²)", 1),
    ("PSM1_idle_fraction", "PSM1 idle fraction", 1),
    ("PSM2_idle_fraction", "PSM2 idle fraction", 1),
    ("PSM1_working_volume", "PSM1 hull volume (cm³)", 1e6),
    ("PSM2_working_volume", "PSM2 hull volume (cm³)", 1e6),
    ("PSM1_gripper_cycles", "PSM1 gripper cycles", 1),
    ("PSM2_gripper_cycles", "PSM2 gripper cycles", 1),
]

FOOTNOTE = ("Each dot is one trial. Box: median and quartiles; whiskers reach 1.5 x IQR. "
            "ρ is the Spearman correlation of the feature with GRS in this task. "
            "Units assumed SI (positions in metres); not verified against the dataset documentation.")


def spearman_table(data: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho between each feature and GRS, per task, with n."""
    rows = []
    for task in TASKS:
        sub = data[data["task"] == task]
        for column, _, _ in PANELS:
            rho = spearmanr(sub[column], sub["grs"])[0]
            rows.append({"task": task, "feature": column, "rho": rho, "n": len(sub)})
    return pd.DataFrame(rows)


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def plot_task(task: str, data: pd.DataFrame, group: str, out_path: Path) -> None:
    """One figure: a 4 x 4 grid of small panels, one per feature, for one task.

    group is "grs_band" (main) or "skill" (secondary). Each panel has its own
    y-axis because the features have different units; nothing is put on a shared
    or a second axis.
    """
    sub = data[data["task"] == task]
    if group == "grs_band":
        order = BAND_ORDER
        names = {b: b for b in BAND_ORDER}
        heading = "GRS band (low / mid / high thirds of this task's scores)"
    else:
        order = SKILL_ORDER
        names = SKILL_NAMES
        heading = "self-declared skill level"

    present = [g for g in order if (sub[group] == g).any()]
    colours = {g: RAMP[order.index(g) * (len(RAMP) - 1) // (len(order) - 1)] for g in order}
    rho = spearman_table(data).query("task == @task").set_index("feature")["rho"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(4, 4, figsize=(13, 11.5), facecolor=SURFACE)
    rng = np.random.default_rng(0)   # fixed seed: the sideways jitter is reproducible

    for ax, (column, title, scale) in zip(axes.ravel(), PANELS):
        _style_axis(ax)
        values = [sub.loc[sub[group] == g, column].to_numpy() * scale for g in present]
        positions = np.arange(len(present))

        boxes = ax.boxplot(
            values, positions=positions, widths=0.5, patch_artist=True,
            showfliers=False, manage_ticks=False,
            medianprops=dict(color=INK, linewidth=2),
            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
            boxprops=dict(linewidth=1.2),
        )
        for g, patch, wh_a, wh_b, cap_a, cap_b in zip(
            present, boxes["boxes"], boxes["whiskers"][::2], boxes["whiskers"][1::2],
            boxes["caps"][::2], boxes["caps"][1::2],
        ):
            patch.set_facecolor(colours[g])
            patch.set_alpha(0.22)
            patch.set_edgecolor(colours[g])
            for line in (wh_a, wh_b, cap_a, cap_b):
                line.set_color(colours[g])
        for g, pos, vals in zip(present, positions, values):
            x = pos + rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(x, vals, s=30, color=colours[g], edgecolor=SURFACE, linewidth=0.8,
                       zorder=3, alpha=0.95)

        counts = [len(v) for v in values]
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{names[g]}\nn={n}" for g, n in zip(present, counts)],
                           color=INK_2, fontsize=8)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
        ax.text(1.0, 1.02, f"ρ = {rho[column]:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK_2)

    fig.suptitle(f"{task.replace('_', ' ')}: features by {heading}", x=0.01, ha="left",
                 fontsize=14, color=INK, y=0.995)
    fig.text(0.01, 0.005, FOOTNOTE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.975), h_pad=2.0, w_pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    data = load_plot_data()

    print("GRS band sizes by task (bands are within-task thirds; ties share a band):")
    print(pd.crosstab(data["task"], data["grs_band"]).to_string())
    print()
    for task in TASKS:
        g = data.loc[data["task"] == task, "grs"]
        by = data[data["task"] == task].groupby("grs_band", observed=True)["grs"].agg(["min", "max"])
        print(f"{task}: GRS range per band ->", {b: (int(r['min']), int(r['max'])) for b, r in by.iterrows()})

    for task in TASKS:
        plot_task(task, data, "grs_band", FIGURES / f"features_by_grs_band_{task}.png")
        plot_task(task, data, "skill", FIGURES / f"features_by_skill_{task}.png")

    table = spearman_table(data)
    table.to_csv(RESULTS / "feature_grs_spearman.csv", index=False)
    print()
    print("wrote 6 figures to", FIGURES)
    print("wrote", RESULTS / "feature_grs_spearman.csv")
