"""Generate the paper's tables from the logged values, into paper/tables.md.

Same rule as src/paper_figures.py: nothing is recomputed here. If a number is not in
results/experiments.csv it does not appear in a table, so every cell traces to a logged
run. The one exception is Table 1, which describes the dataset and is built from the
metadata loader; that is stated in its caption.

Run:  uv run python src/paper_tables.py
"""

import pandas as pd

from baseline import PROJECT_ROOT
from loaders import TASKS, load_all_meta
from paper_figures import PRIMARY_JEV, load_log

OUT = PROJECT_ROOT / "paper" / "tables.md"
LABEL = {"Suturing": "Suturing", "Knot_Tying": "Knot-Tying", "Needle_Passing": "Needle-Passing"}


def ci(row, stem, places=2) -> str:
    """'0.44 [0.25, 0.63]' from a row carrying <stem>, <stem>_low and <stem>_high."""
    return f"{row[stem]:.{places}f} [{row[f'{stem}_low']:.{places}f}, {row[f'{stem}_high']:.{places}f}]"


def table1() -> str:
    meta = load_all_meta()
    lines = ["### Table 1. Dataset",
             "",
             "| task | trials | surgeons | GRS median | GRS range | GRS SD | novice / intermediate / expert trials |",
             "|---|---|---|---|---|---|---|"]
    for task in TASKS:
        sub = meta[meta["task"] == task]
        counts = [int((sub["skill"] == c).sum()) for c in ("N", "I", "E")]
        lines.append(f"| {LABEL[task]} | {len(sub)} | {sub['subject'].nunique()} | {sub['grs'].median():.0f} | "
                     f"{sub['grs'].min()}-{sub['grs'].max()} | {sub['grs'].std():.2f} | "
                     f"{counts[0]} / {counts[1]} / {counts[2]} |")
    lines += [f"| **all** | **{len(meta)}** | **{meta['subject'].nunique()}** | {meta['grs'].median():.0f} | "
              f"{meta['grs'].min()}-{meta['grs'].max()} | {meta['grs'].std():.2f} | "
              f"{int((meta['skill'] == 'N').sum())} / {int((meta['skill'] == 'I').sum())} / "
              f"{int((meta['skill'] == 'E').sum())} |", ""]
    lines.append("GRS is the sum of six 1-5 OSATS elements, so its range is 6-30. Surgeon G contributes no "
                 "usable needle-passing trial. The kinematics directories hold 40 files per task; the "
                 "metadata files, which are authoritative, list 103 usable trials in total. Built from the "
                 "metadata loader rather than the experiment log.")
    return "\n".join(lines)


def table2(log) -> str:
    cls = log[log["kind"] == "addendum_classification_ci"].set_index(["cv_scheme", "task"])
    gap = log[log["kind"] == "addendum_protocol_gap"].set_index("task")
    maj = log[log["kind"] == "addendum_majority_class"].set_index("task")
    sid = log[log["kind"] == "addendum_surgeon_identity_ci"].set_index("task")
    kappa = log[(log["kind"] == "classification") & (log["feature_set"] == "smoothed_5hz")] \
        .set_index(["cv_scheme", "task"])["quadratic_kappa"]

    lines = ["### Table 2. The protocol gap, and surgeon identifiability",
             "",
             "| task | LOSO accuracy | LOUO accuracy | LOSO - LOUO | majority class | LOSO kappa | LOUO kappa | surgeon ID | chance |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        lines.append(
            f"| {LABEL[task]} | {ci(cls.loc[('LOSO', task)], 'accuracy')} | {ci(cls.loc[('LOUO', task)], 'accuracy')} | "
            f"{ci(gap.loc[task], 'accuracy_gap')} | {maj.loc[task, 'majority_class_rate']:.3f} | "
            f"{kappa.loc[('LOSO', task)]:+.2f} | {kappa.loc[('LOUO', task)]:+.2f} | "
            f"{ci(sid.loc[task], 'accuracy')} | {sid.loc[task, 'chance_accuracy']:.3f} |")
    lines += ["",
              "Three-way skill classification (novice / intermediate / expert). Brackets are 95% intervals from "
              "resampling whole surgeons. The LOSO - LOUO column is bootstrapped paired on the same resampled "
              "surgeons. 'Surgeon ID' replaces the target with surgeon identity, leaving features, model and folds "
              "unchanged; it is a post-hoc diagnostic. Every LOUO interval spans its task's majority-class rate."]
    return "\n".join(lines)


def table3(log) -> str:
    base = log[(log["kind"] == "headline_mae_vs_reference") & (log["feature_set"] == "smoothed_5hz")
               & (log["config_file"] == "configs/baseline.yaml")]
    jev = log[(log["kind"] == "jev_headline_mae_vs_reference") & (log["predictor"] == PRIMARY_JEV)]
    time_only = log[log["kind"] == "posthoc_time_only_headline"]
    sources = [("16 features + ridge", base), ("Jev, zero-shot", jev), ("task time alone", time_only)]

    lines = ["### Table 3. GRS error against the feature-free reference",
             "",
             "| scheme | task | reference MAE | method | model MAE | reduction [95%] | beats reference |",
             "|---|---|---|---|---|---|---|"]
    for scheme in ("LOUO", "LOSO"):
        for task in TASKS:
            for i, (name, frame) in enumerate(sources):
                row = frame[(frame["cv_scheme"] == scheme) & (frame["task"] == task)].iloc[-1]
                head = f"| {scheme} | {LABEL[task]} | {row['reference_mae']:.2f} " if i == 0 else "| | | "
                lines.append(f"{head}| {name} | {row['mae']:.2f} | {ci(row, 'mae_improvement')} | "
                             f"{'**yes**' if row['beats_reference'] else 'no'} |")
    lines += ["",
              "The reference predicts each held-out trial with the mean GRS of its own training fold. A positive "
              "reduction means lower error than that reference; 'beats reference' requires the lower bound to "
              "exceed zero, a rule fixed before any interval was computed. Jev's ordinal score is placed on the GRS "
              "scale by a line fitted within each training fold."]
    return "\n".join(lines)


def table4(log) -> str:
    cal = log[log["kind"] == "eval_calibration"]
    lines = ["### Table 4. Calibration",
             "",
             "| path | scheme | judged against | mean confidence | accuracy | ECE [95%] | MCE | Brier | constant Brier | Brier skill [95%] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    against = {"baseline": "self-declared skill (N/I/E)", "jev": "level implied by GRS"}
    for r in cal.itertuples():
        row = r._asdict()
        lines.append(f"| {r.model} | {r.cv_scheme} | {against[r.model]} | {r.mean_confidence:.3f} | "
                     f"{r.accuracy:.3f} | {ci(row, 'ece', 3)} | {r.mce:.2f} | {r.brier:.3f} | "
                     f"{r.brier_reference:.3f} | {ci(row, 'brier_skill', 3)} |")
    lines += ["",
              "Each path is judged against the target it was given, so the rows are not a head-to-head comparison. "
              "The constant forecast assigns every trial the observed class frequencies; a negative Brier skill "
              "score means the path is worse than that constant. Five equal-width bins, 103 pooled trials. Jev is "
              "never trained, so it has a single row. Per-bin counts are logged as `addendum_reliability_bin`."]
    return "\n".join(lines)


def table5(log) -> str:
    sel = log[log["kind"] == "eval_selective"]
    cov = log[log["kind"] == "eval_coverage"]
    lines = ["### Table 5. Selective prediction",
             "",
             "| path | scheme | overall MAE | AURC | oracle AURC | E-AURC | usefulness [95%] | informative |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        row = r._asdict()
        share = r.usefulness / (r.overall_mae - r.oracle_aurc) * 100
        lines.append(f"| {r.model} | {r.cv_scheme} | {r.overall_mae:.2f} | {r.aurc:.2f} | {r.oracle_aurc:.2f} | "
                     f"{r.e_aurc:.2f} | {ci(row, 'usefulness')} ({share:.1f}% of the gap to ideal) | "
                     f"{'yes' if r.informative else 'no'} |")
    lines += ["",
              "Usefulness is the overall MAE minus the area under the risk-coverage curve: positive means ordering "
              "by confidence lowers the error of the trials kept, relative to deferring at random. 'Informative' "
              "requires the lower bound to exceed zero.",
              "",
              "| path | scheme | 100% | 80% | 60% | 40% |",
              "|---|---|---|---|---|---|"]
    for (model, scheme), group in cov.groupby(["model", "cv_scheme"], sort=False):
        cells = []
        for c in (1.0, 0.8, 0.6, 0.4):
            row = group[group["coverage"] == c].iloc[-1]
            cells.append(f"{row['mae']:.2f} ({row['mae_improvement']:+.2f})")
        lines.append(f"| {model} | {scheme} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean absolute GRS error of the trials retained at each coverage, with the reduction against the "
                 "feature-free reference on those same trials in brackets. Coverage levels were fixed in advance.")
    return "\n".join(lines)


def main() -> None:
    log = load_log()
    parts = ["# Tables", "",
             "Generated by `src/paper_tables.py` from `results/experiments.csv`. Every cell except Table 1 traces "
             "to a logged run; regenerate rather than editing by hand.", "",
             table1(), "", table2(log), "", table3(log), "", table4(log), "", table5(log), ""]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
