# Paper outline — audit reframe

Status: outline only, no prose drafted. Awaiting go-ahead.

Every figure below was verified against `results/experiments.csv` and `docs/NOTES.md`
on 2026-09-20. Verification status is marked per claim. **Section 3 lists the places
where the brief I was given does not match the repo.** Read that section before
drafting anything.

---

## 0. Title candidates

Direction given: *"Surgeon identity, not skill: auditing confounds and calibration in
automated JIGSAWS assessment."* That works. Three alternates:

1. **"Recognising the surgeon, not the skill: an audit of evaluation protocol and
   confidence in automated JIGSAWS assessment."** Closest to the given direction;
   "recognising" states the mechanism rather than asserting the negative.
2. **"What does a surgical skill model measure? Quantifying identity confounding and
   miscalibration on JIGSAWS."** Question-form title; foregrounds *quantifying*, which
   is the novelty claim ("flagged but never measured").
3. **"Same surgeon, different story: the evaluation-protocol gap in automated surgical
   skill assessment."** Shortest, most memorable, least precise. Better for the SAGES
   abstract than for IJCARS.

A note on the given title: "Surgeon identity, not skill" asserts the model measures
*only* identity. The evidence supports "largely", not "only" — Knot-Tying retains a real
cross-surgeon signal (Finding 2). Prefer a title that survives that caveat.

---

## 1. Section-by-section outline

### 1. Introduction
**Thesis:** Reported JIGSAWS skill-assessment accuracy is produced under evaluation
protocols that leave the same surgeons in training; we show that most of the apparent
skill signal is surgeon recognition, and that the confidence these models report is not
usable for the human-in-the-loop triage such systems are proposed for.

Beats: (a) automated skill assessment is proposed for training and credentialling, where
a wrong score has consequences; (b) JIGSAWS is the field's standard benchmark and ships
two protocols, LOSO (same surgeons in training) and LOUO (held-out surgeon); (c) results
are commonly reported under LOSO; (d) we audit what that buys, on both a conventional
baseline and a commercial zero-shot decision model; (e) contributions, four bullets, each
mapping to a Results subsection.

### 2. Related work
**Thesis:** The subject-imbalance concern has been flagged in the literature but never
measured; calibration and selective prediction have never been audited for this task at
all.

Structure as "flagged, but never measured, until now": prior work flags that JIGSAWS
results may not transfer to practice because of subject imbalance (2020 IJCARS
motion-analysis paper citing Lefor et al.) but does not quantify it. Then four explicit
gaps: (i) no prior quantification of the same-surgeon vs new-surgeon gap; (ii) no prior
measurement of surgeon identifiability from skill features; (iii) no prior calibration or
selective-prediction audit for surgical skill assessment; (iv) no prior zero-shot
application of an LLM or non-LLM decision model to JIGSAWS skill scoring.

> **TODO(verify) — provenance.** All four novelty claims and the 2020 IJCARS/Lefor
> citation are **supplied by the user and were not verified by me.** The instruction was
> to cite rather than re-verify. Before submission someone must confirm each claim and
> resolve the citation to a specific paper, because a novelty claim that fails review
> damages the rest of the paper. I have not searched the literature for these.

### 3. Methods
**Thesis:** A protocol fixed in advance, with two prediction paths that see identical
information, so the LOSO-to-LOUO gap is attributable to the protocol and not to
implementation differences.

Subsections:
- **3.1 Dataset.** 103 trials (39 Suturing, 36 Knot-Tying, 28 Needle-Passing), 8 surgeons,
  GRS 6-30 as the sum of six 1-5 OSATS elements. Note the exclusions (the meta files, not
  the file listing, are authoritative) and that surgeon G is absent from Needle-Passing.
  Note the self-declared skill label tracks GRS poorly (expert D mean 13.5 vs novice H
  19.1) — this motivates reporting both targets.
- **3.2 Features.** 16 motion features from the PSM (patient-side) channels over the
  annotated task window, positions low-pass filtered at 5 Hz. State the thresholds were
  fixed in a config before use and never tuned against outcomes.
- **3.3 Two prediction paths.** Baseline: ridge for GRS (alpha by leave-one-subject-out
  *inside* each training fold), multinomial logistic for N/I/E. Jev (jev-1.13.0): the same
  16 features rendered as *words only* (percentile phrases, no digits, no identifiers,
  no labels), seven Score questions, zero-shot, never trained.
- **3.4 Evaluation protocol.** LOSO and LOUO from the official splits; our generated folds
  for the two tasks that lack official skill splits reproduce the official Suturing folds
  exactly. Headline metric: MAE against a **feature-free reference** that predicts each
  held-out trial with its training fold's mean GRS.
- **3.5 Protocol scrutiny.** *(See section 3 of this outline — the framing you gave for
  this subsection is factually wrong and must be rewritten.)* What is defensible: design
  rules were fixed and committed before each analysis; a shuffled-label sentinel and
  explicit leakage unit tests were run and **found no leakage**; separately, our
  originally pre-declared headline statistic (pooled out-of-fold Spearman) was found to be
  **unusable under LOUO** because a feature-free predictor scores about -0.8 on it by
  construction, which is why the headline is reported as MAE against that reference.

### 4. Results
**Thesis:** The apparent skill signal is largely surgeon recognition; what survives is one
task of three and reduces to task duration; and neither path's confidence supports
deferral.

- **4.1 The protocol gap (headline).** Finding 1.
- **4.2 What survives a held-out surgeon.** Finding 2.
- **4.3 A trivial baseline matches the commercial model.** Finding 3.
- **4.4 Confidence is miscalibrated on new surgeons.** Finding 4.
- **4.5 Confidence does not support deferral.** Finding 5.

### 5. Discussion
**Thesis:** LOSO on JIGSAWS measures a different quantity than the one the field intends
to report, and confidence-gated deployment of these systems is not currently supported by
evidence.

Beats: (a) what LOSO actually measures when the label is a property of the person;
(b) why this is not a JIGSAWS-specific quirk — any benchmark where a person-level label
is predicted from repeated measures of that person has the same structure; (c) the
deployment reading: a system that is 76% confident and 38% correct on a new surgeon fails
in the direction that matters, since triage requires confidence to track correctness;
(d) what a fix looks like — report LOUO as primary, report the feature-free reference
alongside, report a surgeon-identifiability diagnostic as a standard check.

### 6. Limitations
**Thesis:** This is a single-dataset case study with 8 surgeons; the confound and the
calibration failure are demonstrated here, not established as general.

Stated plainly, not softened: 8 surgeons (7 in Needle-Passing), so every interval is
crude and none reflects refitting variability; skill is confounded with surgeon identity —
this is the paper's central finding rather than a caveat; the surgeon-identifiability
diagnostic was **post-hoc**, not pre-declared (the LOSO/LOUO comparison itself was
pre-declared); Jev's rubric wording was finalised after some Phase 2/3 results were seen,
and it leans on completion time and pauses, which is a plausible route by which it came to
track task time; single dataset, single model version; human inter-rater reliability
(Martin et al. 1997, 0.64-0.72) is context only, the coefficient is unnamed in that
abstract and the setting differs, and **no claim is made that either method approaches
it**; the NHSR ethics determination is still pending; TypeSafe states no fixed
data-retention period for API inputs.

### 7. Ethics / data statement
NHSR determination pending with Cornell RASS-IRB. What was sent to a third-party API:
wording-derived summaries of a public de-identified dataset — no video, no trajectories,
no identifiers, no labels, no digits. TypeSafe's policy states no training on inputs and
no third-party disclosure beyond service providers, but specifies no fixed retention
period. JIGSAWS cited per its condition of use (Gao et al. 2014; Ahmidi et al. 2017).

---

## 2. Figures and tables

Existing assets are in `results/figures/` (gitignored; regenerate with
`uv run python src/plot_features.py` and `src/plot_evaluation.py`).

| # | Asset | Status | Must show |
|---|---|---|---|
| **F1** | The protocol gap | **NEW — build this** | Per task, three bars: LOSO skill accuracy, LOUO skill accuracy, surgeon-ID accuracy. Reference lines for majority-class rate and for 1/n_surgeons chance. This is the paper's headline figure and does not exist yet. |
| **T1** | Dataset | **NEW — trivial** | Trials per task, surgeons, GRS range and SD per task, skill-label counts, the exclusions. |
| **T2** | Protocol gap, numeric | **NEW from CSV** | Per task and scheme: accuracy, balanced accuracy, quadratic kappa, majority-class rate, surgeon-ID accuracy, chance. Backs F1. |
| **T3** | GRS prediction vs the feature-free reference | **NEW from CSV** | Per task and scheme, for baseline / Jev / task-time-only: reference MAE, model MAE, improvement with 95% interval, and whether the interval clears zero. The Finding 2 and 3 table. |
| **F2** | Jev tracks task time | **NEW — build this** | Scatter of Jev's `overall_skill` score against task time, one panel per task, Spearman annotated. Optionally a second row against the next-strongest feature to show the contrast. |
| **F3** | Reliability diagrams | **EXISTS** (`eval_reliability.png`) | Three panels: baseline LOUO, baseline LOSO, Jev. Already carries ECE and accuracy. Caption must state the two paths are judged on different truths and are not a head-to-head comparison. |
| **T4** | Calibration | **NEW from CSV** | Accuracy, ECE, MCE, Brier, best-constant Brier, Brier skill with interval, per path and scheme. |
| **F4** | Risk-coverage | **EXISTS** (`eval_risk_coverage.png`) | Four panels. Coloured curve vs the random-order and oracle references. The point is that the curve hugs the random line. |
| **T5** | Selective prediction | **NEW from CSV** | Overall MAE, AURC, oracle AURC, E-AURC, usefulness with interval, informative yes/no; plus the coverage table at 100/80/60/40%. |
| **F5** | Bland-Altman | **EXISTS** (`eval_bland_altman_LOUO.png`) | Supplementary. Caption must note the downward slope is partly built in for any weak predictor (a constant predictor has slope exactly -2), so it is not independent evidence. |
| **S1** | Feature distributions by GRS band | **EXISTS** (6 files) | Supplementary only. |

Three new figures/tables need code (F1, F2, and the CSV-to-LaTeX table generation). I'd
add a `src/paper_figures.py` that builds them from `experiments.csv` and the saved
predictions, so every number in the paper traces to a logged run.

---

## 3. Mismatches between the brief and the repo

### 3.1 MUST FIX — "a leakage bug was found and corrected" is not what happened

Your brief says: *"a leakage bug was found and corrected — pooled correlation under
leave-one-user-out is biased to about -0.8 for a model that knows nothing, and the
reported numbers above are post-correction."*

**There was no leakage bug.** Putting that sentence in a methods section would be a false
statement about the work, and it would understate what was actually done. The repo records
three separate things that this sentence merges:

1. **Leakage was tested for and not found.** `docs/NOTES.md` (Phase 3 results): *"All six
   means are NEGATIVE, the opposite direction to leakage... Reading: no evidence of
   leakage (which would push the mean positive)."* Unit tests additionally show the
   safeguards can detect a leak: a deliberately leaky variant moved untouched test
   predictions by 16 GRS points where the real pipeline moved them by exactly 0.
2. **A *criterion* I wrote was mis-specified.** The shuffled-label sentinel used a
   two-sided rule (|mean rho| <= 0.10) and so flagged 2 of 6 cells as FAIL in the
   *harmless* direction. It was replaced with a one-sided rule in a new config, with the
   original FAILs left on record. This is a flaw in a check, not a leak in the pipeline.
3. **The pre-declared headline *statistic* was unusable.** Pooled out-of-fold Spearman is
   biased to about -0.8 under LOUO for a feature-free predictor, because each fold is one
   surgeon and the prediction is one constant per fold. This is a property of the metric,
   not a bug.

Also: **the reported numbers are not "post-correction."** The +1.54 / +2.30 / +1.88 figures
are a *different metric* (MAE against a feature-free reference), adopted because the
correlation was uninterpretable. The correlations were never corrected; they are still
reported, always beside their reference.

Suggested accurate framing for §3.5: *"We tested for leakage and found none. Separately,
we found our own pre-declared headline statistic unusable under LOUO and report a
reference-anchored error metric instead."* That is a stronger claim for an audit paper
than "we had a bug and fixed it", and it is true.

### 3.2 Numbers correct but not in `results/experiments.csv`

These are right, but a reader tracing them to the log will not find them. Each is
reproducible; I would add them to the log before drafting.

| Claim | Status |
|---|---|
| Jev score vs task time, -0.81 / -0.86 / -0.91 | Correct — re-derived today, matches exactly. Printed by `src/jev_analysis.py posthoc` but never logged. |
| "76% confident" / "70% confident" | Correct: baseline LOUO mean top-class confidence **0.755**, Jev **0.696**. Not a logged column. |
| Majority-class rates (0.487 / 0.444 / 0.393) | Correct, in NOTES, not in the CSV. |

### 3.3 Minor numeric imprecisions

| Your brief | Repo | Note |
|---|---|---|
| "closes 3-8% of the gap" | **2.5-7.4%** | Rounding 2.5 up to 3 overstates the worst case. Use "about 3-7%" or give the four values. |
| "chance ~12%" | 12.5% (Suturing, Knot-Tying), **14.3%** (Needle-Passing) | Needle-Passing has 7 surgeons, not 8. |
| "at or below always-guess-majority-class" | Suturing 0.436 vs 0.487 (below); Knot-Tying 0.306 vs 0.444 (below); Needle-Passing 0.393 vs 0.393 (**exactly equal**) | The claim is exactly right, and stronger if stated precisely. |

### 3.4 Scope flags

- **The headline diagnostic was post-hoc.** The surgeon-identifiability measurement
  (83-93%) was added *after* seeing the LOSO/LOUO gap, to explain it. The gap itself was
  pre-declared (both schemes were always planned). An audit paper must say which is
  which; presenting the diagnostic as a pre-registered test would misrepresent it.
- **Which config.** Findings 2's baseline numbers are from `configs/baseline.yaml` (v1).
  The v2 config (wider alpha grid) gives identical Knot-Tying numbers but changes Suturing
  (+0.15 → -0.37) and Needle-Passing (-0.50 → -0.09). The paper must state v1 is primary
  and report v2 as a sensitivity, or the numbers will look irreproducible.
- **Jev finding 4's scale offset.** Jev's level usage sits about one level above the
  GRS-implied truth. Genuine overconfidence and a scale offset cannot be separated without
  a recalibration that was ruled out in advance. Do not present the -0.433 Brier skill as
  pure overconfidence.
- **"Widely-used evaluation protocol... appears to reward surgeon recognition."** Our
  evidence is that *our* features and *our* two models behave this way. We have not
  reproduced any specific published system. The claim should be about the protocol's
  structure plus our demonstration, not about other authors' results.

---

## 4. Verified findings, as they should appear

All checked against `results/experiments.csv` on 2026-09-20.

| # | Claim | Repo value | Match |
|---|---|---|---|
| 1 | Same-surgeon (LOSO) skill classification 92-97% | 0.923 / 0.972 / 0.929 | yes |
| 1 | Surgeon ID 83-93% | 0.872 / 0.833 / 0.929 | yes |
| 1 | New-surgeon (LOUO) 31-44% | 0.436 / 0.306 / 0.393 | yes |
| 2 | Knot-Tying LOUO, baseline +1.54 [0.65, 2.52] | +1.537 [0.646, 2.519] | yes |
| 2 | Knot-Tying LOUO, Jev +2.30 [1.14, 3.45] | +2.297 [1.143, 3.452] | yes |
| 2 | Suturing and Needle-Passing show none | Sut +0.151 [-1.037, 1.335]; NP -0.500 [-1.214, -0.037] | yes (NP is significantly *worse* than the mean) |
| 3 | Task-time-only, Knot-Tying LOUO +1.88 [0.94, 2.89] | +1.884 [0.940, 2.890] | yes |
| 4 | Baseline 76% confident / 38% correct | 0.755 / 0.379; Brier skill -0.351 [-0.688, -0.092] | yes |
| 4 | Jev 70% confident / 30% correct | 0.696 / 0.301; Brier skill -0.433 [-0.588, -0.242] | yes |
| 5 | Deferral +0.13 / +0.16, intervals span zero | +0.135 [-0.628, 0.922]; +0.155 [-0.416, 0.606] | yes |
| 5 | Upper ends +0.92 / +0.61 | 0.922 / 0.606 | yes |

---

## 5. SAGES abstract (separate, later)

A 400-word SAGES 2027 abstract is a separate, shorter artifact, to be drafted **after**
the full outline is approved. It should preview one finding only — the protocol gap
(Finding 1) — with F1 as its single figure, and should not attempt the calibration
material.
