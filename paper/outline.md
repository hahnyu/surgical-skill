# Paper outline — audit reframe

Status: outline only. No prose drafted. Title deferred (see §6).

Every figure quoted here was verified against `results/experiments.csv` on 2026-09-20.
Each analysis is labelled **[pre-declared]** or **[post-hoc]** — see §4, which is the
table a reviewer will look for first.

---

## 1. Section-by-section outline

### 1. Introduction  (paragraph plan, 2026-09-20; not yet drafted)

Arc: general phenomenon -> surgical gap -> what a good score must beat -> thesis and
contributions. Each paragraph below names the Related Work or Results claim it rests on, so
the narrowing can be checked before prose.

**¶1 — Stakes. [ADDITION, flagged]** Automated skill assessment is proposed for training
and credentialling; in any such use the system scores a surgeon it has not seen before.
Two or three sentences, no citations from our own results.
*Rests on:* nothing in our data. Pure framing, and the one paragraph at risk of asserting
more about clinical deployment than we can support. Keep modest and non-quantitative.
*Why added:* the fixed arc opens on the general ML phenomenon, which for a surgical
readership means two paragraphs before surgery is mentioned. This paragraph buys that
opening without losing the reader. Remove if you want the phenomenon to open cold.

**¶2 — The general phenomenon.** Identity confounding under naive cross-validation: Saeb
et al. 2017 and Chaibub Neto et al. 2019 for the mechanism and the name; Brookshire et al.
2024 as the closest structural mirror (an EEG classifier above 99% under segment holdout,
indistinguishable from chance on unseen subjects); speaker leakage in depression detection
as a second modality; Geirhos et al. 2020 and Kapoor and Narayanan 2023 as the umbrella.
*Rests on:* no claim of ours. This paragraph exists to establish that we did not discover
the mechanism.
*Seam to manage:* §2.3 already cites Saeb et al. as the study Little et al. responded to.
Here Saeb is cited for the phenomenon; there, for the counter-argument. Do not re-explain
the study in §2.3 — cross-reference it.

**¶3 — The narrowing: suspected in surgery, never tested.** The failure mode has not been
demonstrated in robotic surgical skill assessment, though the field repeatedly gestures at
it. Name the pattern compactly: a documented gap framed as a generalisation limitation, as
representation quality, as missing subject-independent validation, and once as a
"surgeon-specific component" — with no direct test in any of them.
*Rests on:* §2.3 (I3D-SAP, "Looking Beyond the Scale", the Journal of Robotic Surgery
explainable-ML study, the markerless hand-tracking study, Lefor et al.) and its negative
finding; §2.2 for the separation from the label-validity critique.
*Load-bearing:* this paragraph has to carry the transition from "textbook elsewhere" to
"worth a paper here". If it reads as a literature list, ¶2 will have made our result look
derivative.

**¶4 — What a good score would have to beat.** The target is a noisy, training-dependent
human judgment, not a fixed bar: pooled GRS inter-rater ICC 0.78 (Ilgen et al. 2015), but
untrained raters as low as 0.26 with reliability not retained months later (Gawad et al.
2019). Note also that the dataset's self-declared skill label tracks GRS poorly.
*Rests on:* §4.3's revised ceiling passage; §3.1's label observation, itself a replication
of Hendricks et al. and "Looking Beyond the Scale" (§2.2). Motivates the feature-free
reference of §3.5 — if the label is noisy and the metric is not self-interpreting, claims
need an explicit uninformative comparator.
*Do not:* reintroduce "ceiling". The whole point of this paragraph is that there isn't one.

**¶5 — Thesis and contributions.** The thesis: on this benchmark the sanctioned protocol
lets a model succeed without measuring skill, and the confidence such models report is not
usable for the deferral their deployment assumes. Four contributions:
1. The protocol gap quantified with paired surgeon-level intervals — §4.1 (+0.49, +0.67,
   +0.54, all excluding zero).
2. A direct surgeon-identification test that distinguishes confounding from underfitting —
   §4.2 (0.83-0.93 against chance near 0.13), answering Little et al. (§2.3).
3. The first calibration and selective-prediction audit in this application — §4.5, §4.6.
4. A zero-shot decision model evaluated against a one-feature baseline — §4.4.
*Honesty constraints:* contribution 2 is a post-hoc diagnostic (§4.2) and the phrasing must
not imply it was planned; contribution 4 must not imply an ordering between the methods
(§4.4); the novelty claim is the surgical instantiation and quantification, never the
mechanism.

### 2. Related work  (six paragraphs; ¶5 and ¶6 compressed to 3-4 sentences each)

**¶1 — The benchmark and its two sanctioned protocols.** Gao et al. 2014 (dataset),
Ahmidi et al. 2017 (benchmarks), van Amsterdam et al. 2021 (gesture-recognition review,
which notes LOUO is harder because of "subject-specific style variability" and treats that
style as nuisance variance to normalise away rather than as leakage inflating a score).
Purpose: establish that the protocol we audit is the dataset's own, not an unofficial
practice, and that the field already knows subject style exists without treating it as a
validity threat.

**¶2 — Label validity is a different critique from protocol validity. [KEEP SHARP]**
Anchored on Hendricks et al. 2024 (RA-L), with "Looking Beyond the Scale" (arXiv
2608.17519) noted as reporting the same self-proclaimed-skill-versus-GRS mismatch. State
their finding fairly, and note our own data reproduces it (§2.1, `addendum_label_vs_grs`)
rather than presenting it as ours. Then the distinction, which is the paragraph's reason
for existing: *they ask whether the target measures skill; we show the protocol lets a
model succeed without measuring skill, whatever the target's quality.* The two critiques
are independent and compound — **a perfectly valid target would not fix what we describe**,
because identifiability is a property of the features and the fold structure rather than of
the label's meaning; and conversely their critique survives a perfect protocol.

**¶3 — The gap is widely reported in surgery, and framed three other ways.** The richest
paragraph, and the one that carries the novelty claim.
- I3D-SAP (IJCARS): near-ceiling LOSO figures, a "significant performance drop" under LOUO,
  near-chance on external ROSMA, framed as a *generalisation limitation* with a call for
  "surgeon-invariant feature learning" — the goal is named, the mechanism is not measured.
- "Looking Beyond the Scale" (arXiv 2608.17519): JIGSAWS GRS prediction at CCC about -0.03
  under disjoint-participant evaluation, with model-side causes ruled out by search; framed
  as *representation and annotation quality*.
- Markerless hand-tracking (PubMed 42616309): **the closest prior framing.** Accuracy falls
  to 52.3% under leave-one-subject-out and the authors state the gap "quantifies the
  surgeon-specific component captured by video-level splits". Distinguish honestly: same
  intuition, different modality (open-surgery hand tracking, not JIGSAWS kinematics), and
  **no direct surgeon-identification classifier**.
- Lefor et al. 2020 (IJCARS, doi 10.1007/s11548-020-02259-z), cited once, correctly as first
  author: imbalance and inverse correlation among subjects flagged as a concern, unquantified.
- **Close with Little et al. 2017.** Their rebuttal to Saeb holds that a record-wise versus
  subject-wise gap can reflect *underfitting* on heterogeneous data rather than identity
  confounding, so a gap alone proves nothing. This is the strongest counter-argument to our
  §3.1 and we preempt it with §3.2: the same 16 features identify which of eight surgeons
  performed a held-out repetition at 0.83-0.93 against chance near 0.13. A gap is
  ambiguous; recoverable identity at seven times chance is not.
- **Negative finding, stated as a contribution:** no surgical-skill paper located both
  attributes the gap to identity leakage and demonstrates it by classifying the surgeon from
  the same features.

**¶4 — Reported performance, and the missing reference.** Zia & Essa 2018, Wang et al. 2021
(C3D multi-task), VBA-Net. Purpose: establish the reporting norm we depart from — accuracies
and correlations quoted without an uninformative reference predictor — which motivates §2.5
and makes the -0.8 result in §2.7 legible as a property of the metric rather than a quirk.

**¶5 — Calibration and deferral: borrowed lineage. [COMPRESSED, 3-4 sentences]** No
surgical-skill calibration or deferral precedent was located; the methodological lineage is
diagnostic imaging and ICU. Cite CoDoC (Nature Medicine 2023) and "Trust or Escalate"
(ICLR 2025) as the demonstrated-benefit cases, Tian et al. 2025 for LLM-judge
overconfidence, and "Calibrated Triage, Not Autonomy" for the point that deferral cannot
manufacture signal a model lacks. State the thinness as the second negative finding.

**¶6 — Zero-shot decision models. [COMPRESSED, 3-4 sentences]** Distinguish ZEAL
explicitly, since it is the nearest neighbour and a reviewer will raise it: zero-shot
segmentation feeding a *trained* LSTM is a different mechanism from a decision model
scoring a rubric with no training. Then Jev's evidence base, presented evidence-first
rather than vendor-first: the RLCD calibration claim, the Forbes critique noting RLCD is a
coined term and that "zero hallucinations" refers to schema conformance rather than
correctness, and three independent September 2026 evaluations (priorbench, scienthoon,
webofmike) converging on calibrated-in-domain and overconfident-out-of-distribution.
**The non-peer-reviewed status of all three is stated in the body text, not a footnote.**

### 3. Methods
**Thesis:** A protocol fixed in advance, with two prediction paths that see identical
information, so the LOSO-to-LOUO gap is attributable to the protocol and not to
implementation differences.

- **3.1 Dataset.** 103 trials (39 Suturing, 36 Knot-Tying, 28 Needle-Passing), 8 surgeons,
  GRS 6-30 as the sum of six 1-5 OSATS elements. The meta files, not the file listing, are
  authoritative for which trials are usable. Surgeon G is absent from Needle-Passing, so
  that task has 7 surgeons. Note the self-declared skill label tracks GRS poorly (expert D
  mean 13.5 against novice H 19.1), which is why both targets are reported.
- **3.2 Features.** 16 motion features from the PSM channels over the annotated task
  window, positions low-pass filtered at 5 Hz. Thresholds were fixed in a config before
  use and never tuned against outcomes. **[pre-declared]**
- **3.3 Two prediction paths.** Baseline: ridge for GRS (penalty chosen by
  leave-one-subject-out *inside* each training fold), multinomial logistic for N/I/E.
  Jev (jev-1.13.0): the same 16 features rendered as words only — percentile phrases, no
  digits, no identifiers, no labels — seven Score questions, zero-shot, never trained.
  **[pre-declared]**
- **3.4 Evaluation protocol.** LOSO and LOUO from the official splits; our generated folds
  for the two tasks lacking official skill splits reproduce the official Suturing folds
  exactly. **[pre-declared]**
- **3.5 Protocol scrutiny.** Two things belong here, and they are separate:

  **Leakage was tested for and not found.** The scaler is fit inside each training fold;
  hyperparameters are chosen only within the training set; held-out trials are never seen.
  Unit tests confirm the safeguards would detect a leak: a deliberately leaky variant
  (standardising all rows before splitting) moved untouched test-fold predictions by
  16 GRS points where the real pipeline moved them by exactly 0. A shuffled-label sentinel
  was run on all six task-by-scheme cells; every mean was *negative*, the opposite of what
  leakage produces.

  **Separately, the pre-declared headline statistic proved unusable under LOUO and was
  replaced.** Pooled out-of-fold Spearman was pre-registered as the headline. It is not
  interpretable under LOUO: each fold is one surgeon and the prediction is one constant
  per fold, so a predictor that knows nothing scores about -0.8 (measured: -0.78 to -0.86).
  We therefore report **MAE against a feature-free reference** that predicts each held-out
  trial with its training fold's mean GRS. The Spearman values are still reported, always
  beside that reference, never alone. **The substitution is itself post-hoc** — see §4.

  > Do not describe either of these as a bug, and do not use "corrected". No result was
  > corrected: one metric was found uninterpretable and a second was adopted alongside it.
  > A separate point, worth one sentence: our *sentinel criterion* was two-sided and so
  > flagged two cells in the harmless direction; it was replaced with a one-sided rule in a
  > second config, with the original flags left on the record.

### 4. Results
**Thesis:** The apparent skill signal is largely surgeon recognition; what survives is one
task of three and reduces to task duration; and neither path's confidence supports deferral.

- **4.1 The protocol gap.** Finding 1. **[pre-declared]** — both schemes were planned from
  the start.
- **4.2 Why: surgeon identifiability.** Finding 1b. **[post-hoc]** — run to explain the gap
  after seeing it. This must be stated in the text, not only in a table.
- **4.3 What survives a held-out surgeon.** Finding 2. **[pre-declared design, post-hoc
  metric]**.
- **4.4 A trivial baseline matches the commercial model.** Finding 3. **[post-hoc]**.
- **4.5 Confidence is miscalibrated on new surgeons.** Finding 4. **[pre-declared]**.
- **4.6 Confidence does not support deferral.** Finding 5. **[pre-declared]**, including the
  interpretation rule, which was fixed before any interval was computed.

### 5. Discussion
**Thesis:** LOSO on JIGSAWS measures a different quantity than the field intends to report,
and confidence-gated deployment of these systems is not currently supported by evidence.

Beats: (a) what LOSO measures when the label is a property of the person; (b) why this is
not JIGSAWS-specific — any benchmark predicting a person-level label from repeated measures
of that person has the same structure; (c) the deployment reading: a system that is 76%
confident and 38% correct on a new surgeon fails in the direction that matters, since triage
requires confidence to track correctness; (d) what a fix looks like — report LOUO as primary,
report a feature-free reference alongside, and publish a surgeon-identifiability diagnostic
as a standard check.

### 6. Limitations
**Thesis:** A single-dataset case study with 8 surgeons; the confound and the calibration
failure are demonstrated here, not established as general.

Stated plainly: 8 surgeons (7 in Needle-Passing), so every interval is crude and none
reflects refitting variability; skill is confounded with surgeon identity, which is the
paper's central finding rather than a caveat; **the surgeon-identifiability diagnostic and
the headline metric substitution were both post-hoc** (§4); Jev's rubric wording was
finalised after some Phase 2 and 3 results were seen and leans on completion time and
pauses, a plausible route by which it came to track task time; single dataset, single model
version; human inter-rater reliability (Martin et al. 1997, 0.64-0.72) is context only — the
coefficient is unnamed in that abstract and the setting differs — and **no claim is made
that either method approaches it**; the NHSR determination is pending; TypeSafe states no
fixed data-retention period.

### 7. Ethics and data statement
NHSR determination pending with Cornell RASS-IRB. What was sent to a third-party API:
wording-derived summaries of a public de-identified dataset — no video, no trajectories, no
identifiers, no labels, no digits. TypeSafe's policy states no training on inputs and no
third-party disclosure beyond service providers, but specifies no fixed retention period.
JIGSAWS cited per its condition of use (Gao et al. 2014; Ahmidi et al. 2017).

---

## 2. Figures and tables

| # | Asset | Status | Must show |
|---|---|---|---|
| **F1** | `results/figures/fig1_protocol_gap.png` | **BUILT** | Three panels. A: skill accuracy, LOSO vs LOUO, per task, with the always-predict-the-biggest-class line. B: surgeon-identification accuracy, with chance. C: GRS improvement over the feature-free reference under LOUO, for ridge / Jev / task-time-only, with 95% intervals and a zero line. Colour means task in all three panels. |
| **T1** | Dataset | to build | Trials per task, surgeons, GRS range and SD, skill-label counts, exclusions. |
| **T2** | Protocol gap, numeric | to build from log | Per task and scheme: accuracy, balanced accuracy, quadratic kappa, majority-class rate, surgeon-ID accuracy, chance. Backs F1A and F1B. |
| **T3** | GRS vs the feature-free reference | to build from log | Per task and scheme, for ridge / Jev / task-time-only: reference MAE, model MAE, improvement with interval, interval-clears-zero flag. Backs F1C and adds LOSO. |
| **F2** | Jev tracks task time | to build | Jev's `overall_skill` score against task time, one panel per task, Spearman annotated. Now logged as `addendum_jev_score_vs_feature` for all 16 features, so the choice of task time can be shown not to be cherry-picked. |
| **F3** | Reliability diagrams | **EXISTS** | Caption must state the two paths are judged on different truths and are not a head-to-head comparison. |
| **T4** | Calibration | to build from log | Accuracy, mean stated confidence, ECE, MCE, Brier, best-constant Brier, Brier skill with interval. |
| **F4** | Risk-coverage | **EXISTS** | The coloured curve hugging the random-order reference is the point. |
| **T5** | Selective prediction | to build from log | Overall MAE, AURC, oracle, E-AURC, usefulness with interval, informative flag; plus coverage at 100/80/60/40%. |
| **F5** | Bland-Altman | **EXISTS** | Supplementary. Caption must note the downward slope is partly built in for any weak predictor (a constant predictor has slope exactly -2), so it is not independent evidence. |
| **S1** | Feature distributions by GRS band | **EXISTS** (6 files) | Supplementary only. |

**Note on F1's scope.** The brief specified two panels (LOSO vs LOUO vs surgeon-ID). I built
three. Panel C was added because item 8 asks to see Knot-Tying's surviving signal in this
figure, and that signal is **not** in the classification result — in classification,
Knot-Tying collapses hardest of the three (0.97 to 0.31). Knot-Tying's survival is a GRS
regression result, so it needs its own panel. If you want F1 restricted to the original two
panels, panel C moves to F1b and the title discussion below still applies.

---

## 3. Corrections applied to the brief

| Item | Brief said | Repo says | Action |
|---|---|---|---|
| Deferral gap | "3-8%" | 2.5% / 3.9% / 5.5% / 7.4% | Use **"about 3-7%"**, or give all four. Rounding 2.5 up to 3 overstates the worst case. |
| Needle-Passing chance | "~12%" | **14.3%** (7 surgeons; G is absent) | Use 12.5% for Suturing and Knot-Tying, 14.3% for Needle-Passing. Now on F1's footnote. |
| Majority-class comparison | "at or below" | Suturing 0.436 vs 0.487; Knot-Tying 0.306 vs 0.444; **Needle-Passing 0.393 vs 0.393, exactly equal** | Use the sharper fact: two tasks fall *below* the biggest-class rate and one lands *exactly on* it. |
| Leakage | "a leakage bug was found and corrected" | Leakage tested for, not found; sentinel *criterion* mis-specified; headline *statistic* separately unusable | §3.5 rewritten. No "corrected" language anywhere. |

---

## 4. Pre-declared vs post-hoc

Written before drafting because an audit paper that criticises others' protocol discipline
will be read closely on its own. Each item traces to a commit that precedes the analysis.

**Pre-declared** (design committed before the analysis ran):
- Feature thresholds and the 5 Hz smoothing choice, with sensitivity runs at raw and 8 Hz.
- Both evaluation schemes, LOSO and LOUO, from the start.
- Ridge and multinomial logistic; the penalty grid; C fixed at 1.0.
- The surgeon-cluster bootstrap.
- The shuffled-label sentinel (though its criterion was mis-specified, see below).
- The Phase 2 boxplot rules, including expected directions and a stop rule, committed before
  any figure existed.
- Jev's entire prompt, state format and analysis plan, committed before any dataset-derived
  API call, including the direction guard on the fitted slope.
- The whole Phase 5 evaluation design — which truth each path is judged on, every metric,
  the bins, the coverage levels — and the rule that confidence counts as informative only if
  the lower end of its interval clears zero, fixed before any interval was computed.

**Post-hoc** (added after seeing results, to explain them):
- **The surgeon-identifiability diagnostic.** Run after seeing the LOSO/LOUO gap. The gap
  itself was pre-declared; the explanation was not.
- **The headline metric substitution.** Pooled Spearman was pre-registered; MAE against a
  feature-free reference replaced it after the Spearman proved uninterpretable under LOUO.
  Principled, but a post-hoc change, and the paper must say so.
- The feature-free reference itself, and the intercept-only null used to explain the
  sentinel's negative means.
- The task-time-only baseline, and Jev's score-versus-feature correlations.
- The second config (wider penalty grid, one-sided sentinel), decided after seeing the first
  config's results and reported alongside it, never instead of it.

---

## 5. Verified findings

| # | Claim | Repo value | Match |
|---|---|---|---|
| 1 | Same-surgeon (LOSO) skill classification 92-97% | 0.923 / 0.972 / 0.929 | yes |
| 1 | Surgeon ID 83-93% | 0.872 / 0.833 / 0.929 | yes |
| 1 | New-surgeon (LOUO) 31-44% | 0.436 / 0.306 / 0.393 | yes |
| 2 | Knot-Tying LOUO, ridge +1.54 [0.65, 2.52] | +1.537 [0.646, 2.519] | yes |
| 2 | Knot-Tying LOUO, Jev +2.30 [1.14, 3.45] | +2.297 [1.143, 3.452] | yes |
| 2 | Suturing and Needle-Passing show none | Sut +0.151 [-1.037, 1.335]; NP -0.500 [-1.214, -0.037] | yes — Needle-Passing is significantly *worse* than the mean |
| 3 | Jev vs task time, -0.81 to -0.91 | -0.81 / -0.86 / -0.91 | yes |
| 3 | Task-time-only, Knot-Tying LOUO +1.88 [0.94, 2.89] | +1.884 [0.940, 2.890] | yes |
| 4 | Baseline 76% confident / 38% correct | 0.755 / 0.379; Brier skill -0.351 [-0.688, -0.092] | yes |
| 4 | Jev 70% confident / 30% correct | 0.696 / 0.301; Brier skill -0.433 [-0.588, -0.242] | yes |
| 5 | Deferral +0.13 / +0.16, intervals span zero | +0.135 [-0.628, 0.922]; +0.155 [-0.416, 0.606] | yes |
| 5 | Upper ends +0.92 / +0.61 | 0.922 / 0.606 | yes |

Every value above now traces to a row in `results/experiments.csv`.

---

## 6. Title — deferred, read against F1

Panel C is the reason to be careful. Knot-Tying (orange) clears zero under all three
methods, at +1.54, +2.30 and +1.88 GRS points, with intervals that exclude zero. A title
asserting the models measure identity *rather than* skill is contradicted by the panel
directly beneath it.

1. **"Recognising the surgeon, not the skill: an audit of evaluation protocol and confidence
   in automated JIGSAWS assessment."** Same objection as the working title — "not the skill"
   overstates given panel C.
2. **"What does a surgical skill model measure? Quantifying identity confounding and
   miscalibration on JIGSAWS."** Question form, foregrounds *quantifying* (the novelty),
   and survives panel C because it promises a measurement rather than a verdict.
   **My preference.**
3. **"Same surgeon, different story: the evaluation-protocol gap in automated surgical skill
   assessment."** Shortest and most memorable, least precise. Better suited to the SAGES
   abstract than to IJCARS.

A fourth, if you want the confound named while staying accurate:
**"Mostly the surgeon: identity confounding and miscalibration in automated JIGSAWS skill
assessment."** "Mostly" is defensible — one task of three survives, and what survives
reduces to task time.

---

## 7. SAGES abstract

A separate, shorter artifact, to be drafted only after the full outline is approved. One
finding — the protocol gap — with F1 panels A and B as its single figure. It should not
attempt the calibration material.
