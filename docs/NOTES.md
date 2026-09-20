# NOTES.md

Running log of decisions, surprises, and reproduced numbers. This becomes the
methods section later.

## 2026-09-19

Environment set up with uv 0.12.17 (Homebrew). Dependencies installed per
PLAN.md Phase 0; `uv.lock` committed. Note: `netcal` pulls in `torch`/`gpytorch`
as transitive dependencies. Import check (`pandas, sklearn, pingouin, netcal`)
passes. JIGSAWS is unzipped in `data/` (contents not yet inspected).

Phase 0 complete: private repo pushed to `github.com/hahnyu/surgical-skill`.
Jev early access granted. NHSR determination request in progress with Cornell
RASS-IRB.

### meta_file layout — CONFIRMED

Each task folder has a `readme.txt` documenting the layout. It claims:

```
1: filename   2: self-proclaimed skill (E/I/N)   3: GRS total
4-9: Respect for tissue, Suture/needle handling, Time and motion,
     Flow of operation, Overall performance, Quality of final product
```

Verified positions 3 and 4-9 by arithmetic: **GRS equals the sum of columns 4-9
on all 103 rows, all three tasks, zero exceptions.** That confirms which columns
hold the total and which hold the six elements.

Caveat recorded deliberately: the arithmetic cannot confirm the *order of the
six elements among themselves*, since a sum ignores order. Any within-block
permutation sums identically. We take `readme.txt` at its word (decision
2026-09-19). This only matters if we report per-element results in Phase 5; GRS
totals are unaffected. TODO(verify) against the JIGSAWS papers before any
per-element claim goes in the write-up.

Format details: meta files are **tab-separated with CRLF line endings** and
contain *double* tabs after the filename and after the GRS, so splitting on a
single tab yields empty fields — split on whitespace instead. Each meta file
ends with a junk line containing only a carriage return; it is not a trial.
Kinematics files, by contrast, are space-padded with plain LF endings.

### Trial counts — CONFIRMED 103

`kinematics/AllGestures/` holds 40 files per task, but the meta files list
39 / 36 / 28 = **103**. The meta file is the authoritative list of usable
trials; the surplus files on disk (1 Suturing, 4 Knot-Tying, 12 Needle-Passing)
are the dropped recordings. `Knot_Tying_H002.txt` is 0 bytes and is among them.
Every trial named in a meta file has a kinematics file.

Subject G is absent from Needle-Passing entirely (all 5 trials dropped), which
will matter for LOUO on that task.

### Self-proclaimed skill does not track GRS — SURPRISE

Mean GRS by label, pooled: E 15.6 (n=29), I 19.1 (n=28), N 14.8 (n=46). Not
monotonic. Per subject:

| subject | label | mean GRS |
|---|---|---|
| D | expert | 13.5 |
| E | expert | 17.8 |
| C | intermediate | 18.0 |
| F | intermediate | 20.3 |
| B | novice | 12.0 |
| G | novice | 13.5 |
| H | novice | 19.1 |
| I | novice | 15.2 |

Expert D rates below novice H. With only 2 experts and 2 intermediates each
group mean is one or two people, so the group ordering is not a stable effect —
but the label (hours of robotic experience) and the GRS (what a rater saw on
that trial) are clearly measuring different things. This is the
subject-identity confound from CLAUDE.md, visible on day one, and it means
Phase 3's two models have genuinely different targets. Plan: run both as
written, report the discrepancy as a finding.

TODO(verify): I believe this discordance is known in the JIGSAWS literature but
have not checked a source. Do not assert it in the write-up until checked.

### Experimental_setup — two problems

**Pre-computed *skill* splits exist only for Suturing.** Relevant path:

```
Experimental_setup/<task>/unBalanced/SkillDetection/{SuperTrialOut,UserOut}/<i>_Out/itr_1/{Train,Test}.txt
```

`SuperTrialOut` = LOSO, `UserOut` = LOUO. The fold convention is transparent:
LOSO fold *i* holds out repetition *i* across all subjects (5 folds); LOUO fold
*i* holds out subject *i* (8 folds). Folds 2 and 7 of Suturing are short
because `Suturing_H002` was dropped.

State this carefully, because two different claims are easy to conflate:

- The **LOSO/LOUO schemes are dataset-wide standards**, defined for all three
  tasks, and `SuperTrialOut`/`UserOut` fold directories do exist for all three
  tasks. The literature reports both schemes on all three tasks routinely.
- The **`SkillDetection` directory, which is the one whose split files carry a
  skill label and operate on whole trials, exists only under `Suturing`.** For
  Knot-Tying and Needle-Passing the `SuperTrialOut`/`UserOut` directories live
  only under `GestureClassification` and `GestureRecognition`, whose rows are
  *gesture segments* (`Knot_Tying_B001_001627_001735.txt  G11`) or frame
  windows labeled by source trial — no skill score in either, so neither can be
  used as a skill split directly.

Verified this is not a bad extraction: the only structural difference between
`Suturing/` and `Knot_Tying/` under `Experimental_setup` is the SkillDetection
subtree (104 files, exactly the file-count difference: 5512 vs 5408), with the
other 5408 files identical in layout, original 2014-05-20 mtimes preserved, and
no archive leftovers on disk.

Decision (2026-09-19): **generate LOSO/LOUO folds for Knot-Tying and
Needle-Passing by the documented scheme definition.** Justification is stronger
than "we copied a convention": `_generate_splits()` reproduces the official
Suturing `SkillDetection` folds **exactly** — 5 LOSO folds and 8 LOUO folds,
identical train *and* test membership under both schemes. That equivalence is
locked down by a test (`test_generated_splits_match_official_suturing`). So the
generated folds for the other two tasks instantiate the same standard
definition rather than approximating it. Say so plainly in the paper.

**The split files' GRS column contradicts the meta file, systematically by
subject.** For subjects B, C, D, E the two agree on all trials. For F, G, H, I
the split value equals `meta GRS - element_6 (Quality of final product)` on all
19 trials, zero exceptions. Example: `Suturing_F002` is 26 in meta
(4+4+4+5+4+5) and 21 in the split files.

Looks like the split files were generated against an earlier five-element
version of the ratings for half the subjects. Decision (2026-09-19): **the meta
file is authoritative** — it is self-consistent and matches the documented 6-30
range. The split files are used for *fold membership only*; their score column
is ignored.

### Frame windows — RESOLVED

Split filenames embed a frame range (`Suturing_B001_000080_005635.txt`). For 17
of 39 Suturing trials the end frame does not equal the file's row count (e.g.
`Suturing_D001` names 118-2770, file has 2800 rows).

Finding: the window is **first gesture start to last gesture end** from the
trial's `transcriptions/<trial>.txt`. It matches the split filename on **39 of
39** Suturing trials, zero exceptions. Frames are **1-indexed and inclusive**
(`Suturing_B001`'s window ends at 5635 and the file has exactly 5635 rows; some
trials start at frame 1), so the Python slice is `iloc[start - 1 : end]`. No
window runs past the end of its file, on any of the 103 trials.

How much recording lies outside the window (fraction of the file):

| task | median | worst trial | worst trailing time |
|---|---|---|---|
| Suturing | 1% | 10% | 291 frames (~10 s) |
| Needle-Passing | 2% | 9% | 251 frames (~8 s) |
| Knot-Tying | 7% | 42% | 951 frames (~32 s) |

For most trials the choice hardly matters; for a few Knot-Tying trials it
changes task time, idle fraction and path length materially.

Decision (2026-09-19): **compute features over the annotated window**
(`load_kinematics(trial, windowed=True)`). Reason: it measures the task, not
setup or wind-down time, and it is the window the official splits use. The
choice was made before looking at any relationship between window and GRS, and
deliberately not tuned against outcomes.

Consequence to state in the paper: this makes the pipeline depend on human
gesture annotations to mark task start and end. A fully automatic system would
need its own boundary detector. Limitation, not a flaw in the comparison.

### Speed from position — decision and sanity check

Decision (2026-09-19): compute speed by **finite differences of the position
columns** (`step length * fps`), not from the recorded velocity columns.
Reason: it can be checked against a hand calculation on a synthetic path.

Sanity check on `Suturing_B001` (windowed): finite-difference speed against the
norm of the recorded `PSM*_vx/vy/vz`. Correlation 0.875 (PSM1) and 0.926
(PSM2); mean finite-difference speed is ~95% of the recorded mean on both arms
(0.0098 vs 0.0104 for PSM1). Agreement in both shape and level, so neither the
column indexing nor the 30 Hz assumption looks wrong. One trial only, not a
proof. No NaNs in the window.

Units: positions span ~0.01-0.1 and mean speed is ~0.01 per second, consistent
with meters and ~1 cm/s. TODO(verify) against the JIGSAWS documentation before
any units appear in a figure or the paper.

## 2026-09-20

### Acceleration — RAW, unsmoothed. FLAGGED, revisit before Phase 3.

Decision (2026-09-20): compute acceleration **raw**, with no smoothing, for now.
Smoothing needs a window or cutoff, and PLAN.md warns against tuning such
choices against results. Deferred, not dismissed.

Definition used: the **length of the acceleration vector**, from the second
difference of position (`np.diff(positions, n=2) * fps**2`), so it responds to
speeding up, slowing down, *and* changing direction. n_frames - 2 values. The
alternative is the change in speed (signed; blind to constant-speed turns).
Easy to switch; not chosen.

Why it is flagged: differencing twice amplifies measurement noise. Check on
`Suturing_B001` (windowed), acceleration in m/s^2:

| arm | mean | median | p99 | max | max / median |
|---|---|---|---|---|---|
| PSM1 | 0.154 | 0.121 | 0.595 | 1.281 | 10.6 |
| PSM2 | 0.138 | 0.069 | 0.820 | 1.669 | 24.1 |

The tail is heavy: on PSM2 the maximum is 24x the median, so `accel_max` for a
trial may be set by a single frame. I have **not** shown this is noise rather
than a genuine abrupt movement -- do not describe it as noise in the write-up
without evidence.

Ruled out: quantisation. Positions are recorded to 1e-6 m (smallest nonzero
step), which would contribute only ~1e-3 m/s^2, far below the median of ~0.1.

Prefer `accel_mean` over `accel_max` if only one is used. Open questions to
settle before Phase 3, with a choice made in advance of seeing model results:
(1) smooth or not, and if so how; (2) whether to keep `accel_max` at all.
One trial only.

### Idle and movement thresholds — DECIDED before any model results

Chosen once on 2026-09-20 and stored in `configs/features.yaml`. Do not tune.

- **Literature search first, as agreed.** Found no usable numeric threshold.
  The ROVIMAS paper (Motion analysis of JIGSAWS using ROVIMAS, PMC7671974)
  defines a movement qualitatively -- velocity rises to a peak and "returns
  nearly to zero" -- counts local peaks after a Gaussian filter, and states **no**
  numeric threshold or filter settings. The other results had no threshold, or
  did not say how it was computed. So no source to cite. TODO(verify): re-check
  if a source turns up; a cited number would replace ours.
- **Idle:** speed strictly below **1 mm/s (0.001 m/s)**. Justification is
  physical, not data-driven: positions are recorded to 1e-6 m, so the speed
  noise floor at 30 Hz is ~3e-5 m/s and 1 mm/s is ~30x above it.
- **Movement:** a run of consecutive steps at or above the threshold lasting at
  least **0.2 s (6 steps)**. Chosen over peak counting because peak counting on
  raw speed counts noise wiggles and needs three parameters (height,
  prominence, separation); this needs two, and shares the threshold with idle.
- **Robustness check, not features:** repeat the final analysis at 0.5, 1 and
  2 mm/s, declared in advance.

Disclosure: I looked at the pooled speed distribution (no labels) before
proposing 1 mm/s, so the number was not chosen blind to where it falls. It was
never compared against GRS or skill.

The distribution offers no natural cut: speed is one continuous smear, no gap
or second mode. Pooled, PSM1 spent 14% of frames below 1 mm/s and 47% below
5 mm/s. Per-trial median speed varies ~5x across trials (3.0 to 15.2 mm/s on
PSM1), which is why a per-trial relative threshold was rejected: overall pace is
itself a skill signal and a relative threshold would erase it.

Label-free sanity check with the chosen values, all 103 trials, both arms
(206 rows, no NaNs, no trial with zero movements):

| task | arm | idle fraction (min / median / max) | movements per minute (min / median / max) |
|---|---|---|---|
| Knot-Tying | PSM1 | 0.02 / 0.06 / 0.21 | 8.9 / 21.8 / 37.6 |
| Knot-Tying | PSM2 | 0.06 / 0.16 / 0.31 | 16.6 / 28.0 / 40.2 |
| Needle-Passing | PSM1 | 0.09 / 0.15 / 0.24 | 22.2 / 34.4 / 41.4 |
| Needle-Passing | PSM2 | 0.11 / 0.21 / 0.37 | 19.6 / 30.1 / 39.6 |
| Suturing | PSM1 | 0.06 / 0.14 / 0.32 | 19.4 / 32.0 / 41.3 |
| Suturing | PSM2 | 0.13 / 0.26 / 0.37 | 18.4 / 26.3 / 34.9 |

Sensitivity is large, as expected: median PSM1 idle fraction is 0.063 / 0.130 /
0.234 at 0.5 / 1 / 2 mm/s (PSM2: 0.123 / 0.207 / 0.320). The absolute level of
the feature depends heavily on the threshold; that is what the robustness
check tests.

Known limitations, written down now:
- Speed is raw. One slow frame inside a real movement splits it in two, and both
  halves may then fall under the 0.2 s minimum and vanish. No smoothing or
  gap-merging (same open question as acceleration).
- Raw movement count grows with trial duration, so it is partly redundant with
  task time. A per-minute form is available; whether to use it is undecided.

### Working volume, bimanual coordination, smoothness (SPARC) — decisions

Decided 2026-09-20: **working volume** = convex hull volume (`scipy.spatial
.ConvexHull`); **bimanual coordination** = Pearson correlation of the two
instruments' speed profiles; **smoothness** = SPARC on the speed profile of the
**whole windowed trial**, with cutoff 10 Hz, amplitude threshold 0.05, padding 4.
Those SPARC settings are the published defaults as I remember them from the
original paper's reference implementation (Balasubramanian et al. 2015).
TODO(verify) all three against the paper before the write-up.

Edge cases: volume returns 0 when the points span no 3D shape (never moved, or
only along a line or plane). Bimanual coordination and SPARC **raise** on
undefined input (constant speed, all-zero profile) rather than return NaN, so no
NaN can reach the feature matrix.

Label-free sanity check, all 103 trials: no NaNs, no zero volumes. Hull volume
runs from ~9 to 309 cm^3 (a workspace a few centimetres wide); one Suturing PSM2
trial reaches 309, ~4x that arm's median, consistent with a hull being set by a
few extreme frames. Bimanual correlation is modest: medians 0.26-0.34, maximum
0.63.

**SPARC WARNING -- whole-trial SPARC is largely a duration / movement-count
measure.** One smooth movement scores about -1.41 regardless of duration
(1, 2, 4 s all give -1.41 to -1.42), so PLAN.md's claim that SPARC is
insensitive to duration holds for a *single movement*. But five identical,
perfectly smooth movements score -4.66 (two score -2.97). Real whole trials
score -8 to -34, and across the 103 trials (both arms) SPARC correlates with
**movement count at Spearman -0.83** and with **duration at -0.81**; within
each task the correlations are still -0.60 to -0.75. So the whole-trial value
mostly reports how long the trial is and how many movements it holds, not how
smooth each one was. This is a feature-to-feature check; no labels were used.

Chose whole-trial with this caveat known in advance (it was in the option
description); the size of the effect was not. Options still open, to decide
before Phase 3: (a) keep whole-trial and accept the overlap, mindful that a
regression will split credit among collinear features; (b) switch to per-movement
mean SPARC (closer to how SPARC is defined, but movements can be as short as
0.2 s = 6 frames, where the spectrum is poorly resolved, and it ties smoothness
to the speed threshold); (c) drop SPARC. Do not pick by which scores best
against GRS.

### Gripper cycles — hysteresis band, and a known failure case

Decided 2026-09-20: count open/close cycles with a **hysteresis band** on the
gripper angle, `[-0.6, -0.1]` (in `configs/features.yaml`). A cycle needs the
angle to cross the whole band, so jitter near one cut point is not counted. Two
state changes make one cycle; frames before the angle first leaves the band are
skipped; a trial ending part-way through a cycle does not count it. We do not
need to know which end is "open". Units look like radians (range about -1.1 to
+0.53), TODO(verify).

Why the band goes where it does: PSM1's pooled angle distribution is clearly
two-humped (about -0.9 and +0.5) with a sparse gap between, and the band sits in
that gap (6.7% of PSM1 frames inside; PSM2 18.4%, because PSM2's distribution is
smeared with no clean gap, so its count is the less trustworthy one). Disclosure:
the band was placed after looking at the pooled, label-free distribution. It was
never compared against GRS or skill.

Effect of hysteresis, all trials pooled: a single cut at the band midpoint would
give 1.13x (PSM1) and 1.29x (PSM2) as many cycles, i.e. jitter inflation the band
removes.

Label-free check: 206 arm-trials, no NaNs, one trial with 0 cycles.
Cycles per trial (median, range): PSM1 7 (4-21), PSM2 3-8 (0-18), depending on task.

**KNOWN FAILURE:** `Suturing_G002` PSM2 has 0 cycles. Its angle spans -0.521 to
+0.521 and never goes below the -0.6 threshold; 38.7% of its frames sit inside
the band and 61.3% above it. The gripper is moving, but between "inside the
band" and "open", so a fixed absolute band cannot see the cycles (a single
midpoint cut finds several). Hypothesis, NOT verified: this arm does not close
fully, e.g. while holding a needle. So a fixed band assumes every trial shares
the same "closed" angle, and this trial breaks that. The 0 is a wrong value, not
a real absence of gripping. Not fixed, because moving a pre-declared threshold
after seeing one trial is tuning. Options if it matters: a per-trial band (cut
relative to each trial's own range, at the cost of erasing range as a possible
skill signal), or leave it and report the failure. One trial of 206.

### Feature matrix built (2026-09-20) — 103 x 24, with current choices

`build_feature_matrix()` in `src/features.py`; regenerate with
`uv run python src/features.py`, which writes `results/features.csv`. That file
is **gitignored**: it is fully regenerable and derived from JIGSAWS, which we do
not redistribute. Rows are indexed by trial; columns are **features only** --
subject, task, repetition, skill and every label are deliberately excluded so a
model cannot see them (a test enforces this). 24 columns: `task_time_s`,
`bimanual_speed_corr`, and 11 features per arm (path length, speed mean/std/max,
acceleration mean/max, idle fraction, movement count, SPARC, hull volume,
gripper cycles). No NaN or infinite values; a check raises if any appear.

Verification beyond self-consistency: a test recomputes path length, task time,
mean speed and the gripper column for three trials (one per task), both arms,
straight from the raw file with `np.loadtxt` and BARE numeric column indices from
the readme (PSM1 position = Python 38:41, PSM2 = 57:60, grips = 56 and 75) --
none of the named-column machinery -- and it agrees. That checks the column
indexing independently, which agreeing-with-itself tests cannot.

Feature redundancy (label-free): 7 pairs at |Spearman| >= 0.9, e.g.
`PSM1_speed_mean ~ PSM1_accel_mean` (+0.98), `PSM2_speed_max ~ PSM2_accel_max`
(+0.96), `task_time_s ~ PSM1_n_movements` (+0.94). Expect a regression to split
credit among these.

**DATA-QUALITY FINDING -- position discontinuities.** `speed_max` reaches
0.73 m/s (PSM1) and 0.81 m/s (PSM2) against medians near 0.1, and `accel_max`
reaches 21-23 m/s^2. In a workspace a few centimetres wide that implies a tool
moving centimetres in one 1/30 s frame. The largest single-frame steps:
26.9 mm (`Suturing_I004` PSM2), 24.4 mm (`Suturing_D001` PSM1), 16.1 mm
(`Needle_Passing_H002` PSM1), against a typical step of ~0.2 mm. 47 of 206
arm-trials have a step > 5 mm, 7 have > 10 mm, 2 have > 20 mm.

Pattern: in `Suturing_I004` (step 2042) and `Suturing_D001` (step 739) **both
arms jump at the same frame**, by different amounts, and the position does not
return -- it settles at the new value (I004 PSM2 goes from (94, 55, -71) mm to
(89, 34, -55) mm). A simultaneous, sustained jump on both instruments points to a
shared cause, not one arm moving quickly. Two explanations, NEITHER verified:
the endoscope/camera moved (shifting the frame the positions are expressed in),
or frames were dropped from the log so the motion between two distant frames
appears as one step. Either way it is not real single-frame motion.

Effect: `speed_max` and `accel_max` are corrupted by these; do not trust them as
skill features as they stand. Path length is affected by at most ~1-2% in the
worst cases (the jump is 0.6-2.1% of a trial's total path). Working volume is
affected less than feared: I004 PSM2's hull is 181 cm^3 before the jump and
204 cm^3 with it (+13%); D001 PSM1 goes 10.8 -> 24.8 cm^3 but is mid-pack
overall. The largest hulls (`Suturing_C003/C004/C005` PSM2, 175-309 cm^3) are
NOT explained by these jumps. SPARC and gripper cycles were not examined for
this.

Not fixed. Any fix needs a rule (a step threshold, or detecting simultaneous
jumps on both arms) and a decision about what to do with the frames (drop,
interpolate, split the trial into segments), and any such rule is another
pre-declared choice. Decide it before Phase 3, and not by which version scores
best against GRS.

### Five data-quality decisions ACCEPTED (2026-09-20) -- APPLIED, see next entry

Made together, after label-free diagnostics, on validity and not on
performance. No labels were used. The bundle:

1. **Position jumps: drop `speed_max` and `accel_max`** (PSM1 and PSM2). They are
   what the jumps corrupt and they are redundant with each other (0.93-0.96). No
   repair rule, because the cause is unknown. Path length and volume stay, with
   the caveat recorded above.
   CORRECTION to the earlier entry: the "camera moved" explanation is NOT
   supported for the largest jumps. Only 8 trials have a simultaneous jump on
   both arms. In the three largest (`Suturing_D001`, `Needle_Passing_H002`,
   `Suturing_I004`) the tip-to-tip distance changes by 2.0-2.9x the smaller arm's
   step, so the scene did not shift as one rigid block. The three smaller events
   (ratios ~0.1-0.4) do look rigid. Cause is mixed and remains unverified.
2. **SPARC: dropped from the primary feature set.** Whole-trial SPARC correlates
   -0.83 with movement count and fails the "measures smoothness" test.
   Per-movement SPARC is not viable while movements are chatter (see 3). Code and
   config kept for a revisit. This is a deviation from PLAN.md, which lists
   smoothness as a feature. The plan's smoothness item is therefore unmet;
   PLAN.md's acceptance ("no NaNs, >= 3 separating features") does not depend on it.
3. **Smoothing: smooth positions ONCE, before every derived feature, with the raw
   pipeline kept as the sensitivity analysis.** Evidence: **40% of counted
   movements (3570 of 8888) are followed within 2 frames (<= 0.067 s) by another
   counted movement**, so raw speed chatters across the threshold and raw movement
   counts are inflated by noise. Cutoff and filter: see the entry below once chosen.
4. **Movement count: per-minute.** Raw count vs task time: Spearman +0.94 (PSM1),
   +0.91 (PSM2). Per-minute: +0.57 and -0.02. Removes the duplicate of task time.
5. **Gripper: accept the `Suturing_G002` PSM2 failure and report it.** 1 of 206
   arm-trials; a per-trial band would be a rule chosen after seeing that trial and
   would erase gripper range as a possible skill signal.

SPARC settings verified against the authors' reference implementation
(`sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05)`, github.com/siva82kb/SPARC),
closing that TODO(verify) for the VALUES. The paper citation is still TODO(verify).

### Bundle applied: 103 x 18 matrix, smoothed at 5 Hz (2026-09-20)

**Smoothing choice: 5 Hz**, zero-phase Butterworth (`scipy.signal.butter` +
`filtfilt`, second order per pass, so effectively fourth order), applied once to
tool-tip positions before every derived feature; the gripper angle is not
smoothed. Values live in `configs/features.yaml`. Because it runs forward and
backward the gain at the cutoff is 0.5 (-6 dB), not 0.707; tests pin this.
Behaviour (tested): 1 Hz and 2 Hz motion pass at 99% / 98%, 5 Hz at 50%, 8 Hz at
7%, 12 Hz (tremor band) at 0.1%.

Basis: a literature search found no JIGSAWS-specific cutoff. Physiological tremor
sits at 8-12 Hz (several sources); voluntary hand movement is low-frequency
(~0-2 Hz per a search summary of abstracts; NOT confirmed in full text); the
surgical-tracking hits for 5-10 Hz Butterworth were patents, weak evidence. The
evidence pins a RANGE of roughly 3-8 Hz, not a number; 5 Hz is a judgment call,
like the 1 mm/s idle threshold. TODO(verify) if a source turns up.

Pre-declared sensitivity analysis (not extra features): the primary result at
5 Hz, repeated on raw positions and at 8 Hz. `uv run python src/features.py`
writes `results/features.csv` (5 Hz), `features_raw.csv`, `features_8hz.csv`, all
gitignored.

Matrix now 103 x 18: `task_time_s`, `bimanual_speed_corr`, and 8 per arm
(path length, speed mean, speed std, accel mean, idle fraction, movements per
minute, hull volume, gripper cycles). Removed: `speed_max`, `accel_max`,
`sparc`, raw `n_movements`. No NaN under raw, 8 Hz or 5 Hz.

What smoothing did, label-free (median raw -> 5 Hz, rank agreement raw vs 5 Hz):
- Barely changed: path length (0.999), speed mean (0.999), idle fraction
  (0.99), hull volume (1.000), gripper cycles (1.000), bimanual correlation
  (0.983). These do not hinge on the cutoff.
- **Mean acceleration: PSM1 0.155 -> 0.066, PSM2 0.163 -> 0.061 m/s^2** (rank
  agreement 0.97 / 0.93). More than half of raw acceleration was measurement
  noise; the earlier "raw acceleration is noisy" flag is confirmed and now
  addressed.
- **Movements per minute: PSM1 31.1 -> 24.8, PSM2 28.1 -> 25.0**, rank agreement
  only 0.85 / 0.78. The feature most sensitive to the smoothing choice.
- The largest single-frame step falls from 26.9 mm (raw) to 14.8 mm (8 Hz) to
  10.1 mm (5 Hz): jump artifacts are cut by more than half but not removed.
- Collinearity: pairs at |Spearman| >= 0.9 fell from 7 to 2
  (`PSM2_speed_mean ~ PSM2_speed_std` 0.91, `PSM1_speed_mean ~ PSM1_accel_mean`
  0.91).

**NEGATIVE RESULT -- smoothing did NOT fix movement chatter.** The reason given
for smoothing (decision 3) was that 40% of counted movements were followed
within <= 2 frames by another. After smoothing that share is **35.5% (8 Hz) and
34.6% (5 Hz)**, down from 40.2% raw; total counted movements fell only from
8888 to 7810 (-12%). So the chatter is not mainly high-frequency noise. Speed
hovers around the 1 mm/s threshold during slow, fine motion, and that slow drift
crosses the threshold repeatedly however the positions are filtered. Per-minute
movement counts are therefore still inflated by threshold crossings, and remain
the least trustworthy feature. Idle fraction (a share of time below threshold)
does not suffer this way, which is why it is stable (rank agreement 0.99).

Options for movement counting, NOT yet chosen (each adds a parameter set after
seeing this result): (a) keep `movements_per_min` and report the caveat;
(b) use hysteresis on speed (start above one threshold, end below a lower one),
as already done for the gripper; (c) merge gaps of <= g frames; (d) drop
`movements_per_min` and keep idle fraction. Decide before Phase 3, by validity
and not by which scores best against GRS.

Decision 3's rationale was therefore only partly borne out: smoothing is still
justified by the acceleration result and the jump reduction, but not by the
movement-splitting one.

### Movement count DROPPED from the matrix (2026-09-20) -- 103 x 16

Decision: option (d) of the four offered after the smoothing negative result.
`movements_per_min` is removed from the feature matrix (both arms). Reasons:
threshold crossings chatter even after smoothing (34.6% of counted movements
still followed within <= 2 frames by another), it is the feature most sensitive
to the smoothing choice (rank agreement 0.78-0.85 raw vs 5 Hz), and every
repair (hysteresis, gap merging) adds parameters chosen after seeing this
result. Idle fraction, which is a share of time and not a count of crossings,
stays and is stable (rank agreement 0.99).

Cost: PLAN.md lists "number of movements" as a Phase 2 feature; that item is now
unmet, as is smoothness (SPARC). PLAN.md's acceptance (103 rows, no NaNs, >= 3
features that visibly separate skill levels) does not depend on either.
`count_movements()` and its tests, `min_movement_duration_s` in the config, and
`sparc()` are kept in case we revisit; `moving_speed_threshold` is still used by
idle fraction.

Final matrix: **103 x 16**, no NaN under raw, 8 Hz or 5 Hz. Columns:
`task_time_s`, `bimanual_speed_corr`, and per arm (PSM1, PSM2) path length, speed
mean, speed std, accel mean, idle fraction, hull volume, gripper cycles.
Pairs at |Spearman| >= 0.9: two (`PSM1_speed_mean ~ PSM1_accel_mean` 0.91,
`PSM2_speed_mean ~ PSM2_speed_std` 0.91). 132 tests pass.

### Phase 2 boxplots: rules PRE-DECLARED before any figure exists (2026-09-20)

This is the first time the labels are looked at against the features. The rules
below are fixed now, in a commit that precedes the plotting code, so they cannot
bend to what the plots show.

**Main grouping: GRS bands, computed WITHIN each task** (user's choice; GRS
distributions differ by task -- median 19 Suturing, 14.5 Knot-Tying, 13
Needle-Passing -- so a pooled cut would make "high" mean different things and
would mostly separate tasks). Cut points are the 33.3rd and 66.7th percentiles of
that task's GRS: **low = GRS <= q33; high = GRS > q67; mid = otherwise**. Ties
share a band (no arbitrary tie-breaking), so band sizes are unequal; they are
reported. "High" therefore means a different absolute GRS in each task.
Secondary grouping: the self-declared E/I/N label, same layout.

**Layout:** one figure per task, one small panel per feature (16), each with its
own y-axis, showing every trial as a point over a box. No pooling across tasks.

**Numeric supplement:** Spearman rho between each feature and GRS, per task, with
n. 16 features x 3 tasks = 48 comparisons; no p-values are used to select
features, and nothing here is a significance claim.

**Pre-stated directions (higher GRS = more skilled), and their basis:**
- `task_time_s`: lower with higher GRS (PLAN.md: experts show shorter times).
- `*_path_length`: lower (PLAN.md: shorter paths).
- `*_idle_fraction`: lower (from the literature hits on idle time, unverified;
  TODO(verify) with a source).
Every other feature is exploratory: NO direction is stated, and none may be
claimed afterwards as "expected".

**Stop rule (PLAN.md):** if the pre-stated-direction features run the OPPOSITE way
in most tasks, stop and investigate the pipeline; do not adjust features.

**No numeric pass/fail cutoff is set** for PLAN.md's "at least three features
that visibly separate skill levels". That is a visual judgment; the plots and rho
table are reported and the user judges. No cutoff will be invented afterwards.

**Not allowed after seeing the plots:** changing any feature, threshold, the
smoothing cutoff, the bands, or the set of features shown. Anything that looks
wrong is logged and decided by the user, not fixed to make the picture better.

### Phase 2 boxplots: what they show (2026-09-20)

`uv run python src/plot_features.py` writes 6 figures to `results/figures/`
(GRS bands: main, `features_by_grs_band_<task>.png`; self-declared skill:
secondary, `features_by_skill_<task>.png`) and `results/feature_grs_spearman.csv`.
Both are gitignored (regenerable, derived). Palette: three steps of the dataviz
skill's blue ramp (250/450/650), validated with `validate_palette.js --ordinal`,
all four checks pass. The band rule has 7 tests (`tests/test_plot_features.py`).
Rules were pre-declared in the entry above and were followed: no feature,
threshold, band or panel was changed after seeing a figure.

Band sizes (ties share a band, so unequal): Suturing 16/11/12, Knot-Tying
14/12/10, Needle-Passing 12/7/9 (low/mid/high). GRS ranges of the bands:
Suturing low 8-17, mid 18-21, high 23-30; Knot-Tying 6-11, 12-18, 19-22;
Needle-Passing 7-12, 13-15, 17-24. "High" is a different absolute GRS in each task.

**Stop rule: NOT triggered.** Of the 15 feature-task cells with a pre-stated
direction (task time, both arms' path length, both arms' idle fraction; expected
rho < 0), **12 run the expected way**, and every task has >= 3 of 5:
Suturing 3, Knot-Tying 5, Needle-Passing 4.

Spearman rho with GRS (n = 39 / 36 / 28):

| feature | Suturing | Knot-Tying | Needle-Passing |
|---|---|---|---|
| task_time_s | -0.34 | -0.79 | -0.18 |
| PSM1_path_length | -0.31 | -0.22 | -0.35 |
| PSM1_speed_mean | +0.12 | +0.68 | -0.16 |
| PSM1_speed_std | +0.26 | +0.63 | +0.02 |
| PSM1_accel_mean | +0.12 | +0.68 | -0.26 |
| PSM1_idle_fraction | +0.03 | -0.45 | +0.55 |
| PSM1_working_volume | -0.05 | +0.43 | -0.04 |
| PSM1_gripper_cycles | -0.30 | -0.46 | -0.27 |
| PSM2_path_length | +0.11 | -0.50 | -0.13 |
| PSM2_speed_mean | +0.55 | +0.28 | +0.07 |
| PSM2_speed_std | +0.60 | +0.20 | +0.03 |
| PSM2_accel_mean | +0.44 | +0.24 | +0.07 |
| PSM2_idle_fraction | -0.37 | -0.18 | -0.21 |
| PSM2_working_volume | +0.47 | -0.12 | +0.10 |
| PSM2_gripper_cycles | -0.47 | -0.63 | -0.39 |
| bimanual_speed_corr | +0.44 | +0.56 | +0.20 |

**Pre-stated features that went the WRONG way (logged, not fixed):**
- `PSM1_idle_fraction` in **Needle-Passing: rho = +0.55**, the opposite of the
  expectation, and not small: the band medians rise from about 0.125 (low) to
  0.185 (high). In Suturing it is +0.03 (no relationship); only in Knot-Tying is
  it the expected -0.45. PSM2's idle fraction goes the expected way in all three
  tasks (-0.37, -0.18, -0.21), so the two arms disagree.
- `PSM2_path_length` in Suturing: +0.11 (weak, wrong way).
Neither has an explanation yet, and none is claimed.

Strongest expected-direction result: `task_time_s` in Knot-Tying, rho = -0.79.
It is weak in Needle-Passing (-0.18).

**Exploratory, NO direction was pre-stated, so none may be called "expected":**
features with the same sign in all three tasks are, negative: task time, PSM1
path length, both arms' gripper cycles, PSM2 idle fraction; positive: bimanual
speed correlation, PSM2 mean speed, PSM2 speed SD, PSM2 mean acceleration, and PSM1
speed SD (but only +0.02 in Needle-Passing, effectively zero there).
Gripper cycles (rho -0.47 / -0.63 / -0.39 on PSM2) is the notable one among
those with no prior. Speed features flip sign across tasks on PSM1 (mean speed
+0.12 / +0.68 / -0.16), so do not read a single "faster = better".

**Read these rhos with care:**
- Roughly, a single correlation reaches the usual 5% line at |rho| >=
  1.96 / sqrt(n - 1) = 0.32 (Suturing), 0.33 (Knot-Tying),
  0.38 (Needle-Passing). 20 of 48 cells clear it
  (8 / 10 / 2 by task); about 2-3 would by
  chance alone with 48 comparisons. A rough guide only, NOT used to select features.
- **The trials are not independent.** Each surgeon contributes ~5 trials per
  task, so the effective sample is closer to 7-8 surgeons than to 28-39 trials,
  and these rhos look more precise than they are. Skill is confounded with
  surgeon identity (the main threat to validity, CLAUDE.md): a correlation here
  can reflect a surgeon's style, not skill. Descriptive only; LOUO in Phase 3 is
  the check that bites.
- Units are assumed SI and unverified (footnote on every figure).

PLAN.md's "at least three features that visibly separate skill levels": no
numeric cutoff was set in advance and none is invented here. The judgment is
left to the user, with the figures and this table as the evidence.

## Phase 3 -- baseline: design PRE-DECLARED before any model runs (2026-09-20)

Fixed in `configs/baseline.yaml` and committed ahead of the code. As in Phase 2,
nothing here may change after results are seen; a change is a new logged decision
under a new config file.

**Inputs and scope.** All 16 features from `results/features.csv` (5 Hz smoothed),
no feature selection. One model per task; nothing pooled across tasks. LOSO and
LOUO folds from `loaders.load_splits`. Pre-declared robustness runs: the raw and
8 Hz matrices.

**Leakage safeguards.** The scaler is fit on the training fold only (inside a
scikit-learn Pipeline); hyperparameters are chosen only inside the training set;
held-out trials are never seen. Tests check that the fitted model cannot depend
on test-fold labels. Plus a sentinel: refit 50 times with labels shuffled; the
mean Spearman must be within +/-0.10 of zero, else suspect a leak. (The 0.10 is
my judgment, declared now; it is a sanity check, not a result.)

**Regression (GRS).** Ridge on standardized features. Alpha in {0.1, 1, 10, 100},
chosen by leave-one-SUBJECT-out inside each training set (scoring: negative mean
squared error). Grouping by subject stops the inner search from rewarding a model
that recognizes a surgeon's style. This is stricter than LOSO itself, on purpose.

**Classification (N/I/E).** Multinomial logistic regression (scikit-learn, no new
dependency), L2, **C = 1.0 FIXED, not searched.** Reason: the classes are held by
few surgeons (2 experts, 2 intermediates, 3-4 novices), so in LOUO the training set
can have a single expert; a leave-one-subject-out inner search would then hold that
surgeon out and leave the inner training set with NO experts, making the search
ill-defined. C = 1.0 is scikit-learn's default and an arbitrary choice; expect it
to matter. Ignores that N < I < E is ordered (limitation; ordinal model not chosen).
No class weighting although classes are unbalanced (46 N / 28 I / 29 E).
This asymmetry with the regression (searched vs fixed) is a consequence of the
data, not a preference, and the user did not specify the value of C.

**Headline statistic.** ONE Spearman rho per task and scheme, from the pooled
out-of-fold predictions (every trial predicted once by a model that never saw it),
plus Kendall tau, MAE and RMSE for context. Not the mean of per-fold correlations:
a LOUO fold has ~4-5 trials. Classification: accuracy, balanced accuracy,
quadratic-weighted kappa.

**Uncertainty.** 95% percentile interval from resampling whole surgeons with
replacement (2000 resamples, seed 0), holding each trial's out-of-fold prediction
fixed. Limits, stated now: with 7-8 surgeons the interval is crude and will be
wide; it captures who the evaluation surgeons are, NOT variability from refitting
on different training sets; resamples where a statistic is undefined (e.g. a
constant y) are skipped and counted.

**Logging.** One row per run in `results/experiments.csv`: timestamp, config file,
git commit, feature set, CV scheme, task, model, seed, library versions, metrics.
Out-of-fold predictions and `predict_proba` saved under `results/predictions/`
(gitignored: derived from JIGSAWS) for Phase 5.

**Acceptance (PLAN.md).** LOSO Spearman rho on Suturing in the rough
neighbourhood of 0.5-0.7. Comparison figures in PLAN.md (Zia & Essa, IJCARS 2018:
~0.61 average, ~0.75 / 0.63 / 0.46 per task under LOSO, ~0.41 under LOUO) are
carried from PLAN.md and NOT verified by me. TODO(verify) against the paper before
any comparison goes in the write-up. Their protocol and features differ from ours.

**Known threat.** Skill is confounded with surgeon identity. LOSO leaves the same
surgeons in training, so a model can score well by recognizing style; the LOSO to
LOUO gap is the honest measure of that. Report both.

## Phase 3 results (2026-09-20)

Run from a clean tree at commit `cb813a2`; the diagnostics at a clean tree one
commit later. 42 rows in `results/experiments.csv` (18 regression, 18
classification, 6 sentinel) plus 9 post-hoc diagnostic rows. Out-of-fold
predictions and `predict_proba` are in `results/predictions/` (gitignored). Design
was pre-declared (entry above) and NOT changed after seeing results.

**GRS regression, primary features (5 Hz).** Pooled out-of-fold Spearman rho, 95%
interval from resampling surgeons (7-8 of them, so crude):

| task | LOSO rho [95% interval] | LOUO rho [95% interval] |
|---|---|---|
| Suturing | +0.66 [+0.21, +0.84] | +0.21 [-0.26, +0.67] |
| Knot-Tying | +0.79 [+0.41, +0.87] | +0.64 [+0.15, +0.82] |
| Needle-Passing | +0.46 [+0.15, +0.73] | -0.13 [-0.39, +0.27] |

MAE (GRS points): LOSO 3.20 / 2.88 / 3.79; LOUO 4.56 / 3.56 / 4.91. Kendall tau is
in the log. LOUO intervals for Suturing and Needle-Passing include zero; Knot-Tying
LOUO excludes it. The average LOUO rho is about 0.24. Bootstrap resamples skipped
for an undefined Spearman: 0 in every run.

**Acceptance (PLAN.md): LOSO Spearman on Suturing in the neighbourhood of
0.5-0.7 -> 0.66, MET.** Read it with the caveat below: LOSO leaves the same
surgeons in training. PLAN.md's reference figures (Zia & Essa 2018: per-task LOSO
~0.75 / 0.63 / 0.46, LOUO average ~0.41) are UNVERIFIED (TODO(verify)) and their
protocol and features differ; ours are 0.66 / 0.79 / 0.46 and ~0.24.

**Skill classification (N/I/E), multinomial logistic, C = 1 fixed:**

| task | LOSO accuracy / balanced / kappa | LOUO accuracy / balanced / kappa | majority-class rate |
|---|---|---|---|
| Suturing | 0.92 / 0.92 / +0.77 | 0.44 / 0.41 / -0.19 | 0.49 |
| Knot-Tying | 0.97 / 0.97 / +0.98 | 0.31 / 0.28 / +0.13 | 0.44 |
| Needle-Passing | 0.93 / 0.93 / +0.80 | 0.39 / 0.40 / -0.46 | 0.39 |

LOUO is at or BELOW the rate for always guessing the biggest class (0.44 vs 0.49,
0.31 vs 0.44, 0.39 vs 0.39), and kappa is negative for Suturing and
Needle-Passing (worse than chance). Balanced accuracy and kappa intervals are
approximate: bootstrap resamples that omit a class are scored on the classes
present (scikit-learn warned about this many times).

**MAIN FINDING -- the LOSO skill result is largely surgeon recognition.**
(Post-hoc diagnostic, `src/diagnostics.py`, not pre-declared.) Trained on the
other repetitions, the same features name the right SURGEON with accuracy 0.87 /
0.83 / 0.93 (Suturing / Knot-Tying / Needle-Passing) against chance 0.12 / 0.12 /
0.14. The self-declared skill label is a property of the surgeon, constant across
their trials, so a model that recognises a surgeon can look up the label. That is
consistent with LOSO skill accuracy of 0.92-0.97 collapsing to 0.31-0.44 under
LOUO. It is consistent with, not proof of, the classifier using identity; LOUO is
the number that means something. The regression is exposed to the same problem
(GRS has a surgeon-level component): LOSO to LOUO is 0.66 to 0.21, 0.79 to 0.64,
0.46 to -0.13. Knot-Tying is the task where the signal survives LOUO best, and
task time is its strongest single feature (Phase 2, rho -0.79). This is the
subject-identity confound named in CLAUDE.md, now measured.

**Leakage sentinel: the pre-declared criterion FAILED in 2 of 6 cells.** Criterion
declared in advance: mean Spearman under 50 shuffled-label refits within +/-0.10 of
zero.

| scheme | task | ridge sentinel mean rho | intercept-only mean rho | criterion |
|---|---|---|---|---|
| LOSO | Suturing | -0.147 | -0.280 | FAIL |
| LOSO | Knot-Tying | -0.116 | -0.284 | FAIL |
| LOSO | Needle-Passing | -0.082 | -0.331 | pass |
| LOUO | Suturing | -0.092 | -0.392 | pass |
| LOUO | Knot-Tying | -0.097 | -0.395 | pass |
| LOUO | Needle-Passing | -0.067 | -0.426 | pass |

All six means are NEGATIVE, the opposite direction to leakage. A post-hoc check
(`src/diagnostics.py`) shows why: a predictor that ignores every feature and just
predicts the training-label mean, under the same shuffles, scores -0.28 to -0.43.
So cross-validated rank correlation is pushed below zero by construction (a test
trial with an above-average label pulls the training average below it), and ridge
sits between zero and that null. The criterion was mis-specified: I wrote a
two-sided "near zero" without accounting for this bias. It is NOT being changed
retroactively; the two FAILs are recorded as failures of the criterion as written.
Reading: no evidence of leakage (which would push the mean positive), and the
headline rhos above are, if anything, slightly UNDERSTATED by this bias. A
corrected criterion (flag only a mean above +0.10, and compare with the
intercept-only null) would be a new logged decision.

The unit tests also confirm the safeguards can detect a leak: changing test-fold
labels leaves that fold's predictions bit-identical, and a deliberately leaky
variant (standardizing all rows before splitting) moved untouched test
predictions by 16 GRS points where the real pipeline moved them by exactly 0.

**Ridge alpha hit the edge of the declared grid.** alpha = 100, the largest value,
was chosen in 25 of 38 outer folds (LOSO Suturing 5/5, LOSO Knot-Tying 5/5, LOSO
Needle-Passing 2/5, LOUO Suturing 6/8, LOUO Knot-Tying 4/8, LOUO Needle-Passing
3/7). The inner search wanted more shrinkage than the grid allowed, so the models
may be under-regularised at the edge. Not changed. Widening the grid would be a new
logged decision under a new config; it would use only inner-CV evidence, not
outer-fold results, but the temptation should be acknowledged.

**Sensitivity to smoothing (pre-declared): conclusions do not change.** GRS rho,
5 Hz / raw / 8 Hz:
- LOSO: Suturing 0.66 / 0.66 / 0.66; Knot-Tying 0.79 / 0.80 / 0.79;
  Needle-Passing 0.46 / 0.46 / 0.30 (the one place the cutoff matters).
- LOUO: Suturing 0.21 / 0.10 / 0.15; Knot-Tying 0.64 / 0.73 / 0.62;
  Needle-Passing -0.13 / -0.24 / -0.27.

**Limitations to carry into the paper.** 7-8 surgeons per task, so surgeon-level
intervals are crude and do not reflect refitting variability; skill confounded with
surgeon identity (measured above); C fixed at 1.0 and not searched (LOUO
classification is at or below the majority rate, so this was not rescued by
tuning); the classifier ignores that N < I < E is ordered; no class weighting
despite unbalanced classes; units unverified.

**Decisions now open (each would be a new logged decision):** widen the alpha grid;
replace the sentinel criterion; whether LOUO is the headline in the paper; what to
do about the classifier (fixed C, unordered).

## Decisions after the Phase 3 results (2026-09-20)

The user accepted my recommendations on the four open decisions, after I
re-explained them in plain terms. Made AFTER seeing the results, and logged as such.

1. **Widen the ridge alpha grid -> YES, in a NEW config, `configs/baseline_v2.yaml`.**
   Grid becomes 0.1, 1, 10, 100, 1000, 10000. The original config and every result
   under it are untouched; v2 is reported ALONGSIDE, never instead. Rows are
   distinguished by `config_file`; prediction files by the `v2_` prefix. The
   motivation used only inner-cross-validation evidence (alpha = 100 chosen in 25 of
   38 folds, i.e. the optimum sat on the grid boundary), not outer-fold results, but
   the decision was made after seeing both, and that is disclosed.
   **Rule fixed before the run: this is the last extension.** If 10000 is chosen
   often it is reported as is; the grid is not widened again. Only regression is
   re-run (classification is unaffected: C is fixed).
2. **Replace the leakage criterion -> YES, one-sided, in v2.** Flag only a shuffled-
   label mean Spearman ABOVE +0.10, and log the feature-free null alongside. The
   original two-sided rule and its two recorded FAILs stand as they are.
3. **Headline number for the paper -> LOUO, with LOSO reported alongside.** LOUO
   (a new surgeon) is what a real assessor would face; LOSO (a new trial from a
   known surgeon) is partly surgeon recognition. CLAUDE.md already requires
   reporting both. The LOSO-to-LOUO gap is presented as a finding about the
   subject-identity confound. Phase 5 tables list LOUO first.
4. **The skill classifier -> LEAVE ALONE and report honestly.** C stays 1.0, no
   ordinal model, no class weights. With 2 experts and 2 intermediates any tuning to
   improve LOUO would mostly fit noise. It is reported as a limitation and a
   finding. Consequence for Phase 5: the calibration and selective-prediction
   analysis on the baseline's `predict_proba` will be run on a classifier that does
   no better than the majority class on new surgeons. That is still a valid analysis
   (a weak model can be miscalibrated, and confidence can still rank trials), but it
   must be described as such.

## Phase 3, second config (v2) and the feature-free reference (2026-09-20)

v2 run at commit `ee96645` (regression only, primary + raw + 8 Hz, plus the sentinel).
Reported ALONGSIDE the original, never instead. The feature-free reference was added
afterwards (post-hoc, `src/diagnostics.py C`, commit-clean run); its log rows appear
twice (first Spearman only, then Spearman and MAE): use the later ones.

**The wider grid did not help.** Three of six primary cells are IDENTICAL to the
original (Suturing LOSO 0.66, Knot-Tying LOSO 0.79, Knot-Tying LOUO 0.64; alpha = 100
chosen exactly as before), so alpha = 100 was a true optimum there and the
"edge of the grid" worry was unfounded for them. In the other three the search chose
alpha = 10000 and the correlations FELL: Needle-Passing LOSO +0.46 -> +0.00,
Needle-Passing LOUO -0.13 -> -0.44, Suturing LOUO +0.21 -> -0.04. The new maximum
(10000) was chosen in 9 of 38 primary folds (Needle-Passing LOSO 3/5, Needle-Passing
LOUO 5/7, Suturing LOUO 1/8). **Rule declared beforehand: this was the last
extension; the grid is not widened again.**

Leakage sentinel, one-sided: **PASS in all 6 cells** (means -0.16 to -0.26; the
feature-free null is -0.28 to -0.43). The means are more negative than under the
original grid because heavier shrinkage moves the model toward the feature-free
predictor.

**Feature-free reference: what the numbers actually mean.** Predict every held-out
trial with the mean GRS of its fold's training trials, on the REAL labels:

| scheme | task | rho: feature-free / v1 / v2 | MAE (GRS points): feature-free / v1 / v2 |
|---|---|---|---|
| LOSO | Suturing | -0.18 / +0.66 / +0.66 | 4.35 / 3.20 / 3.20 |
| LOSO | Knot-Tying | -0.16 / +0.79 / +0.79 | 4.66 / 2.88 / 2.88 |
| LOSO | Needle-Passing | -0.41 / +0.46 / +0.00 | 4.22 / 3.79 / 4.29 |
| LOUO | Suturing | -0.78 / +0.21 / -0.04 | 4.71 / 4.56 / 5.09 |
| LOUO | Knot-Tying | -0.86 / +0.64 / +0.64 | 5.10 / 3.56 / 3.56 |
| LOUO | Needle-Passing | -0.79 / -0.13 / -0.44 | 4.41 / 4.91 / 4.50 |

1. **Pooled Spearman is dominated by an artifact under LOUO.** A predictor that
   knows nothing scores -0.78 to -0.86. Mechanism: each LOUO fold is one surgeon and
   the prediction is one constant per fold; surgeons differ a lot in mean GRS, and a
   surgeon with a high mean gets a LOWER training mean, so prediction and truth move
   oppositely across folds. A model's LOUO rho therefore cannot be read against zero;
   e.g. Knot-Tying's +0.64 is against a null of -0.86. This is a flaw in the
   pre-declared headline statistic (pooled out-of-fold rho), found after the fact.
   It stays reported, but never without the reference.
2. **MAE has no such artifact, and says:** a model has learned something only if its
   MAE is below the feature-free MAE.
   - LOSO: clear gains in Suturing (3.20 vs 4.35) and Knot-Tying (2.88 vs 4.66);
     small in Needle-Passing under v1 (3.79 vs 4.22) and none under v2 (4.29).
   - **LOUO: only Knot-Tying shows a real gain over predicting the mean (3.56 vs
     5.10, about 30% lower).** Suturing is essentially no better (v1 4.56 vs 4.71)
     or worse (v2 5.09); Needle-Passing is no better (v1 4.91, v2 4.50 vs 4.41).
   Consistent with Phase 2, where task time was strongest in Knot-Tying (rho -0.79).
3. In Needle-Passing the inner search chose maximum shrinkage (alpha = 10000), i.e.
   it turned the model into the feature-free predictor. That is the search reporting
   "nothing here generalises across surgeons".
4. **Design tension, recorded not fixed:** alpha is chosen by mean squared error but
   the headline was a rank correlation. Heavy shrinkage minimises squared error yet
   produces near-constant predictions that score badly on rank correlation by
   construction. Changing the inner criterion would be a new decision.

**What this does to the conclusions.** The LOSO results are real improvements over
the mean but partly reflect surgeon recognition (83-93% surgeon-ID accuracy). The
honest cross-surgeon result is: a signal in Knot-Tying, and none demonstrable in
Suturing or Needle-Passing. No confidence interval was computed on the MAE
differences; with 7-8 surgeons they would be wide, and the Knot-Tying gain should be
described as a point estimate until one is.

**Recommendation awaiting the user (a new decision):** report LOUO as the headline
in terms of MAE relative to the feature-free reference, with pooled rho shown beside
the reference, and add a surgeon-bootstrap interval on the MAE difference.

### Headline metric: decided, rule fixed before the intervals exist (2026-09-20)

The user accepted the recommendation. **The paper's headline is LOUO, reported as
the model's MAE against the feature-free reference (predict each held-out trial
with the training-fold mean GRS), as an improvement in GRS points with a 95%
surgeon-bootstrap interval. Pooled Spearman is reported beside its own reference,
never alone.** LOSO is reported alongside. Implemented in `src/report.py`;
statistic in `metrics.mae_improvement`.

**Interpretation rule, fixed BEFORE the intervals were computed:** a model beats the
reference across new surgeons only if the LOWER end of its 95% interval for the MAE
improvement is above zero. With 7-8 surgeons that interval is crude, so "excludes
zero" is suggestive and not proof; nothing is selected by it. The interval holds both
sets of predictions fixed (no refit; the reference's training mean is not recomputed
inside a resample), so it reflects which surgeons were evaluated, not training
variability.

### Headline results: MAE improvement over predicting the mean (2026-09-20)

`uv run python src/report.py` (clean tree, commit after the rule was committed);
36 rows in `results/headline_table.csv` (gitignored) and in `experiments.csv`
(`kind = headline_mae_vs_reference`). Improvement = reference MAE - model MAE in GRS
points; positive = model better. Interval: 95%, resampling whole surgeons, predictions
held fixed. Rule (fixed beforehand): the model beats the reference only if the LOWER
end of the interval is above zero.

Primary features (5 Hz), **LOUO (the headline)**:

| task | config | reference MAE | model MAE | improvement [95% interval] | beats? | rho model / reference |
|---|---|---|---|---|---|---|
| Suturing | v1 | 4.71 | 4.56 | +0.15 [-1.04, +1.33] | no | +0.21 / -0.78 |
| Knot-Tying | v1 | 5.10 | 3.56 | **+1.54 [+0.65, +2.52]** | **YES** | +0.64 / -0.86 |
| Needle-Passing | v1 | 4.41 | 4.91 | **-0.50 [-1.21, -0.04]** | no | -0.13 / -0.79 |
| Suturing | v2 | 4.71 | 5.09 | -0.37 [-1.29, +0.33] | no | -0.04 / -0.78 |
| Knot-Tying | v2 | 5.10 | 3.56 | +1.54 [+0.65, +2.52] | YES | +0.64 / -0.86 |
| Needle-Passing | v2 | 4.41 | 4.50 | -0.09 [-0.29, +0.08] | no | -0.44 / -0.79 |

LOSO (reported alongside; includes surgeon recognition, see the surgeon-ID finding):
v1 beats the reference in all three tasks (Suturing +1.15 [+0.47, +1.92], Knot-Tying
+1.78 [+1.00, +2.58], Needle-Passing +0.43 [+0.11, +0.76]); v2 beats it in Suturing
and Knot-Tying (identical numbers) but not Needle-Passing (-0.07 [-0.67, +0.49]).

**Reading, by the pre-declared rule:**
- **Knot-Tying is the only task where the model beats predicting the mean on a NEW
  surgeon**: about 1.5 GRS points, or roughly 30% lower error (3.56 vs 5.10).
- **Suturing: no demonstrable gain** across new surgeons (point estimates +0.15 and
  -0.37, intervals spanning zero widely).
- **Needle-Passing under v1: the model is WORSE than the mean** (-0.50, interval
  entirely below zero, though only just: upper end -0.04). Under v2's heavier
  shrinkage it collapses to about the reference (-0.09). More flexibility hurts on
  new surgeons here.

**Sensitivity to smoothing, LOUO (pre-declared runs, all six config x feature-set
combinations):** Knot-Tying's improvement is positive in all six (+1.18 to +1.69) but
its interval excludes zero in only 4 of 6: at 8 Hz the lower end is -0.26, so that
run just misses the rule. Suturing never beats the reference (points -0.76 to +0.15).
Needle-Passing is negative in all six (-0.09 to -0.80), with the interval entirely
below zero in 2 of 6. So the Knot-Tying claim is "consistently positive, clears the
bar in most but not all variants"; it should not be stated more strongly than that.

Caveats to carry forward: 7-8 surgeons, so intervals are crude; the interval ignores
training variability; the reference is one specific baseline (the training mean), and
a stronger one (e.g. a per-task median or a task-time-only model) would be a harder
bar and has not been run. The improvement of about 1.5 points is against a scale of
6-30 with a per-task GRS standard deviation of 4.8-5.4 points (Suturing 5.40,
Knot-Tying 5.11, Needle-Passing 4.82), so it is modest in absolute terms even where
it is credible: 0.23 to 0.33 of a standard deviation for Knot-Tying (1.18 and 1.69
divided by 5.11), i.e. about a quarter to a third.

## Phase 4 -- Jev: design PRE-DECLARED before any dataset-derived call (2026-09-20)

**API facts, verified from TypeSafe's own docs (docs.typesafe.ai) and the installed SDK,
not from memory.** `POST https://api.typesafe.ai/v1/systemone`, bearer key; request
`{state, model, questions}`; a Score question is `{type: "score", instructions,
criteria}` with 2-10 ordered levels, level numbers from 0; response `{model, answers,
usage}` with, per Score, `score` (the probability-weighted level, so it can fall between
levels), `probabilities`, `legend` and `confidence` (0-1, from the shape of the
distribution). `jev-latest` is an alias of `jev-1.13.0`; the response's `model` field
reports the versioned id. Limits: 64k tokens per request, 32k for state plus the longest
question; 1,200 requests/min (adjusting dynamically); about $0.042 per million input
tokens, output free. Errors: 401 bad key, 422 malformed, 429/529 retry with backoff (the
SDK does; we set max_retries 5). The docs do NOT specify determinism, state-size limits
beyond the above, or a fixed retention period. Package: `typesafe-sdk` 0.7.0 (PyPI,
Python >= 3.10), added 2026-09-20 with the user's approval.

**The model's documented weaknesses drive the design** (docs, "Jev 1.13 jaggedness"): it
"does not count reliably", cannot reliably judge whether two numbers are near each other,
its score levels "are weak in numerical calibration" and "will not be able to help you
reconstruct the exact number by interpolating", and accuracy "falls as the state grows
with content unrelated to the decision". Hence: no number is ever sent, the state is
short, and the continuous score is treated as a rank signal, with a weak magnitude
estimate at best.

**Smoke test:** one made-up sentence and one Score question, model pinned to
`jev-1.13.0`; the API reported `jev-1.13.0`; score 1.9, confidence 0.86, 320 input
tokens. Raw response saved as `results/jev_raw/_smoke_test.json`. No project data sent.

**Data terms (read 2026-09-20; for the ethics statement).** Privacy policy: "We will not
train or fine tune any artificial intelligence or machine learning models on your prompts
or other Input"; inputs are not disclosed "to a third party other than our service
providers" (sub-processor list at trust.typesafe.ai, NOT read). **Retention has no fixed
period** ("as long as reasonably necessary"); the DPA says nothing on deletion on request;
research use and sensitive data are not addressed. NHSR determination still pending.
Judgement recorded: what is sent is wording derived from a de-identified public dataset,
not the dataset (no video, no trajectories, no identifiers, no labels, no numbers).

**Prompt v1, fixed in `configs/jev.yaml` and hashed into every cached response.**
- State: the task in words, and for each of the 16 features one phrase placing it against
  the other trials of the SAME task ("typical of comparable attempts", ...). Percentile =
  share of the task's trials below it, ties counted half; bands cut at 0.10 / 0.30 / 0.70 /
  0.90 (arbitrary, conventional, declared now). The reference set is label-free but
  includes the trial itself and other surgeons' trials (transductive, no labels).
  No trial name, surgeon, repetition, skill label, score or digit; tests enforce this.
- Questions: 7 Score questions, 5 levels each, one dimension each, situations not
  degrees, in ONE request per trial: `overall_skill` (primary) and the six OSATS
  elements in the readme's order. The element order is on the readme's word, so
  element-level comparisons are EXPLORATORY. Two elements (respect for tissue, quality of
  final product) concern things motion data cannot show; kept for completeness.
- Model pinned `jev-1.13.0`; `response.model` checked against it on every fresh AND cached
  record; each trial scored 3 times (repeatability); raw JSON cached to
  `results/jev_raw/<trial>__run<k>.json` BEFORE parsing, with the exact state sent; a
  cached trial is never re-requested. Cost estimate: about 309 requests, cents in total.
- **No revision.** The wording was written after the Phase 2 and Phase 3 results were
  seen; it is from general OSATS ideas and encodes no specific finding, but contamination
  cannot be ruled out. With 103 trials there is no held-out set to tune on, so the wording
  is not iterated against GRS. A change would be a new version (v2, new file) run alongside
  and reported alongside.

**Analysis plan, fixed before any score exists:**
1. Predictor: `overall_skill` score from run 0 (the single-call use a real deployment
   would have). Sensitivities: mean of the 3 runs; the mean of the six element scores.
2. **Rank agreement:** Spearman of the score with GRS per task, surgeon-bootstrap interval.
   Jev is never trained, so its pooled correlation has none of the cross-validation bias
   that pulls the baseline's LOUO correlation toward -0.8; the two must NOT be compared
   as raw correlations.
3. **Common yardstick (the headline):** MAE improvement over the feature-free reference,
   as for the baseline. Within each training fold, fit an ordinary least-squares line
   GRS ~ a + b * score on the TRAINING trials only and predict the held-out trials, under
   LOUO (headline) and LOSO; surgeon bootstrap; same rule (lower end of the 95% interval
   above zero). Reported side by side with the baseline.
4. **Direction guard:** the sign of the fitted slope b is reported for every fold. A
   negative slope means Jev ranks skill backwards; it is reported as that, and is NOT
   silently corrected by the line (a fitted line would otherwise turn an inverted score
   into an apparently good predictor).
5. **Repeatability:** between-run agreement on `overall_skill` (Spearman; mean absolute
   score difference). Identical answers would be reported as apparent determinism.
6. Jev's `confidence` and level probabilities are logged for Phase 5.
7. Element-level agreement with the six OSATS columns: exploratory, order unverified.

Not run: a full-file sensitivity analysis.Not run: a full-file sensitivity analysis. Offered as an option, not chosen.
Cheap to add later since `windowed=False` is the default.
