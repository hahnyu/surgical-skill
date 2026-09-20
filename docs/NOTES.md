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

### Five data-quality decisions ACCEPTED (2026-09-20) -- rebuild pending

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

Not run: a full-file sensitivity analysis. Offered as an option, not chosen.
Cheap to add later since `windowed=False` is the default.
