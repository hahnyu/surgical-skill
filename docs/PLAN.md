# PLAN.md

Phased task list. Each phase has acceptance criteria. **Do not start a phase
until the previous one's criteria are met.** Mark items done as you go.

Estimated total: 4–5 months at roughly 10 hours/week.

---

## Phase 0 — Setup (≈1 week)

Two things have external lead time. Do them first, today, because they run in
the background while you code.

- [ ] Join the TypeSafe Jev early-access waitlist with a Cornell email; mention
      academic research use.
- [ ] File a Not-Human-Subjects-Research determination request with Cornell
      RASS-IRB, with the faculty advisor listed. JIGSAWS is public, de-identified
      data from consenting surgeons and is almost certainly NHSR — but Cornell
      requires the determination to come from them, not from you. Keep the
      written result for the paper's ethics statement.

Then the local environment:

- [ ] Install `uv`. Run `uv init`, then `uv add numpy pandas scipy
      scikit-learn matplotlib pingouin netcal jupyterlab pytest`.
- [ ] Install VS Code with the Python and Jupyter extensions.
- [ ] `git init`. Write a `.gitignore` that excludes `data/`, `.env`,
      `results/jev_raw/`, `.venv/`, `__pycache__/`.
- [ ] Create a **private** GitHub repo and push.
- [ ] Unzip JIGSAWS into `data/`.
- [ ] Create `docs/NOTES.md` with a first dated entry.

**Acceptance:** `uv run python -c "import pandas, sklearn, pingouin, netcal;
print('ok')"` prints `ok`, and `git log` shows at least one commit on a remote.

---

## Phase 1 — Data loading (1–2 weeks)

The goal is a trustworthy path from files on disk to a labeled table. Nothing
later can be correct if this is wrong.

- [ ] Load one kinematics file. Confirm it has 76 columns and that row count
      divided by 30 gives a plausible trial duration in seconds.
- [ ] **Verify the meta_file layout by eye.** Print the first 5 lines of
      `meta_file_Suturing.txt` and identify: trial name, self-proclaimed skill
      (N/I/E), GRS total, and the six OSATS sub-scores. Confirm GRS equals the
      sum of the six sub-scores on several rows — that arithmetic check is how
      you know you have the right columns. Record the confirmed layout in
      `docs/NOTES.md`.
- [ ] Write `src/loaders.py`: `load_kinematics(trial)`, `load_meta(task)`,
      `load_splits(task, scheme)` where scheme is `"LOSO"` or `"LOUO"`.
- [ ] Build one tidy DataFrame: one row per trial, columns for task, subject,
      repetition, skill, GRS, six OSATS sub-scores, frame count.
- [ ] Sanity-check counts: 39 Suturing, 36 Knot-Tying, 28 Needle-Passing.
- [ ] Write tests for the loaders.

**Acceptance:** a DataFrame with 103 rows; GRS equals the sum of sub-scores for
every row; every trial named in the `Experimental_setup` split files resolves to
a real kinematics file.

**Gotcha:** if trial counts come out at 40 per task, you're picking up files that
should have been excluded. If split files reference trials you can't find, your
name parsing is off.

---

## Phase 2 — Feature extraction (2–3 weeks)

Compute descriptive motion features from the PSM columns. These are the features
the literature uses, and they're computable with numpy and scipy alone.

Per trial, per instrument (PSM1 and PSM2):

- [ ] Total task time (frames ÷ 30)
- [ ] Path length — summed Euclidean distance between consecutive tool-tip
      positions
- [ ] Mean, std, and max speed
- [ ] Mean and max acceleration
- [ ] Number of movements — count of velocity peaks above a threshold
- [ ] Idle fraction — proportion of frames below a low velocity threshold
- [ ] Smoothness — integrated squared jerk, or spectral arc length (prefer
      spectral arc length; it's less sensitive to trial duration)
- [ ] Working volume — bounding-box or convex-hull volume of the tool-tip path
- [ ] Gripper activity — number of open/close cycles

And bimanual:

- [ ] Correlation or coordination measure between the two PSM speed profiles

Then:

- [ ] Write tests: feed a synthetic straight-line trajectory of known length and
      confirm path length matches; feed a constant-position trajectory and
      confirm speed is zero and idle fraction is 1.
- [ ] Plot feature distributions by skill level. Experts should show shorter
      times, shorter paths, smoother motion. **If they don't, stop and debug —
      something is wrong.**

**Acceptance:** a feature matrix of 103 rows × N features, no NaNs, and at least
three features that visibly separate experts from novices in a boxplot.

**Gotcha:** thresholds for "number of movements" and "idle" are arbitrary. Pick
them once, write them in a config, and don't tune them against your results.

---

## Phase 3 — Baseline reproduction (2–3 weeks)

Reproduce something known before attempting something new. This de-risks the
whole project and gives you a fallback probability source for Phase 5.

- [ ] Ridge regression or SVR predicting GRS, under both LOSO and LOUO.
- [ ] Ordinal or multinomial logistic regression classifying N/I/E, under both.
- [ ] Standardize features inside the CV loop, never before it.
- [ ] Report Spearman ρ between predicted and true GRS, per task, per scheme.
- [ ] Compare to Zia & Essa (IJCARS 2018), who report roughly ρ ≈ 0.61 average
      GRS correlation under LOSO, with per-task LOSO values near 0.75 Suturing,
      0.63 Knot-Tying, 0.46 Needle-Passing. LOUO is much weaker (≈0.41 average).
- [ ] Save `predict_proba` outputs for the classifier — Phase 5 needs them.
- [ ] Log every run to `results/experiments.csv`.

**Acceptance:** LOSO Spearman ρ in the rough neighborhood of 0.5–0.7 on
Suturing. You do not need to match published numbers exactly — different
features, different implementation. You need to be in the same territory.

**Gotcha:** leaking the scaler or feature selection outside the CV loop will
inflate results and is the single most common beginner error in this exact
setup. If LOSO comes out suspiciously near-perfect, suspect leakage.

---

## Phase 4 — Jev integration (≈2 weeks, gated on API access)

Only start when access arrives. If it hasn't after 3–4 weeks, skip to Phase 5
using the baseline's probabilities and treat Jev as a later addition.

- [ ] Put `TYPESAFE_API_KEY` in `.env`. Confirm `.env` is gitignored.
- [ ] `uv add typesafe-sdk`. Make one trivial call and confirm a response.
- [ ] Write `src/jev_client.py` with a `state_for_trial(features) -> dict`
      function producing a compact record: task name, timing, path lengths,
      movement counts, smoothness, idle fraction, bimanual coordination. Keep it
      well under 1K tokens. **No raw trajectories.**
- [ ] Define one `Score` question per OSATS dimension plus one for overall GRS.
      Each takes `instructions` (one specific judgment) and `criteria` (an
      ordered list of 2–10 level descriptions, low to high). Describe
      *situations*, not degrees — "many unnecessary movements, long idle times"
      beats "poor efficiency."
- [ ] Batch all questions into a single call per trial. Questions are evaluated
      in parallel at negligible extra cost, so there's no reason to loop.
- [ ] **Cache the raw JSON response** to `results/jev_raw/<trial>.json` before
      parsing anything.
- [ ] Record `response.model` (the exact version) with every result, then pin it.
- [ ] Handle errors: retry 429 and 529 with backoff; 401 is a bad key; 422 means
      a malformed request and names the offending field.

**Acceptance:** 103 cached JSON responses, each with a score, a probability
distribution, and a confidence value per question, all parsed into a DataFrame.

**Gotcha:** re-running the full set costs almost nothing (~$0.0004 per call), but
if you don't cache, an API change mid-project silently invalidates your numbers.

---

## Phase 5 — Evaluation (3–4 weeks)

This is the paper. Run every analysis on both the baseline and Jev.

**Agreement with the expert rating:**

- [ ] Spearman ρ and Kendall τ (`scipy.stats`)
- [ ] ICC(2,1), two-way random, single rater, absolute agreement
      (`pingouin.intraclass_corr`, read the `ICC2` row)
- [ ] Quadratic weighted Cohen's κ for the ordinal sub-scores
      (`sklearn.metrics.cohen_kappa_score(..., weights='quadratic')`)
- [ ] Bland–Altman plot: mean vs. difference of model and expert scores

**Calibration:**

- [ ] Reliability diagram (`sklearn.calibration.calibration_curve` or
      `netcal.presentation.ReliabilityDiagram`)
- [ ] Expected Calibration Error and Maximum Calibration Error (`netcal.metrics`)
- [ ] Brier score

**Selective prediction — the differentiator:**

- [ ] Risk–coverage curve: sort by confidence, sweep a threshold, plot coverage
      against selective risk
- [ ] AURC, and E-AURC (AURC minus the oracle) to isolate ranking quality
- [ ] The headline experiment: **as you defer the least-confident trials to a
      human, does agreement on the remainder climb toward or past the human
      inter-rater ceiling?**

**The ceiling:**

- [ ] Frame everything against human OSATS inter-rater reliability, ICC ≈ 0.6–0.8.
      Martin et al. (1997) reported 0.64–0.72; more recent work reports ≈0.71–0.79
      for trained raters. Matching a second human is the realistic target, not
      perfection.

**Acceptance:** a reliability diagram, a risk–coverage curve, and a table of
agreement statistics for both methods across both CV schemes.

---

## Phase 6 — Writing (3–4 weeks)

- [ ] Draft the methods section from `docs/NOTES.md` — it should already be
      mostly written if you've been logging.
- [ ] Limitations, stated plainly: 8 subjects, skill confounded with identity,
      LOUO instability, a single expert rating per trial, a closed-API model with
      no published architecture, and Jev's documented weakness at magnitude.
- [ ] Ethics statement: NHSR determination, data handling, what was sent to a
      third-party API.
- [ ] Cite JIGSAWS properly — Gao et al. (2014, M2CAI/MICCAI workshop) and
      Ahmidi et al. (2017, IEEE TBME). Citation is the dataset's explicit
      condition of use.
- [ ] Release question schemas, state serialization, and recombination code so
      the work survives the API changing.

**First target venues**, easiest entry first: a SAGES 2027 abstract (400 words,
meeting April 2027), an MLHC clinical abstract (~2 pages), or an ML4H Findings
submission. Then develop toward IPCAI/MICCAI 2027 or IJCARS.

---

## If things go wrong

- **Jev access never arrives** → the baseline's `predict_proba` gives you
  calibrated probabilities. The calibration and selective-prediction paper
  stands on its own. Jev becomes a follow-up.
- **Correlations are near zero** → column indexing or units bug in Phase 2.
  Go back and check against the boxplots.
- **LOSO looks perfect** → leakage in the CV loop, or you're accidentally
  training on test trials.
- **LOUO is wildly unstable** → expected with 2 experts. Report intervals and
  say so. That instability is itself a finding worth discussing.
