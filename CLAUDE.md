# CLAUDE.md

Project instructions for Claude Code. Read this at the start of every session.
The phased task list lives in `docs/PLAN.md` — read it before starting work.

## Project

Automated surgical skill assessment on the JIGSAWS dataset. We predict expert
GRS/OSATS ratings from robot kinematics, then study **calibration and selective
prediction**: when the model is confident, is it right, and can its confidence
tell us which trials to hand back to a human rater?

Two prediction paths, both required:

1. **Baseline** — hand-crafted kinematic features + scikit-learn. Calibrated
   probabilities come from `predict_proba`.
2. **Jev** — TypeSafe AI's "System One" model. Same features, serialized into a
   compact state, scored with the `Score` primitive.

The calibration analysis is the paper's contribution. It must run on either
path, so **the project must never be blocked on Jev API access.**

## Who you are working with

The user is an undergraduate in Biometry and Statistics with strong statistics
but **minimal programming experience**. This changes how you work:

- Explain what code does and why, in plain language, as you write it.
- Prefer boring, readable code over clever code. No metaclasses, no decorators
  we don't need, no premature abstraction.
- When you use a library function they likely haven't seen, say what it does in
  one line.
- When there's a real choice to make (a design tradeoff, a statistical
  decision), surface it and let them decide. Don't quietly pick.
- Never generate a large block of code they haven't been walked through.

They are learning to do research, not just to get results. Optimize for their
understanding.

## Hard rules

- **Never commit secrets.** `TYPESAFE_API_KEY` lives in `.env`, which is
  gitignored. Never write a key into source, a notebook, or a config file.
- **Never commit data.** `data/` is gitignored. JIGSAWS is not ours to
  redistribute.
- **Never invent a number.** If a baseline figure, a dataset statistic, or an
  API detail isn't verified, say so and leave a `TODO(verify)`.
- **All arithmetic happens in Python, never in Jev.** Jev is documented as not
  being a calculator. Every feature is computed by our code; Jev only judges.
- **Cache every Jev response to disk as raw JSON** before parsing, under
  `results/jev_raw/`. Re-running analysis must never require re-calling the API.
- **Pin the Jev model version** once results collection starts (e.g.
  `jev-1.13.0`, not `jev-latest`) and record it in every run's config.
- **Set and record random seeds.** Every experiment must be re-runnable to the
  same numbers.

## Environment

Python managed with `uv`. Assume macOS, Apple Silicon.

```bash
uv sync                      # install/refresh the environment from uv.lock
uv run python src/...        # run any script
uv run pytest                # run tests
uv run jupyter lab           # notebooks
```

Do not use `pip install` directly, `conda`, or a global Python. Add dependencies
with `uv add <pkg>` so `uv.lock` stays authoritative.

Core deps: numpy, pandas, scipy, scikit-learn, matplotlib, pingouin, netcal.
Jev: `typesafe-sdk` (only needed for Phase 4).

## Repo layout

```
.
├── CLAUDE.md
├── docs/
│   ├── PLAN.md           # phased task list — read this
│   └── NOTES.md          # running log of decisions and surprises
├── data/                 # JIGSAWS, unzipped. GITIGNORED.
│   ├── Suturing/
│   ├── Knot_Tying/
│   ├── Needle_Passing/
│   └── Experimental_setup/
├── src/
│   ├── loaders.py        # read kinematics, meta files, CV splits
│   ├── features.py       # kinematic feature extraction
│   ├── baseline.py       # sklearn models
│   ├── jev_client.py     # state serialization + API calls + caching
│   └── metrics.py        # agreement, calibration, selective prediction
├── configs/              # one YAML per experiment
├── results/
│   ├── jev_raw/          # cached API responses
│   ├── figures/
│   └── experiments.csv   # one row per run: config + metrics
├── notebooks/            # exploration and figures only
└── tests/
```

Notebooks are for looking at things. Any logic that gets reused moves into
`src/` and is imported. Do not let a notebook become the project.

## Dataset facts (verified — trust these)

- Kinematics files: whitespace-delimited, **no header**, **76 columns**, 30 Hz.
- Column layout: 1–19 left MTM, 20–38 right MTM, 39–57 PSM1, 58–76 PSM2. Within
  each block: 3 position, 9 rotation matrix, 3 linear velocity, 3 angular
  velocity, 1 gripper angle.
- **Use the PSM columns (39–76)** for skill features. Those are the instruments
  inside the patient. MTM columns are the surgeon's hands at the console.
- Usable trials: **39 Suturing, 36 Knot-Tying, 28 Needle-Passing** (103 total).
  Some trials were dropped for corrupt recordings — do not assume 40 per task.
- GRS total range is **6–30** (six OSATS elements, 1–5 each). Some papers print
  6–35; that is an error.
- `Experimental_setup/` holds the official CV splits. **Always use them.**
  LOSO = leave-one-supertrial-out (easier, saturated in the literature).
  LOUO = leave-one-user-out (harder, clinically meaningful, noisy with only 8
  subjects). Report both.

## Known gotchas

- **The meta_file column order is not confirmed from an official source.** Do
  not write a loader against an assumed layout. Open the file, print the first
  few lines, and have the user visually confirm which column is GRS and which
  are the six OSATS sub-scores before anything depends on it. This is the most
  likely source of a silent, invisible labeling bug.
- **Don't dump raw trajectories into Jev state.** State is capped around 32K
  tokens, and thousands of rows of 30 Hz kinematics is both over the limit and
  the wrong input. Summarize to a small feature record first.
- **Jev's `Score` is documented as good for ranking and thresholding, not for
  reconstructing a magnitude.** Report rank correlation (Spearman, Kendall) as
  primary. Treat the continuous `.score` as a weak magnitude estimate and say so
  in the paper.
- **LOUO results will be unstable.** There are only 2 experts and 2
  intermediates. Report confidence intervals, don't chase a point estimate.
- **Skill is confounded with subject identity.** A model can score well by
  recognizing a surgeon's style. This is the main threat to validity and belongs
  in the limitations section.
- If a correlation comes out near zero on Suturing LOSO, suspect a column
  indexing or units bug before suspecting the method.

## Working style

- **Work one phase at a time.** Do not skip ahead to Jev before the baseline
  reproduces. `docs/PLAN.md` has acceptance criteria for each phase; meet them
  before moving on.
- **Small commits with real messages.** Commit after each working piece, not at
  the end of a session.
- **Write tests for the loaders and feature functions.** If a feature function
  is wrong, every downstream number is wrong and nothing will look obviously
  broken. At minimum: a test that path length of a known synthetic trajectory
  equals the hand-computed answer.
- **Append to `docs/NOTES.md`** whenever something surprising happens, a
  decision gets made, or a number gets reproduced. This becomes the methods
  section later.
- **Log every experiment run** as a row in `results/experiments.csv`: timestamp,
  config file, feature set, CV scheme, seed, model version, and the resulting
  metrics.
- Before a large refactor or adding a dependency, ask first.

## Offline / second-machine transfer (only if needed)

The project does not require a GPU and the Jev API requires network access, so
default to running everything locally. If work must move to another machine:

```bash
# On the Mac
git bundle create transfer.bundle --all      # full history, one file
uv export --format requirements-txt > requirements.txt
uv pip download -r requirements.txt -d wheels/   # wheels for offline install

# On the target machine
git clone transfer.bundle project && cd project
pip install --no-index --find-links ../wheels -r requirements.txt

# Sending work back
git bundle create return.bundle --all
```

Copying a folder is not an acceptable substitute — it loses history and drifts
dependencies. Note that wheels are platform-specific: Apple Silicon wheels will
not install on a Linux GPU box, so download them on or for the target platform.
