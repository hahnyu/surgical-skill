# NOTES.md

Running log of decisions, surprises, and reproduced numbers. This becomes the
methods section later.

## 2026-09-19

Environment set up with uv 0.12.17 (Homebrew). Dependencies installed per
PLAN.md Phase 0; `uv.lock` committed. Note: `netcal` pulls in `torch`/`gpytorch`
as transitive dependencies. Import check (`pandas, sklearn, pingouin, netcal`)
passes. JIGSAWS is unzipped in `data/` (contents not yet inspected).
