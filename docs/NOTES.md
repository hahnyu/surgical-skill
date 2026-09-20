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

### Frame windows — open question for Phase 2

Split filenames embed a frame range (`Suturing_B001_000080_005635.txt`). For 17
of 39 Suturing trials the end frame does not equal the file's row count (e.g.
`Suturing_D001` names 1-2770, file has 2800 rows). These windows probably come
from the gesture transcriptions. Unresolved: compute features over the full
file or over the annotated window? It changes task time directly. Decide in
Phase 2.
