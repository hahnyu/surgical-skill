# Cover letter

> Draft. Author names, affiliations and the ethics sentence must be completed before
> sending — see the notes at the end, which are not part of the letter.

---

To the Editors,
*International Journal of Computer Assisted Radiology and Surgery*

We submit for your consideration **"What does a surgical skill model measure? Quantifying
identity confounding and miscalibration on JIGSAWS."**

Automated surgical skill assessment is proposed for training and credentialling, and JIGSAWS
is the field's standard benchmark. The benchmark sanctions two cross-validation protocols:
one holds out a repetition while every surgeon remains in training, the other holds out a
surgeon entirely. We asked what a model evaluated under the first is measuring, and whether
the confidence such models report would support deferring uncertain cases to a human rater.

Evaluating two prediction paths on identical information — sixteen kinematic features with
ridge and multinomial logistic regression, and a commercial zero-shot decision model given
the same features as text — we find that changing only the protocol removes at least a third
of reported skill-classification accuracy, with paired surgeon-level intervals excluding
zero on all three tasks. A follow-up diagnostic shows the same features identify which of
eight surgeons performed a held-out trial with accuracy between 0.83 and 0.93 against chance
near 0.13. On unseen surgeons, only one task of three improves on a reference predictor that
uses no motion information at all, and on that task a single-feature regression on trial
duration matches both models. Both paths are overconfident on unseen surgeons, scoring worse
than a constant forecast, and neither path's confidence identified the trials it got wrong.

On novelty, we are deliberately narrow. Identity confounding under naive cross-validation is
a well-established failure mode in mobile health, EEG and speech, and we do not claim to have
discovered it; nor are we the first to identify individual surgeons from surgical signals,
which has been demonstrated from intraoperative force data. What we believe is new, and what
we offer the journal, is threefold: the protocol gap on this benchmark quantified with
paired surgeon-level intervals rather than reported as a limitation; the use of a
surgeon-identification test as a validity diagnostic on the same features, model and folds
as the skill task, which distinguishes identity confounding from the underfitting
explanation that a performance gap alone leaves open; and what is, to our knowledge from a
recorded search, the first calibration and selective-prediction audit of a surgical skill
assessment model built from motion data.

We should be equally clear about the limits. This is a single-dataset case study with eight
surgeons, so every interval is wide and the selective-prediction result is an underpowered
null rather than a demonstration that confidence cannot help. The confound and the
calibration failure are demonstrated here, not established as general properties of the
field, and we say so in the paper. We reproduced no published system, so our claims concern
the protocol's structure and our own demonstration of it, not the validity of specific prior
results.

The analysis rules for each stage were fixed in a version-controlled log before the
corresponding analysis was run, and analyses added afterwards are labelled as such
throughout, including the surgeon-identification diagnostic and one substitution of the
headline metric. All code, configurations and the full experiment log are available so that
every number in the manuscript can be traced to the run that produced it.

The manuscript is original, is not under consideration elsewhere, and all authors have
approved the submission. We declare no competing interests. [ETHICS SENTENCE — see note.]
The JIGSAWS dataset is public and cited per its conditions of use; no patient data were
handled.

Thank you for considering our work.

Yours sincerely,

[Corresponding author name]
[Affiliation]
[Email]
on behalf of all authors

---

## Notes before sending — not part of the letter

1. **Ethics sentence.** The Cornell RASS-IRB Not-Human-Subjects-Research determination is
   still pending. Do not assert a determination that has not been issued. Once it arrives,
   replace the placeholder with its actual wording and reference. If it has not arrived,
   state plainly that a determination has been requested and is pending.
2. **Third-party API disclosure.** The zero-shot path sent wording-derived summaries of the
   public dataset to a commercial API. No video, trajectories, identifiers, labels or digits
   were transmitted. Consider whether the editor should be told in the letter as well as in
   the paper's data statement; the vendor states no fixed retention period.
3. **Author list, affiliations and the corresponding author** must be completed here and in
   `main.tex`.
4. **Suggested reviewers.** Not included. If IJCARS requests them, avoid anyone from the
   groups whose work Section 2.3 distinguishes, since the paper's novelty claim turns on
   how their framing differs from ours.
