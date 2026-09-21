> **IJCARS submission copy.** Derived from `paper/manuscript_full.md`, which is retained as
> the Electronic Supplementary Material. Compressed to this venue's 5,000-word all-inclusive
> limit; supplementary sections are referenced as S1-S4.

## Abstract

**Purpose** JIGSAWS sanctions two cross-validation protocols: one holds out a repetition while every surgeon remains in training, the other
holds out a surgeon entirely. We ask what a model evaluated under the first measures, and
whether reported confidence would support deferring uncertain trials to a human rater.

**Methods** Two prediction paths were evaluated on identical information across 103 trials,
8 surgeons and 3 tasks: 16 kinematic features with ridge and multinomial logistic
regression, and a zero-shot commercial decision model given the same features as text. Both
protocols were run. Errors are reported against a feature-free reference predicting each
held-out trial with its training-fold mean; intervals resample whole surgeons. Analysis
rules were fixed in advance; post-hoc additions are labelled.

**Results** Changing only the protocol removed at least a third of skill-classification
accuracy (paired gap 0.49-0.67, intervals excluding zero). A post-hoc diagnostic found the same
features identified which of eight surgeons performed a held-out trial (accuracy 0.83-0.93, chance near 0.13), separating identity confounding from underfitting.
On unseen surgeons one task of three beat the reference, where a single-feature
regression on trial duration matched both models. Both paths were overconfident (confidence
0.755 against accuracy 0.379; 0.696 against 0.301), scoring worse than a constant forecast,
and neither identified deferrable trials.

**Conclusion** Accuracy under the permissive protocol is consistent with surgeon
recognition rather than skill measurement. With eight surgeons these are single-dataset
demonstrations, not established generalities. We recommend reporting held-out-surgeon
results against an uninformative reference, with a subject-identifiability diagnostic.

**Keywords** Surgical skill assessment; JIGSAWS; identity confounding; cross-validation;
calibration; selective prediction

## 1. Introduction

Automated assessment of surgical technical skill promises scalable, consistent feedback
during training. If such a system is to inform training or credentialling decisions, it must
generalise to surgeons not seen during its development. Whether current evaluation practice
on the field's standard benchmark establishes that it does is the question we take up.

The concern is not specific to surgery. A model trained and tested on the same individuals
can succeed by recognising the individual rather than the property of interest, a failure
mode named identity confounding [Chaibub Neto et al. 2019] and demonstrated by Saeb et al.
[2017], whose activity-recognition error rose from about 2% to 7% when whole subjects rather
than records were held out. Electroencephalography supplies the closest
parallel to what we report: an Alzheimer's classifier reaching 99.8% under segment-based
holdout fell to 53.0% on unseen subjects, an interval spanning chance [Brookshire et al.
2024]. These are instances of shortcut learning [Geirhos et al. 2020], documented across 17
fields and 294 papers [Kapoor and Narayanan 2023].

The benchmark used throughout this literature ships two cross-validation protocols: one
holds out a repetition while every surgeon remains in training, the other holds out a
surgeon entirely. Both are sanctioned and both are used, so what follows is not a criticism
of an informal practice. That performance falls under the second is not in dispute. What has
not settled is why. Three studies of this benchmark attribute the same gap to three
different causes, and a fourth, on a different modality, describes its own gap as
quantifying "the surgeon-specific component captured by video-level splits" and stops there.
Three different attributions, none of them measured, and a fourth naming a mechanism without
measuring it. The test is inexpensive: train the same features to predict which surgeon
performed the trial. Such classifiers exist, but they have been built to enable per-surgeon
feedback, not to ask whether a skill benchmark measures skill.

Judging such a system also requires care about its target. Human ratings of technical skill
are neither perfectly reliable nor uniformly so: pooled inter-rater reliability for global
rating scales is 0.78 (95% CI 0.71-0.83, 23 studies) [Ilgen et al. 2015], while untrained
raters have been measured at 0.26, the benefit of training lost during non-use [Gawad et al.
2019]. The model is fitted to a noisy, training-dependent judgment, so agreement statistics
cannot be weighed against a fixed bar and a reported accuracy is not self-interpreting.

We evaluate two prediction paths on identical information under both protocols. We make four
contributions: the protocol gap quantified with paired surgeon-level intervals; a
surgeon-identification diagnostic, added post hoc, that separates identity confounding from
the underfitting explanation a gap alone leaves open; the first calibration and
selective-prediction audit of a skill model built from motion data; and a comparison of both
paths against a single-feature regression on trial duration. We do not claim to have
discovered identity confounding, nor to be the first to identify surgeons from surgical
signals; what we contribute is the demonstration and quantification of its consequences for
this benchmark.

## 2. Related work

### 2.1 The benchmark, its protocols, and how results are reported

JIGSAWS [Gao et al. 2014] is the standard public benchmark and its suite [Ahmidi et al.
2017] defines the two schemes above. The field knows individual style exists: a review notes
that leave-one-user-out is harder "due to subject-specific style variability" [van Amsterdam
et al. 2021] and surveys methods to normalise it away, treating identity as nuisance
variance rather than as something a permissive protocol rewards. In the studies we examined [Zia and Essa 2018; Wang et al.
2021], results are reported as accuracies or correlations without the value the same metric
takes for a predictor using no motion information; we did not survey systematically enough
to call that a norm.

### 2.2 Label validity and protocol validity are separate critiques

The closest existing critique concerns the target: Hendricks et al. [2024] show the
self-declared skill label correlates poorly with the expert rating, and "Looking Beyond the
Scale" [2026] reports the same mismatch; our data reproduce it, and we report that as
replication. The distinction from our contribution matters. **They ask whether the target
measures skill; we ask whether the protocol lets a model succeed without measuring skill,
whatever the target's quality.** These compound. A perfectly valid target would not repair
what we describe, because identifiability is a property of the features and folds, not of
what the label means; conversely their critique survives a flawless protocol.

### 2.3 The gap is widely reported, and explained three other ways

A whole-stream video framework [Chen et al. 2026] reports near-perfect supertrial-out
accuracy, a drop under user-out evaluation and near-chance external transfer, calling for
surgeon-invariant features: the goal named, the mechanism unmeasured. "Looking Beyond the
Scale" [2026] reports a concordance correlation of -0.03 (+/-0.08) under disjoint-participant
evaluation against 0.22 for its best baselines and, having excluded model-side causes by
architecture search, attributes the failure to annotations. An explainable-machine-learning
study [Xuan et al. 2026] criticises prior work for lacking subject-independent validation,
and Lefor et al. [2020] flagged subject imbalance without quantifying it. Closest to our
framing, Constans et al. [2026] find accuracy falling to 52.3% under leave-one-subject-out
and state that this gap "quantifies the surgeon-specific component captured by video-level
splits". Little separates that sentence from our thesis: had they run a
subject-identification probe, the attribution would already exist. Our contribution relative
to that study is narrower than to the others: a different modality, and a direct measurement
of the component they name.

That our own null agrees with "Looking Beyond the Scale" strengthens rather than
duplicates it: two pipelines differing in features, models, metric and reference reach the
same place, which is harder to attribute to one pipeline's idiosyncrasy than either alone.

One counter-argument must be met. Responding to Saeb et al. [2017], Little et al. [2017]
argued that a record-wise versus subject-wise gap can arise from *underfitting* rather than
confounding, so a gap alone establishes nothing. None of the studies above can answer that,
because each reports only a gap. Our diagnostic can: a gap is ambiguous; recoverable
identity at roughly seven times chance is not.

Surgeon identifiability is itself not new: from intraoperative force data, Baghdadi et al.
[2023] identify surgeons with AUC 0.97 and 83.8% accuracy across 89 neurosurgical cases,
framing it as a capability enabling per-surgeon feedback rather than a threat to any
protocol's validity. We are not the first to identify surgeons from surgical signals; we add
the use of identifiability as a validity diagnostic tied to the gap it explains. To our
knowledge no study both attributes that gap to identity leakage and demonstrates it from the
features used for skill. This follows a recorded PubMed and web search; Scholar and IEEE
Xplore could not be queried programmatically, so the claim is bounded, not exhaustive.

### 2.4 Calibration, deferral and zero-shot scoring

Calibration has begun to reach surgery: SurgiQ [2026] benchmarks language models on surgical
knowledge and includes a calibration analysis. We located no calibration or
selective-prediction study of a surgical *skill assessment* model built from motion,
kinematic or video data, so our lineage is imaging and intensive care, where deferral is
demonstrably useful [Dvijotham et al. 2023; ICLR 2025], with the caution that deferral
cannot manufacture reliability a model lacks ["Calibrated Triage, Not Autonomy", 2026]. Foundation models have reached this benchmark but not in our role: the
nearest work,
ZEAL [ZEAL 2024], feeds zero-shot segmentation to a *trained* network. The model we evaluate is
marketed on calibration, a claim a trade critique tempers [Eliot 2026]; three independent
2026 evaluations [priorbench; scienthoon; webofmike] find calibration holding in
distribution and degrading outside it. **All three are non-peer-reviewed**, and their
version strings may not match the build we pinned, so we treat them as evidence about the
model family.

## 3. Methods

### 3.1 Data, features and models

JIGSAWS comprises 103 usable trials from 8 surgeons across suturing, knot-tying and
needle-passing, each rated on a modified OSATS instrument whose six 1-5 elements sum to a
global rating score (GRS, 6-30). The metadata files, not the directory listing, are
authoritative; surgeon G has no usable needle-passing trial, and GRS equals the sum of its
elements on all 103 rows.

We compute 16 motion features per trial from the patient-side manipulator channels over the
annotated task window, with tool-tip positions low-pass filtered once at 5 Hz before any
derived quantity: task time, the correlation between the two instruments' speed profiles,
and per instrument the path length, speed mean and standard deviation, mean acceleration,
idle fraction below 1 mm/s, hull volume and gripper cycles. Thresholds were fixed before use
and never tuned against outcomes; three features were removed before modelling for stated
reasons unrelated to performance (S1).

The baseline applies ridge regression to GRS and multinomial logistic regression to the
skill label, standardised within a pipeline so scaling uses training rows only; the ridge
penalty is chosen inside each training set by leave-one-subject-out search, and the logistic
regulariser is fixed at C = 1.0, a subject-level inner search being ill-defined when a class
is held by two surgeons. The second path uses a commercial zero-shot
decision model (jev-1.13.0), never trained, given the same 16 features rendered as ordinal
phrases describing each feature's percentile within its task. No number, identifier or label
is transmitted; tests enforce this. Seven five-level scoring questions are sent per trial
and each trial was scored three times (S2).

### 3.2 Protocols, reference predictor and uncertainty

The two schemes hold out the *i*-th repetition with every surgeon still in training, or a
surgeon entirely. Official skill splits ship only for suturing; our generated folds reproduce
them exactly there, verified by test. Every trial is predicted once per scheme.

Claims are stated against a **feature-free reference** predicting each held-out trial with
the mean GRS of its own training fold, so the headline quantity is the reduction in mean
absolute error relative to that reference, in GRS points. **Interpretation rule, fixed before
any interval was computed:** a method beats the reference only if the lower bound of its 95%
interval exceeds zero. With eight surgeons this is a weak test and we treat it as suggestive.
The zero-shot score is placed on the GRS scale by a line fitted within each training fold,
and we record the sign of that slope per fold as a direction guard.

All intervals resample **whole surgeons** with replacement (2000 resamples), never
individual trials, with predictions held fixed; an interval therefore reflects which surgeons
were evaluated, not refitting variability. For the central claim we bootstrap the
between-protocol difference paired on the same resampled surgeons.

### 3.3 Protocol scrutiny and pre-registration

Analysis rules for each stage were committed to a version-controlled log before the
corresponding analysis ran; analyses added afterwards are labelled post hoc below. Two
findings about our own protocol belong on record. **Leakage was tested for and not found:**
altering test-fold labels leaves that fold's predictions bit-identical, a deliberately leaky
variant moves untouched predictions by 16 GRS points where ours moves them by zero, and 50
shuffled-label refits per cell gave uniformly negative mean correlations, the opposite of
leakage. **Separately, our pre-registered headline statistic proved uninterpretable and was
replaced:** under leave-one-user-out each fold is one surgeon and the prediction one constant,
so the feature-free reference scores -0.78 to -0.86 on pooled rank correlation by
construction. We therefore report error against that reference, retaining correlations only
beside it. This substitution is itself post hoc (S3).

## 4. Results

Subsections are labelled by whether the analysis was specified before results were seen. The
protocol comparison, calibration and selective prediction were pre-declared; the
surgeon-identification diagnostic, the duration comparison and the headline metric were not.

### 4.1 The protocol gap [pre-declared]

Skill classification with every surgeon in training reaches 0.92 (95% interval 0.85-0.97),
0.97 (0.92-1.00) and 0.93 (0.85-1.00) on suturing, knot-tying and needle-passing. On a
held-out surgeon the same features and model give 0.44 (0.25-0.63), 0.31 (0.08-0.58) and
0.39 (0.16-0.69). Bootstrapping the difference paired on the same resampled surgeons gives
+0.49 (0.35-0.62), +0.67 (0.41-0.89) and +0.54 (0.27-0.71); all three exclude zero. Changing
only the protocol removes at least a third of reported accuracy and plausibly more than half
(Table 1).

The held-out-surgeon point estimates fall at or below the majority-class rate (0.487, 0.444,
0.393), but we state this as a point estimate: those intervals span the majority rate in all
three tasks, so we cannot establish that accuracy falls below that floor, only that it falls
far below the permissive figure. Quadratic-weighted kappa is negative on suturing (-0.19) and
needle-passing (-0.47).

### 4.2 The same features identify the surgeon [post-hoc]

Added after observing the gap, to explain it. Substituting surgeon identity for skill and
leaving features, model, folds and regularisation unchanged, the features identify which of
eight surgeons performed a held-out repetition with accuracy 0.87 (0.77-0.97), 0.83
(0.67-0.97) and 0.93 (0.83-1.00) against chance of 0.125, 0.125 and 0.143. Because the skill label is constant across a
surgeon's trials, a model that identifies the surgeon can retrieve the label without
representing skill. This is a sufficient mechanism, not proof the classifier uses identity;
establishing that needs an intervention on the features, which we did not perform.

### 4.3 What survives a held-out surgeon [pre-declared design, post-hoc metric]

On knot-tying, ridge reduces error against the feature-free reference by +1.54 GRS points
(0.65-2.52), from 5.10 to 3.56; on suturing +0.15 (-1.04 to 1.34) and on needle-passing -0.50
(-1.21 to -0.04), the fitted model being *worse* than predicting the training mean. The
zero-shot path behaves alike (+2.30, 1.14-3.45 on knot-tying; negative elsewhere), with a
positive fitted slope in all 38 folds, so the direction guard passed. The two knot-tying
reductions are not distinguishable and nothing here ranks the paths (Table 2).

The zero-shot score correlates with task time at -0.91, -0.81 and -0.86, and a single
ordinary-least-squares line on task time alone reduces knot-tying error by +1.88 (0.94-2.89).
All three knot-tying intervals overlap across almost their length, so we report them as three
estimates of one effect. This study does not demonstrate that 16 features or a commercial
model add anything over trial duration on the one task where signal survives. Agreement
follows suit: ICC(2,1) on knot-tying is 0.60 (0.35-0.78) for ridge and 0.73 (0.53-0.85) for
the zero-shot path, and indistinguishable from zero elsewhere. Neither Ilgen et al. nor
Martin et al. names the coefficient it pooled, so comparison with those figures is
approximate and we claim no equivalence to human raters.

### 4.4 Calibration [pre-declared]

Each path is judged against the target it was given and the two are not compared. On a
held-out surgeon the baseline states mean confidence 0.755 while achieving 0.379 accuracy
(expected calibration error 0.377, 0.237-0.541), scoring worse than a constant forecast of
the observed class frequencies: Brier skill -0.351 (-0.688 to -0.092), entirely below zero.
Its most confident bin, 46 of 103 trials at mean confidence 0.92, is correct 39% of the
time. With every surgeon in training the same
classifier appears well calibrated and slightly underconfident (0.837 against 0.942, Brier
skill +0.778), so the protocol manufactures the calibration verdict as it does the accuracy.
The zero-shot path is also overconfident (0.696 against 0.301; skill -0.433, -0.588 to
-0.242), though usually close: within one level on 84% of trials, with a mean predicted level
of 2.05 against a true 1.71. That offset means its exact-level miscalibration mixes genuine
overconfidence with a scale mismatch, which we cannot separate without a recalibration
excluded in advance.

### 4.5 Selective prediction [pre-declared]

**The two arms do not support the same inference, and we state that before the numbers.**
Ridge emits no native confidence, so we declared in advance that the baseline would pair its
regression prediction with the classifier's top-class probability. Those are different models
fitted to different targets, and the targets disagree. The baseline's confidence therefore
carries no information about the regression's error beyond what the targets share, and **its
null is uninformative about whether ridge knows when it is wrong.** The zero-shot path has no
such defect: score and confidence come from one model answering one question.

The clean test is null. Ordering by the zero-shot model's own confidence lowers the area
under the risk-coverage curve by 0.155 GRS points relative to random ordering (-0.416 to
0.606); substituting its maximum level probability changes nothing (0.149). The
uninterpretable arm is also null (0.135, -0.628 to 0.922). Deferring 20% of trials by baseline
confidence raises retained error from 4.31 to 4.50. Upper bounds reach +0.61 and +0.92 GRS
points, a fifth of overall error, so this is an under-powered null rather than evidence that
confidence is useless. Conclusions are unchanged under unfiltered and 8 Hz
features and a wider penalty grid (S4).

## 5. Discussion

The skill label here is a property of the surgeon, constant across their trials. Under the
permissive protocol every test surgeon also appears in training, so a model that recognises
an individual can retrieve that individual's label without representing skill, and will be
scored correct for it. Our results are consistent with this: the protocol alone accounts for
at least a third of reported accuracy, and the features identify the operator far above
chance. We have not shown the classifier internally relies on identity, and a reader may
hold the weaker view that both results reflect some third property of individuals; our data
do not separate those accounts. Either way, a figure obtained with the surgeon present in
training does not estimate performance on a new surgeon, the only situation in which such a
tool would be deployed. Any dataset predicting a person-level label from repeated
measurements of the same people admits the same shortcut, though we measured it only here.

What survives is modest: one task of three beats a reference using no motion information, by
roughly a quarter to a third of the per-task standard deviation, and there a single-feature
regression on duration matches both models with heavily overlapping intervals. The zero-shot model's output correlates with duration at -0.81 to -0.91, so the
convergence is not an artefact of fitting. We disclose that its rubric, written before any
trial was scored but after we had seen which features carried signal, refers to completion
time and pauses; this is a plausible route by which it came to track duration and we cannot
exclude it.

For deployment the relevant finding is that confidence does not track correctness on
unfamiliar operators. Both paths are overconfident and score worse than a constant forecast;
a system confidently wrong on new surgeons is worse for triage than one uncertain
everywhere, because the errors it passes through unreviewed are the ones it will not flag.
Neither path's confidence identified deferrable trials, but the two nulls are not equal
evidence: one is clean, one uninterpretable by construction, and both under-powered. A
predictor with native uncertainty, through quantile or conformal methods, would remove the
pairing defect and is the improvement a follow-up should adopt.

Three cheap changes would have caught this: report the held-out-surgeon protocol as primary
with the paired difference beside it; anchor claims to an uninformative reference, since a
positive correlation can accompany larger error than predicting the training mean; and
publish a subject-identifiability diagnostic alongside a one-feature baseline. We note the
tension in recommending protocol discipline from a study that substituted its own headline
metric and added its central diagnostic post hoc; we have labelled both rather than
presenting the final analysis as the planned one.

This is a single-dataset case study. Eight surgeons, seven on one task, make every interval
crude, and those intervals express which surgeons were evaluated rather than refitting
variability or the sampling of these eight. We examined one baseline and one commercial
model at one pinned version with one rubric. We reproduced no published system,
so our claims concern the protocol's structure and our own demonstration of it, not the
validity of specific prior results. The confound and the calibration failure are demonstrated
here, not established as general, and both should be checked on a second, larger dataset
before being treated as properties of the field.
