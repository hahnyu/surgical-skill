# Systematic search supporting the two negative claims

Run 2026-09-21. Raw API responses are in `results/searches/` (gitignored is *not* applied;
they are small JSON files kept as evidence).

## What was searched, and what was not

**Searched.** PubMed/MEDLINE via NCBI E-utilities (`esearch`/`esummary`/`efetch`), with the
exact query strings below. Europe PMC (which additionally covers PMC, preprint servers and
Agricola) was queried first but returned HTTP 502/503 for most of the session; the five
queries that completed there before the outage are recorded as A1-A5 in
`results/searches/A*.json` and were consistent with the PubMed results. General web search
was used as a supplementary sweep.

**Not searched directly, and this is a limitation.** Google Scholar and IEEE Xplore have no
open programmatic interface and block automated querying, so neither was queried directly.
IEEE Xplore matters here because much robotic-surgery methods work appears in ICRA, IROS
and T-BME. Web search surfaced IEEE-indexed and arXiv material indirectly, but a manual
pass through Scholar and Xplore before submission would strengthen both claims. **Neither
negative claim below should be described as exhaustive.**

## Claim A (§2.3): no study attributes the protocol gap to identity leakage *and* tests it

| ID | Query (PubMed) | Hits | Relevant after screening |
|---|---|---|---|
| A1 | `JIGSAWS AND (surgeon OR subject) AND (identification OR identity)` | 6 | 0 |
| A2 | `("surgical skill" OR "technical skill") AND ("identity confounding" OR "subject identification" OR "surgeon identification")` | **0** | 0 |
| A3 | `("surgeon identification" OR "surgeon recognition" OR "surgeon-specific signature")` | 32 | **1** |
| A4 | `JIGSAWS AND (leakage OR confounding)` | 4 | 0 |
| A5 | `("surgical skill" OR "skill assessment") AND ("leave-one-user-out" OR "leave-one-subject-out")` | 9 | 0 (all already cited) |

All 32 A3 titles were screened. Thirty-one use "surgeon recognition" in an unrelated sense
(a surgeon recognising a condition, surgeon-volume outcome studies, and so on). One is
relevant, and it is important:

**Force Profile as Surgeon-Specific Signature.** Baghdadi A, Guo E, Lama S, Singh R, Chow
M, Sutherland GR. *Annals of Surgery Open* 2023 Sep;4(3):e326. doi
10.1097/AS9.0000000000000326. PMID 37746608. (Authors and venue verified via E-utilities.)
Identifies individual surgeons from time-series tool-tissue force data using deep networks,
reporting micro-average AUC 0.97 and 83.8% accuracy for identifying an expert surgeon across
2,819 ten-second segments from 89 neurosurgical cases, and concluding that "each surgeon has
a unique force profile amenable to identification using deep learning methods."

This is the closest prior work to our §4.2 and it was not in the supplied literature review.
It changes what we may claim. Surgeon identification from intraoperative signals is *not*
novel. What remains novel is the use to which it is put: that study frames identifiability
as a capability, proposing it "will enable a quantitative framework to provide bespoke
feedback to surgeons" and allow "correlating outcome to surgeon performance". It does not
connect identifiability to the validity of a skill-assessment protocol, is measured on
intraoperative force data rather than benchmark kinematics, and reports no protocol gap.
The claim has been narrowed accordingly in §2.3 and in the Introduction.

A5's nine hits were all studies already cited or already known (the markerless hand-tracking
study, the explainable-ML kinematic-features study, the interpretable whole-stream video
study, and older JIGSAWS benchmark work). None runs a surgeon-identification test.

## Claim B (§2.5): no calibration or selective-prediction study of a *skill assessment* model built from motion, kinematic or video data

| ID | Query (PubMed) | Hits | Relevant after screening |
|---|---|---|---|
| B1 | `("surgical skill" OR "technical skill") AND (calibration OR miscalibration) AND (machine learning OR deep learning OR automated)` | 4 | 0 — all use "calibration" in the sensor or marker sense, not probability calibration |
| B2 | `("surgical skill" OR "skill assessment") AND ("selective prediction" OR deferral OR abstention)` | **0** | 0 |
| B3 | `"expected calibration error" AND (surgical OR surgery)` | 21 | 0 for this claim |
| B4 | `"skill assessment" AND ("Brier score" OR "reliability diagram")` | **0** | 0 |

B3 is the informative one. All 21 hits apply calibration to surgical *imaging, diagnosis or
prognosis* — cystoscopy, thyroid cytology, glioma segmentation, surgical visual question
answering, intraoperative hypotension prediction, surgical-site-infection risk models. None
concerns skill assessment from motion, kinematic or video data. This is precisely the
distinction §2.5 draws, and it now rests on a recorded search rather than an impression.
SurgiQ, the LLM surgical-knowledge calibration study identified earlier, is the near-miss and
is cited in the text.

## Conclusion

Claim B stands as written. Claim A required narrowing: we are not the first to identify
surgeons from surgical signals, and the text now says so and cites the prior demonstration.
Both claims remain bounded by the Scholar and Xplore limitation stated above.
