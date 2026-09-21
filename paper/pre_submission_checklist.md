# Pre-submission checklist

Open items that do not block drafting but must be closed before the paper is submitted.
Each names what is currently unverified and what would close it.

## Citations and novelty

- [x] **Systematic search backing the §2.3 negative claim — DONE, and it narrowed the
  claim.** PubMed queried via E-utilities with five recorded query strings; all 32 hits of
  the critical query screened. Protocol and results in `paper/systematic_search.md`, raw
  responses in `results/searches/`. **The search found prior work the supplied literature
  review had missed:** Baghdadi et al. (Ann Surg Open 2023, doi 10.1097/AS9.0000000000000326)
  identify individual surgeons from intraoperative force data at 83.8% accuracy, framing
  identifiability as a capability rather than a validity threat. Surgeon identification from
  surgical signals is therefore *not* novel; §2.3 and the Introduction now say so and cite it,
  and the claim is narrowed to the use of identifiability as a validity diagnostic tied to
  the protocol gap.
- [x] **Systematic search backing the §2.5 negative claim — DONE, claim stands.** Four
  recorded queries. Two returned zero hits; the informative query (`"expected calibration
  error" AND (surgical OR surgery)`) returned 21, all applying calibration to imaging,
  diagnosis or prognosis rather than to skill assessment from motion data. The narrowed
  wording is now supported by a recorded search.
- [ ] **Remaining search limitation.** Google Scholar and IEEE Xplore could not be queried
  programmatically and were not searched directly. IEEE Xplore matters because much
  robotic-surgery methods work appears in ICRA, IROS and T-BME. A manual pass through both
  before submission would strengthen the two negative claims; neither is described as
  exhaustive in the text.

- [x] **Verify every borrowed figure against its primary source — DONE for what is
  reachable; two remain.** All figures attributed to prior work originally came from a
  supplied literature review. Status as of 2026-09-20:

  | Figure | Source | Status |
  |---|---|---|
  | Record-wise ~2% vs subject-wise ~7% (30 subjects) and 27% (2 subjects) | Saeb et al. 2017, GigaScience | **verified** against the article |
  | Alzheimer's 99.8% (99.1-100.0) to 53.0% (43.1-64.8); seizure 79.1% (78.8-79.4) to 65.1% (61.3-69.1) | Brookshire et al. 2024, Frontiers | **verified**; more precise than the review stated, and the draft now uses the exact figures |
  | 294 papers across 17 fields | Kapoor and Narayanan 2023, Patterns | **verified** via Europe PMC: "17 fields where leakage has been found, collectively affecting 294 papers". The preprint's 329 is not used |
  | CCC -0.03 (+/-0.08); best baselines 0.22 | "Looking Beyond the Scale", arXiv 2608.17519 | **verified** against the full text. Authors: Hoffmann, von Bechtolsheim, Speidel, Hisey. Not in the abstract — only the full text |
  | 52.3% under leave-one-subject-out; the "surgeon-specific component" quote | Constans et al. 2026, IJCARS (PubMed 42616309) | **verified** via Europe PMC, quote confirmed verbatim |
  | 25% false-positive reduction; 66% workload reduction | Dvijotham et al. 2023, Nature Medicine (CoDoC) | **verified** via Europe PMC abstract |
  | ICC 0.26 untrained; 0.76 post-training; retention -0.17 to 0.70 | Gawad et al. 2019, J Surg Educ | **verified** via Europe PMC. Note: retention is *variable*, not monotonically lost, so the draft now cites the authors' own summary rather than implying decay |
  | 0.64-0.72 inter-rater reliability | Martin et al. 1997, Br J Surg | **verified** earlier via Europe PMC |
  | **ICC 0.78 (95% CI 0.71-0.83), 23 studies** | **Ilgen et al. 2015, Medical Education** | **NEEDS LIBRARY ACCESS.** Citation metadata independently confirmed (doi 10.1111/medu.12621, 49(2):161-173, 45 studies) but the figure itself is unconfirmed against the primary abstract; Medical Education is paywalled and Europe PMC returned 502/503 on two attempts. **This is the most load-bearing unverified number in the draft** — §1 and §4.3 both compare our agreement results against it. Check via Cornell library access |
  | I3D-SAP per-task accuracies | IJCARS | **not needed** — the draft describes them qualitatively and quotes no figure |

  Rule going forward: a verified figure and an unverified one must not be presented alike.
  The only unverified figure now in the draft is Ilgen's 0.78, and it is flagged here rather
  than silently equated with the rest.

- [ ] **Resolve incomplete citations to venue and DOI:** I3D-SAP, VBA-Net, ZEAL, SurgiQ,
  the Journal of Robotic Surgery explainable-ML study, and the markerless hand-tracking
  study (PubMed 42616309).
- [ ] **Confirm preprint status** for the 2026 arXiv items before publication; several may
  have appeared in a venue since.

## Model and data provenance

- [ ] **Jev version strings.** The three independent evaluations identify the model as
  `typesafe/jev-1.13-20260917` and `typesafe-ai/jev`; we pinned `jev-1.13.0`. Confirm
  whether these are the same build. The draft currently treats them as evidence about the
  model family rather than replications, which is the safe reading either way.
- [ ] **NHSR determination** from Cornell RASS-IRB still pending. The ethics statement
  cannot be finalised until it arrives.
- [ ] **Rotate the TypeSafe API key** used for data collection.

## Reporting

- [ ] Decide venue (IJCARS or a MICCAI-family workshop) once the Introduction and Abstract
  exist and the real total is visible. If something must shrink, Methods is the preferred
  candidate; §2.3 carries the novelty argument and is the least willing cut.
- [ ] Regenerate `paper/tables.md` and all figures from the committed scripts immediately
  before submission, so every number in the manuscript matches the log at that commit.
- [ ] Confirm the pre-declared versus post-hoc labelling survives copy-editing. It is the
  paper's own protocol-discipline claim and the easiest thing to lose in a length cut.
