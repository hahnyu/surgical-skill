# Pre-submission checklist

Open items that do not block drafting but must be closed before the paper is submitted.
Each names what is currently unverified and what would close it.

## Citations and novelty

- [ ] **Systematic search backing the §2.3 negative claim.** The claim that no surgical
  skill assessment study both attributes the protocol gap to identity leakage and
  demonstrates it with a surgeon-identification test currently rests on targeted searches,
  not a systematic pass. Run Scholar, PubMed and IEEE Xplore; record the search strings,
  databases and dates; keep the record so the claim is defensible in review. Two
  stress-test searches have already failed to break it, which is encouraging but is not
  the same as a recorded protocol.
- [ ] **Systematic search backing the §2.5 negative claim**, now narrowed to "no
  calibration or selective-prediction study of a surgical *skill assessment* model built
  from motion, kinematic or video data". SurgiQ is the known near-miss and is cited.
- [ ] **Verify every borrowed figure against its primary source.** All bibliographic
  details and quantitative figures attributed to prior work come from a supplied
  literature review and have not been checked by us. Specifically: Ilgen et al. ICC 0.78
  (0.71-0.83); Gawad et al. ICC 0.26 and the retention figures; Saeb et al. error rates;
  Brookshire et al. 99%-to-chance and 79.1%-to-65.1%; the markerless hand-tracking 52.3%;
  "Looking Beyond the Scale" CCC -0.03; CoDoC's 25% false-positive reduction; Kapoor and
  Narayanan's 294 papers across 17 fields (the Patterns count, not the preprint's 329).
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
