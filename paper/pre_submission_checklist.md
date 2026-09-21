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

- [x] **Verify every borrowed figure — DONE, all reachable sources confirmed.** Full
  register with per-citation status in `paper/citations.md`. Every quantitative figure
  attributed to prior work has now been checked against a primary record; **nothing remains
  paywalled or unverified.** Ilgen et al., the last open item, was resolved through the
  PubMed-hosted abstract (PMID 25626747), which made the journal paywall irrelevant.
  Four corrections came out of this pass, all applied to both manuscripts:
  - **Ilgen et al. and Martin et al. do not name the coefficient they pooled.** Neither is
    strictly an ICC, so our comparison against ICC(2,1) is approximate; the text now says so
    and no longer labels either figure "ICC".
  - **Chen et al. 2026 is in the International Journal of Surgery, not IJCARS** as the
    supplied review stated. Venue corrected.
  - **van Amsterdam et al. is published in IEEE TBME** (10.1109/TBME.2021.3054828), not a
    preprint; cite the journal version.
  - **SurgiQ is a knowledge benchmark that includes a calibration analysis**, not a
    dedicated calibration study. Characterisation corrected in §2.5.
- [x] **Resolve incomplete citations — mostly done, four remain.** Eighteen references now
  carry a verified venue and DOI. Still unresolved and listed in `paper/citations.md`:
  Gao et al. 2014 (MICCAI M2CAI workshop, no DOI located), ZEAL, the Forbes critique, and
  the three Jev evaluations (which need dated URLs and an explicit non-peer-reviewed
  designation in the reference list, not only in the body text).
- [x] **Confirm preprint versus published status — DONE.** Checked against the arXiv API.
  Seven cited arXiv items carry no `journal_ref` and are preprints as of 2026-09-21.
  **One caveat:** "Trust or Escalate" is cited as ICLR 2025 on the strength of the supplied
  review, but arXiv records no journal reference; confirm before citing the venue.
- [x] **Jev version-string caveat reads correctly.** §2.6 states that the independent
  evaluations "identify the model by version strings that may not correspond exactly to the
  build we pinned, so we treat them as evidence about the model family rather than as
  replications of our setup", alongside the bolded non-peer-reviewed designation.
- [x] **Regenerate tables and figures — DONE at this commit.** `paper/tables.md`,
  `fig1_protocol_gap.png`, the three evaluation figures and the six supplementary
  feature figures were all rebuilt from the committed scripts.

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
