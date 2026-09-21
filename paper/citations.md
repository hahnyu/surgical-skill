# Citation register

Every reference, with verification status. Built 2026-09-21 by resolving each against a
primary source (PubMed E-utilities, Crossref, arXiv API). **Verified** means I retrieved the
record and confirmed venue, year and any quoted figure. Nothing here is taken on the
literature review's word alone.

## Verified: venue, identifiers and quoted figures all confirmed

| Citation | Venue | DOI / ID | Figure quoted in our text | Status |
|---|---|---|---|---|
| Ilgen et al. 2015 | Medical Education 49(2):161-73 | 10.1111/medu.12621 | inter-rater reliability 0.78 (95% CI 0.71-0.83, n=23); 45 studies | **verified** via PubMed abstract. **The coefficient is not named**, so it is not strictly an ICC; our text now says so |
| Gawad et al. 2019 | J Surgical Education | PMID 30709756 | ICC 0.26 untrained; 0.76 post-training; retention -0.17 to 0.70 | verified |
| Martin et al. 1997 | British Journal of Surgery 84(2):273-8 | 10.1046/j.1365-2168.1997.02502.x | "Mean interrater reliability across stations ranged from 0.64 to 0.72" | verified; coefficient also unnamed |
| Baghdadi et al. 2023 | Annals of Surgery Open 4(3):e326 | 10.1097/AS9.0000000000000326 | AUC 0.97; 83.8% surgeon identification; 89 cases | verified |
| Constans et al. 2026 | IJCARS | 10.1007/s11548-026-03770-5 | 52.3% under leave-one-subject-out; "surgeon-specific component" quote | verified |
| Hendricks et al. 2024 | IEEE Robotics and Automation Letters | 10.1109/lra.2024.3421190 | label/GRS mismatch (direction only) | verified |
| Chen et al. 2026 | **International Journal of Surgery** 112:12565-78 | 10.1097/JS9.0000000000005082 | described qualitatively | verified. **The supplied review listed this as IJCARS; it is not.** Our text has been corrected |
| Ahmidi et al. 2017 | IEEE Trans Biomedical Engineering | 10.1109/tbme.2016.2647680 | benchmark definition | verified |
| Zia and Essa 2018 | IJCARS | 10.1007/s11548-018-1735-5 | none quoted | verified |
| Wang et al. 2021 | IJCARS | 10.1007/s11548-021-02448-4 | none quoted | verified |
| Lefor et al. 2020 | IJCARS | 10.1007/s11548-020-02259-z | described qualitatively | verified |
| van Amsterdam et al. 2021 | **IEEE Trans Biomedical Engineering** | 10.1109/TBME.2021.3054828 | "subject-specific style variability" | verified. **Published**, not a preprint; cite the journal version, not arXiv 2102.00027 |
| Saeb et al. 2017 | GigaScience 6(5):gix019 | 10.1093/gigascience/gix019 | ~2% record-wise; ~7% (30 subjects); 27% (2 subjects) | verified |
| Chaibub Neto et al. 2019 | npj Digital Medicine | 10.1038/s41746-019-0178-x | term "identity confounding" | verified |
| Brookshire et al. 2024 | Frontiers in Neuroscience 18:1373515 | 10.3389/fnins.2024.1373515 | 99.8% (99.1-100.0) to 53.0% (43.1-64.8); 79.1% to 65.1% | verified; more precise than the review reported |
| Geirhos et al. 2020 | Nature Machine Intelligence | 10.1038/s42256-020-00257-z | shortcut learning framing | verified |
| Kapoor and Narayanan 2023 | Patterns 4(9):100804 | 10.1016/j.patter.2023.100804 | "17 fields ... collectively affecting 294 papers" | verified; the preprint's 329 is not used |
| Dvijotham et al. 2023 | Nature Medicine | 10.1038/s41591-023-02437-x | 25% false positives; 66% workload | verified |

## Verified as preprints (no journal reference as of 2026-09-21)

Checked against the arXiv API; none carries a `journal_ref`. Re-check before submission.

| Citation | arXiv | Note |
|---|---|---|
| "Looking Beyond the Scale" (Hoffmann, von Bechtolsheim, Speidel, Hisey) | 2608.17519 (2026-08-18) | CCC -0.03 (+/-0.08) and best baseline 0.22 verified **in the full text only, not the abstract** |
| SurgiQ | 2606.08071 (2026-06-06) | A knowledge benchmark of 13,055 MCQs **that includes a calibration analysis**; not a dedicated calibration study. Our text now says this precisely |
| Speaker leakage in depression detection | 2604.14354 (2026-04-15) | qualitative use only |
| "Calibrated Triage, Not Autonomy" | 2606.15910 (2026-06-14) | qualitative use only |
| "Trust or Escalate" | 2407.18370 (2024-07-25) | **ICLR 2025 acceptance is from the supplied review; arXiv records no journal_ref.** Confirm before citing as ICLR |
| Tian et al., overconfidence in LLM-as-a-judge | 2508.06225 (2025-08-08) | qualitative use only |
| VBA-Net | 2203.09589 (2022-03-17) | none quoted |

## Unresolved — must be completed before submission

- **Gao et al. 2014** (the JIGSAWS dataset paper, MICCAI M2CAI workshop). No DOI located;
  workshop proceedings frequently lack one. Needs a manual citation from the workshop
  proceedings.
- **ZEAL** (zero-shot segmentation feeding a trained LSTM). Cited in §2.6 as the nearest
  zero-shot neighbour; full reference not yet resolved.
- **Eliot (Forbes, 2026)** critique of the RLCD claim. Trade press; needs a dated URL.
- **The three Jev evaluations** (priorbench, scienthoon, webofmike). GitHub repositories and
  a dev.to post; need dated URLs and an explicit non-peer-reviewed designation in the
  reference list as well as in the body text.
