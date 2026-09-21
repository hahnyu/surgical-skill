"""Build the LaTeX submission from the Markdown manuscript.

Generates paper/ijcars/latex/ with one .tex per section, tables converted from
paper/tables.md (which is itself generated from the experiment log), figures copied from
results/figures/, and references.bib. Re-runnable: it overwrites its own outputs and never
edits anything by hand.

On the document class: IJCARS uses Springer Nature's `sn-jnl.cls`, which Springer
distributes through its author-support pages and which is not on CTAN. It is therefore not
vendored here. main.tex is written so that dropping the official class in requires changing
one line; until then it builds with `article` plus matching packages, so the content can be
proofed and the bibliography checked.

Run:  uv run python src/build_latex.py
"""

import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "paper" / "ijcars" / "manuscript.md"
TABLES_MD = PROJECT_ROOT / "paper" / "tables.md"
FIGURES_SRC = PROJECT_ROOT / "results" / "figures"
OUT = PROJECT_ROOT / "paper" / "ijcars" / "latex"

# Inline "[Author et al. Year]" and "(Author et al. Year)" mapped to BibTeX keys. Keys for
# references that are still unresolved (see paper/citations.md) are included so the build
# fails loudly on a missing entry rather than dropping a citation silently.
CITATIONS = {
    "Gao et al. 2014": "gao2014jigsaws",
    "Ahmidi et al. 2017": "ahmidi2017dataset",
    "van Amsterdam et al. 2021": "vanamsterdam2021gesture",
    "Hendricks et al. 2024": "hendricks2024exploring",
    "Chen et al. 2026": "chen2026interpretable",
    "Lefor et al. 2020": "lefor2020motion",
    "Saeb et al. 2017": "saeb2017need",
    "Little et al. 2017": "little2017using",
    "Chaibub Neto et al. 2019": "chaibubneto2019detecting",
    "Brookshire et al. 2024": "brookshire2024leakage",
    "Geirhos et al. 2020": "geirhos2020shortcut",
    "Kapoor and Narayanan 2023": "kapoor2023leakage",
    "Ilgen et al. 2015": "ilgen2015checklists",
    "Gawad et al. 2019": "gawad2019interrater",
    "Martin et al. 1997": "martin1997osats",
    "Baghdadi et al. 2023": "baghdadi2023force",
    "Zia and Essa 2018": "zia2018automated",
    "Wang et al. 2021": "wang2021towards",
    "Dvijotham et al. 2023": "dvijotham2023codoc",
    "ICLR 2025": "zheng2024trust",
    "Eliot 2026": "eliot2026forbes",
    "Xuan et al. 2026": "xuan2026explainable",
    "ZEAL 2024": "zeal"
}
BRACKET_EXTRA = {
    '["Calibrated Triage, Not Autonomy", 2026]': "calibratedtriage2026",
    '"Looking Beyond the Scale" [2026]': "hoffmann2026looking",
    "SurgiQ [2026]": "surgiq2026",
    "[priorbench; scienthoon; webofmike]": "priorbench2026,scienthoon2026,webofmike2026"
}

# The manuscript also uses the narrative form "Saeb et al. [2017]", where only the year is
# bracketed. These are matched by surname and year rather than by an exact phrase.
NARRATIVE = {
    ("Saeb", "2017"): "saeb2017need",
    ("Little", "2017"): "little2017using",
    ("Lefor", "2020"): "lefor2020motion",
    ("Hendricks", "2024"): "hendricks2024exploring",
    ("Baghdadi", "2023"): "baghdadi2023force",
    ("Chen", "2026"): "chen2026interpretable",
    ("Xuan", "2026"): "xuan2026explainable",
    ("Constans", "2026"): "constans2026markerless"
}

# Bracketed text that is deliberately NOT a citation and must survive untouched.
NOT_CITATIONS = {"pre-declared", "post-hoc", "pre-declared design, post-hoc metric"}

# Figures, in the order they are referenced.
FIGURES = [
    ("fig1_protocol_gap.png", "fig:protocol_gap", 1.0,
     "\\textbf{A} skill classification accuracy under each protocol, dashed line the "
     "majority-class rate. \\textbf{B} accuracy of identifying the surgeon from the same "
     "features, dashed line chance. \\textbf{C} reduction in GRS error against the "
     "feature-free reference, held-out surgeon. Intervals 95\\%, resampling surgeons."),
    ("eval_reliability.png", "fig:reliability", 1.0,
     "Reliability. Each path is judged against its own target, so the panels are not a "
     "head-to-head comparison. Points below the diagonal are overconfident."),
    ("eval_risk_coverage.png", "fig:risk_coverage", 0.92,
     "Risk--coverage. If confidence were informative the coloured curve would fall below "
     "the dotted random-order reference as coverage drops; dashed is the best possible."),
]


def escape(text: str) -> str:
    """Escape LaTeX specials in running prose. Code spans are handled separately."""
    for char in ("\\", "{", "}"):
        text = text.replace(char, {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}"}[char])
    for char in "%&#$_":
        text = text.replace(char, "\\" + char)
    return text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def inline(text: str) -> str:
    """Convert one paragraph of Markdown to LaTeX, protecting code spans."""
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)

    for phrase, key in BRACKET_EXTRA.items():
        text = text.replace(phrase, f"\x01{key}\x01")
    # Multi-citation groups: "[Zia and Essa 2018; Wang et al. 2021; VBA-Net 2022]".
    def group(m):
        parts = [p.strip() for p in m.group(1).split(";")]
        keys = [CITATIONS.get(p) for p in parts]
        return f"\x01{','.join(keys)}\x01" if all(keys) else m.group(0)
    text = re.sub(r"\[([^\]]*;[^\]]*)\]", group, text)
    for phrase, key in CITATIONS.items():
        text = text.replace(f"[{phrase}]", f"\x01{key}\x01").replace(f"({phrase})", f"\x01{key}\x01")
    # Narrative form: "Saeb et al. [2017]" -> "Saeb et al. \cite{...}".
    def narrative(m):
        key = NARRATIVE.get((m.group(1), m.group(2)))
        return f"{m.group(0)[:m.group(0).index('[')]}\x01{key}\x01" if key else m.group(0)
    text = re.sub(r"([A-Z][A-Za-z]+)((?: et al\.)? )\[(\d{4})\]",
                  lambda m: (lambda k: f"{m.group(1)}{m.group(2)}\x01{k}\x01" if k else m.group(0))(
                      NARRATIVE.get((m.group(1), m.group(3)))), text)

    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)
    text = re.sub(r"§(\d)\.(\d)", r"Section~\1.\2", text)
    text = re.sub(r"§(\d)", r"Section~\1", text)
    text = text.replace("—", "---").replace("–", "--")
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    text = re.sub(r"\x01([a-z0-9,]+)\x01", r"\\cite{\1}", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: r"\texttt{" + escape(spans[int(m.group(1))]) + "}", text)


def convert_section(body: str) -> str:
    """Markdown body of one section to LaTeX, keeping provenance lines as comments."""
    out, paragraph = [], []

    def flush():
        if paragraph:
            out.append(inline(" ".join(paragraph).strip()))
            out.append("")
            paragraph.clear()

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            flush()
            title = re.sub(r"^\d+\.\d+\s*", "", stripped[4:]).strip()
            out += [r"\subsection{" + inline(title) + "}", ""]
        elif stripped.startswith("*Source") or stripped.startswith("*Figure"):
            flush()
            out.append("% PROVENANCE: " + re.sub(r"[*`]", "", stripped))
        elif stripped.startswith(">") or stripped.startswith("|"):
            continue                                   # header notes and raw tables
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out)


def markdown_table_to_latex(block: list, label: str) -> str:
    """One Markdown table (plus the caption paragraph that follows it) to a LaTeX table."""
    header, rows = block[0], [r for r in block[2:] if r.strip().startswith("|")]
    cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    ncol = len(cells(header))
    body = [" & ".join(inline(c) for c in cells(r)) + r" \\" for r in rows]
    # Wide tables are scaled to the text width; narrow ones are left at natural size so
    # their type stays the same size as the body text.
    wide = ncol >= 6
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        (r"\resizebox{\textwidth}{!}{%" if wide else ""),
        r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule",
        " & ".join(r"\textbf{" + inline(c) + "}" for c in cells(header)) + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}",
        ("}" if wide else ""),
        r"\caption{" + label + "}", r"\label{tab:" + label.split(".")[0].lower().replace(" ", "") + "}",
        r"\end{table}", ""])


MAIN_TABLES = ("Table 2.", "Table 3.")   # the others move to the Supplementary Material


def build_tables() -> list:
    """Convert the main-text tables from paper/tables.md (generated from the log)."""
    text = TABLES_MD.read_text().split("\n")
    tables, current, title, caption = [], [], None, []
    for line in text + [""]:
        if line.startswith("### "):
            if current:
                tables.append((title, current, " ".join(caption)))
            title, current, caption = line[4:].strip(), [], []
        elif line.strip().startswith("|"):
            current.append(line)
        elif current and line.strip():
            caption.append(line.strip())
    if current:
        tables.append((title, current, " ".join(caption)))

    tables = [t for t in tables if t[0].startswith(MAIN_TABLES)]
    written = []
    for i, (name, block, cap) in enumerate(tables, start=1):
        tex = markdown_table_to_latex(block, name)
        tex = tex.replace(r"\caption{" + name + "}", r"\caption{" + inline(name.split(". ", 1)[-1]) +
                          (". " + inline(cap) if cap else "") + "}")
        path = OUT / "tables" / f"table{i}.tex"
        path.write_text(tex)
        written.append(path.name)
    return written


REFERENCES = r"""
@inproceedings{gao2014jigsaws,
  author = {Gao, Yixin and Vedula, S. Swaroop and Reiley, Carol E. and others},
  title = {{JHU-ISI} Gesture and Skill Assessment Working Set ({JIGSAWS}): A Surgical
           Activity Dataset for Human Motion Modeling},
  booktitle = {MICCAI Workshop: M2CAI},
  year = {2014},
  note = {MICCAI M2CAI workshop}
}

@article{ahmidi2017dataset,
  author = {Ahmidi, Narges and Tao, Lingling and Sefati, Shahin and others},
  title = {A Dataset and Benchmarks for Segmentation and Recognition of Gestures in Robotic
           Surgery},
  journal = {IEEE Transactions on Biomedical Engineering},
  year = {2017}, volume = {64}, number = {9}, pages = {2025--2041},
  doi = {10.1109/TBME.2016.2647680}
}

@article{vanamsterdam2021gesture,
  author = {van Amsterdam, Beatrice and Clarkson, Matthew J. and Stoyanov, Danail},
  title = {Gesture Recognition in Robotic Surgery: A Review},
  journal = {IEEE Transactions on Biomedical Engineering},
  year = {2021}, volume = {68}, number = {6}, pages = {2021--2035},
  doi = {10.1109/TBME.2021.3054828}
}

@article{hendricks2024exploring,
  author = {Hendricks, Ashley and Panoff, Rebecca and Xiao, Yu and others},
  title = {Exploring the Limitations and Implications of the {JIGSAWS} Dataset for
           Robot-Assisted Surgery},
  journal = {IEEE Robotics and Automation Letters},
  year = {2024},
  doi = {10.1109/LRA.2024.3421190}
}

@article{chen2026interpretable,
  author = {Chen, K. and Weng, Y. and Zhang, Y. and others},
  title = {An Interpretable Whole-Stream Video Based Deep Learning Framework for Surgical
           Skill Assessment of Basic Procedural Elements},
  journal = {International Journal of Surgery},
  year = {2026}, volume = {112}, pages = {12565--12578},
  doi = {10.1097/JS9.0000000000005082}
}

@article{lefor2020motion,
  author = {Lefor, Alan Kawarai and Harada, Kanako and Dosis, Aristotelis and Mitsuishi,
            Mamoru},
  title = {Motion Analysis of the {JHU-ISI} Gesture and Skill Assessment Working Set Using
           Robotics Video and Motion Assessment Software},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year = {2020}, doi = {10.1007/s11548-020-02259-z}
}

@article{saeb2017need,
  author = {Saeb, Sohrab and Lonini, Luca and Jayaraman, Arun and others},
  title = {The Need to Approximate the Use-Case in Clinical Machine Learning},
  journal = {GigaScience}, year = {2017}, volume = {6}, number = {5}, pages = {gix019},
  doi = {10.1093/gigascience/gix019}
}

@article{little2017using,
  author = {Little, Max A. and Varoquaux, Ga{\"e}l and Saeb, Sohrab and others},
  title = {Using and Understanding Cross-Validation Strategies},
  journal = {GigaScience}, year = {2017}, volume = {6}, number = {5}
}

@article{chaibubneto2019detecting,
  author = {Chaibub Neto, Elias and Perumal, Thanneer M. and Pratap, Abhishek and others},
  title = {Detecting the Impact of Subject Characteristics on Machine Learning-Based
           Diagnostic Applications},
  journal = {npj Digital Medicine}, year = {2019}, volume = {2}, pages = {99},
  doi = {10.1038/s41746-019-0178-x}
}

@article{brookshire2024leakage,
  author = {Brookshire, Geoffrey and Kasper, Jake and Blauch, Nicholas M. and others},
  title = {Data Leakage in Deep Learning Studies of Translational {EEG}},
  journal = {Frontiers in Neuroscience}, year = {2024}, volume = {18}, pages = {1373515},
  doi = {10.3389/fnins.2024.1373515}
}

@article{geirhos2020shortcut,
  author = {Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and others},
  title = {Shortcut Learning in Deep Neural Networks},
  journal = {Nature Machine Intelligence}, year = {2020}, volume = {2}, number = {11},
  pages = {665--673}, doi = {10.1038/s42256-020-00257-z}
}

@article{kapoor2023leakage,
  author = {Kapoor, Sayash and Narayanan, Arvind},
  title = {Leakage and the Reproducibility Crisis in Machine-Learning-Based Science},
  journal = {Patterns}, year = {2023}, volume = {4}, number = {9}, pages = {100804},
  doi = {10.1016/j.patter.2023.100804}
}

@article{ilgen2015checklists,
  author = {Ilgen, Jonathan S. and Ma, Irene W. Y. and Hatala, Rose and others},
  title = {A Systematic Review of Validity Evidence for Checklists Versus Global Rating Scales
           in Simulation-Based Assessment},
  journal = {Medical Education}, year = {2015}, volume = {49}, number = {2}, pages = {161--173},
  doi = {10.1111/medu.12621}
}

@article{gawad2019interrater,
  author = {Gawad, Nada and Fowler, Amanda and Mimeault, Richard and others},
  title = {The Inter-Rater Reliability of Technical Skills Assessment and Retention of Rater
           Training},
  journal = {Journal of Surgical Education}, year = {2019}, volume = {76}, number = {4},
  pages = {1088--1093}, note = {PMID 30709756}
}

@article{martin1997osats,
  author = {Martin, J. A. and Regehr, G. and Reznick, R. and others},
  title = {Objective Structured Assessment of Technical Skill ({OSATS}) for Surgical Residents},
  journal = {British Journal of Surgery}, year = {1997}, volume = {84}, number = {2},
  pages = {273--278}, doi = {10.1046/j.1365-2168.1997.02502.x}
}

@article{baghdadi2023force,
  author = {Baghdadi, Amir and Guo, Eddie and Lama, Sanju and others},
  title = {Force Profile as Surgeon-Specific Signature},
  journal = {Annals of Surgery Open}, year = {2023}, volume = {4}, number = {3}, pages = {e326},
  doi = {10.1097/AS9.0000000000000326}
}

@article{xuan2026explainable,
  author = {Xuan, W. and Rui, Z. and Mingxu, Y. and others},
  title = {Explainable Machine Learning for Automated Robotic Surgical Skill Assessment
           Using Rich Kinematic Features},
  journal = {Journal of Robotic Surgery}, year = {2026},
  doi = {10.1007/s11701-026-03669-y}
}

@article{constans2026markerless,
  author = {Constans, N. and Mitrea, M. and Neumann, N. and others},
  title = {Assessment of Open Surgery Suturing Skills via Markerless Hand Tracking: A
           Multistage Approach Using {3D} Hand Skeleton and Kinematic Metrics},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year = {2026}, doi = {10.1007/s11548-026-03770-5}
}

@article{zia2018automated,
  author = {Zia, Aneeq and Essa, Irfan},
  title = {Automated Surgical Skill Assessment in {RMIS} Training},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year = {2018}, doi = {10.1007/s11548-018-1735-5}
}

@article{wang2021towards,
  author = {Wang, Tianyu and Wang, Yijie and Li, Mian},
  title = {Towards Accurate and Interpretable Surgical Skill Assessment},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year = {2021}, doi = {10.1007/s11548-021-02448-4}
}

@article{dvijotham2023codoc,
  author = {Dvijotham, Krishnamurthy and Winkens, Jim and Barsbey, Melih and others},
  title = {Enhancing the Reliability and Accuracy of {AI}-Enabled Diagnosis via
           Complementarity-Driven Deferral to Clinicians},
  journal = {Nature Medicine}, year = {2023}, volume = {29}, pages = {1814--1820},
  doi = {10.1038/s41591-023-02437-x}
}

@misc{hoffmann2026looking,
  author = {Hoffmann, Hanna and von Bechtolsheim, Felix and Speidel, Stefanie and Hisey, Rebecca},
  title = {Looking Beyond the Scale: Do Surgical Skill Models Learn Transferable
           Representations Across Assessment Rubrics?},
  year = {2026}, eprint = {2608.17519}, archivePrefix = {arXiv},
  note = {Preprint}
}

@misc{surgiq2026,
  title = {{SurgiQ}: A Large-Scale Multi-Domain Benchmark for Evaluating Large Language Models
           in Surgery},
  year = {2026}, eprint = {2606.08071}, archivePrefix = {arXiv},
  note = {Preprint}
}

@misc{calibratedtriage2026,
  title = {Calibrated Triage, Not Autonomy: Confidence Estimation for Medical
           Vision-Language Models},
  year = {2026}, eprint = {2606.15910}, archivePrefix = {arXiv},
  note = {Preprint}
}

@misc{zheng2024trust,
  title = {Trust or Escalate: {LLM} Judges with Provable Guarantees for Human Agreement},
  year = {2024}, eprint = {2407.18370}, archivePrefix = {arXiv},
  note = {Preprint}
}

@misc{tian2025overconfidence,
  author = {Tian, Zailong and others},
  title = {Overconfidence in {LLM}-as-a-Judge: Diagnosis and Confidence-Driven Solution},
  year = {2025}, eprint = {2508.06225}, archivePrefix = {arXiv},
  note = {Preprint}
}

@misc{vbanet2022video,
  title = {Video-Based Formative and Summative Assessment of Surgical Tasks Using Deep Learning},
  year = {2022}, eprint = {2203.09589}, archivePrefix = {arXiv}, note = {Preprint}
}

@misc{eliot2026forbes,
  author = {Eliot, Lance},
  title = {Critique of {TypeSafe}'s {RLCD} Training Claim},
  howpublished = {Forbes}, year = {2026},
  note = {Trade commentary}
}

@misc{priorbench2026,
  title = {Pre-Registered Independent Evaluation of {Jev}},
  howpublished = {\url{https://github.com/priorbench/jev}}, year = {2026},
  note = {Non-peer-reviewed independent evaluation}
}

@misc{scienthoon2026,
  title = {{Jev} Out-of-Distribution Calibration Test},
  howpublished = {\url{https://github.com/scienthoon/jev-ood-calibration}}, year = {2026},
  note = {Non-peer-reviewed independent evaluation}
}

@misc{webofmike2026,
  title = {Benchmarking {Jev} on Agent Tool-Call Risk},
  howpublished = {dev.to}, year = {2026},
  note = {Non-peer-reviewed independent evaluation}
}

@misc{zeal,
  title = {{ZEAL}: Zero-Shot Surgical Video Segmentation for Skill Assessment},
  year = {2024},
  note = {Reference to be completed}
}
"""

MAIN = r"""%% IJCARS submission.
%%
%% DOCUMENT CLASS: IJCARS uses Springer Nature's sn-jnl.cls, distributed from
%% springernature.com/gp/authors/campaigns/latex-author-support and not available on CTAN,
%% so it is not vendored in this repository. To switch to it, download the official
%% template (v3.1, Dec 2024), place sn-jnl.cls and the .bst files beside this file, and
%% replace the \documentclass line below with:
%%
%%     \documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
%%
%% Everything else in this file is written to be compatible with that swap.
%%
%% Generated by src/build_latex.py -- do not edit by hand; edit the Markdown and rebuild.

\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphics}   % for \resizebox on wide tables
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
%% microtype is omitted: not present in this TeX installation and purely cosmetic.

\graphicspath{{figures/}}
\setlength{\parskip}{0.4em}

\title{What does a surgical skill model measure? Quantifying identity confounding and
miscalibration on JIGSAWS}
\author{}
\date{}

\begin{document}
\maketitle

\input{sections/00_abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_methods}
\input{sections/04_results}
\input{sections/05_discussion}

%% Figures and tables are collected after the text; move them inline if the class prefers.
FIGURES_HERE

TABLES_HERE

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""


def main() -> None:
    (OUT / "sections").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    text = SOURCE.read_text()
    text = re.sub(r"^> .*$", "", text, flags=re.M)          # drop the editorial header block

    parts = re.split(r"(?m)^## ", text)
    names = {"Abstract": "00_abstract", "1. Introduction": "01_introduction",
             "2. Related work": "02_related_work", "3. Methods": "03_methods",
             "4. Results": "04_results", "5. Discussion": "05_discussion"}
    written = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        heading = heading.strip()
        stem = names.get(heading)
        if not stem:
            continue
        if heading == "Abstract":
            tex = convert_section(body)
            tex = "\\begin{abstract}\n" + tex + "\n\\end{abstract}\n"
        else:
            tex = "\\section{" + inline(heading.split(". ", 1)[1]) + "}\n\n" + convert_section(body)
        (OUT / "sections" / f"{stem}.tex").write_text(tex)
        written.append(stem)

    tables = build_tables()

    figure_blocks = []
    for filename, label, width, caption in FIGURES:
        src = FIGURES_SRC / filename
        if not src.exists():
            print(f"  WARNING: {filename} missing; figure omitted")
            continue
        shutil.copy2(src, OUT / "figures" / filename)
        figure_blocks.append("\n".join([
            r"\begin{figure*}[htbp]", r"\centering",
            rf"\includegraphics[width={width}\textwidth]{{{filename}}}",
            r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure*}", ""]))

    main_tex = MAIN.replace("FIGURES_HERE", "\n".join(figure_blocks))
    main_tex = main_tex.replace("TABLES_HERE", "\n".join(rf"\input{{tables/{t[:-4]}}}" for t in tables))
    (OUT / "main.tex").write_text(main_tex)
    (OUT / "references.bib").write_text(REFERENCES.lstrip())

    leftovers = []
    for tex_file in sorted((OUT / "sections").glob("*.tex")):
        for m in re.finditer(r"\[([^\]]{1,70})\]", tex_file.read_text()):
            inner = m.group(1)
            if inner in NOT_CITATIONS or not re.search(r"\d{4}", inner):
                continue
            leftovers.append(f"{tex_file.name}: [{inner}]")
    if leftovers:
        raise SystemExit("unconverted citations remain:\n  " + "\n  ".join(leftovers))
    print("citation check: no unconverted citation-shaped brackets remain")

    print(f"sections: {', '.join(written)}")
    print(f"tables:   {len(tables)}")
    print(f"figures:  {len(figure_blocks)}")
    print(f"wrote     {OUT}")


if __name__ == "__main__":
    main()
