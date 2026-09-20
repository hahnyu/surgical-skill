"""Loading JIGSAWS kinematics, expert ratings, and cross-validation splits.

Everything downstream depends on this module being correct, so it verifies its
own assumptions on every load and raises rather than returning something
plausible-looking but wrong. See docs/NOTES.md (2026-09-19) for how the file
layouts were confirmed.
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Project root is the parent of src/, so the module works regardless of the
# directory a script happens to be run from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]

FPS = 30  # kinematics sampling rate, Hz

# Number of usable trials per task. These come from the meta files, which are
# the authoritative list; there are 40 kinematics files per task on disk but
# some trials were dropped for corrupt recordings.
EXPECTED_TRIAL_COUNTS = {"Suturing": 39, "Knot_Tying": 36, "Needle_Passing": 28}

# The six modified-OSATS elements, in the order given by each task's readme.txt.
# NOTE: this order is taken on the readme's word. The arithmetic check below
# confirms that these six columns sum to GRS, but a sum cannot confirm their
# order among themselves. Do not report per-element results without first
# verifying this against the JIGSAWS papers. See docs/NOTES.md.
OSATS_ELEMENTS = [
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]


def _kinematic_column_names():
    """Build the 76 column names, in file order.

    The file has four 19-column blocks, one per arm, in this order:
    left master (MTML), right master (MTMR), left slave (PSM1), right slave
    (PSM2). Within each block: 3 position, 9 rotation matrix, 3 linear
    velocity, 3 angular velocity, 1 gripper angle.

    Use the PSM blocks for skill features -- those are the instruments inside
    the patient. The MTM blocks are the surgeon's hands at the console.
    """
    names = []
    for arm in ["MTML", "MTMR", "PSM1", "PSM2"]:
        names += [f"{arm}_{axis}" for axis in "xyz"]                    # position
        names += [f"{arm}_R{i}{j}" for i in range(1, 4) for j in range(1, 4)]
        names += [f"{arm}_v{axis}" for axis in "xyz"]                   # linear velocity
        names += [f"{arm}_w{axis}" for axis in "xyz"]                   # angular velocity
        names += [f"{arm}_grip"]                                        # gripper angle
    return names


KINEMATIC_COLUMNS = _kinematic_column_names()
assert len(KINEMATIC_COLUMNS) == 76

# Convenience groupings for feature code, so nothing downstream needs a raw index.
PSM_POSITION = {arm: [f"{arm}_{a}" for a in "xyz"] for arm in ["PSM1", "PSM2"]}
PSM_LINEAR_VELOCITY = {arm: [f"{arm}_v{a}" for a in "xyz"] for arm in ["PSM1", "PSM2"]}


def task_of(trial: str) -> str:
    """'Suturing_B001' -> 'Suturing'. Trial names are <task>_<subject><rep>."""
    for task in TASKS:
        if trial.startswith(task + "_"):
            return task
    raise ValueError(f"cannot determine task from trial name {trial!r}")


def parse_trial_name(trial: str):
    """'Suturing_B001' -> ('Suturing', 'B', 1)."""
    task = task_of(trial)
    code = trial[len(task) + 1:]          # e.g. 'B001'
    return task, code[0], int(code[1:])


def load_kinematics(trial: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one trial's kinematics as a DataFrame with named columns.

    One row per frame at 30 Hz, 76 named columns. Values are space-padded with
    no header, so we tell pandas to split on runs of whitespace (sep=r"\\s+")
    and supply the names ourselves.
    """
    task = task_of(trial)
    path = data_dir / task / "kinematics" / "AllGestures" / f"{trial}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no kinematics file for trial {trial!r}: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=KINEMATIC_COLUMNS)

    if df.shape[1] != 76:
        raise ValueError(f"{trial}: expected 76 columns, got {df.shape[1]}")
    if df.empty:
        raise ValueError(f"{trial}: kinematics file is empty")
    return df


def count_frames(trial: str, data_dir: Path = DATA_DIR) -> int:
    """Number of frames in a trial, without parsing the numbers.

    Counting lines is much cheaper than reading 76 floats per row, and we only
    need the count when building the trial table.
    """
    task = task_of(trial)
    path = data_dir / task / "kinematics" / "AllGestures" / f"{trial}.txt"
    with path.open() as f:
        return sum(1 for _ in f)


def load_meta(task: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one task's expert ratings.

    Returns one row per usable trial, with columns: trial, task, subject,
    repetition, skill (self-proclaimed N/I/E), grs, and the six OSATS elements.

    The file is tab-separated with CRLF endings and contains double tabs, so we
    split on whitespace rather than on tabs. A junk final line holding only a
    carriage return is dropped.
    """
    path = data_dir / task / f"meta_file_{task}.txt"
    columns = ["trial", "skill", "grs"] + OSATS_ELEMENTS

    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < len(columns):
            continue  # the trailing carriage-return-only line
        rows.append(parts[:len(columns)])

    df = pd.DataFrame(rows, columns=columns)
    df[["grs"] + OSATS_ELEMENTS] = df[["grs"] + OSATS_ELEMENTS].astype(int)

    # Split 'Suturing_B001' into subject 'B' and repetition 1.
    parsed = df["trial"].map(parse_trial_name)
    df["task"] = [p[0] for p in parsed]
    df["subject"] = [p[1] for p in parsed]
    df["repetition"] = [p[2] for p in parsed]

    _check_meta(df, task)
    return df[["trial", "task", "subject", "repetition", "skill", "grs"] + OSATS_ELEMENTS]


def _check_meta(df: pd.DataFrame, task: str) -> None:
    """Fail loudly if the ratings are not what we verified them to be."""
    expected = EXPECTED_TRIAL_COUNTS[task]
    if len(df) != expected:
        raise ValueError(f"{task}: expected {expected} trials in meta file, got {len(df)}")

    # The check that tells us we have the right columns: the total must equal
    # the sum of its parts, on every row.
    sums = df[OSATS_ELEMENTS].sum(axis=1)
    bad = df.loc[sums != df["grs"], "trial"].tolist()
    if bad:
        raise ValueError(f"{task}: GRS does not equal sum of OSATS elements for {bad}")

    if not df["grs"].between(6, 30).all():
        raise ValueError(f"{task}: GRS outside the valid 6-30 range")


def load_all_meta(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """All 103 trials across the three tasks, one row each."""
    df = pd.concat([load_meta(t, data_dir) for t in TASKS], ignore_index=True)
    if len(df) != sum(EXPECTED_TRIAL_COUNTS.values()):
        raise ValueError(f"expected 103 trials in total, got {len(df)}")
    return df


def build_trial_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The Phase 1 deliverable: ratings plus trial length, one row per trial."""
    df = load_all_meta(data_dir)
    df["n_frames"] = df["trial"].map(lambda t: count_frames(t, data_dir))
    df["duration_s"] = df["n_frames"] / FPS
    return df


class Fold(NamedTuple):
    """One cross-validation fold: which trials to train on, which to test on."""
    name: str
    train: list
    test: list


def load_splits(task: str, scheme: str, data_dir: Path = DATA_DIR) -> list:
    """Cross-validation folds for a task.

    scheme is "LOSO" (leave-one-supertrial-out; hold out repetition i across
    all subjects) or "LOUO" (leave-one-user-out; hold out subject i).

    Only Suturing ships official skill-detection splits. For Knot-Tying and
    Needle-Passing we generate folds by the same rule the Suturing files
    follow -- see docs/NOTES.md. Use splits_are_official() to report this
    honestly.

    The official files also carry a score column. We ignore it: it disagrees
    with the meta file for subjects F-I. The meta file is authoritative.
    """
    if scheme not in ("LOSO", "LOUO"):
        raise ValueError(f"scheme must be 'LOSO' or 'LOUO', got {scheme!r}")

    if splits_are_official(task, data_dir):
        return _load_official_splits(task, scheme, data_dir)
    return _generate_splits(task, scheme, data_dir)


def splits_are_official(task: str, data_dir: Path = DATA_DIR) -> bool:
    """True if JIGSAWS ships SkillDetection splits for this task."""
    return (data_dir / "Experimental_setup" / task / "unBalanced" / "SkillDetection").is_dir()


def _load_official_splits(task: str, scheme: str, data_dir: Path) -> list:
    dirname = {"LOSO": "SuperTrialOut", "LOUO": "UserOut"}[scheme]
    base = data_dir / "Experimental_setup" / task / "unBalanced" / "SkillDetection" / dirname

    folds = []
    for fold_dir in sorted(base.glob("*_Out"), key=lambda p: int(p.name.split("_")[0])):
        train = _read_split_file(fold_dir / "itr_1" / "Train.txt")
        test = _read_split_file(fold_dir / "itr_1" / "Test.txt")
        folds.append(Fold(name=fold_dir.name, train=train, test=test))
    if not folds:
        raise ValueError(f"no {scheme} folds found under {base}")
    return folds


def _read_split_file(path: Path) -> list:
    """Trial names from a split file, dropping the frame range and the score.

    Lines look like 'Suturing_B001_000080_005635.txt    13'. We keep only
    'Suturing_B001' -- the trailing two underscore-separated fields are a frame
    window, and the number is a GRS we deliberately ignore.
    """
    trials = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        stem = parts[0].replace(".txt", "")
        trials.append("_".join(stem.split("_")[:-2]))
    return trials


def _generate_splits(task: str, scheme: str, data_dir: Path) -> list:
    """Build folds by the convention the official Suturing files follow."""
    meta = load_meta(task, data_dir)
    key = "repetition" if scheme == "LOSO" else "subject"

    folds = []
    for i, held_out in enumerate(sorted(meta[key].unique()), start=1):
        is_test = meta[key] == held_out
        folds.append(Fold(
            name=f"{i}_Out",
            train=meta.loc[~is_test, "trial"].tolist(),
            test=meta.loc[is_test, "trial"].tolist(),
        ))
    return folds
