"""Tests for src/loaders.py.

These guard the facts everything downstream rests on. A feature function that
is wrong produces numbers that look fine; so does a loader that reads the wrong
column. These tests are what makes that visible.
"""

import pytest

import loaders
from loaders import (
    EXPECTED_TRIAL_COUNTS,
    KINEMATIC_COLUMNS,
    OSATS_ELEMENTS,
    TASKS,
    build_trial_table,
    count_frames,
    load_all_meta,
    load_kinematics,
    load_meta,
    load_splits,
    parse_trial_name,
    splits_are_official,
)


# --- column layout -------------------------------------------------------


def test_there_are_76_columns():
    assert len(KINEMATIC_COLUMNS) == 76
    assert len(set(KINEMATIC_COLUMNS)) == 76, "column names must be unique"


def test_psm_blocks_start_where_the_readme_says():
    """readme.txt: slave left starts at column 39, slave right at 58.

    Those are 1-indexed, so 38 and 57 as Python indices. Getting this wrong is
    the single most likely cause of a silently wrong result downstream.
    """
    assert KINEMATIC_COLUMNS[38] == "PSM1_x"
    assert KINEMATIC_COLUMNS[57] == "PSM2_x"
    # Each arm block is 19 wide: 3 + 9 + 3 + 3 + 1.
    assert KINEMATIC_COLUMNS[0] == "MTML_x"
    assert KINEMATIC_COLUMNS[19] == "MTMR_x"
    assert KINEMATIC_COLUMNS[56] == "PSM1_grip"
    assert KINEMATIC_COLUMNS[75] == "PSM2_grip"


def test_parse_trial_name():
    assert parse_trial_name("Suturing_B001") == ("Suturing", "B", 1)
    assert parse_trial_name("Knot_Tying_H005") == ("Knot_Tying", "H", 5)
    assert parse_trial_name("Needle_Passing_I003") == ("Needle_Passing", "I", 3)
    with pytest.raises(ValueError):
        parse_trial_name("Nonsense_A001")


# --- kinematics ----------------------------------------------------------


def test_kinematics_shape_and_duration():
    df = load_kinematics("Suturing_B001")
    assert df.shape[1] == 76
    assert len(df) == 5635
    duration = len(df) / loaders.FPS
    assert 10 < duration < 600, "a trial should be seconds-to-minutes long"


def test_kinematics_missing_trial_raises():
    with pytest.raises(FileNotFoundError):
        load_kinematics("Suturing_Z999")


def test_count_frames_matches_a_full_read():
    assert count_frames("Suturing_B001") == len(load_kinematics("Suturing_B001"))


# --- ratings -------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_meta_trial_counts(task):
    assert len(load_meta(task)) == EXPECTED_TRIAL_COUNTS[task]


@pytest.mark.parametrize("task", TASKS)
def test_grs_equals_sum_of_osats_elements(task):
    """The arithmetic check that confirms we read the right columns."""
    df = load_meta(task)
    assert (df[OSATS_ELEMENTS].sum(axis=1) == df["grs"]).all()


@pytest.mark.parametrize("task", TASKS)
def test_scores_are_in_range(task):
    df = load_meta(task)
    assert df["grs"].between(6, 30).all()
    for element in OSATS_ELEMENTS:
        assert df[element].between(1, 5).all()
    assert set(df["skill"]) <= {"N", "I", "E"}


def test_trial_table_has_103_rows():
    df = build_trial_table()
    assert len(df) == 103
    assert df["trial"].is_unique
    assert not df.isna().any().any(), "no missing values anywhere in the table"
    assert (df["n_frames"] > 0).all()


def test_every_meta_trial_has_a_kinematics_file():
    for trial in load_all_meta()["trial"]:
        assert count_frames(trial) > 0, f"{trial} has no usable kinematics"


# --- cross-validation splits --------------------------------------------


def test_only_suturing_ships_official_skill_splits():
    """Documents the state of the data, so a future re-download that adds the
    other two tasks fails here loudly instead of changing results silently."""
    assert splits_are_official("Suturing")
    assert not splits_are_official("Knot_Tying")
    assert not splits_are_official("Needle_Passing")


@pytest.mark.parametrize("scheme", ["LOSO", "LOUO"])
def test_generated_splits_match_official_suturing(scheme):
    """The load-bearing test.

    We generate folds for Knot-Tying and Needle-Passing because JIGSAWS ships
    skill splits only for Suturing. That is only defensible if our generation
    rule reproduces the official folds where they do exist. It does -- exactly,
    train and test, under both schemes. If this ever fails, the generated folds
    for the other two tasks are no longer the standard definition.
    """
    official = loaders._load_official_splits("Suturing", scheme, loaders.DATA_DIR)
    generated = loaders._generate_splits("Suturing", scheme, loaders.DATA_DIR)

    assert len(official) == len(generated)
    for off, gen in zip(official, generated):
        assert set(off.test) == set(gen.test)
        assert set(off.train) == set(gen.train)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("scheme", ["LOSO", "LOUO"])
def test_splits_have_no_train_test_overlap(task, scheme):
    """Leakage check. CLAUDE.md names this as the most common beginner error
    in exactly this setup, so it gets a test rather than a code review."""
    for fold in load_splits(task, scheme):
        assert not (set(fold.train) & set(fold.test)), f"{task} {scheme} {fold.name} leaks"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("scheme", ["LOSO", "LOUO"])
def test_every_trial_is_tested_exactly_once(task, scheme):
    folds = load_splits(task, scheme)
    tested = [t for fold in folds for t in fold.test]
    assert len(tested) == len(set(tested)), "a trial appears in two test folds"
    assert set(tested) == set(load_meta(task)["trial"])


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("scheme", ["LOSO", "LOUO"])
def test_split_trials_resolve_to_real_files(task, scheme):
    """Phase 1 acceptance: every trial named in a split resolves to a file."""
    for fold in load_splits(task, scheme):
        for trial in set(fold.train) | set(fold.test):
            assert count_frames(trial) > 0


@pytest.mark.parametrize("task", TASKS)
def test_fold_counts(task):
    """LOSO holds out one repetition (5), LOUO one subject (8)."""
    assert len(load_splits(task, "LOSO")) == 5
    n_subjects = load_meta(task)["subject"].nunique()
    assert len(load_splits(task, "LOUO")) == n_subjects


@pytest.mark.parametrize("task", TASKS)
def test_loso_folds_hold_out_one_repetition(task):
    for fold in load_splits(task, "LOSO"):
        reps = {parse_trial_name(t)[2] for t in fold.test}
        assert len(reps) == 1, f"{task} {fold.name} mixes repetitions"


@pytest.mark.parametrize("task", TASKS)
def test_louo_folds_hold_out_one_subject(task):
    for fold in load_splits(task, "LOUO"):
        subjects = {parse_trial_name(t)[1] for t in fold.test}
        assert len(subjects) == 1, f"{task} {fold.name} mixes subjects"


def test_bad_scheme_raises():
    with pytest.raises(ValueError):
        load_splits("Suturing", "LOO")
