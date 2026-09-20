"""Tests for src/features.py.

Each test feeds a trajectory whose answer can be worked out by hand. If a
feature function is wrong, every downstream number is wrong and nothing looks
obviously broken -- these tests are what makes it visible.
"""

import numpy as np
import pytest

from scipy.signal import butter, filtfilt

import loaders

from features import (
    acceleration,
    acceleration_features,
    count_movements,
    gripper_cycles,
    idle_fraction,
    load_feature_config,
    path_length,
    speed,
    speed_features,
    step_lengths,
    task_time,
    bimanual_coordination,
    build_feature_matrix,
    smooth_positions,
    smoothness,
    sparc,
    working_volume,
)


def straight_line(n_frames, step=(0.001, 0.0, 0.0)):
    """A tool tip moving at a constant step per frame, starting at the origin."""
    return np.arange(n_frames)[:, None] * np.array(step)


def test_task_time():
    assert task_time(300, fps=30) == 10.0
    assert task_time(45, fps=30) == 1.5


def test_path_length_of_a_straight_line():
    # 11 frames, 10 steps of 0.001 m along x: the path is 10 * 0.001 = 0.01 m.
    assert path_length(straight_line(11)) == pytest.approx(0.01)


def test_path_length_uses_all_three_axes():
    # A 3-4-5 triangle: one step of (0.003, 0.004, 0) has length 0.005.
    positions = np.array([[0.0, 0.0, 0.0], [0.003, 0.004, 0.0]])
    assert path_length(positions) == pytest.approx(0.005)


def test_path_length_counts_a_return_trip_twice():
    # Out 0.01 m and back again is 0.02 m travelled, even though we end where
    # we started. Path length is distance travelled, not displacement.
    out = straight_line(11)
    there_and_back = np.vstack([out, out[-2::-1]])
    assert path_length(there_and_back) == pytest.approx(0.02)


def test_path_length_of_a_stationary_tool_is_zero():
    assert path_length(np.zeros((100, 3))) == 0.0


def test_step_lengths_has_one_fewer_entry_than_frames():
    assert len(step_lengths(straight_line(11))) == 10


def test_speed_of_constant_motion():
    # 0.001 m per frame at 30 frames/s is 0.03 m/s on every step.
    s = speed(straight_line(50), fps=30)
    assert np.allclose(s, 0.03)


def test_speed_of_a_stationary_tool_is_zero():
    feats = speed_features(np.zeros((100, 3)))
    assert feats == {"speed_mean": 0.0, "speed_std": 0.0, "speed_max": 0.0}


def test_speed_features_on_known_values():
    # Steps of 0.001 then 0.003 along x, at 30 fps: speeds 0.03 and 0.09 m/s.
    positions = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.004, 0, 0]])
    feats = speed_features(positions, fps=30)
    assert feats["speed_mean"] == pytest.approx(0.06)
    assert feats["speed_max"] == pytest.approx(0.09)
    # np.std divides by n (not n-1): values 0.03, 0.09 have mean 0.06 and
    # population std 0.03.
    assert feats["speed_std"] == pytest.approx(0.03)


def test_speed_is_independent_of_direction():
    # The same motion along y, or diagonally, must give the same speed.
    along_x = speed(straight_line(20, step=(0.001, 0, 0)))
    along_y = speed(straight_line(20, step=(0, 0.001, 0)))
    diagonal = speed(straight_line(20, step=(0.001 / np.sqrt(3),) * 3))
    assert np.allclose(along_x, along_y)
    assert np.allclose(along_x, diagonal)


def test_acceleration_of_constant_velocity_is_zero():
    # Moving at a steady speed in a straight line: no acceleration at all.
    assert np.allclose(acceleration(straight_line(50)), 0.0)


def test_acceleration_of_constant_acceleration():
    # Position x = 0.5 * a * t^2 has acceleration exactly a. With a = 2 m/s^2
    # and t = k / 30, the second difference recovers a exactly.
    a, fps = 2.0, 30
    t = np.arange(40) / fps
    positions = np.zeros((40, 3))
    positions[:, 0] = 0.5 * a * t**2
    assert np.allclose(acceleration(positions, fps), a)


def test_acceleration_has_two_fewer_entries_than_frames():
    assert len(acceleration(straight_line(11))) == 9


def test_acceleration_picks_up_a_change_of_direction():
    # Constant speed around a circle still accelerates (toward the centre):
    # magnitude v^2 / r = omega^2 * r. Sampled finely, the finite-difference
    # answer is close to that.
    fps, radius, omega = 30, 0.02, 1.5  # m, rad/s
    t = np.arange(400) / fps
    positions = np.column_stack([radius * np.cos(omega * t), radius * np.sin(omega * t), np.zeros_like(t)])
    expected = omega**2 * radius
    assert np.allclose(acceleration(positions, fps), expected, rtol=0.01)
    assert np.allclose(speed(positions, fps), omega * radius, rtol=0.01)  # speed is constant


def test_acceleration_of_a_stationary_tool_is_zero():
    assert acceleration_features(np.zeros((100, 3))) == {"accel_mean": 0.0, "accel_max": 0.0}


def test_acceleration_features_on_known_values():
    # Positions 0, 0, 0.001, 0.001: differences 0, 0.001, 0; second differences
    # +0.001 and -0.001, each of size 0.001 -> 0.001 * 30^2 = 0.9 m/s^2.
    positions = np.array([[0.0, 0, 0], [0, 0, 0], [0.001, 0, 0], [0.001, 0, 0]])
    feats = acceleration_features(positions, fps=30)
    assert feats["accel_mean"] == pytest.approx(0.9)
    assert feats["accel_max"] == pytest.approx(0.9)


def path_from_steps(step_lengths_m):
    """Build a trajectory along x from a list of per-frame step sizes (metres)."""
    x = np.concatenate([[0.0], np.cumsum(step_lengths_m)])
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


THRESHOLD = 0.001   # m/s, as in configs/features.yaml
MIN_DURATION = 0.2  # s, as in configs/features.yaml (0.2 s * 30 fps = 6 steps)
FAST = 0.002        # m per frame = 0.06 m/s, far above the threshold
SLOW = 0.00001      # m per frame = 0.0003 m/s, safely below the threshold


def test_idle_fraction_of_a_stationary_tool_is_one():
    assert idle_fraction(np.zeros((100, 3)), THRESHOLD) == 1.0


def test_idle_fraction_of_constant_motion_is_zero():
    assert idle_fraction(straight_line(100, step=(FAST, 0, 0)), THRESHOLD) == 0.0


def test_idle_fraction_on_known_values():
    # 30 idle steps then 70 fast steps: idle fraction is exactly 0.3.
    positions = path_from_steps([SLOW] * 30 + [FAST] * 70)
    assert idle_fraction(positions, THRESHOLD) == pytest.approx(0.3)


def test_movements_of_a_stationary_tool_is_zero():
    assert count_movements(np.zeros((100, 3)), THRESHOLD, MIN_DURATION) == 0


def test_movements_of_constant_motion_is_one():
    positions = straight_line(100, step=(FAST, 0, 0))
    assert count_movements(positions, THRESHOLD, MIN_DURATION) == 1


def test_movements_counts_separate_runs():
    # idle, 10 fast, idle, 8 fast, idle: two runs, both >= 6 steps.
    steps = [SLOW] * 5 + [FAST] * 10 + [SLOW] * 5 + [FAST] * 8 + [SLOW] * 5
    assert count_movements(path_from_steps(steps), THRESHOLD, MIN_DURATION) == 2


def test_movements_ignores_runs_shorter_than_the_minimum():
    # A 3-step burst (0.1 s) is below the 0.2 s minimum, so it is not a movement.
    steps = [SLOW] * 5 + [FAST] * 3 + [SLOW] * 5 + [FAST] * 10 + [SLOW] * 5
    assert count_movements(path_from_steps(steps), THRESHOLD, MIN_DURATION) == 1


def test_movements_at_the_minimum_length_counts():
    # Exactly 6 steps (0.2 s) is long enough; 5 steps is not.
    six = [SLOW] * 5 + [FAST] * 6 + [SLOW] * 5
    five = [SLOW] * 5 + [FAST] * 5 + [SLOW] * 5
    assert count_movements(path_from_steps(six), THRESHOLD, MIN_DURATION) == 1
    assert count_movements(path_from_steps(five), THRESHOLD, MIN_DURATION) == 0


def test_movements_at_the_very_start_and_end_are_counted():
    # Runs touching the first or last step must still be found (the padding).
    steps = [FAST] * 8 + [SLOW] * 5 + [FAST] * 8
    assert count_movements(path_from_steps(steps), THRESHOLD, MIN_DURATION) == 2


def test_a_one_frame_dip_splits_a_movement():
    """Documents the known limitation of raw speed: one slow frame in the middle
    of a long movement cuts it in two. Here both halves survive the minimum."""
    steps = [FAST] * 10 + [SLOW] + [FAST] * 10
    assert count_movements(path_from_steps(steps), THRESHOLD, MIN_DURATION) == 2


def test_config_holds_the_pre_declared_values():
    """Locks the thresholds decided on 2026-09-20. If this fails, someone
    changed a threshold: that must be a deliberate, logged decision in
    docs/NOTES.md, not a quiet edit."""
    cfg = load_feature_config()
    assert cfg["moving_speed_threshold"] == 0.001
    assert cfg["min_movement_duration_s"] == 0.2
    assert cfg["idle_threshold_grid"] == [0.0005, 0.001, 0.002]


def test_working_volume_of_a_unit_cube():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    assert working_volume(corners) == pytest.approx(1.0)


def test_working_volume_ignores_points_inside_the_shape():
    # Fill the cube with random interior points: the hull is still the cube.
    rng = np.random.default_rng(0)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    interior = rng.uniform(0.1, 0.9, size=(200, 3))
    assert working_volume(np.vstack([corners, interior])) == pytest.approx(1.0)


def test_working_volume_scales_with_the_cube_of_length():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    assert working_volume(corners * 0.02) == pytest.approx(0.02**3)


def test_working_volume_of_a_tetrahedron():
    # Corners (0,0,0), (1,0,0), (0,1,0), (0,0,1): volume 1/6.
    tetra = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert working_volume(tetra) == pytest.approx(1 / 6)


def test_working_volume_is_zero_when_there_is_no_3d_shape():
    assert working_volume(np.zeros((50, 3))) == 0.0            # never moved
    assert working_volume(straight_line(50)) == 0.0            # moved along a line
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    assert working_volume(flat) == 0.0                         # flat square


def test_bimanual_coordination_of_identical_motion_is_one():
    steps = [FAST, SLOW, FAST, FAST, SLOW, SLOW, FAST] * 5
    a = path_from_steps(steps)
    assert bimanual_coordination(a, a) == pytest.approx(1.0)


def test_bimanual_coordination_of_opposite_motion_is_minus_one():
    # Where one arm is fast the other is slow, and vice versa.
    a = path_from_steps([FAST, SLOW] * 20)
    b = path_from_steps([SLOW, FAST] * 20)
    assert bimanual_coordination(a, b) == pytest.approx(-1.0)


def test_bimanual_coordination_ignores_overall_scale_and_direction():
    # Correlation compares the shape of the speed profiles, so an arm moving
    # twice as fast in the same rhythm, along another axis, still correlates 1.
    steps = np.array([FAST, SLOW, FAST, FAST, SLOW, SLOW, FAST] * 5)
    a = path_from_steps(steps)
    b = path_from_steps(steps * 2)[:, [1, 0, 2]]  # swap x and y
    assert bimanual_coordination(a, b) == pytest.approx(1.0)


def test_bimanual_coordination_raises_when_an_arm_never_changes_speed():
    moving = path_from_steps([FAST, SLOW] * 20)
    with pytest.raises(ValueError):
        bimanual_coordination(moving, np.zeros_like(moving))
    with pytest.raises(ValueError):
        bimanual_coordination(moving, moving[:-5])   # different lengths


def bell(duration_s, fps=30, width=0.15):
    """A smooth bell-shaped speed profile: one clean movement."""
    t = np.linspace(0, 1, int(duration_s * fps))
    return np.exp(-((t - 0.5) / width) ** 2 / 2)


def test_sparc_is_negative():
    assert sparc(bell(2.0)) < 0


def test_sparc_ignores_overall_speed():
    # The spectrum is normalised, so moving 100x faster scores the same.
    assert sparc(bell(2.0) * 100) == pytest.approx(sparc(bell(2.0)))


def test_sparc_of_one_smooth_movement_does_not_depend_on_its_duration():
    """PLAN.md's reason for preferring SPARC. True for a SINGLE movement, which
    is what this tests. It is NOT true of a whole multi-movement trial; see
    test_sparc_worsens_as_more_movements_are_added and docs/NOTES.md."""
    values = [sparc(bell(d)) for d in (1.0, 2.0, 4.0)]
    assert max(values) - min(values) < 0.02


def test_sparc_scores_a_wiggly_movement_as_less_smooth():
    t = np.linspace(0, 3, 90)
    smooth = bell(3.0)
    wiggly = smooth + 0.15 * np.sin(2 * np.pi * 4 * t)   # add a 4 Hz tremor
    assert sparc(wiggly) < sparc(smooth)


def test_sparc_worsens_as_more_movements_are_added():
    """Documents a limitation of whole-trial SPARC (decision 2026-09-20): five
    identical, perfectly smooth movements score much lower than one, so the
    whole-trial value partly counts movements. Measured: one bell -1.41, two
    -2.97, five -4.66."""
    one = sparc(bell(3.0))
    five = sparc(np.concatenate([bell(0.6)] * 5))
    assert five < one - 2.0


def test_sparc_raises_on_an_all_zero_profile():
    with pytest.raises(ValueError):
        sparc(np.zeros(100))


def test_smoothness_is_sparc_of_the_speed_profile():
    positions = path_from_steps([FAST, SLOW, FAST, FAST, SLOW, SLOW, FAST] * 10)
    assert smoothness(positions) == pytest.approx(sparc(speed(positions)))


def test_config_holds_the_sparc_settings():
    cfg = load_feature_config()
    assert cfg["sparc_cutoff_hz"] == 10.0
    assert cfg["sparc_amp_threshold"] == 0.05
    assert cfg["sparc_padlevel"] == 4


GRIP_LOW, GRIP_HIGH = -0.6, -0.1   # as in configs/features.yaml
CLOSED, OPEN = -0.9, 0.4           # the two humps of the PSM1 angle distribution


def test_one_full_gripper_cycle():
    angles = np.array([CLOSED] * 5 + [OPEN] * 5 + [CLOSED] * 5)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 1


def test_two_full_gripper_cycles():
    angles = np.array(([CLOSED] * 4 + [OPEN] * 4) * 2 + [CLOSED] * 4)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 2


def test_a_half_cycle_is_not_counted():
    # Closed then open, and the trial ends: only one transition, no full cycle.
    angles = np.array([CLOSED] * 5 + [OPEN] * 5)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 0


def test_a_gripper_that_never_moves_has_no_cycles():
    assert gripper_cycles(np.full(100, CLOSED), GRIP_LOW, GRIP_HIGH) == 0


def test_jitter_inside_the_band_is_not_counted():
    """The point of hysteresis. An angle flickering across one cut point (-0.35,
    the middle of the band) would inflate a single-threshold count, but here it
    never leaves the band, so nothing changes."""
    angles = np.array([CLOSED] * 5 + [-0.40, -0.30] * 20 + [CLOSED] * 5)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 0


def test_jitter_just_outside_one_edge_is_not_counted():
    # Wobbling between the band and the closed side never reaches the far side.
    angles = np.array([CLOSED, -0.65, -0.55, -0.65, -0.55] * 10)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 0


def test_a_slow_crossing_through_the_band_counts_once():
    ramp_up = np.linspace(CLOSED, OPEN, 30)      # passes through the whole band
    ramp_down = np.linspace(OPEN, CLOSED, 30)
    angles = np.concatenate([np.full(5, CLOSED), ramp_up, ramp_down])
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 1


def test_frames_before_the_first_exit_from_the_band_are_skipped():
    # Starts inside the band (state unknown); the first exit is not a transition.
    angles = np.array([-0.3] * 5 + [OPEN] * 5 + [CLOSED] * 5 + [OPEN] * 5)
    assert gripper_cycles(angles, GRIP_LOW, GRIP_HIGH) == 1


def test_gripper_cycles_rejects_a_reversed_band():
    with pytest.raises(ValueError):
        gripper_cycles(np.zeros(10), -0.1, -0.6)


def test_config_holds_the_gripper_band():
    cfg = load_feature_config()
    assert cfg["gripper_low_threshold"] == -0.6
    assert cfg["gripper_high_threshold"] == -0.1


# --- the assembled feature matrix -----------------------------------------


@pytest.fixture(scope="module")
def matrix():
    return build_feature_matrix()


@pytest.fixture(scope="module")
def raw_matrix():
    """The same matrix with smoothing switched off."""
    cfg = load_feature_config()
    cfg["smoothing_cutoff_hz"] = None
    return build_feature_matrix(cfg)


def test_feature_matrix_shape_and_index(matrix):
    assert matrix.shape == (103, 16)
    assert matrix.index.is_unique
    assert set(matrix.index) == set(loaders.load_all_meta()["trial"])


def test_feature_matrix_has_no_missing_or_infinite_values(matrix):
    assert np.isfinite(matrix.to_numpy(dtype=float)).all()


def test_feature_matrix_holds_only_features_not_identifiers_or_labels(matrix):
    """A model must not be able to see subject, task, repetition or a label.
    Subject identity is the main threat to validity (CLAUDE.md)."""
    forbidden = {"task", "subject", "repetition", "skill", "grs", "trial"}
    assert not forbidden & set(matrix.columns)
    assert not any(c in matrix.columns for c in loaders.OSATS_ELEMENTS)


def test_feature_matrix_is_reproducible(matrix):
    again = build_feature_matrix()
    assert matrix.equals(again)


@pytest.mark.parametrize("trial", ["Suturing_B001", "Knot_Tying_D003", "Needle_Passing_B001"])
@pytest.mark.parametrize("arm, pos_cols, grip_col", [("PSM1", slice(38, 41), 56), ("PSM2", slice(57, 60), 75)])
def test_matrix_agrees_with_an_independent_read_of_the_raw_file(raw_matrix, trial, arm, pos_cols, grip_col):
    """The load-bearing check on column indexing and windowing.

    Recompute path length, task time and the gripper range straight from the raw
    file with np.loadtxt and BARE numeric column indices taken from the readme
    (PSM1 position = columns 39-41 => Python 38:41; PSM2 = 58-60 => 57:60;
    gripper = 57 and 76 => 56 and 75). None of the named-column machinery in
    loaders.py is used. If that machinery had an off-by-one, everything else
    would agree with itself and still be wrong; this would not.
    """
    matrix = raw_matrix   # built with smoothing OFF, so raw positions compare exactly
    task = loaders.task_of(trial)
    raw = np.loadtxt(loaders.DATA_DIR / task / "kinematics" / "AllGestures" / f"{trial}.txt")
    assert raw.shape[1] == 76

    starts, ends = [], []
    for line in (loaders.DATA_DIR / task / "transcriptions" / f"{trial}.txt").read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            starts.append(int(parts[0]))
            ends.append(int(parts[1]))
    window = raw[min(starts) - 1: max(ends)]

    positions = window[:, pos_cols]
    expected_path = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    assert matrix.loc[trial, f"{arm}_path_length"] == pytest.approx(expected_path)
    assert matrix.loc[trial, "task_time_s"] == pytest.approx(len(window) / 30)

    # Gripper: the raw column must match what the loader called <arm>_grip.
    grip = loaders.load_kinematics(trial, windowed=True)[f"{arm}_grip"].to_numpy()
    assert np.array_equal(grip, window[:, grip_col])

    # Speed mean recomputed from raw positions.
    expected_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean() * 30
    assert matrix.loc[trial, f"{arm}_speed_mean"] == pytest.approx(expected_speed)


# --- smoothing ------------------------------------------------------------


def sine(freq_hz, n=300, fps=30):
    t = np.arange(n) / fps
    return np.sin(2 * np.pi * freq_hz * t)[:, None] * np.ones((1, 3))


def test_smoothing_leaves_a_constant_position_alone():
    x = np.full((100, 3), 0.05)
    assert np.allclose(smooth_positions(x, 5.0), 0.05)


def test_smoothing_leaves_a_steady_ramp_alone():
    x = straight_line(100)
    core = slice(10, 90)   # ignore the edges, where any filter has to guess
    assert np.allclose(smooth_positions(x, 5.0)[core], x[core], atol=1e-6)


def gain(freq_hz, cutoff_hz=5.0):
    """Fraction of a sine's amplitude that survives smoothing, away from the edges.
    Divides by the input's own biggest sample: a fast sine sampled at 30 Hz never
    lands exactly on its peak, so comparing with 1.0 would be wrong."""
    core = slice(60, 240)
    x = sine(freq_hz)
    return np.abs(smooth_positions(x, cutoff_hz)[core]).max() / np.abs(x[core]).max()


def test_smoothing_keeps_slow_motion_and_removes_tremor_band_motion():
    assert gain(1.0) > 0.99      # slow, deliberate motion passes
    assert gain(2.0) > 0.95
    assert gain(12.0) < 0.01     # 12 Hz, inside the 8-12 Hz tremor band, is gone
    assert gain(8.0) < 0.10


def test_smoothing_has_gain_one_half_at_the_cutoff():
    """Running the filter forward and backward squares its gain, so at the cutoff
    the gain is 0.5, not the single-pass 0.707. Documented in the config."""
    assert gain(5.0) == pytest.approx(0.5, abs=0.02)


def test_smoothing_adds_no_time_shift():
    """Forward-and-backward filtering cancels the delay: a single bump keeps its
    peak at the same frame. (A bump has one unambiguous peak; a multi-cycle sine
    has several near-equal ones, so argmax would be arbitrary.)"""
    t = np.arange(200)
    bump = np.exp(-((t - 100) / 12.0) ** 2 / 2)[:, None] * np.ones((1, 3))
    smoothed = smooth_positions(bump, 5.0)
    assert np.argmax(bump[:, 0]) == np.argmax(smoothed[:, 0]) == 100


def test_smoothing_keeps_the_shape_and_does_not_change_its_input():
    x = sine(1.0)
    before = x.copy()
    y = smooth_positions(x, 5.0)
    assert y.shape == x.shape
    assert np.array_equal(x, before)


def test_smoothing_shrinks_a_single_frame_position_spike():
    """A one-frame excursion (the kind of glitch behind the huge speed_max) has
    its speed reduced but not removed. Documented limitation."""
    x = np.zeros((200, 3))
    x[100, 0] = 0.02                                     # 20 mm for one frame
    raw_peak = speed(x).max()
    smooth_peak = speed(smooth_positions(x, 5.0)).max()
    assert 0 < smooth_peak < raw_peak / 3


def test_config_holds_the_smoothing_choice():
    cfg = load_feature_config()
    assert cfg["smoothing_cutoff_hz"] == 5.0
    assert cfg["smoothing_order"] == 2
    assert cfg["smoothing_sensitivity_cutoffs_hz"] == [None, 8.0]


# --- the bundle of 2026-09-20 --------------------------------------------


def test_matrix_columns_match_the_bundle(matrix):
    per_arm = ["path_length", "speed_mean", "speed_std", "accel_mean",
               "idle_fraction", "working_volume", "gripper_cycles"]
    expected = ["task_time_s"] + [f"{a}_{f}" for a in ("PSM1", "PSM2") for f in per_arm] + ["bimanual_speed_corr"]
    assert list(matrix.columns) == expected


def test_dropped_features_are_absent(matrix):
    for arm in ("PSM1", "PSM2"):
        for gone in ("speed_max", "accel_max", "sparc", "n_movements", "movements_per_min"):
            assert f"{arm}_{gone}" not in matrix.columns


@pytest.mark.parametrize("trial", ["Suturing_B001", "Knot_Tying_D003", "Needle_Passing_B001"])
def test_smoothed_path_length_agrees_with_an_independent_filter_call(matrix, trial):
    """Recompute the smoothed path length from the raw file with bare column
    indices and a direct scipy filter call, independent of smooth_positions."""
    task = loaders.task_of(trial)
    raw = np.loadtxt(loaders.DATA_DIR / task / "kinematics" / "AllGestures" / f"{trial}.txt")
    starts, ends = [], []
    for line in (loaders.DATA_DIR / task / "transcriptions" / f"{trial}.txt").read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            starts.append(int(parts[0]))
            ends.append(int(parts[1]))
    window = raw[min(starts) - 1: max(ends)]
    b, a = butter(2, 5.0, btype="low", fs=30)
    for arm, cols in (("PSM1", slice(38, 41)), ("PSM2", slice(57, 60))):
        smoothed = filtfilt(b, a, window[:, cols], axis=0)
        expected = np.linalg.norm(np.diff(smoothed, axis=0), axis=1).sum()
        assert matrix.loc[trial, f"{arm}_path_length"] == pytest.approx(expected)


def test_smoothing_reduces_path_length_on_every_trial(matrix, raw_matrix):
    """Removing jitter can only shorten a path. A violation would mean the two
    matrices are not comparing the same thing."""
    for arm in ("PSM1", "PSM2"):
        assert (matrix[f"{arm}_path_length"] <= raw_matrix[f"{arm}_path_length"] + 1e-12).all()
