"""Tests for src/features.py.

Each test feeds a trajectory whose answer can be worked out by hand. If a
feature function is wrong, every downstream number is wrong and nothing looks
obviously broken -- these tests are what makes it visible.
"""

import numpy as np
import pytest

from features import (
    acceleration,
    acceleration_features,
    count_movements,
    idle_fraction,
    path_length,
    speed,
    speed_features,
    step_lengths,
    task_time,
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
