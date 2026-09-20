"""Tests for src/features.py.

Each test feeds a trajectory whose answer can be worked out by hand. If a
feature function is wrong, every downstream number is wrong and nothing looks
obviously broken -- these tests are what makes it visible.
"""

import numpy as np
import pytest

from features import path_length, speed, speed_features, step_lengths, task_time


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
