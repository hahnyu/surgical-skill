"""Kinematic features for skill assessment.

Every function here takes a plain NumPy array of tool-tip positions with shape
(n_frames, 3) -- one row per frame, columns x, y, z -- so it can be tested on a
small trajectory whose answer we can work out by hand.

Units: positions are assumed to be in meters, so speed is m/s.
TODO(verify): confirm the position units against the JIGSAWS documentation.
The observed values (a few centimeters of workspace, ~1 cm/s mean speed)
are consistent with meters but that is not a source.
"""

import numpy as np

from loaders import FPS  # kinematics sampling rate, 30 Hz


def task_time(n_frames: int, fps: int = FPS) -> float:
    """Task duration in seconds: number of frames divided by the frame rate."""
    return n_frames / fps


def step_lengths(positions: np.ndarray) -> np.ndarray:
    """Distance the tool tip moved between each pair of consecutive frames.

    np.diff subtracts each row from the next, giving the (dx, dy, dz) change per
    step. np.linalg.norm then turns each (dx, dy, dz) into a straight-line
    distance: sqrt(dx^2 + dy^2 + dz^2). Result has n_frames - 1 entries.
    """
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def path_length(positions: np.ndarray) -> float:
    """Total distance travelled by the tool tip: the sum of all step lengths."""
    return float(step_lengths(positions).sum())


def speed(positions: np.ndarray, fps: int = FPS) -> np.ndarray:
    """Speed at each step: distance moved divided by the time between frames.

    Time between frames is 1/fps, so speed = step length * fps. There is one
    value per step, so n_frames - 1 in total.
    """
    return step_lengths(positions) * fps


def acceleration(positions: np.ndarray, fps: int = FPS) -> np.ndarray:
    """Size of the acceleration vector at each frame, from positions alone.

    np.diff(..., n=2) takes the difference of the differences: the change in
    (dx, dy, dz) from one step to the next. Dividing by the time per frame
    squared (1/fps)^2 turns that into a real acceleration, so we multiply by
    fps**2. We then take the length of each acceleration vector.

    This is the size of the full vector, so it also picks up changes of
    direction, not just speeding up and slowing down. Result has n_frames - 2
    entries.

    This is RAW: no smoothing. Differencing twice amplifies measurement noise,
    so these values are noisy. See docs/NOTES.md (2026-09-20).
    """
    return np.linalg.norm(np.diff(positions, n=2, axis=0), axis=1) * fps**2


def acceleration_features(positions: np.ndarray, fps: int = FPS) -> dict:
    """Mean and maximum acceleration magnitude (raw, unsmoothed)."""
    a = acceleration(positions, fps)
    return {
        "accel_mean": float(a.mean()),
        "accel_max": float(a.max()),
    }


def speed_features(positions: np.ndarray, fps: int = FPS) -> dict:
    """Mean, standard deviation, and maximum speed."""
    s = speed(positions, fps)
    return {
        "speed_mean": float(s.mean()),
        "speed_std": float(s.std()),
        "speed_max": float(s.max()),
    }
