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


def speed_features(positions: np.ndarray, fps: int = FPS) -> dict:
    """Mean, standard deviation, and maximum speed."""
    s = speed(positions, fps)
    return {
        "speed_mean": float(s.mean()),
        "speed_std": float(s.std()),
        "speed_max": float(s.max()),
    }
