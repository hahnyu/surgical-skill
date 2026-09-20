"""Kinematic features for skill assessment.

Every function here takes a plain NumPy array of tool-tip positions with shape
(n_frames, 3) -- one row per frame, columns x, y, z -- so it can be tested on a
small trajectory whose answer we can work out by hand.

Units: positions are assumed to be in meters, so speed is m/s.
TODO(verify): confirm the position units against the JIGSAWS documentation.
The observed values (a few centimeters of workspace, ~1 cm/s mean speed)
are consistent with meters but that is not a source.
"""

from pathlib import Path

import numpy as np
import yaml

from loaders import FPS  # kinematics sampling rate, 30 Hz

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "features.yaml"


def load_feature_config(path: Path = CONFIG_PATH) -> dict:
    """Read the thresholds from configs/features.yaml.

    yaml.safe_load turns the YAML text into a plain Python dict, e.g.
    {"moving_speed_threshold": 0.001, ...}. Feature code takes these values as
    arguments, so nothing is typed in twice and the config stays the one
    record of what was chosen.
    """
    with open(path) as f:
        return yaml.safe_load(f)


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


def idle_fraction(positions: np.ndarray, threshold: float, fps: int = FPS) -> float:
    """Fraction of steps where the tool is moving slower than `threshold` (m/s).

    "Idle" means speed strictly below the threshold. The threshold is chosen
    once and recorded in configs/features.yaml; do not tune it.
    """
    return float((speed(positions, fps) < threshold).mean())


def count_movements(
    positions: np.ndarray, threshold: float, min_duration_s: float, fps: int = FPS
) -> int:
    """Number of movements: runs of steps at or above `threshold` that last at
    least `min_duration_s` seconds.

    Step by step: mark each step as moving (speed >= threshold) or not. Pad the
    marks with a "not moving" at each end, so a run touching the start or the
    end of the trial still has an edge. np.diff of the marks is +1 where a run
    begins and -1 where it ends; subtracting the two positions gives each run's
    length. Keep runs at least min_duration_s * fps steps long, and count them.

    Known limitation: speed is raw. If it dips below the threshold for a single
    frame in the middle of one real movement, this counts two shorter runs (and
    may drop both if they fall under the minimum duration).
    """
    moving = speed(positions, fps) >= threshold
    padded = np.concatenate([[False], moving, [False]]).astype(int)
    edges = np.diff(padded)
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    run_lengths = run_ends - run_starts
    min_steps = int(round(min_duration_s * fps))
    return int((run_lengths >= min_steps).sum())
