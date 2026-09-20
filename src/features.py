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
from scipy.spatial import ConvexHull, QhullError

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


def working_volume(positions: np.ndarray) -> float:
    """Volume of the smallest convex shape enclosing every tool-tip position.

    Think of stretching shrink-wrap around all the points the tool tip visited:
    the volume inside is the convex hull volume. scipy.spatial.ConvexHull does
    the geometry (via the Qhull library). Result is in m^3 if positions are in m.

    If the points do not span three dimensions -- for example a tool that never
    moved, or moved along a single line or flat plane -- there is no 3D shape
    and Qhull raises an error; the true volume in that case is 0.
    """
    try:
        return float(ConvexHull(positions).volume)
    except QhullError:
        return 0.0


def bimanual_coordination(positions_a: np.ndarray, positions_b: np.ndarray, fps: int = FPS) -> float:
    """Pearson correlation between the two instruments' speed over time.

    +1 means the two arms speed up and slow down together, 0 means unrelated,
    -1 means one speeds up as the other slows. np.corrcoef returns a 2x2 table
    of correlations; the off-diagonal entry is the one we want.

    Raises if either arm's speed never varies, because the correlation is then
    undefined (0/0). Better to fail loudly than to put a NaN in the feature
    matrix.
    """
    speed_a = speed(positions_a, fps)
    speed_b = speed(positions_b, fps)
    if len(speed_a) != len(speed_b):
        raise ValueError("the two instruments must have the same number of frames")
    if speed_a.std() == 0 or speed_b.std() == 0:
        raise ValueError("speed is constant on one instrument; correlation is undefined")
    return float(np.corrcoef(speed_a, speed_b)[0, 1])


def sparc(
    speed_profile: np.ndarray,
    fps: int = FPS,
    cutoff_hz: float = 10.0,
    amp_threshold: float = 0.05,
    padlevel: int = 4,
) -> float:
    """Spectral arc length (SPARC): a smoothness score. Negative; closer to 0 is smoother.

    Takes a speed profile (one value per time step), not positions.

    1. Fourier-transform the speed profile, zero-padding to a power of two for a
       finely resolved spectrum, and normalise so the peak magnitude is 1. The
       normalising makes the score independent of how fast the movement was.
    2. Keep frequencies up to cutoff_hz, then trim the spectrum to the span
       where the magnitude stays at or above amp_threshold.
    3. Measure the arc length of that curve, with frequency scaled to 0..1 and
       magnitude on the other axis: sum of sqrt(d_freq^2 + d_magnitude^2).
       A smooth movement has a simple spectrum and a short arc; a jerky one
       has many bumps and a long arc.
    4. Negate, so that a smoother movement scores higher (nearer zero).

    Raises on an all-zero speed profile, where the spectrum cannot be normalised.
    """
    speed_profile = np.asarray(speed_profile, dtype=float)
    if not np.any(speed_profile):
        raise ValueError("speed profile is all zeros; SPARC is undefined")

    n_fft = int(2 ** (np.ceil(np.log2(len(speed_profile))) + padlevel))
    freqs = np.arange(n_fft) * fps / n_fft
    magnitude = np.abs(np.fft.fft(speed_profile, n_fft))
    magnitude = magnitude / magnitude.max()

    keep = freqs <= cutoff_hz
    freqs, magnitude = freqs[keep], magnitude[keep]

    above = np.flatnonzero(magnitude >= amp_threshold)
    freqs = freqs[above[0]: above[-1] + 1]
    magnitude = magnitude[above[0]: above[-1] + 1]

    d_freq = np.diff(freqs) / (freqs[-1] - freqs[0])
    d_mag = np.diff(magnitude)
    return float(-np.sum(np.sqrt(d_freq**2 + d_mag**2)))


def smoothness(positions: np.ndarray, fps: int = FPS, **sparc_settings) -> float:
    """Whole-trial SPARC of the tool tip's speed profile (decision 2026-09-20).

    Pass the settings from configs/features.yaml as keyword arguments
    (cutoff_hz, amp_threshold, padlevel).
    """
    return sparc(speed(positions, fps), fps, **sparc_settings)


def gripper_cycles(angles: np.ndarray, low: float, high: float) -> int:
    """Number of complete gripper open/close cycles, using a hysteresis band.

    The gripper is treated as "low" once its angle drops below `low` and as
    "high" once it rises above `high`. Angles between the two thresholds (inside
    the band) change nothing, so jitter around a single cut point cannot count
    as cycles: the angle has to cross the whole band. We do not need to know
    which end is "open"; a cycle is any low -> high -> low (or high -> low ->
    high) excursion.

    The state before the angle first leaves the band is unknown, so those first
    frames are skipped. Each change of state is one transition; two transitions
    make one cycle, so a trial ending part-way through a cycle does not count it.
    """
    if low >= high:
        raise ValueError("low threshold must be below the high threshold")

    state = None            # None until the angle first leaves the band
    transitions = 0
    for angle in angles:
        if angle < low:
            new_state = "low"
        elif angle > high:
            new_state = "high"
        else:
            continue        # inside the band: no change
        if state is not None and new_state != state:
            transitions += 1
        state = new_state
    return transitions // 2
