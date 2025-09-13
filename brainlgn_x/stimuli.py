"""
Visual stimuli generators and a minimal Movie container.

Goals
- Stronger than the legacy: support parametric stimuli (grating, flash) and
  easy integration from arrays/NPY/videos (optional).
- Keep outputs as numpy arrays (T, H, W) with an attached frame_rate/t_range.
"""

from typing import Optional, Tuple
import math
import numpy as np


class Movie:
    """
    Minimal movie container to pair (T,H,W) array with frame_rate and t_range.
    """

    def __init__(self, data: np.ndarray, frame_rate: float):
        data = np.asarray(data)
        assert data.ndim == 3, "Movie data must be (T,H,W)."
        self.data = data
        self.frame_rate = float(frame_rate)
        self.t_range = np.arange(data.shape[0], dtype=np.float64) * (1.0 / self.frame_rate)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def as_array(self) -> np.ndarray:
        return self.data


def drifting_grating(row_size: int,
                     col_size: int,
                     frame_rate: float,
                     duration: float,
                     gray_screen: float = 0.0,
                     cpd: float = 0.04,
                     temporal_f: float = 4.0,
                     theta: float = 0.0,
                     contrast: float = 1.0,
                     phase: float = 0.0) -> Movie:
    """
    Generate a drifting sinusoidal grating movie.

    Args:
        row_size, col_size: spatial size (pixels/linear degrees after mapping).
        frame_rate: Hz.
        duration: stimulus duration (excluding gray), seconds.
        gray_screen: leading gray duration (seconds, intensity 0).
        cpd: cycles per degree (spatial frequency along theta direction).
        temporal_f: temporal frequency (Hz).
        theta: orientation in degrees (0: along x-axis of array).
        contrast: amplitude in [0, 1].
        phase: initial phase in degrees.

    Returns:
        Movie with data in [-contrast, contrast]. Gray frames are zeros.
    """
    assert contrast >= 0 and contrast <= 1.0
    assert duration > 0 and frame_rate > 0

    # Build time axis
    n_gray = int(round(gray_screen * frame_rate))
    n_stim = int(round(duration * frame_rate))
    time_range = np.arange(n_stim, dtype=np.float64) / float(frame_rate)

    # Spatial grid (use 1.0 spacing units; cpd handles cycles per "degree")
    yy, xx = np.meshgrid(np.arange(row_size, dtype=np.float64),
                         np.arange(col_size, dtype=np.float64),
                         indexing='ij')
    # Orientation
    theta_rad = math.radians(theta)
    phase_rad = math.radians(phase)
    xy = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)

    # Compute sinusoid over time and space
    tt = time_range[:, None, None]
    stim = contrast * np.sin(2.0 * math.pi * (cpd * xy[None, :, :] + temporal_f * tt) + phase_rad)

    if n_gray > 0:
        gray_block = np.zeros((n_gray, row_size, col_size), dtype=stim.dtype)
        data = np.concatenate([gray_block, stim], axis=0)
    else:
        data = stim

    return Movie(data, frame_rate=frame_rate)


def full_field_flash(row_size: int,
                     col_size: int,
                     frame_rate: float,
                     pre: float,
                     on: float,
                     off: float,
                     post: float,
                     max_intensity: float = 1.0) -> Movie:
    """
    Generate full-field flash sequence: pre-gray, ON, inter-gray, OFF, post-gray.

    All gray periods are intensity 0; ON is +max_intensity; OFF is -max_intensity.
    """
    def _blk(sec: float, val: float) -> np.ndarray:
        n = int(round(sec * frame_rate))
        return np.full((n, row_size, col_size), val, dtype=np.float32) if n > 0 else np.zeros((0, row_size, col_size), dtype=np.float32)

    pre_b = _blk(pre, 0.0)
    on_b = _blk(on, float(max_intensity))
    inter_b = _blk(0.0, 0.0)  # reserved for future extension if needed
    off_b = _blk(off, -float(max_intensity))
    post_b = _blk(post, 0.0)

    data = np.concatenate([pre_b, on_b, inter_b, off_b, post_b], axis=0)
    return Movie(data, frame_rate=frame_rate)


def from_array(data: np.ndarray, frame_rate: float) -> Movie:
    """Wrap an array (T,H,W) into a Movie."""
    return Movie(np.asarray(data), frame_rate=float(frame_rate))


def from_npy(path: str, frame_rate: Optional[float] = None) -> Movie:
    """
    Load a (T,H,W) array from .npy. If the array has a different layout, the
    caller should preprocess before wrapping into Movie.
    """
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError("NPY must contain 3D array (T,H,W)")
    if frame_rate is None:
        raise ValueError("frame_rate must be provided for from_npy().")
    return Movie(arr, frame_rate=float(frame_rate))


def from_video(path: str, to_gray: bool = True, target_frame_rate: Optional[float] = None) -> Movie:
    """
    Load frames from a video file using imageio if available.
    Converts to grayscale by simple luminance average if requested.
    If target_frame_rate is given, it's recorded; resampling is not performed.
    """
    try:
        import imageio.v3 as iio
    except Exception as exc:  # pragma: no cover
        raise ImportError("imageio is required to load videos. Install imageio.") from exc

    frames = iio.imread(path)  # returns (T,H,W[,C])
    if frames.ndim == 4:
        # Convert to grayscale by average
        if to_gray:
            frames = frames.mean(axis=3)
        else:
            # if color, pick first channel to maintain (T,H,W)
            frames = frames[..., 0]
    elif frames.ndim != 3:
        raise ValueError("Unsupported video array shape.")

    # Normalize to [-1,1] range by simple scaling if needed
    frames = frames.astype(np.float32)
    if frames.max() > 1.0:
        frames = (frames - frames.min())
        denom = frames.max()
        if denom > 0:
            frames = frames / denom
        frames = 2.0 * frames - 1.0

    if target_frame_rate is None:
        # best-effort: try metadata
        fps = None
        try:
            meta = iio.immeta(path)
            fps = meta.get('fps', None)
        except Exception:
            fps = None
        if fps is None:
            raise ValueError("target_frame_rate not provided and fps metadata missing.")
        frame_rate = float(fps)
    else:
        frame_rate = float(target_frame_rate)

    return Movie(frames, frame_rate=frame_rate)

