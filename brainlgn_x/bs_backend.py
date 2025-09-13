"""
BrainState/JAX backend for LGN separable pipeline.

Implements a JAX-based evaluation path that mirrors BMTK's separable
cursor semantics: spatial dot per-frame, followed by temporal convolution
with the same padding/valid convention, then transfer function.

Notes
- Uses BMTK filter objects (spatial/temporal) to build kernels to ensure
  identical parameterization and grids.
- Requires jax and jax.numpy. Does not depend on brainstate APIs directly.
"""

from typing import Optional
import os
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    jax = None
    jnp = None


def _require_jax():  # pragma: no cover
    if jnp is None:
        raise ImportError("JAX is required for the BrainState/JAX backend. Install jax/jaxlib.")


def eval_separable(linear_filter,
                   transfer_function,
                   stimulus: np.ndarray,
                   frame_rate: float,
                   downsample: int = 1) -> np.ndarray:
    """
    JAX separable evaluation matching BMTK's SeparableLNUnitCursor.evaluate().

    Args:
        linear_filter: BMTK SpatioTemporalFilter instance (contains spatial/temporal filters and amplitude)
        transfer_function: BMTK ScalarTransferFunction (callable: f(s))
        stimulus: numpy array (T, H, W)
        frame_rate: sampling frequency (Hz)
        downsample: output stride (>=1); applied after full evaluation

    Returns:
        rate: numpy array (T//downsample,)
    """
    _require_jax()

    T, H, W = stimulus.shape
    row_range = np.arange(H)
    col_range = np.arange(W)

    # Spatial kernel (dense), multiply amplitude as in BMTK separable cursor
    spatial_k = linear_filter.spatial_filter.get_kernel(row_range, col_range, threshold=-1)
    spatial_full = spatial_k.full() * float(linear_filter.amplitude)

    # Temporal kernel on movie time grid (reverse=True like BMTK cursor)
    t_range = np.arange(T) * (1.0 / float(frame_rate))
    temporal_k = linear_filter.temporal_filter.get_kernel(t_range=t_range, threshold=0, reverse=True)
    temporal_full = temporal_k.full()

    # JAX arrays
    stim_j = jnp.asarray(stimulus)
    sk_j = jnp.asarray(spatial_full)
    tk_j = jnp.asarray(temporal_full)

    # Spatial stage: per-frame dot of (H,W) with kernel (H,W)
    # result s[t] = sum_{y,x} stimulus[t,y,x] * sk[y,x]
    # Vectorized: (T,H,W) * (H,W) -> (T,H,W) -> sum over (H,W)
    s_t = jnp.sum(stim_j * sk_j, axis=(1, 2))  # shape (T,)

    # Temporal stage: replicate BMTK convention
    # sig_tmp = zeros(len(tk)+len(s_t)-1); sig_tmp[len(tk)-1:] = s_t
    # y = convolve(sig_tmp, tk[::-1], mode='valid')
    Lk = tk_j.shape[0]
    sig_tmp = jnp.zeros((Lk + s_t.shape[0] - 1,), dtype=s_t.dtype)
    sig_tmp = sig_tmp.at[Lk - 1 :].set(s_t)
    y_lin = jnp.convolve(sig_tmp, tk_j[::-1], mode='valid')  # shape (T,)

    # Transfer (use numpy callable; convert to np for compatibility)
    y_np = np.asarray(y_lin)
    rate = transfer_function(y_np)

    if downsample and downsample > 1:
        rate = rate[:: int(downsample)]

    return rate


def eval_nonseparable(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("BrainState/JAX backend non-separable path not implemented yet.")

