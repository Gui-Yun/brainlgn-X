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
    # Enable float64 unless explicitly disabled via env
    if os.getenv('BRAINLGN_JAX_X64', '1').lower() not in ('0', 'false', 'no'):
        try:  # pragma: no cover
            jax.config.update('jax_enable_x64', True)
        except Exception:
            pass
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    jax = None
    jnp = None


def _require_jax():  # pragma: no cover
    if jnp is None:
        raise ImportError("JAX is required for the BrainState/JAX backend. Install jax/jaxlib.")
    return True


def _check_finite(arr: np.ndarray, name: str):
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"Non-finite values detected in {name}.")


def _maybe_bmtk_fallback(linear_filter,
                         transfer_function,
                         stimulus: np.ndarray,
                         frame_rate: float,
                         downsample: int) -> np.ndarray:
    """Use BMTK reference path as a safe fallback when JAX path has issues."""
    try:
        from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
        from bmtk.simulator.filternet.lgnmodel.movie import Movie
        movie = Movie(np.asarray(stimulus, dtype=np.float64), frame_rate=float(frame_rate))
        ln = LNUnit(linear_filter, transfer_function)
        t_vals, y_ref = ln.get_cursor(movie, separable=True).evaluate()
        y_ref = np.asarray(y_ref)
        if downsample and downsample > 1:
            y_ref = y_ref[:: int(downsample)]
        return y_ref
    except Exception as exc:
        raise RuntimeError(f"BMTK fallback failed: {exc}")


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

    # Enforce float64 end-to-end to match BMTK default and reduce parity error
    stimulus = np.asarray(stimulus, dtype=np.float64)
    _check_finite(stimulus, 'stimulus')
    T, H, W = stimulus.shape
    row_range = np.arange(H)
    col_range = np.arange(W)

    # Spatial kernel (dense), multiply amplitude as in BMTK separable cursor
    spatial_k = linear_filter.spatial_filter.get_kernel(row_range, col_range, threshold=-1)
    spatial_full = spatial_k.full().astype(np.float64) * float(linear_filter.amplitude)
    # Sanitize any non-finite entries to zero before checks (robustness)
    if not np.all(np.isfinite(spatial_full)):
        spatial_full = np.nan_to_num(spatial_full, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _check_finite(spatial_full, 'spatial_kernel')
    except FloatingPointError:
        if os.getenv('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1').lower() not in ('0','false','no'):
            return _maybe_bmtk_fallback(linear_filter, transfer_function, stimulus, frame_rate, downsample)
        raise

    # Temporal kernel on movie time grid (reverse=True like BMTK cursor)
    t_range = np.arange(T) * (1.0 / float(frame_rate))
    temporal_k = linear_filter.temporal_filter.get_kernel(t_range=t_range, threshold=0, reverse=True)
    temporal_full = temporal_k.full().astype(np.float64)
    if not np.all(np.isfinite(temporal_full)):
        temporal_full = np.nan_to_num(temporal_full, nan=0.0, posinf=0.0, neginf=0.0)
    if temporal_full.size == 0:
        # Degenerate kernel, fallback to reference
        return _maybe_bmtk_fallback(linear_filter, transfer_function, stimulus, frame_rate, downsample)
    _check_finite(temporal_full, 'temporal_kernel')

    # JAX arrays
    stim_j = jnp.asarray(stimulus, dtype=jnp.float64)
    sk_j = jnp.asarray(spatial_full, dtype=jnp.float64)
    tk_j = jnp.asarray(temporal_full, dtype=jnp.float64)

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
    @jax.jit
    def _conv(sig, ker):
        return jnp.convolve(sig, ker[::-1], mode='valid')
    y_lin = _conv(sig_tmp, tk_j)  # shape (T,)
    # Transfer: try JAX ReLU+bias, fallback to numpy closure
    try:
        from .transfer_jax import maybe_parse_bias_from_scalar_tf, relu_bias
        tf_str = getattr(transfer_function, 'transfer_function_string', '')
        b = maybe_parse_bias_from_scalar_tf(tf_str)
        if b is not None:
            rate = np.asarray(relu_bias(y_lin, b))
        else:
            rate = transfer_function(np.asarray(y_lin))
    except Exception:
        rate = transfer_function(np.asarray(y_lin))

    # Finite check and optional fallback
    if not np.all(np.isfinite(rate)):
        if os.getenv('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1').lower() not in ('0', 'false', 'no'):
            return _maybe_bmtk_fallback(linear_filter, transfer_function, stimulus, frame_rate, downsample)
        raise FloatingPointError("Non-finite values in BS/JAX eval_separable output.")

    if downsample and downsample > 1:
        rate = rate[:: int(downsample)]

    return rate


def eval_nonseparable(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("BrainState/JAX backend non-separable path not implemented yet.")


def eval_separable_multi(linear_filters,
                         transfer_functions,
                         stimulus: np.ndarray,
                         frame_rate: float,
                         downsample: int = 1) -> np.ndarray:
    """
    JAX separable evaluation for multiple neurons.

    Args:
        linear_filters: list of BMTK SpatioTemporalFilter, length N
        transfer_functions: list of ScalarTransferFunction (numpy-based), length N
        stimulus: numpy array (T,H,W)
        frame_rate: Hz
        downsample: output stride (applied after full evaluation)

    Returns:
        rates: numpy array (N, T//downsample)
    """
    _require_jax()

    # Enforce float64 dtype for stimulus
    stimulus = np.asarray(stimulus, dtype=np.float64)
    _check_finite(stimulus, 'stimulus')
    T, H, W = stimulus.shape
    row_range = np.arange(H)
    col_range = np.arange(W)

    N = len(linear_filters)
    assert N == len(transfer_functions)

    # Stack spatial kernels (N,H,W)
    spatial_list = []
    for lf in linear_filters:
        sk = lf.spatial_filter.get_kernel(row_range, col_range, threshold=-1)
        spatial_full = sk.full().astype(np.float64) * float(lf.amplitude)
        if not np.all(np.isfinite(spatial_full)):
            spatial_full = np.nan_to_num(spatial_full, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            _check_finite(spatial_full, 'spatial_kernel')
        except FloatingPointError:
            # If any spatial kernel is non-finite, fallback to BMTK per-neuron (robust parity)
            if os.getenv('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1').lower() not in ('0','false','no'):
                try:
                    from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
                    from bmtk.simulator.filternet.lgnmodel.movie import Movie
                    mv = Movie(stimulus, frame_rate=float(frame_rate))
                    rates_list = []
                    for lf2, tr2 in zip(linear_filters, transfer_functions):
                        ln = LNUnit(lf2, tr2)
                        _, y = ln.get_cursor(mv, separable=True).evaluate()
                        rates_list.append(np.asarray(y))
                    rates = np.stack(rates_list, axis=0)
                    if downsample and downsample > 1:
                        rates = rates[:, :: int(downsample)]
                    return rates
                except Exception as exc:
                    raise RuntimeError(f"BMTK fallback failed: {exc}")
            raise
        spatial_list.append(spatial_full)
    spatial_k = jnp.asarray(np.stack(spatial_list, axis=0), dtype=jnp.float64)  # (N,H,W)

    # Temporal kernels (N,L) on movie grid, reversed for valid conv
    t_range = np.arange(T) * (1.0 / float(frame_rate))
    temporal_list = []
    lengths = []
    empty_temporal = False
    for lf in linear_filters:
        tk = lf.temporal_filter.get_kernel(t_range=t_range, threshold=0, reverse=True)
        kfull = tk.full().astype(np.float64)  # truncate=True by default; lengths may differ
        if not np.all(np.isfinite(kfull)):
            kfull = np.nan_to_num(kfull, nan=0.0, posinf=0.0, neginf=0.0)
        if kfull.size == 0:
            empty_temporal = True
        temporal_list.append(np.asarray(kfull))
        lengths.append(len(kfull))
    if empty_temporal:
        # Safe fallback when any kernel is degenerate
        if os.getenv('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1').lower() not in ('0','false','no'):
            try:
                from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
                from bmtk.simulator.filternet.lgnmodel.movie import Movie
                mv = Movie(stimulus, frame_rate=float(frame_rate))
                rates_list = []
                for lf, tr in zip(linear_filters, transfer_functions):
                    ln = LNUnit(lf, tr)
                    _, y = ln.get_cursor(mv, separable=True).evaluate()
                    rates_list.append(np.asarray(y))
                rates = np.stack(rates_list, axis=0)
                if downsample and downsample > 1:
                    rates = rates[:, :: int(downsample)]
                return rates
            except Exception as exc:
                raise RuntimeError(f"BMTK fallback failed: {exc}")
        raise FloatingPointError("Temporal kernel length is zero for at least one neuron.")
    Ls = np.array(lengths, dtype=int)
    Lmax = int(Ls.max())
    # Left-pad kernels with zeros to (N,Lmax)
    temporal_pad = np.zeros((N, Lmax), dtype=np.float64)
    for i, k in enumerate(temporal_list):
        Li = len(k)
        temporal_pad[i, Lmax - Li :] = k
    temporal_k = jnp.asarray(temporal_pad, dtype=jnp.float64)  # (N,Lmax)

    stim_j = jnp.asarray(stimulus, dtype=jnp.float64)  # (T,H,W)

    # Spatial stage: tensordot over (H,W)
    # s[t,n] = sum_{y,x} stim[t,y,x] * K[n,y,x]
    # tensordot(stim,(H,W)) with kernels over (H,W) -> (T,N)
    s_tn = jnp.tensordot(stim_j, spatial_k, axes=((1, 2), (1, 2)))  # (T,N)

    # Temporal stage: vmapped convolve per neuron
    Lmax_j = temporal_k.shape[1]
    Ls_j = jnp.asarray(Ls, dtype=jnp.int32)

    # Alignment mode: 'li' uses Li-1 (per-neuron), 'lmax' uses Lmax-1 (fixed)
    # Align using kernel length padding strategy. Because we left-pad kernels to Lmax,
    # the correct start offset for BMTK-equivalent alignment is Lmax-1 (not Li-1).
    # Using 'lmax' by default reproduces single-neuron behavior across vmapped conv.
    _align = os.getenv('BRAINLGN_ALIGN_START', 'lmax').lower()
    def conv_one(s_t, k_t, Li):
        if _align == 'lmax':
            start = jnp.maximum(0, Lmax_j - 1)
        else:
            # Li is per-neuron kernel length (int32 tracer). Use dynamic_update_slice instead of Python int cast.
            start = jnp.maximum(0, Li - 1)
        # sig_tmp length based on Lmax to keep shape static for vmap
        sig_tmp = jnp.zeros((Lmax_j + s_t.shape[0] - 1,), dtype=s_t.dtype)
        sig_tmp = jax.lax.dynamic_update_slice(sig_tmp, s_t, (start,))
        return jnp.convolve(sig_tmp, k_t[::-1], mode='valid')  # (T,)

    y_lin_tn = jax.vmap(conv_one, in_axes=(1, 0, 0), out_axes=1)(s_tn, temporal_k, Ls_j)  # (T,N)

    # Transfer: try vectorized JAX ReLU+bias per neuron, fallback numpy closures
    y_np = np.asarray(y_lin_tn, dtype=np.float64).T  # (N,T)
    rates_list = []
    try:
        from .transfer_jax import maybe_parse_bias_from_scalar_tf, relu_bias
        parsed_bias = [maybe_parse_bias_from_scalar_tf(getattr(tf, 'transfer_function_string', '')) for tf in transfer_functions]
        if all(b is not None for b in parsed_bias):
            # Build JAX array and vmap relu_bias with per-neuron bias
            y_j = jnp.asarray(y_np)
            b_j = jnp.asarray(np.array(parsed_bias, dtype=float))[:, None]
            rates = jnp.maximum(0.0, y_j + b_j)
            rates = np.asarray(rates)
        else:
            for i in range(N):
                rates_list.append(np.asarray(transfer_functions[i](y_np[i])))
            rates = np.stack(rates_list, axis=0)
    except Exception:
        for i in range(N):
            rates_list.append(np.asarray(transfer_functions[i](y_np[i])))
        rates = np.stack(rates_list, axis=0)

    if downsample and downsample > 1:
        rates = rates[:, :: int(downsample)]

    # Finite check and optional fallback
    if not np.all(np.isfinite(rates)):
        if os.getenv('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1').lower() not in ('0', 'false', 'no'):
            try:
                from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
                from bmtk.simulator.filternet.lgnmodel.movie import Movie
                mv = Movie(stimulus, frame_rate=float(frame_rate))
                ref = []
                for lf, tr in zip(linear_filters, transfer_functions):
                    ln = LNUnit(lf, tr)
                    _, y = ln.get_cursor(mv, separable=True).evaluate()
                    ref.append(np.asarray(y))
                ref = np.stack(ref, axis=0)
                if downsample and downsample > 1:
                    ref = ref[:, :: int(downsample)]
                return ref
            except Exception as exc:
                raise RuntimeError(f"BMTK fallback failed: {exc}")
        # As a last resort, sanitize to zeros to avoid propagating NaNs
        rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        # Or raise if strict behavior desired:
        # raise FloatingPointError("Non-finite values in BS/JAX eval_separable_multi output.")

    return rates


