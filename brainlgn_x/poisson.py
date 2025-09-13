"""
Poisson spike generation from rate (Hz) time series.

Implements a simple inhomogeneous Poisson process using per-bin Bernoulli
approximation: p(t) = 1 - exp(-rate(t)*dt), with at most one spike per bin.

Supports single neuron (T,) or multi-neuron (N,T) inputs. Returns flattened
gid/time arrays for convenience and IO writers.
"""

from typing import Tuple
import numpy as np


def _gen_seeds(n: int, base_seed: int) -> np.ndarray:
    rs = np.random.RandomState(base_seed)
    # Avoid collisions across processes (simple approach):
    return rs.randint(0, 2**31 - 1, size=n)


def generate_inhomogeneous_poisson(rates_hz: np.ndarray,
                                   dt: float,
                                   base_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Poisson spikes from rate time series.

    Args:
        rates_hz: shape (T,) or (N,T), firing rates in Hz.
        dt: timestep in seconds.
        base_seed: base random seed; per-neuron seeds derived from it.

    Returns:
        gids: 1D array of node ids per spike (int64)
        times: 1D array of spike times in seconds (float64)
    """
    rates = np.asarray(rates_hz)
    if rates.ndim == 1:
        rates = rates[None, :]

    n, t = rates.shape
    # Per-bin spike probability
    p = 1.0 - np.exp(-rates * float(dt))
    p = np.clip(p, 0.0, 1.0)

    gids_list = []
    times_list = []
    seeds = _gen_seeds(n, base_seed)

    for gid in range(n):
        rng = np.random.RandomState(int(seeds[gid]))
        mask = rng.rand(t) < p[gid]
        if np.any(mask):
            ti = np.nonzero(mask)[0]
            tt = (ti.astype(np.float64)) * float(dt)
            gids_list.append(np.full(ti.size, gid, dtype=np.int64))
            times_list.append(tt)

    if len(times_list) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    gids = np.concatenate(gids_list, axis=0)
    times = np.concatenate(times_list, axis=0)

    # Sort by time (stable)
    order = np.argsort(times, kind='mergesort')
    return gids[order], times[order]

