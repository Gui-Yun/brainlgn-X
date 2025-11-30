"""
LGN population generator (Phase A, minimal).

Generates a population of LGN filters (SpatioTemporalFilter) and transfer
functions based on a simple schema:

layout:
  X_grids, Y_grids, X_len, Y_len (visual plane units)

cell_types: list of types, each:
  name: string
  n_per_tile: int
  spatial: { sigma_range: [min,max] }
  temporal: { weights: [w0,w1], kpeaks: [k0,k1], delays: [d0,d1], jitter_percent: 0.05 }
  amplitude_range: [min,max]
  transfer: { bias_range: [min,max] }

Returns lists of (SpatioTemporalFilter, ScalarTransferFunction) and a metadata
dict with gids, positions, and types.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from .filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from .transfer import ScalarTransferFunction


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def _sample_uniform(rng, low: float, high: float, size=None):
    return rng.uniform(low, high, size=size)


def _jitter_tuple(base: Tuple[float, float], jitter_percent: float, rng) -> Tuple[float, float]:
    jp = float(jitter_percent)
    low = 1.0 - jp
    high = 1.0 + jp
    return tuple(float(b * _sample_uniform(rng, low, high)) for b in base)


def generate_population(cell_types_cfg: List[Dict[str, Any]],
                        layout_cfg: Dict[str, Any],
                        base_seed: int = 0):
    """
    Generate LGN population (filters + transfers) and metadata.

    Args:
        cell_types_cfg: list of cell type dicts (see module docstring)
        layout_cfg: dict with X_grids, Y_grids, X_len, Y_len
        base_seed: int

    Returns:
        lfs: list of SpatioTemporalFilter
        trs: list of ScalarTransferFunction
        meta: dict with 'gids', 'positions', 'types'
    """
    Xg = int(layout_cfg.get('X_grids', 10))
    Yg = int(layout_cfg.get('Y_grids', 10))
    X_len = float(layout_cfg.get('X_len', 240.0))
    Y_len = float(layout_cfg.get('Y_len', 120.0))

    tile_w = X_len / Xg
    tile_h = Y_len / Yg

    rs = _rng(base_seed)

    lfs: List[SpatioTemporalFilter] = []
    trs: List[ScalarTransferFunction] = []
    gids: List[int] = []
    positions: List[Tuple[float, float]] = []
    types: List[str] = []

    gid = 0
    for i in range(Xg):
        for j in range(Yg):
            x0, x1 = i * tile_w, (i + 1) * tile_w
            y0, y1 = j * tile_h, (j + 1) * tile_h
            for ct in cell_types_cfg:
                name = ct.get('name', 'LGN')
                npt = int(ct.get('n_per_tile', 1))
                # Spatial sigma
                s_cfg = ct.get('spatial', {})
                sig_range = s_cfg.get('sigma_range', [2.0, 2.0])
                if isinstance(sig_range, (int, float)):
                    sig_range = [float(sig_range), float(sig_range)]
                # Temporal
                t_cfg = ct.get('temporal', {})
                weights_base = tuple(t_cfg.get('weights', (0.4, -0.3)))
                kpeaks_base = tuple(t_cfg.get('kpeaks', (20.0, 60.0)))
                delays_base = tuple(t_cfg.get('delays', (0, 0)))
                jitter = float(t_cfg.get('jitter_percent', 0.05))
                # Amplitude & bias
                amp_min, amp_max = ct.get('amplitude_range', [1.0, 1.0])
                b_min, b_max = ct.get('transfer', {}).get('bias_range', [0.0, 0.0])

                for _ in range(npt):
                    # Position uniform in tile
                    tx = float(_sample_uniform(rs, x0, x1))
                    ty = float(_sample_uniform(rs, y0, y1))
                    # Sigma triangular between range endpoints
                    sig_min, sig_max = float(sig_range[0]), float(sig_range[1])
                    if sig_max == sig_min:
                        sig = (sig_min, sig_min)
                    else:
                        s_val = float(rs.triangular(sig_min, (sig_min + sig_max) * 0.5, sig_max))
                        sig = (s_val, s_val)
                    spatial = GaussianSpatialFilter(translate=(tx, ty), sigma=sig)
                    # Temporal with jitter
                    w = _jitter_tuple(weights_base, jitter, rs)
                    kp = _jitter_tuple(kpeaks_base, jitter, rs)
                    dl = delays_base  # keep delays fixed for now
                    temporal = TemporalFilterCosineBump(weights=w, kpeaks=kp, delays=dl)
                    # Amplitude & bias
                    amp = float(_sample_uniform(rs, amp_min, amp_max))
                    bias = float(_sample_uniform(rs, b_min, b_max))
                    lf = SpatioTemporalFilter(spatial, temporal, amplitude=amp)
                    tr = ScalarTransferFunction(f"Max(0, s + {bias})")

                    lfs.append(lf)
                    trs.append(tr)
                    gids.append(gid)
                    positions.append((tx, ty))
                    types.append(name)
                    gid += 1

    meta = {'gids': np.array(gids, dtype=np.int64),
            'positions': np.array(positions, dtype=np.float32),
            'types': np.array(types)}
    return lfs, trs, meta

