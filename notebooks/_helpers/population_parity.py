"""
Population-level parity helper between BrainState/JAX backend and BMTK.

Usage (from a notebook cell):

    %env BRAINLGN_BACKEND=brainstate
    %env BRAINLGN_JAX_X64=1
    %env BRAINLGN_FALLBACK_BMTK_ON_NAN=1

    from _helpers.population_parity import run_population_parity

    layout = {"X_grids": 6, "Y_grids": 4, "X_len": 240.0, "Y_len": 120.0}
    cell_types = [
        {"name": "sON",  "n_per_tile": 2, "spatial": {"sigma_range": [2.0, 3.0]},
         "temporal": {"weights": [0.6, -0.4], "kpeaks": [15.0, 45.0], "delays": [0, 0], "jitter_percent": 0.05},
         "amplitude_range": [1.5, 2.0], "transfer": {"bias_range": [0.8, 1.2]}},
        {"name": "sOFF", "n_per_tile": 1, "spatial": {"sigma_range": [2.0, 3.0]},
         "temporal": {"weights": [0.5, -0.3], "kpeaks": [20.0, 60.0], "delays": [0, 0], "jitter_percent": 0.05},
         "amplitude_range": [-2.0, -1.5], "transfer": {"bias_range": [0.5, 1.0]}}
    ]

    run_population_parity(
        row_size=120, col_size=240, frame_rate=1000.0, duration=0.5, gray=0.2,
        layout_cfg=layout, cell_types_cfg=cell_types,
        subset_ref=128,  # compute BMTK reference on this many neurons for parity plots
        backend='brainstate', downsample=1, base_seed=123,
        save_dir='../notebooks/_outputs_population'  # optional, relative to this file
    )

This runs BS/JAX on the full population, and BMTK reference on a configurable
subset for parity comparison and plots. It prints parity metrics and shows
several visualizations: overlay per-neuron traces, residual histograms, scatter.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt

# Add package root if executed from notebooks directory
import sys
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PKG_ROOT = os.path.abspath(os.path.join(_ROOT, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from brainlgn_x.stimuli import drifting_grating
from brainlgn_x.generator import generate_population
from brainlgn_x.bs_backend import eval_separable_multi
from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def run_population_parity(row_size: int,
                          col_size: int,
                          frame_rate: float,
                          duration: float,
                          gray: float,
                          layout_cfg: Dict[str, Any],
                          cell_types_cfg: List[Dict[str, Any]],
                          subset_ref: int = 128,
                          backend: str = 'brainstate',
                          downsample: int = 1,
                          base_seed: int = 0,
                          save_dir: Optional[str] = None):
    # Stimulus
    movie = drifting_grating(row_size, col_size, frame_rate, duration, gray_screen=gray,
                             cpd=0.04, temporal_f=4.0, theta=0.0, contrast=0.8)
    stim = movie.as_array()

    # Population
    lfs, trs, meta = generate_population(cell_types_cfg, layout_cfg, base_seed=base_seed)
    N = len(lfs)
    print(f"Population N={N}; stimulus shape={stim.shape}; frame_rate={movie.frame_rate}")

    # BS/JAX full evaluation (with the backend already selected via env in bs_backend)
    rates_bs = eval_separable_multi(lfs, trs, stim, frame_rate=movie.frame_rate, downsample=downsample)
    print(f"BS rates shape: {rates_bs.shape}")

    # BMTK reference on subset
    K = min(int(subset_ref), N)
    mv = Movie(stim, frame_rate=movie.frame_rate)
    ref = []
    for i in range(K):
        ln = LNUnit(lfs[i], trs[i])
        _, y = ln.get_cursor(mv, separable=True).evaluate()
        ref.append(np.asarray(y))
    rates_ref = np.stack(ref, axis=0)
    if downsample and downsample > 1:
        rates_ref = rates_ref[:, :: int(downsample)]
    print(f"Ref(BMTK subset) shape: {rates_ref.shape}")

    # Parity metrics on subset
    rs = rates_bs[:K]
    mae = np.mean(np.abs(rs - rates_ref))
    mx = np.max(np.abs(rs - rates_ref))
    print("Parity subset: MAE=%.3e MaxAbs=%.3e" % (mae, mx))

    # Optional save
    if save_dir:
        _ensure_dir(save_dir)
        np.save(os.path.join(save_dir, 'rates_bs.npy'), rates_bs)
        np.save(os.path.join(save_dir, 'rates_ref_subset.npy'), rates_ref)
        print(f"Saved rates to {save_dir}")

    # Plots
    T = rs.shape[1]
    t = np.arange(T) / (movie.frame_rate / max(1, int(downsample)))

    # 1) Overlay few neurons
    n_show = min(6, K)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2*n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i].plot(t, rates_ref[i], label='BMTK', lw=1)
        axes[i].plot(t, rs[i], '--', label='BS', lw=1)
        axes[i].set_ylabel(f'n{i}')
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Overlay (K={K}) MAE={mae:.2e} MaxAbs={mx:.2e}')
    plt.show()

    # 2) Residual histogram
    resid = (rs - rates_ref).ravel()
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=100, alpha=0.8)
    plt.title('Residuals (BS - BMTK)')
    plt.xlabel('Error (Hz)'); plt.ylabel('Count')
    plt.show()

    # 3) Scatter
    plt.figure(figsize=(5,5))
    plt.scatter(rates_ref.ravel(), rs.ravel(), s=2, alpha=0.3)
    lim = max(rates_ref.max(), rs.max())
    plt.plot([0, lim], [0, lim], 'k--', lw=1)
    plt.xlabel('BMTK (Hz)'); plt.ylabel('BS (Hz)')
    plt.title('Rate scatter (subset)')
    plt.show()

    return {
        'N': N,
        'subset': K,
        'mae': float(mae),
        'max_abs': float(mx),
        'rates_bs': rates_bs,
        'rates_ref_subset': rates_ref,
        'meta': meta,
    }

