"""
Quick smoke test to verify BS/JAX and BMTK paths run end-to-end
and produce closely matched rates for a small LGN population.

Usage:
  python brainlgn-X/scripts/smoke_compare.py

It will:
  - Build a small drifting grating movie
  - Generate a tiny LGN population via generator (using BMTK filters/TFs)
  - Evaluate rates with BS/JAX multi path (separable)
  - Evaluate reference rates with BMTK LNUnit per neuron
  - Print MAE/MaxAbs and timing for both

Requirements:
  - bmtk installed
  - jax/jaxlib installed (for BS/JAX path); if missing, the test will skip BS
"""

from __future__ import annotations
import os
import time
import numpy as np

os.environ.setdefault('BRAINLGN_BACKEND', 'brainstate')
os.environ.setdefault('BRAINLGN_JAX_X64', '1')
os.environ.setdefault('BRAINLGN_FALLBACK_BMTK_ON_NAN', '1')
os.environ.setdefault('BRAINLGN_ALIGN_START', 'lmax')

try:
    import jax  # noqa: F401
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

from brainlgn_x.stimuli import drifting_grating
from brainlgn_x.generator import generate_population
from brainlgn_x.bs_backend import eval_separable_multi

from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def main():
    # Small stimulus
    row, col = 60, 120
    frame_rate = 1000.0
    movie = drifting_grating(row, col, frame_rate, duration=0.3, gray_screen=0.1,
                             cpd=0.04, temporal_f=4.0, theta=0.0, contrast=0.8)
    stim = movie.as_array()

    # Tiny population (6 neurons)
    layout = {"X_grids": 3, "Y_grids": 2, "X_len": 240.0, "Y_len": 120.0}
    cell_types = [
        {"name": "sON",  "n_per_tile": 1, "spatial": {"sigma_range": [2.0, 3.0]},
         "temporal": {"weights": [0.6, -0.4], "kpeaks": [15.0, 45.0], "delays": [0, 0], "jitter_percent": 0.0},
         "amplitude_range": [1.5, 2.0],  "transfer": {"bias_range": [0.8, 1.2]}},
    ]
    lfs, trs, meta = generate_population(cell_types, layout, base_seed=123)
    N = len(lfs)
    print(f"Population N={N}; stimulus shape={stim.shape}")

    # BS/JAX evaluation
    if HAVE_JAX:
        t0 = time.perf_counter()
        rates_bs = eval_separable_multi(lfs, trs, stim, frame_rate=movie.frame_rate, downsample=1)
        t1 = time.perf_counter()
        bs_time = t1 - t0
        print(f"BS/JAX time(s)={bs_time:.4f}; rates_bs shape={rates_bs.shape}; NaNs={np.isnan(rates_bs).sum()}")
    else:
        rates_bs = None
        print("JAX not installed; skipping BS/JAX evaluation.")

    # BMTK reference per neuron
    mv = Movie(stim, frame_rate=movie.frame_rate)
    ref = []
    t0 = time.perf_counter()
    for lf, tr in zip(lfs, trs):
        ln = LNUnit(lf, tr)
        _, y = ln.get_cursor(mv, separable=True).evaluate()
        ref.append(np.asarray(y))
    rates_ref = np.stack(ref, axis=0)
    t1 = time.perf_counter()
    bmtk_time = t1 - t0
    print(f"BMTK time(s)={bmtk_time:.4f}; rates_ref shape={rates_ref.shape}")

    # Compare
    if rates_bs is not None:
        K = min(rates_bs.shape[0], rates_ref.shape[0])
        rates_bs = rates_bs[:K]
        rates_ref = rates_ref[:K]
        mae = float(np.mean(np.abs(rates_bs - rates_ref)))
        mx = float(np.max(np.abs(rates_bs - rates_ref)))
        print(f"Parity: MAE={mae:.3e} MaxAbs={mx:.3e}; speedup={bmtk_time/max(bs_time,1e-12):.2f}x")
        # Basic thresholds (allow small FP diff)
        ok = (mae < 5e-7 and mx < 5e-6)
        return 0 if ok else 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

