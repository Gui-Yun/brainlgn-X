"""
CLI entry: python -m brainlgn_x.run path/to/config.json

Runs Stimulus -> Rates -> Poisson spikes -> Outputs (CSV/H5).
"""

from __future__ import annotations
import sys, os
from typing import Any, Dict

import numpy as np

from .config_parser import load_config
from .stimuli import drifting_grating, full_field_flash, from_npy, from_video, Movie
from .filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from .transfer import ScalarTransferFunction
from .neuron import LGNNeuron
from .poisson import generate_inhomogeneous_poisson
from .io_output import write_spikes_csv, write_spikes_h5, write_rates_h5


def _ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite:
            return
    else:
        os.makedirs(path, exist_ok=True)


def build_movie(cfg: Dict[str, Any]) -> Movie:
    inp = cfg['inputs']
    mod = inp.get('module', 'grating').lower()
    run = cfg.get('run', {})
    fr = float(run.get('frame_rate', inp.get('frame_rate', 1000.0)))

    if mod == 'grating':
        return drifting_grating(
            row_size=int(inp.get('row_size', 120)),
            col_size=int(inp.get('col_size', 240)),
            frame_rate=fr,
            duration=float(inp.get('duration', cfg.get('run', {}).get('tstop', 1.0) / 1000.0)),
            gray_screen=float(inp.get('gray_screen_dur', 0.0)),
            cpd=float(inp.get('cpd', 0.04)),
            temporal_f=float(inp.get('temporal_f', 4.0)),
            theta=float(inp.get('theta', 0.0)),
            contrast=float(inp.get('contrast', 1.0)),
            phase=float(inp.get('phase', 0.0)),
        )
    elif mod == 'flash':
        return full_field_flash(
            row_size=int(inp.get('row_size', 120)),
            col_size=int(inp.get('col_size', 240)),
            frame_rate=fr,
            pre=float(inp.get('pre', 0.2)),
            on=float(inp.get('on', 0.2)),
            off=float(inp.get('off', 0.2)),
            post=float(inp.get('post', 0.2)),
            max_intensity=float(inp.get('max_intensity', 1.0)),
        )
    elif mod == 'npy':
        return from_npy(inp['path'], frame_rate=fr)
    elif mod == 'video':
        return from_video(inp['path'], to_gray=bool(inp.get('to_gray', True)), target_frame_rate=fr)
    else:
        raise ValueError(f"Unknown input module: {mod}")


def build_neuron(cfg: Dict[str, Any], movie: Movie) -> LGNNeuron:
    ncfg = cfg.get('neuron', {})
    # Spatial
    sdict = ncfg.get('spatial', {})
    sigma = sdict.get('sigma', (2.0, 2.0))
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))
    translate = sdict.get('translate', (0.0, 0.0))
    spatial = GaussianSpatialFilter(translate=tuple(translate), sigma=tuple(sigma))

    # Temporal (defaults matching our tests)
    tdict = ncfg.get('temporal', {})
    weights = tuple(tdict.get('weights', (0.4, -0.3)))
    kpeaks = tuple(tdict.get('kpeaks', (20.0, 60.0)))
    delays = tuple(tdict.get('delays', (0, 0)))
    temporal = TemporalFilterCosineBump(weights=weights, kpeaks=kpeaks, delays=delays)

    amplitude = float(ncfg.get('amplitude', 1.0))
    st = SpatioTemporalFilter(spatial, temporal, amplitude=amplitude)

    # Transfer: ReLU with bias
    bias = float(ncfg.get('transfer', {}).get('bias', 0.0))
    transfer = ScalarTransferFunction(f"Max(0, s + {bias})")

    return LGNNeuron(spatial_filter=spatial, temporal_filter=temporal, transfer_function=transfer, amplitude=amplitude)


def run_config(cfg: Dict[str, Any]) -> None:
    # Output setup
    out = cfg.get('output', {})
    out_dir = out.get('output_dir', cfg['manifest']['OUTPUT_DIR'])
    overwrite = bool(out.get('overwrite_output_dir', True))
    _ensure_dir(out_dir, overwrite)

    # Build movie
    movie = build_movie(cfg)

    # Build neuron (MVP: single)
    neuron = build_neuron(cfg, movie)

    # Evaluate rates
    backend = cfg.get('run', {}).get('backend', os.getenv('BRAINLGN_BACKEND', 'bmtk'))
    separable = bool(cfg.get('run', {}).get('separable', True))
    downsample = int(cfg.get('run', {}).get('downsample', 1))
    rates = neuron.evaluate(movie.data, separable=separable, frame_rate=movie.frame_rate, backend=backend, downsample=downsample)

    # Write rates (optional)
    if out.get('rates_h5'):
        write_rates_h5(os.path.join(out_dir, out['rates_h5']), rates, frame_rate=movie.frame_rate / downsample)

    # Poisson spikes
    dt = 1.0 / (movie.frame_rate / downsample)
    base_seed = int(cfg.get('run', {}).get('base_seed', 0))
    gids, times = generate_inhomogeneous_poisson(rates, dt=dt, base_seed=base_seed)

    # Write spikes
    if out.get('spikes_csv'):
        write_spikes_csv(os.path.join(out_dir, out['spikes_csv']), gids, times)
    if out.get('spikes_h5'):
        write_spikes_h5(os.path.join(out_dir, out['spikes_h5']), gids, times)

    print(f"Wrote outputs to: {out_dir}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: python -m brainlgn_x.run path/to/config.json")
        return 1
    cfg = load_config(argv[0])
    run_config(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

