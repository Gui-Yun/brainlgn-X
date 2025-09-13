"""
Minimal JSON config parser for end-to-end pipeline.

Schema (subset):
- manifest: BASE_DIR, OUTPUT_DIR
- run: frame_rate, backend ('bmtk'|'brainstate'), separable (bool), downsample (int), base_seed (int)
- inputs: module ('grating'|'flash'|'npy'|'video'), fields depending on module
- neuron: spatial {sigma, translate}, temporal {weights, kpeaks, delays}, transfer {bias}, amplitude
- output: output_dir, spikes_csv, spikes_h5, rates_h5, overwrite_output_dir
"""

from __future__ import annotations
from typing import Any, Dict
import os, json


def _expand_vars(s: str, manifest: Dict[str, str]) -> str:
    out = s
    for k, v in manifest.items():
        out = out.replace('${' + k + '}', v).replace('$' + k, v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    base_dir = os.path.abspath(os.path.dirname(path))
    manifest = cfg.get('manifest', {})
    manifest.setdefault('BASE_DIR', base_dir)
    manifest.setdefault('OUTPUT_DIR', os.path.join(base_dir, 'results'))

    # Expand simple variables in cfg where string
    def expand(obj):
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand(v) for v in obj]
        elif isinstance(obj, str):
            return _expand_vars(obj, manifest)
        else:
            return obj

    cfg = expand(cfg)
    cfg['manifest'] = manifest
    return cfg

