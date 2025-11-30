"""
Compatibility utilities to run a BMTK FilterNet-style LGN config with brainlgn-x.

Supports a common subset of fields from BMTK's filternet JSON schema and
converts them into brainlgn_x.run.run_config() compatible configuration.

Notes
- Time units: BMTK uses ms for tstop/dt/gray_screen_dur, while brainlgn-x
  uses seconds for durations and Hz for frame_rate.
- Networks: full import of BMTK node files is not implemented here; instead
  provide population via brainlgn-x's `layout` + `cell_types` in the same JSON.
  If you want to run the exact BMTK network nodes, either:
    * use BMTK directly (run_filternet.py), or
    * extend this module to parse nodes.h5/node_types.csv and build filters.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os, csv, h5py
import numpy as np

def _parse_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def _parse_tuple2_str(s: str, default: Tuple[float, float]):
    try:
        parts = [p.strip() for p in str(s).split(',')]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return default

def _parse_int_tuple2_str(s: str, default: Tuple[int, int]):
    try:
        parts = [p.strip() for p in str(s).split(',')]
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        pass
    return default


def _ms_to_s(x):
    return float(x) / 1000.0


def convert_bmtk_to_brainlgn(cfg_bmtk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a BMTK-like filternet LGN JSON dict to brainlgn-x config dict.

    Required subset:
      - manifest.{BASE_DIR, OUTPUT_DIR}
      - run.{tstop (ms), dt (ms)}
      - inputs.LGN_spikes: {module: 'grating'|'flash', ...; frame_rate}
        with evaluation_options.{downsample, separable}

    Optional:
      - layout/cell_types: brainlgn-x population generator settings.

    Returns brainlgn-x config with keys: manifest/run/inputs/neuron(s) or
    layout+cell_types/output.
    """
    out: Dict[str, Any] = {}

    man = cfg_bmtk.get('manifest', {})
    out['manifest'] = {
        'BASE_DIR': man.get('$BASE_DIR', '.'),
        'OUTPUT_DIR': man.get('$OUTPUT_DIR', './results')
    }

    def _expand_vars(s: str) -> str:
        if not isinstance(s, str):
            return s
        res = s
        # Create a combined dictionary for expansion
        all_vars = {}

        # Add converted manifest variables
        for k, v in out['manifest'].items():
            all_vars[k] = v

        # Add original BMTK manifest variables (including $-prefixed keys)
        for k, v in man.items():
            if isinstance(v, str):
                key_clean = k.strip('$')
                all_vars[key_clean] = v
                all_vars[k] = v  # Also keep original key

        # Recursive expansion until no more changes
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            old_res = res
            for k, v in all_vars.items():
                if isinstance(v, str):
                    # Expand both ${KEY} and $KEY patterns
                    res = res.replace('${' + k + '}', v).replace('$' + k, v)
            # If no changes, we're done
            if res == old_res:
                break

        return res

    # Run: derive frame_rate (Hz) from dt (ms) and separable/downsample
    run = cfg_bmtk.get('run', {})
    dt_ms = float(run.get('dt', 1.0))
    frame_rate = 1000.0 / dt_ms
    out_run = {
        'frame_rate': frame_rate,
        'backend': run.get('backend', 'brainstate'),
        'separable': True,
        'downsample': 1,
        'base_seed': int(run.get('base_seed', 0)),
    }

    # Inputs: map module and parameters; prefer inputs' frame_rate if present
    inputs = cfg_bmtk.get('inputs', {})
    lgn = inputs.get('LGN_spikes', {})
    module = lgn.get('module', 'grating').lower()
    fr_in = float(lgn.get('frame_rate', frame_rate))
    out_run['frame_rate'] = fr_in
    out['run'] = out_run

    eval_opts = lgn.get('evaluation_options', {})
    if 'downsample' in eval_opts:
        out['run']['downsample'] = int(eval_opts.get('downsample', 1))
    if 'separable' in eval_opts:
        out['run']['separable'] = bool(eval_opts.get('separable', True))

    out_in = {'module': module, 'frame_rate': fr_in}
    # Common spatial sizes
    if 'row_size' in lgn:
        out_in['row_size'] = int(lgn['row_size'])
    if 'col_size' in lgn:
        out_in['col_size'] = int(lgn['col_size'])

    if module == 'grating':
        out_in.update({
            'duration': _ms_to_s(cfg_bmtk.get('run', {}).get('tstop', 1000.0)),
            'gray_screen_dur': _ms_to_s(lgn.get('gray_screen_dur', 0.0)),
            'cpd': float(lgn.get('cpd', 0.04)),
            'temporal_f': float(lgn.get('temporal_f', 4.0)),
            'theta': float(lgn.get('theta', 0.0)),
            'contrast': float(lgn.get('contrast', 1.0)),
            'phase': float(lgn.get('phase', 0.0)),
        })
    elif module == 'flash':
        # Map typical ON/OFF flash sequence durations
        out_in.update({
            'pre': _ms_to_s(lgn.get('pre', 100.0)),
            'on': _ms_to_s(lgn.get('on', 200.0)),
            'off': _ms_to_s(lgn.get('off', 200.0)),
            'post': _ms_to_s(lgn.get('post', 100.0)),
            'max_intensity': float(lgn.get('max_intensity', 1.0)),
        })
    else:
        # passthrough unsupported modules as-is
        out_in.update(lgn)

    out['inputs'] = out_in

    # Output mapping
    out_cfg = cfg_bmtk.get('output', {})
    out['output'] = {
        'output_dir': out_cfg.get('output_dir', '${OUTPUT_DIR}'),
        'spikes_csv': out_cfg.get('spikes_csv', 'spikes.csv'),
        'spikes_h5': out_cfg.get('spikes_h5', 'spikes.h5'),
        'rates_h5': out_cfg.get('rates_h5', 'rates.h5'),
        'overwrite_output_dir': bool(out_cfg.get('overwrite_output_dir', True)),
    }

    # Population: expect brainlgn-x layout/cell_types or neurons in the same JSON
    if 'layout' in cfg_bmtk:
        out['layout'] = cfg_bmtk['layout']
    if 'cell_types' in cfg_bmtk:
        out['cell_types'] = cfg_bmtk['cell_types']
    if 'neurons' in cfg_bmtk:
        out['neurons'] = cfg_bmtk['neurons']

    # SONATA network: if provided in BMTK-style 'networks', import H5/CSV to neurons
    nets = cfg_bmtk.get('networks', {})
    if isinstance(nets, dict) and 'nodes' in nets:
        nodes_list = nets.get('nodes', [])
        if nodes_list:
            nd = nodes_list[0]
            nodes_path = _expand_vars(nd.get('nodes_file'))
            types_path = _expand_vars(nd.get('node_types_file'))
            if nodes_path and types_path:
                try:
                    neurons = sonata_to_neuron_cfgs(nodes_path, types_path)
                    if neurons:
                        out['neurons'] = neurons
                        # If neurons are provided via SONATA, prefer them over layout/cell_types
                        out.pop('layout', None)
                        out.pop('cell_types', None)
                        # Optional: log successful import
                        import warnings
                        warnings.warn(f"SONATA import successful: Loaded {len(neurons)} neurons for multi-neuron simulation.")
                except Exception as e:
                    # Leave population unspecified; caller may still supply layout/cell_types
                    import warnings
                    warnings.warn(f"SONATA import failed: {e}. Using single neuron mode instead.")

    return out


def sonata_to_neuron_cfgs(nodes_h5_path: str, node_types_csv: str) -> List[Dict[str, Any]]:
    """
    Load SONATA-style LGN nodes and node_types into a list of brainlgn-x neuron cfgs.

    Expected minimal schema:
      - nodes H5: /nodes/lgn/node_id, node_type_id with subgroups containing x,y positions
      - node_types CSV: node_type_id plus optional columns mapping to neuron params:
        sigma_x,sigma_y | sigma; weights or w0,w1; kpeaks or k0,k1; delays or d0,d1;
        amplitude or on_off(+/-); bias.
    Unspecified fields fall back to reasonable defaults.
    """
    # Load node_types into dict by id
    types: Dict[str, Dict[str, str]] = {}
    with open(node_types_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(row.get('node_type_id') or row.get('node_type'))
            if key is None:
                continue
            types[key] = row

    # Load nodes - handle grouped SONATA format
    with h5py.File(nodes_h5_path, 'r') as f:
        # Find the main nodes group
        if '/nodes/lgn' in f:
            lgn_grp = f['/nodes/lgn']
        elif '/nodes/default' in f:
            lgn_grp = f['/nodes/default']
        else:
            lgn_grp = list(f['/nodes'].values())[0]

        # Get global node info
        node_ids = lgn_grp['node_id'][...]
        type_ids = lgn_grp['node_type_id'][...]
        node_group_ids = lgn_grp.get('node_group_id', None)

        # Collect positions from subgroups
        xs = []
        ys = []

        # Check if positions are directly in the main group
        if 'x' in lgn_grp and 'y' in lgn_grp:
            xs = lgn_grp['x'][...]
            ys = lgn_grp['y'][...]
        else:
            # Positions are in subgroups - reconstruct based on node_group_id
            if node_group_ids is not None:
                # Sort by node_id to maintain order
                sorted_indices = np.argsort(node_ids)
                xs = np.zeros_like(node_ids, dtype=float)
                ys = np.zeros_like(node_ids, dtype=float)

                # Map positions from subgroups
                current_idx = 0
                for grp_id in sorted(set(node_group_ids)):
                    subgrp_name = str(grp_id)
                    if subgrp_name in lgn_grp:
                        subgrp = lgn_grp[subgrp_name]
                        if 'x' in subgrp and 'y' in subgrp:
                            subgrp_xs = subgrp['x'][...]
                            subgrp_ys = subgrp['y'][...]
                            # Find nodes belonging to this group
                            mask = node_group_ids == grp_id
                            indices = np.where(mask)[0]
                            xs[indices] = subgrp_xs[:len(indices)]
                            ys[indices] = subgrp_ys[:len(indices)]
            else:
                # Fallback: concatenate all subgroup positions
                for subgrp_name in lgn_grp.keys():
                    if subgrp_name.isdigit():
                        subgrp = lgn_grp[subgrp_name]
                        if 'x' in subgrp and 'y' in subgrp:
                            xs.extend(subgrp['x'][...])
                            ys.extend(subgrp['y'][...])
                xs = np.array(xs)
                ys = np.array(ys)

    neurons: List[Dict[str, Any]] = []
    for nid, x, y, ntid in zip(node_ids, xs, ys, type_ids):
        tr = types.get(str(ntid), {})
        # Spatial
        sigma = (
            _parse_float(tr.get('sigma_x'), None),
            _parse_float(tr.get('sigma_y'), None),
        )
        if sigma[0] is None or sigma[1] is None:
            if tr.get('sigma'):
                sigma = _parse_tuple2_str(tr['sigma'], (2.0, 2.0))
            else:
                sigma = (2.0, 2.0)
        # Temporal
        if tr.get('weights'):
            weights = _parse_tuple2_str(tr['weights'], (0.6, -0.4))
        else:
            w0 = _parse_float(tr.get('w0'), 0.6)
            w1 = _parse_float(tr.get('w1'), -0.4)
            weights = (w0, w1)

        if tr.get('kpeaks'):
            kpeaks = _parse_tuple2_str(tr['kpeaks'], (15.0, 45.0))
        else:
            k0 = _parse_float(tr.get('k0'), 15.0)
            k1 = _parse_float(tr.get('k1'), 45.0)
            kpeaks = (k0, k1)

        if tr.get('delays'):
            delays = _parse_int_tuple2_str(tr['delays'], (0, 0))
        else:
            d0 = int(_parse_float(tr.get('d0'), 0))
            d1 = int(_parse_float(tr.get('d1'), 0))
            delays = (d0, d1)

        # Amplitude and bias
        amp = _parse_float(tr.get('amplitude'), None)
        if amp is None:
            onoff = str(tr.get('on_off') or tr.get('polarity') or '').lower()
            base = _parse_float(tr.get('amp_abs'), 1.5)
            if onoff.startswith('off'):
                amp = -abs(base)
            else:
                amp = abs(base)
        bias = _parse_float(tr.get('bias'), 1.0)

        ncfg = {
            'spatial': {'sigma': [float(sigma[0]), float(sigma[1])], 'translate': [float(x), float(y)]},
            'temporal': {'weights': [float(weights[0]), float(weights[1])], 'kpeaks': [float(kpeaks[0]), float(kpeaks[1])], 'delays': [int(delays[0]), int(delays[1])]},
            'amplitude': float(amp),
            'transfer': {'bias': float(bias)}
        }
        neurons.append(ncfg)

    return neurons
