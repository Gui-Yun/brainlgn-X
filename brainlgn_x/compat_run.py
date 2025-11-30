"""
CLI entry to run a BMTK FilterNet-style LGN config using brainlgn-x.

Usage:
  python -m brainlgn_x.compat_run path/to/bmtk_filternet.json

This converts the BMTK-like JSON to brainlgn-x config, then calls run_config().
Provide `layout`+`cell_types` or `neurons` in the same JSON for population.
"""

from __future__ import annotations
import sys
from .bmtk_compat import convert_bmtk_to_brainlgn
from .config_parser import load_config
from .run import run_config


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: python -m brainlgn_x.compat_run simulation_config.json [circuit_config.json]")
        return 1
    # Load raw JSON, then convert
    import json, os
    sim_path = argv[0]
    with open(sim_path, 'r', encoding='utf-8') as f:
        cfg_bmtk = json.load(f)

    # If a circuit config is provided, merge its networks/components/manifest into sim config
    if len(argv) > 1:
        circ_path = argv[1]
        try:
            with open(circ_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Fix Windows path separators for JSON compatibility
                content = content.replace('\\', '/')
                circ = json.loads(content)
            if 'networks' in circ and 'networks' not in cfg_bmtk:
                cfg_bmtk['networks'] = circ['networks']
            if 'components' in circ and 'components' not in cfg_bmtk:
                cfg_bmtk['components'] = circ['components']
            # Merge manifest: circuit config paths take precedence for absolute paths
            if 'manifest' in circ:
                cfg_bmtk.setdefault('manifest', {})
                for k, v in circ['manifest'].items():
                    # If circuit config has absolute paths, prefer them
                    if isinstance(v, str) and (v.startswith('/') or (len(v) > 1 and v[1] == ':')):
                        cfg_bmtk['manifest'][k] = v
                    elif k not in cfg_bmtk['manifest']:
                        cfg_bmtk['manifest'][k] = v
        except Exception:
            pass
    cfg = convert_bmtk_to_brainlgn(cfg_bmtk)

    # Save to a temp-expanded JSON (optional)
    tmp_dir = os.path.dirname(os.path.abspath(sim_path))
    tmp_path = os.path.join(tmp_dir, '_compat_expanded.json')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    # Load with variable expansion and run
    cfg_loaded = load_config(tmp_path)
    run_config(cfg_loaded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
