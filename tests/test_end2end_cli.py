import os, sys, json

_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def make_config(tmpdir):
    base = str(tmpdir)
    cfg = {
        "manifest": {
            "$BASE_DIR": base,
            "$OUTPUT_DIR": f"{base}/results"
        },
        "run": {
            "frame_rate": 200.0,
            "backend": "bmtk",
            "separable": True,
            "downsample": 1,
            "base_seed": 42
        },
        "inputs": {
            "module": "grating",
            "row_size": 32,
            "col_size": 32,
            "duration": 0.2,
            "gray_screen_dur": 0.1,
            "cpd": 0.05,
            "temporal_f": 4.0,
            "theta": 0.0,
            "contrast": 1.0
        },
        "neuron": {
            "spatial": {"sigma": [2.0, 2.0], "translate": [0.0, 0.0]},
            "temporal": {"weights": [0.6, -0.4], "kpeaks": [15.0, 45.0], "delays": [0, 0]},
            "amplitude": 2.0,
            "transfer": {"bias": 1.0}
        },
        "output": {
            "output_dir": "${OUTPUT_DIR}",
            "spikes_csv": "spikes.csv",
            "spikes_h5": "spikes.h5",
            "rates_h5": "rates.h5",
            "overwrite_output_dir": True
        }
    }
    cfg_path = os.path.join(base, 'config.json')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    return cfg_path, cfg


def test_end2end_minimal(tmp_path):
    cfg_path, cfg = make_config(tmp_path)
    from brainlgn_x.config_parser import load_config
    from brainlgn_x.run import run_config
    cfg_loaded = load_config(cfg_path)
    run_config(cfg_loaded)
    out_dir = cfg_loaded['output']['output_dir']
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, 'spikes.csv'))
    assert os.path.isfile(os.path.join(out_dir, 'spikes.h5'))
    assert os.path.isfile(os.path.join(out_dir, 'rates.h5'))
