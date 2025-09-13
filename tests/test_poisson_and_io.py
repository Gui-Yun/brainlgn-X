import os, sys, tempfile
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from brainlgn_x.poisson import generate_inhomogeneous_poisson
from brainlgn_x.io_output import write_spikes_csv, write_spikes_h5
import h5py


def test_poisson_mean_and_reproducibility():
    # Constant rate: 20 Hz, dt=1 ms, T=10s => expected ~200 spikes per neuron
    rate_hz = 20.0
    dt = 0.001
    T = 10_000  # 10 seconds
    rates = np.full((2, T), rate_hz, dtype=float)

    gids1, times1 = generate_inhomogeneous_poisson(rates, dt, base_seed=123)
    gids2, times2 = generate_inhomogeneous_poisson(rates, dt, base_seed=123)
    gids3, times3 = generate_inhomogeneous_poisson(rates, dt, base_seed=456)

    # Reproducibility for same seed
    assert np.array_equal(gids1, gids2)
    assert np.allclose(times1, times2)

    # Different seed should differ (probabilistic; very unlikely to be identical)
    assert not (np.array_equal(gids1, gids3) and np.allclose(times1, times3))

    # Mean rate check (allow 15% tolerance per neuron)
    duration = T * dt
    for gid in [0, 1]:
        n_spikes = np.sum(gids1 == gid)
        est_rate = n_spikes / duration
        assert abs(est_rate - rate_hz) / rate_hz < 0.15


def test_io_writers_roundtrip(tmp_path):
    gids = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    times = np.array([0.001, 0.005, 0.002, 0.010, 0.011], dtype=np.float64)

    csv_path = tmp_path / 'spikes.csv'
    h5_path = tmp_path / 'spikes.h5'

    write_spikes_csv(str(csv_path), gids, times)
    write_spikes_h5(str(h5_path), gids, times)

    # CSV readback
    data = np.loadtxt(str(csv_path), delimiter=',', skiprows=1)
    gids_csv = data[:, 0].astype(int)
    times_csv = data[:, 1]
    assert np.array_equal(gids_csv, gids)
    assert np.allclose(times_csv, times)

    # H5 readback
    with h5py.File(str(h5_path), 'r') as f:
        assert 'spikes' in f
        grp = f['spikes']
        gids_h5 = grp['node_ids'][...]
        times_h5 = grp['timestamps'][...]
    assert np.array_equal(gids_h5, gids)
    assert np.allclose(times_h5, times)

