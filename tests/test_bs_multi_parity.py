import os, sys
import numpy as np
import pytest

_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

try:
    import jax  # noqa: F401
    _HAVE_JAX = True
except Exception:
    _HAVE_JAX = False

from brainlgn_x.filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from brainlgn_x.transfer import ScalarTransferFunction
from brainlgn_x.bs_backend import eval_separable_multi
from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def _make_cell(translate, sigma=(2.0,2.0), weights=(0.5,-0.3), kpeaks=(20.0,60.0), delays=(0,0), amp=1.5, bias=0.3):
    spatial = GaussianSpatialFilter(translate=translate, sigma=sigma)
    temporal = TemporalFilterCosineBump(weights=weights, kpeaks=kpeaks, delays=delays)
    st = SpatioTemporalFilter(spatial, temporal, amplitude=amp)
    tr = ScalarTransferFunction(f"Max(0, s + {bias})")
    return st, tr


@pytest.mark.skipif(not _HAVE_JAX, reason="JAX not installed; skipping BS multi parity tests")
def test_bs_multi_parity_small():
    rng = np.random.RandomState(11)
    T, H, W = 200, 24, 24
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.05

    # 4 neurons with different translates
    cells = [
        _make_cell((0.0, 0.0)),
        _make_cell((5.0, 3.0)),
        _make_cell((10.0, 2.0)),
        _make_cell((7.0, 8.0)),
    ]
    lfs, trs = zip(*cells)

    # Reference: BMTK per-neuron, stack
    movie = Movie(stim, frame_rate=frame_rate)
    ref = []
    for lf, tr in cells:
        ln = LNUnit(lf, tr)
        t_ref, y_ref = ln.get_cursor(movie, separable=True).evaluate()
        ref.append(np.array(y_ref))
    ref = np.stack(ref, axis=0)  # (N,T)

    # BS multi
    bs_rates = eval_separable_multi(lfs, trs, stim, frame_rate=frame_rate, downsample=1)

    np.testing.assert_allclose(bs_rates, ref, rtol=0, atol=1e-7)
