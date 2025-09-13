import os, sys
import numpy as np
import pytest

# Ensure package root is on path when running tests directly
_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

try:
    import jax  # noqa: F401
    _HAVE_JAX = True
except Exception:
    _HAVE_JAX = False

from brainlgn_x.neuron import LGNNeuron
from brainlgn_x.filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from brainlgn_x.transfer import ScalarTransferFunction

from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def _build_components(amplitude=1.0, bias=0.0):
    spatial = GaussianSpatialFilter(translate=(0.0, 0.0), sigma=(2.0, 2.0))
    tfilt = TemporalFilterCosineBump(weights=(0.4, -0.3), kpeaks=(20.0, 60.0), delays=(0, 0))
    st = SpatioTemporalFilter(spatial, tfilt, amplitude=float(amplitude))
    tr = ScalarTransferFunction(f"Max(0, s + {bias})")
    return st, tr


@pytest.mark.skipif(not _HAVE_JAX, reason="JAX not installed; skipping BS backend parity tests")
def test_bs_backend_parity_separable():
    rng = np.random.RandomState(7)
    T, H, W = 256, 16, 16
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.05

    st, tr = _build_components()

    # BMTK reference
    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate()
    y_ref = np.array(y_ref)

    # BrainState backend
    neuron = LGNNeuron(st.spatial_filter, st.temporal_filter, tr, amplitude=st.amplitude)
    y_bs = neuron.evaluate(stim, separable=True, frame_rate=frame_rate, backend='brainstate')

    # Allow tiny FP differences from JAX convolution
    np.testing.assert_allclose(y_bs, y_ref, rtol=0, atol=1e-9)


@pytest.mark.skipif(not _HAVE_JAX, reason="JAX not installed; skipping BS backend parity tests")
def test_bs_backend_parity_downsample():
    rng = np.random.RandomState(8)
    T, H, W = 300, 12, 12
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.1
    ds = 4

    st, tr = _build_components()

    # BMTK reference full -> slice
    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate()
    y_ref_ds = np.array(y_ref)[::ds]

    # BrainState backend with downsample
    neuron = LGNNeuron(st.spatial_filter, st.temporal_filter, tr, amplitude=st.amplitude)
    y_bs = neuron.evaluate(stim, separable=True, frame_rate=frame_rate, backend='brainstate', downsample=ds)

    np.testing.assert_allclose(y_bs, y_ref_ds, rtol=0, atol=5e-9)
