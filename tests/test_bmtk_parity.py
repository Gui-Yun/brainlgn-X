import os, sys
import numpy as np

# Ensure package root is on path when running tests directly
_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from brainlgn_x.neuron import LGNNeuron
from brainlgn_x.filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from brainlgn_x.transfer import ScalarTransferFunction

from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def build_common_components():
    # Spatial: centered Gaussian, sigma in pixels
    spatial = GaussianSpatialFilter(translate=(0.0, 0.0), sigma=(2.0, 2.0))
    # Temporal: cosine bump (weights, kpeaks, delays)
    tfilt = TemporalFilterCosineBump(weights=(0.4, -0.3), kpeaks=(20.0, 60.0), delays=(0, 0))
    # Linear filter amplitude (ON)
    st = SpatioTemporalFilter(spatial, tfilt, amplitude=1.0)
    # Transfer: ReLU (avoid Heaviside two-arg signature issue in sympy->numpy mapping)
    tr = ScalarTransferFunction('Max(0, s)')
    return st, tr


def test_parity_separable_small_random():
    rng = np.random.RandomState(0)
    T, H, W = 200, 16, 16
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.1

    st, tr = build_common_components()

    # BMTK reference
    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate(downsample=1)
    y_ref = np.array(y_ref)

    # Our neuron
    neuron = LGNNeuron(spatial_filter=st.spatial_filter, temporal_filter=st.temporal_filter,
                       transfer_function=tr, amplitude=st.amplitude)
    y_new = neuron.evaluate(stim, separable=True, downsample=1, frame_rate=frame_rate)

    assert y_ref.shape == y_new.shape
    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-12)
