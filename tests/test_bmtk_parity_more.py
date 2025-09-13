import os, sys
import numpy as np

# Ensure package root on path
_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from brainlgn_x.neuron import LGNNeuron, TwoSubfieldLinearCell
from brainlgn_x.filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter
from brainlgn_x.transfer import ScalarTransferFunction, MultiTransferFunction

from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit, MultiLNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie


def _build_components(amplitude=1.0, bias=0.0):
    spatial = GaussianSpatialFilter(translate=(0.0, 0.0), sigma=(2.0, 2.0))
    tfilt = TemporalFilterCosineBump(weights=(0.4, -0.3), kpeaks=(20.0, 60.0), delays=(0, 0))
    st = SpatioTemporalFilter(spatial, tfilt, amplitude=float(amplitude))
    # Use Max to avoid Heaviside two-arg mapping issues
    tr = ScalarTransferFunction(f"Max(0, s + {bias})")
    return st, tr


def test_parity_nonseparable_small_random():
    rng = np.random.RandomState(1)
    T, H, W = 128, 16, 16
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.05

    st, tr = _build_components()

    # BMTK reference (non-separable cursor)
    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=False, threshold=0.0).evaluate(downsample=1)
    y_ref = np.array(y_ref)

    # Our neuron (non-separable path)
    neuron = LGNNeuron(spatial_filter=st.spatial_filter, temporal_filter=st.temporal_filter,
                       transfer_function=tr, amplitude=st.amplitude)
    y_new = neuron.evaluate(stim, separable=False, downsample=1, threshold=0.0, frame_rate=frame_rate)

    assert y_ref.shape == y_new.shape
    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-12)


def test_parity_downsample_separable():
    rng = np.random.RandomState(2)
    T, H, W = 200, 12, 12
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.1

    st, tr = _build_components()

    # BMTK separable reference
    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate(downsample=1)
    y_ref = np.array(y_ref)

    # Downsample factor
    ds = 5
    y_ref_ds = y_ref[::ds]

    neuron = LGNNeuron(spatial_filter=st.spatial_filter, temporal_filter=st.temporal_filter,
                       transfer_function=tr, amplitude=st.amplitude)
    y_new = neuron.evaluate(stim, separable=True, downsample=ds, frame_rate=frame_rate)

    assert y_ref_ds.shape == y_new.shape
    np.testing.assert_allclose(y_new, y_ref_ds, rtol=0, atol=1e-12)


def test_parity_off_unit_relu():
    rng = np.random.RandomState(3)
    T, H, W = 160, 14, 14
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.1

    st_off, tr = _build_components(amplitude=-1.0)

    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st_off, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate(downsample=1)
    y_ref = np.array(y_ref)

    neuron = LGNNeuron(spatial_filter=st_off.spatial_filter, temporal_filter=st_off.temporal_filter,
                       transfer_function=tr, amplitude=st_off.amplitude)
    y_new = neuron.evaluate(stim, separable=True, downsample=1, frame_rate=frame_rate)

    assert y_ref.shape == y_new.shape
    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-12)


def test_parity_with_bias():
    rng = np.random.RandomState(4)
    T, H, W = 180, 10, 10
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.05

    st, tr = _build_components(amplitude=1.0, bias=0.2)

    movie = Movie(stim, frame_rate=frame_rate)
    ln_ref = LNUnit(st, tr)
    t_ref, y_ref = ln_ref.get_cursor(movie, separable=True).evaluate(downsample=1)
    y_ref = np.array(y_ref)

    neuron = LGNNeuron(spatial_filter=st.spatial_filter, temporal_filter=st.temporal_filter,
                       transfer_function=tr, amplitude=st.amplitude)
    y_new = neuron.evaluate(stim, separable=True, downsample=1, frame_rate=frame_rate)

    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-12)


def test_two_subfield_internal_sum():
    rng = np.random.RandomState(5)
    T, H, W = 120, 8, 8
    frame_rate = 1000.0
    stim = rng.randn(T, H, W) * 0.05

    st_dom, tr_dom = _build_components(amplitude=1.0)
    st_nd, tr_nd = _build_components(amplitude=-0.5)

    unit_dom = LGNNeuron(st_dom.spatial_filter, st_dom.temporal_filter, tr_dom, amplitude=st_dom.amplitude)
    unit_nd = LGNNeuron(st_nd.spatial_filter, st_nd.temporal_filter, tr_nd, amplitude=st_nd.amplitude)

    cell = TwoSubfieldLinearCell(unit_dom, unit_nd)

    y_dom = unit_dom.evaluate(stim, separable=True, frame_rate=frame_rate)
    y_nd = unit_nd.evaluate(stim, separable=True, frame_rate=frame_rate)
    y_sum = y_dom + y_nd
    y_cell = cell.evaluate(stim, separable=True)

    np.testing.assert_allclose(y_cell, y_sum, rtol=0, atol=1e-12)

