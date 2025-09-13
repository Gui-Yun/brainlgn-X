import os, sys
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from brainlgn_x.stimuli import drifting_grating, full_field_flash, Movie


def test_grating_shape_and_range():
    row, col = 32, 64
    frame_rate = 1000.0
    duration = 1.0
    gray = 0.5
    contrast = 0.8
    mov = drifting_grating(row, col, frame_rate, duration, gray_screen=gray,
                           cpd=0.05, temporal_f=4.0, theta=30.0, contrast=contrast)
    T = int(round((gray + duration) * frame_rate))
    assert mov.shape == (T, row, col)
    data = mov.as_array()
    assert np.all(data <= contrast + 1e-7)
    assert np.all(data >= -contrast - 1e-7)


def test_grating_temporal_peak():
    # Check temporal frequency around target at a fixed pixel
    row, col = 32, 32
    frame_rate = 1000.0
    duration = 1.0
    tf = 8.0
    mov = drifting_grating(row, col, frame_rate, duration, gray_screen=0.0,
                           cpd=0.05, temporal_f=tf, theta=0.0, contrast=1.0)
    sig = mov.as_array()[:, row // 2, col // 2]
    # FFT and find peak (ignore dc bin)
    f = np.fft.rfftfreq(sig.size, d=1.0 / frame_rate)
    sp = np.abs(np.fft.rfft(sig))
    sp[0] = 0.0
    peak = f[np.argmax(sp)]
    assert abs(peak - tf) < 0.5


def test_flash_sequence():
    row, col = 16, 16
    frame_rate = 1000.0
    pre, on, off, post = 0.1, 0.2, 0.15, 0.05
    max_i = 0.9
    mov = full_field_flash(row, col, frame_rate, pre, on, off, post, max_intensity=max_i)
    T = int(round((pre + on + off + post) * frame_rate))
    assert mov.shape == (T, row, col)
    data = mov.as_array()
    # Check segment means
    n_pre = int(round(pre * frame_rate))
    n_on = int(round(on * frame_rate))
    n_off = int(round(off * frame_rate))
    assert np.allclose(data[:n_pre].mean(), 0.0, atol=1e-6)
    assert np.allclose(data[n_pre:n_pre + n_on].mean(), max_i, atol=1e-6)
    assert np.allclose(data[n_pre + n_on:n_pre + n_on + n_off].mean(), -max_i, atol=1e-6)


def test_movie_t_range():
    arr = np.zeros((100, 10, 10), dtype=np.float32)
    mov = Movie(arr, frame_rate=500.0)
    dt = np.diff(mov.t_range)
    assert np.allclose(dt, dt[0])
    assert abs(dt[0] - 1.0 / 500.0) < 1e-12

