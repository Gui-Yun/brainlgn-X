import os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainlgn_x.stimuli import drifting_grating
from brainlgn_x.generator import generate_population
from brainlgn_x.bs_backend import eval_separable_multi
from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
from bmtk.simulator.filternet.lgnmodel.movie import Movie as BMovie

try:
    import jax
    jax.config.update('jax_enable_x64', True)
except Exception:
    pass


def main():
    row, col, frame_rate = 120, 240, 1000.0
    movie = drifting_grating(row, col, frame_rate, duration=0.3, gray_screen=0.2, cpd=0.04, temporal_f=4.0, theta=0.0, contrast=0.8)
    stim = movie.as_array()

    layout = { 'X_grids': 4, 'Y_grids': 3, 'X_len': 240.0, 'Y_len': 120.0 }
    cell_types = [
        { 'name': 'sON',  'n_per_tile': 2, 'spatial': {'sigma_range':[2.0,3.0]}, 'temporal': {'weights':[0.6,-0.4], 'kpeaks':[15.0,45.0], 'delays':[0,0], 'jitter_percent':0.0}, 'amplitude_range':[1.5,2.0],  'transfer': {'bias_range':[0.8,1.2]} },
        { 'name': 'sOFF', 'n_per_tile': 1, 'spatial': {'sigma_range':[2.0,3.0]}, 'temporal': {'weights':[0.5,-0.3], 'kpeaks':[20.0,60.0], 'delays':[0,0], 'jitter_percent':0.0}, 'amplitude_range':[-2.0,-1.5], 'transfer': {'bias_range':[0.5,1.0]} },
    ]
    lfs, trs, meta = generate_population(cell_types, layout, base_seed=123)
    N = len(lfs)
    print('Population N=', N)

    # BS multi
    t0 = time.perf_counter();
    rates_bs = eval_separable_multi(lfs, trs, stim, frame_rate=movie.frame_rate, downsample=1)
    t1 = time.perf_counter();
    print('BS multi time(s)=', t1-t0, 'rates shape=', rates_bs.shape)

    # BMTK reference on subset
    K = min(64, N)
    mv = BMovie(stim, frame_rate=movie.frame_rate)
    ref = []
    t0 = time.perf_counter()
    for i in range(K):
        ln = LNUnit(lfs[i], trs[i])
        _, y = ln.get_cursor(mv, separable=True).evaluate()
        ref.append(np.array(y))
    t1 = time.perf_counter()
    rates_bmtk = np.stack(ref, axis=0)
    print('BMTK subset time(s)=', t1-t0, 'shape=', rates_bmtk.shape)

    # Parity metrics
    rb, rs = rates_bmtk, rates_bs[:K]
    mae = np.mean(np.abs(rb - rs))
    mx = np.max(np.abs(rb - rs))
    print('Parity subset: MAE=%.3e MaxAbs=%.3e' % (mae, mx))

    # NaN diagnostics
    print('NaNs in rb:', np.isnan(rb).sum(), 'NaNs in rs:', np.isnan(rs).sum())
    print('dtype rb:', rb.dtype, 'dtype rs:', rs.dtype)


if __name__ == '__main__':
    main()

