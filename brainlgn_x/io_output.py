"""
Minimal SONATA-like output writers for spikes.

CSV: two columns (gid,time). HDF5: group 'spikes' with datasets 'node_ids' and
'timestamps'. These match common readers' expectations sufficiently for tests.
"""

from typing import Iterable
import numpy as np
import csv
import h5py


def write_spikes_csv(path: str, gids: Iterable[int], times: Iterable[float]) -> None:
    gids = np.asarray(gids)
    times = np.asarray(times)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gid', 'time'])
        for g, t in zip(gids, times):
            w.writerow([int(g), float(t)])


def write_spikes_h5(path: str, gids: Iterable[int], times: Iterable[float]) -> None:
    gids = np.asarray(gids, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    with h5py.File(path, 'w') as f:
        grp = f.create_group('spikes')
        grp.create_dataset('node_ids', data=gids, compression='gzip')
        grp.create_dataset('timestamps', data=times, compression='gzip')

