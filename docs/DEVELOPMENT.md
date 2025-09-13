BrainLGN-X Development Log
==========================

@author: gray
@date: 2025-09-12
@e-mail: oswin0001@qq.com
@Liu-Lab

---------------------------
Strategy (foundations first)
- Phase 1: Single-neuron MVP (current)
  - Implement LGN LNUnit interface
  - Validate numerics against BMTK (parity tests)
  - Add unit tests
- Phase 2: Visual stimuli
  - GratingMovie, FullFieldFlashMovie
  - Stimulus validation
- Phase 3: Network
  - Multi-neuron LGN network, connectivity, batch simulation
- Phase 4: BMTK compatibility
  - API wrapper, JSON config parser, HDF5 output parity

Current Tasks
- Single LGN neuron: accept stimulus, spatio-temporal filter, produce rates (Hz), match BMTK numerics.
- Parity visualization: notebooks/visualize_parity.ipynb
- Files of interest: brainlgn_x/neuron.py, brainlgn_x/filters.py, brainlgn_x/transfer.py,
  tests/test_bmtk_parity.py, tests/test_bmtk_parity_more.py

Validation Plan
- Build identical inputs/stimuli and filter stacks.
- Compare outputs vs BMTK; target error <= 1e-12 (absolute) in unit tests.
- Visualize overlay/residual/scatter via the notebook for sanity.

Progress Checklist
- [x] Project structure
- [x] Development strategy
- [x] Single neuron interface (MVP) aligned
- [x] Neuron numerical validation (parity with BMTK, separable and non-separable)
- [x] Parity visualization notebook
- [x] Poisson spikes + CSV/H5 writers (minimal SONATA-like)
- [x] Visual stimuli (Grating/Flash generators)
- [ ] Network scaffold
- [ ] BMTK/SONATA output layer (HDF5/CSV)

Change Log
----------

2025-09-13
- Added numerical parity tests:
  - tests/test_bmtk_parity.py (baseline separable)
  - tests/test_bmtk_parity_more.py (non-separable, downsample handling, OFF unit, bias, two-subfield sum)
- Separable downsample handling: evaluate full BMTK path then slice locally to mirror reference.
- Added visualization notebook: notebooks/visualize_parity.ipynb (overlay/residual/scatter/hist, metrics printout).
- BrainState import made optional with stub fallback to avoid env import-time issues during parity phase.
- New conda env (Python 3.11) prepared; bmtk/brainstate installed for development.
- Added JAX backend (separable) under brainlgn_x/bs_backend.py and backend switch via env/arg.
- Added Poisson + IO:
  - brainlgn_x/poisson.py: inhomogeneous Poisson (per-bin Bernoulli).
  - brainlgn_x/io_output.py: write_spikes_csv/write_spikes_h5 (spikes/node_ids,timestamps).
 - tests/test_poisson_and_io.py: mean-rate tolerance, reproducibility, IO roundtrip.

- Added visual stimuli module:
  - brainlgn_x/stimuli.py: Movie container, drifting_grating(), full_field_flash(), array/NPY/video loaders.
  - tests/test_stimuli.py: shape/range checks, temporal peak (FFT) for grating, flash segment correctness, t_range dt.

Status (honest snapshot)
- Compute: BMTK parity path is default; JAX backend (separable) available and parity-tested (allow tiny FP tolerance).
- Outputs: Poisson spikes + CSV/H5 minimal writers ready; full SONATA conventions TBD.
- Next: visual stimuli generators, batch/multi-neuron separable pipeline, HDF5/CSV schema hardening.

2025-09-12
- Added BMTK pass-through wrappers to ensure numerical/API parity:
  - brainlgn_x/filters.py re-exports GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter, etc.
  - brainlgn_x/transfer.py re-exports ScalarTransferFunction, MultiTransferFunction.
- neuron pipeline alignment:
  - ON/OFF via SpatioTemporalFilter amplitude sign.
  - evaluate(stimulus, separable=True, downsample=1, threshold=None); update() aliases evaluate().
- Notes:
  - Requires bmtk installed (imports from bmtk.simulator.filternet.lgnmodel.*).
  - Temporal kernel default step is 1 ms (nkt=600). Align external frame_rate/dt at caller.
  - Spatial translate uses (x,y) while stimulus is indexed [t,y,x], consistent with BMTK.

Status (honest snapshot)
- Compute path: currently uses BMTK LNUnit + Cursor under-the-hood for exact parity. BrainState backend not yet wired.
- Inputs: random stimuli used for tests; dedicated visual stimulus generators are TODO.
- Outputs: rates (Hz) validated; Poisson spikes + HDF5/CSV writers pending.
- Backend toggle: not yet exposed; plan to add env switch (e.g., BRAINLGN_BACKEND=bmtk/brainstate).
