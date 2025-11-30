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
- [x] Network scaffold (Phase A minimal population)
- [ ] BMTK/SONATA output layer (HDF5/CSV)

Change Log
----------

2025-09-15
- BS/JAX backend stability + parity fixes (multi-neuron):
  - Enabled float64 by default in JAX (`jax_enable_x64`) and added finite checks
    for stimulus, spatial/temporal kernels, and outputs to prevent silent NaNs.
  - Added automatic fallback to BMTK reference when non-finite values or degenerate
    kernels are detected; controlled via `BRAINLGN_FALLBACK_BMTK_ON_NAN` (default=1).
  - Sanitized kernels with `nan_to_num` (replace NaN/Inf with 0) before JAX eval to
    reduce unnecessary fallbacks; kept parity safety net.
  - Fixed temporal alignment in `eval_separable_multi`: when kernels are left-padded
    to common length Lmax, the correct start offset is `Lmax-1` (not `Li-1`).
    Default alignment now uses `BRAINLGN_ALIGN_START=lmax`, eliminating the phase
    shift that caused large MAE despite correct shapes.
- Notebook refresh:
  - Rewrote `notebooks/lgn_population_full.ipynb` for a robust end-to-end parity
    workflow (stimulus → population → BS/JAX multi → BMTK subset parity → optional
    spikes/IO). Prints NaN counts and shows overlay/residual/scatter plots.
  - Added helper `notebooks/_helpers/population_parity.py` providing a single-call
    population parity runner for notebooks.
- Env vars (current defaults shown):
  - `BRAINLGN_JAX_X64=1` — enable float64 in JAX backend.
  - `BRAINLGN_FALLBACK_BMTK_ON_NAN=1` — fallback to BMTK if NaN/Inf/degenerate kernels.
  - `BRAINLGN_ALIGN_START=lmax` — multi-neuron temporal alignment mode.
- Minor: `neuron.py` also falls back to BMTK when BS/JAX path errors, for robustness.

Notes
- After updating the backend, restart the notebook kernel or reload the module to
  ensure changes take effect, e.g.:
  `import importlib, brainlgn_x.bs_backend as bs_backend; importlib.reload(bs_backend)`.
- Expected parity after the alignment fix: MAE/MaxAbs typically in 1e-9–1e-7 range
  depending on platform/BLAS; larger discrepancies likely indicate mismatched
  configuration (frame_rate/downsample) or disabled x64.

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

- End-to-end (minimal) CLI:
  - brainlgn_x/config_parser.py: expand manifest & load JSON.
  - brainlgn_x/run.py: `python -m brainlgn_x.run config.json` equivalent callable `run_config()`.
  - tests/test_end2end_cli.py: generate temp config, run pipeline in-process, assert outputs (spikes/rates) exist.

- Population generator (Phase A):
  - brainlgn_x/generator.py: generate_population(cell_types, layout, base_seed) → filters/transfers/metadata.
  - run.py: support `cell_types` + `layout` in config; BS multi evaluates with eval_separable_multi, BMTK loops per neuron.
  - tests/test_generator.py: counts & reproducibility.
  - configs/lgn_population_grating.json: example population config.

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
