BrainLGN-X
==========

Modern LGN (lateral geniculate nucleus) modeling on BrainState/BrainPy with numerical/API parity to BMTK FilterNet.

Overview
- Reimplements the classic LN (linear–nonlinear) LGN pipeline using BrainState/BrainPy.
- Guarantees parity by reusing BMTK’s LGN filters and transfer functions via thin pass-through wrappers.
- Minimal MVP: grating stimulus → separable LN (space→time) → firing rates (Hz) → Poisson spikes.

Highlights
- Parity with BMTK: identical spatial/temporal/spatio-temporal filters and transfer functions.
- Clean neuron interface: evaluate(stimulus, separable=True, downsample=1, threshold=None, frame_rate=...).
- Tests for parity at 1e-12 absolute tolerance (separable path).

Repository Layout
- brainlgn_x/
  - neuron.py: LGN neuron (LNUnit-equivalent) interface and pipeline.
  - filters.py: pass-through wrappers for BMTK LGN filters (GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter, ...).
  - transfer.py: pass-through wrappers for BMTK transfer functions (ScalarTransferFunction, MultiTransferFunction).
- tests/
  - test_bmtk_parity.py: numerical parity test (separable path) vs BMTK.
- docs/
  - DEVELOPMENT.md: development log and change notes.

Dependencies
- Python 3.9+
- brainstate, brainpy, jax/jaxlib, numpy, scipy, pytest
- bmtk (required for pass-through parity wrappers)

Install (conda example)
- conda create -n brainlgn python=3.11 -y
- conda activate brainlgn
- pip install brainstate brainpy "jax[cpu]" numpy scipy pytest bmtk

Quick Validation
- Run tests: pytest -q
- The parity test builds a shared filter stack and compares outputs between:
  - BMTK LNUnit + Cursor (separable)
  - BrainLGN-X LGNNeuron.evaluate(separable=True)
  Absolute tolerance: 1e-12

Usage Notes
- Stimulus shape: (t, y, x), time sampled by frame_rate (seconds).
- Output rates: non-negative (Hz); enforced by transfer function (e.g., Heaviside(s)*s).
- ON/OFF: select via SpatioTemporalFilter amplitude (>0 ON, <0 OFF).
- Temporal kernel in BMTK defaults to 1 ms sampling (nkt=600). Align external frame_rate/dt at the caller.
- Spatial translate uses (x, y) while stimulus is indexed [t, y, x], matching BMTK.

Roadmap
- MVP: separable LN parity, Poisson spikes, minimal HDF5/CSV outputs.
- Next: stimulus generators (grating, flash), multi-neuron network, JSON config parsing, SONATA-compatible outputs.

Notes
- See docs/DEVELOPMENT.md for detailed plan and change log.

