"""
Pass-through wrappers to reuse BMTK FilterNet LGN filter implementations.

This guarantees numerical and API-level consistency with BMTK while
developing BrainState-based components around them.
"""

try:
    from bmtk.simulator.filternet.lgnmodel.spatialfilter import (
        GaussianSpatialFilter,
        ArrayFilter,
    )
    from bmtk.simulator.filternet.lgnmodel.temporalfilter import (
        TemporalFilterCosineBump,
        ArrayTemporalFilter,
    )
    from bmtk.simulator.filternet.lgnmodel.linearfilter import (
        SpatioTemporalFilter,
    )
except ImportError as e:
    raise ImportError(
        "BMTK is required for pass-through filters. Install bmtk to proceed."
    ) from e

