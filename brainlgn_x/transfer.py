"""
Pass-through wrappers to reuse BMTK FilterNet LGN transfer functions.
"""

try:
    from bmtk.simulator.filternet.lgnmodel.transferfunction import (
        ScalarTransferFunction,
        MultiTransferFunction,
    )
except ImportError as e:
    raise ImportError(
        "BMTK is required for pass-through transfer functions. Install bmtk to proceed."
    ) from e

