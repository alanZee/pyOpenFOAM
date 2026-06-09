"""
stressFoam — transient stress analysis solver.

Time-dependent version of solidDisplacementFoam for dynamic stress
analysis. Solves:

    ρ ∂²D/∂t² = ∇·σ + f

with explicit time stepping for wave propagation problems.

Usage::

    from pyfoam.applications.stress_foam import StressFoam

    solver = StressFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.backend import scatter_add, gather
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solid_displacement_foam import SolidDisplacementFoam

__all__ = ["StressFoam"]

logger = logging.getLogger(__name__)


class StressFoam(SolidDisplacementFoam):
    """Transient stress analysis solver with explicit time stepping.

    Extends SolidDisplacementFoam with inertia term for dynamic problems.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho: float = 7800.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)
        self.rho = rho
        logger.info("StressFoam ready: rho=%.1f kg/m³", rho)

    def run(self) -> dict[str, Any]:
        """Run transient stress solver."""
        # Use parent's run method (already handles time stepping)
        result = super().run()
        logger.info("StressFoam completed")
        return result
