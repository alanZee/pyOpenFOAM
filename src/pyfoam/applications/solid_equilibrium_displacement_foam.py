"""
solidEquilibriumDisplacementFoam — steady-state linear elasticity solver.

Steady-state version of solidDisplacementFoam. Solves the equilibrium
equation without time stepping:

    ∇·σ + f = 0

This is equivalent to solidDisplacementFoam with a single time step.

Usage::

    from pyfoam.applications.solid_equilibrium_displacement_foam import SolidEquilibriumDisplacementFoam

    solver = SolidEquilibriumDisplacementFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

from .solid_displacement_foam import SolidDisplacementFoam

__all__ = ["SolidEquilibriumDisplacementFoam"]

logger = logging.getLogger(__name__)


class SolidEquilibriumDisplacementFoam(SolidDisplacementFoam):
    """Steady-state linear elasticity solver.

    Inherits from SolidDisplacementFoam and forces a single time step
    for equilibrium solving.
    """

    def __init__(self, case_path: Union[str, Path], **kwargs) -> None:
        super().__init__(case_path, **kwargs)
        # Force single time step for equilibrium
        self.end_time = self.start_time + self.delta_t
        logger.info("SolidEquilibriumDisplacementFoam ready (steady-state)")
