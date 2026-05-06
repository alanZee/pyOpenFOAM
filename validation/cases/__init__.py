"""
Validation test cases with analytical solutions.

Provides validation cases that compare pyOpenFOAM solver results against
known analytical solutions:

- :class:`CouetteFlowCase` — plane Couette flow (linear velocity profile)
- :class:`PoiseuilleFlowCase` — plane Poiseuille flow (parabolic velocity profile)
- :class:`LidDrivenCavityCase` — lid-driven cavity (Ghia et al. benchmark)

Each case generates a mesh programmatically, sets boundary conditions,
runs the solver, and computes the analytical reference solution.
"""

from validation.cases.couette_flow import CouetteFlowCase
from validation.cases.poiseuille_flow import PoiseuilleFlowCase
from validation.cases.lid_driven_cavity import LidDrivenCavityCase

__all__ = [
    "CouetteFlowCase",
    "PoiseuilleFlowCase",
    "LidDrivenCavityCase",
]
