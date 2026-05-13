"""
pyfoam.thermophysical — Thermodynamic and transport models.

Provides:

**Equation of State:**
- :class:`PerfectGas` — ideal gas EOS (p = ρRT)
- :class:`IncompressiblePerfectGas` — incompressible ideal gas (ρ = p_ref/RT)

**Transport Models:**
- :class:`ConstantViscosity` — constant dynamic viscosity
- :class:`Sutherland` — Sutherland's law for temperature-dependent viscosity
- :class:`PolynomialTransport` — polynomial viscosity model

**Thermodynamic Models:**
- :class:`JanafThermo` — JANAF polynomial Cp model
- :class:`HConstThermo` — constant specific heat model

**Combined Thermo:**
- :class:`BasicThermo` — basic combined model (EOS + transport)
- :class:`HePsiThermo` — ψ-based thermo for compressible solvers
- :class:`HeRhoThermo` — ρ-based thermo for compressible solvers
- :func:`create_thermo` — factory for creating thermophysical models
- :func:`create_air_thermo` — convenience for air at standard conditions
"""

from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    PerfectGas,
    IncompressiblePerfectGas,
)
from pyfoam.thermophysical.transport_model import (
    TransportModel,
    ConstantViscosity,
    Sutherland,
)
from pyfoam.thermophysical.polynomial_transport import PolynomialTransport
from pyfoam.thermophysical.janaf_thermo import JanafThermo
from pyfoam.thermophysical.hconst_thermo import HConstThermo
from pyfoam.thermophysical.he_psi_thermo import HePsiThermo
from pyfoam.thermophysical.he_rho_thermo import HeRhoThermo
from pyfoam.thermophysical.thermo import (
    BasicThermo,
    create_thermo,
    create_air_thermo,
)

__all__ = [
    # Equation of state
    "EquationOfState",
    "PerfectGas",
    "IncompressiblePerfectGas",
    # Transport
    "TransportModel",
    "ConstantViscosity",
    "Sutherland",
    "PolynomialTransport",
    # Thermodynamic models
    "JanafThermo",
    "HConstThermo",
    # Combined thermo
    "BasicThermo",
    "HePsiThermo",
    "HeRhoThermo",
    "create_thermo",
    "create_air_thermo",
]
