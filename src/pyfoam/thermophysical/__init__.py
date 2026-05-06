"""
pyfoam.thermophysical — Thermodynamic and transport models.

Provides:

- :class:`PerfectGas` — ideal gas EOS (p = ρRT)
- :class:`IncompressiblePerfectGas` — incompressible ideal gas (ρ = p_ref/RT)
- :class:`ConstantViscosity` — constant dynamic viscosity
- :class:`Sutherland` — Sutherland's law for temperature-dependent viscosity
- :class:`BasicThermo` — combined thermophysical model (EOS + transport)
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
    # Combined thermo
    "BasicThermo",
    "create_thermo",
    "create_air_thermo",
]
