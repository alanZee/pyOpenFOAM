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
- :class:`ConstantTransport` — constant viscosity + optional constant kappa
- :class:`SutherlandTransport` — Sutherland viscosity + optional polynomial kappa

**Thermodynamic Models:**
- :class:`JanafThermo` — JANAF polynomial Cp model
- :class:`HConstThermo` — constant specific heat model (enthalpy-based)
- :class:`EConstThermo` — constant specific heat model (energy-based)
- :class:`HPowerThermo` — power-law Cp model
- :class:`JanafMultiThermo` — multi-phase JANAF model

**Combined Thermo:**
- :class:`BasicThermo` — basic combined model (EOS + transport)
- :class:`HePsiThermo` — ψ-based thermo for compressible solvers
- :class:`HeRhoThermo` — ρ-based thermo for compressible solvers
- :func:`create_thermo` — factory for creating thermophysical models
- :func:`create_air_thermo` — convenience for air at standard conditions

**Thermophysical Transport:**
- :class:`FourierTransport` — Fourier law: q = -k * grad(T)
- :class:`FickianTransport` — Fick's law: j = -rho * D * grad(Y)

**Reaction Kinetics:**
- :class:`ReactionRateModel` — abstract base with RTS registry
- :class:`ArrheniusReaction` — k = A * T^b * exp(-Ea/RT)
- :class:`ThirdBodyReaction` — third-body efficiency wrapper
- :class:`FallOffReaction` — Lindemann fall-off pressure dependence

**Combustion Models:**
- :class:`CombustionModel` — abstract base with RTS registry
- :class:`PaSRModel` — Partially Stirred Reactor
- :class:`EDCModel` — Eddy Dissipation Concept
- :class:`InfinitelyFastChemistry` — mixing-limited combustion
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
from pyfoam.thermophysical.constant_transport import ConstantTransport
from pyfoam.thermophysical.sutherland_transport import SutherlandTransport
from pyfoam.thermophysical.janaf_thermo import JanafThermo
from pyfoam.thermophysical.hconst_thermo import HConstThermo
from pyfoam.thermophysical.econst_thermo import EConstThermo
from pyfoam.thermophysical.hpower_thermo import HPowerThermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafMultiThermo, JanafPhase
from pyfoam.thermophysical.he_psi_thermo import HePsiThermo
from pyfoam.thermophysical.he_rho_thermo import HeRhoThermo
from pyfoam.thermophysical.thermo import (
    BasicThermo,
    create_thermo,
    create_air_thermo,
)
from pyfoam.thermophysical.thermophysical_transport import (
    ThermophysicalTransportModel,
    FourierTransport,
    FickianTransport,
)
from pyfoam.thermophysical.reaction import (
    ReactionRateModel,
    ArrheniusReaction,
    ThirdBodyReaction,
    FallOffReaction,
    CombustionModel,
    PaSRModel,
    EDCModel,
    InfinitelyFastChemistry,
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
    "ConstantTransport",
    "SutherlandTransport",
    # Thermodynamic models
    "JanafThermo",
    "HConstThermo",
    "EConstThermo",
    "HPowerThermo",
    "JanafMultiThermo",
    "JanafPhase",
    # Combined thermo
    "BasicThermo",
    "HePsiThermo",
    "HeRhoThermo",
    "create_thermo",
    "create_air_thermo",
    # Thermophysical transport
    "ThermophysicalTransportModel",
    "FourierTransport",
    "FickianTransport",
    # Reaction kinetics
    "ReactionRateModel",
    "ArrheniusReaction",
    "ThirdBodyReaction",
    "FallOffReaction",
    # Combustion models
    "CombustionModel",
    "PaSRModel",
    "EDCModel",
    "InfinitelyFastChemistry",
]
