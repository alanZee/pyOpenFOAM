"""
pyfoam.thermophysical — Thermodynamic and transport models.

Provides:

**Equation of State:**
- :class:`PerfectGas` — ideal gas EOS (p = rho*RT)
- :class:`IncompressiblePerfectGas` — incompressible ideal gas (rho = p_ref/RT)
- :class:`CubicEOS` — generic cubic EOS base class
- :class:`PengRobinsonEOS` — Peng-Robinson cubic EOS
- :class:`RedlichKwongEOS` — Redlich-Kwong cubic EOS
- :class:`VanDerWaalsEOS` — van der Waals cubic EOS
- :class:`IcoTabulatedEOS` — incompressible tabulated EOS (bilinear interpolation)

**Transport Models:**
- :class:`ConstantViscosity` — constant dynamic viscosity
- :class:`Sutherland` — Sutherland's law for temperature-dependent viscosity
- :class:`PolynomialTransport` — polynomial viscosity model
- :class:`ConstantTransport` — constant viscosity + optional constant kappa
- :class:`SutherlandTransport` — Sutherland viscosity + optional polynomial kappa
- :class:`TabulatedTransport` — tabulated viscosity/conductivity interpolation
- :class:`WilkeTransport` — Wilke mixing rule for gas mixture viscosity

**Thermodynamic Models:**
- :class:`JanafThermo` — JANAF polynomial Cp model
- :class:`HConstThermo` — constant specific heat model (enthalpy-based)
- :class:`EConstThermo` — constant specific heat model (energy-based)
- :class:`HPowerThermo` — power-law Cp model
- :class:`JanafMultiThermo` — multi-phase JANAF model
- :class:`JanafMultiThermoEnhanced` — enhanced multi-phase JANAF with Gibbs blending
- :class:`JanafMultiThermoEnhanced2` — enhanced JANAF v2 with latent heat and phase fractions

**Enhanced Transport Models:**
- :class:`TabulatedTransportEnhanced` — tabulated transport with T and P dependence
- :class:`WilkeTransportEnhanced` — Wilke mixing with multi-component diffusion
- :class:`ConstantTransportEnhanced` — constant transport with optional T correction
- :class:`SutherlandTransportEnhanced` — multi-species Sutherland transport

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
- :class:`FSDModel` — Flame Surface Density model for premixed flames

**Chemistry Models:**
- :class:`ChemistryModel` — abstract base for chemistry
- :class:`ODEChemistrySolver` — ODE-based stiff chemistry integration
- :class:`SRMChemistrySolver` — Simplified Reaction Mechanism (progress variable)

**Phase 11 — Enhanced v2/v3 Models:**
- :class:`JanafMultiThermoEnhanced3` — JANAF v3 with Gibbs energy, multi-order departure
- :class:`TabulatedTransportEnhanced2` — tabulated transport with Hermite/monotone interpolation
- :class:`WilkeTransportEnhanced2` — Wilke v2 with FSG diffusion correlation
- :class:`ConstantTransportEnhanced2` — constant transport v2 with exponential/piecewise correction
- :class:`SutherlandTransportEnhanced2` — Sutherland v2 with polar collision correction
- :class:`TwuAlphaPR` — Peng-Robinson with Twu alpha function
- :class:`MathiasCopemanPR` — Peng-Robinson with Mathias-Copeman alpha
- :class:`VirialEOS` — Truncated virial equation of state
- :class:`SoaveRedlichKwongEOS` — Soave-Redlich-Kwong cubic EOS
"""

from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    PerfectGas,
    IncompressiblePerfectGas,
    CubicEOS,
    PengRobinsonEOS,
    RedlichKwongEOS,
    VanDerWaalsEOS,
    IcoTabulatedEOS,
)
from pyfoam.thermophysical.transport_model import (
    TransportModel,
    ConstantViscosity,
    Sutherland,
)
from pyfoam.thermophysical.polynomial_transport import PolynomialTransport
from pyfoam.thermophysical.constant_transport import ConstantTransport
from pyfoam.thermophysical.sutherland_transport import SutherlandTransport
from pyfoam.thermophysical.tabulated_transport import TabulatedTransport
from pyfoam.thermophysical.wilke_transport import WilkeTransport
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
    FSDModel,
)

# Phase 9: Enhanced multi-phase JANAF
from pyfoam.thermophysical.janaf_multi_thermo_enhanced import JanafMultiThermoEnhanced

# Phase 10: Enhanced models
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_2 import JanafMultiThermoEnhanced2
from pyfoam.thermophysical.tabulated_transport_enhanced import TabulatedTransportEnhanced
from pyfoam.thermophysical.wilke_transport_enhanced import WilkeTransportEnhanced
from pyfoam.thermophysical.constant_transport_enhanced import ConstantTransportEnhanced
from pyfoam.thermophysical.sutherland_transport_enhanced import SutherlandTransportEnhanced

# Phase 11: Enhanced models v2/v3
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_3 import JanafMultiThermoEnhanced3
from pyfoam.thermophysical.tabulated_transport_enhanced_2 import TabulatedTransportEnhanced2
from pyfoam.thermophysical.wilke_transport_enhanced_2 import WilkeTransportEnhanced2
from pyfoam.thermophysical.constant_transport_enhanced_2 import ConstantTransportEnhanced2
from pyfoam.thermophysical.sutherland_transport_enhanced_2 import SutherlandTransportEnhanced2
from pyfoam.thermophysical.equation_of_state_enhanced import (
    TwuAlphaPR,
    MathiasCopemanPR,
    VirialEOS,
    SoaveRedlichKwongEOS,
)

# Chemistry models
from pyfoam.thermophysical.chemistry import (
    ChemistryModel,
    ODEChemistrySolver,
    SRMChemistrySolver,
)

__all__ = [
    # Equation of state
    "EquationOfState",
    "PerfectGas",
    "IncompressiblePerfectGas",
    "CubicEOS",
    "PengRobinsonEOS",
    "RedlichKwongEOS",
    "VanDerWaalsEOS",
    "IcoTabulatedEOS",
    # Transport
    "TransportModel",
    "ConstantViscosity",
    "Sutherland",
    "PolynomialTransport",
    "ConstantTransport",
    "SutherlandTransport",
    "TabulatedTransport",
    "WilkeTransport",
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
    "FSDModel",
    # Phase 9: Enhanced multi-phase JANAF
    "JanafMultiThermoEnhanced",
    # Phase 10: Enhanced models
    "JanafMultiThermoEnhanced2",
    "TabulatedTransportEnhanced",
    "WilkeTransportEnhanced",
    "ConstantTransportEnhanced",
    "SutherlandTransportEnhanced",
    # Phase 11: Enhanced models v2/v3
    "JanafMultiThermoEnhanced3",
    "TabulatedTransportEnhanced2",
    "WilkeTransportEnhanced2",
    "ConstantTransportEnhanced2",
    "SutherlandTransportEnhanced2",
    "TwuAlphaPR",
    "MathiasCopemanPR",
    "VirialEOS",
    "SoaveRedlichKwongEOS",
    # Chemistry models
    "ChemistryModel",
    "ODEChemistrySolver",
    "SRMChemistrySolver",
]
