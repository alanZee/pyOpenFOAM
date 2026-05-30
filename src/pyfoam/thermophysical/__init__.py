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

**Phase 12 — Enhanced v3/v4 Models:**
- :class:`JanafMultiThermoEnhanced4` — JANAF v4 with phase transition decomposition, fugacity, vapour quality
- :class:`TabulatedTransportEnhanced3` — tabulated transport with Catmull-Rom and extrapolation
- :class:`WilkeTransportEnhanced3` — Wilke v3 with Knudsen correction and Lewis number
- :class:`ConstantTransportEnhanced3` — constant transport v3 with VFT and WLF models
- :class:`SutherlandTransportEnhanced3` — Sutherland v3 with collision diameter and Mason-Saxena mixing
- :class:`PatelTejaEOS` — Patel-Teja three-parameter cubic EOS
- :class:`VolumeTranslatedPR` — Peng-Robinson with volume translation
- :class:`PatelTejaValderramaEOS` — Patel-Teja-Valderrama variant for polar fluids

**Phase 13 — Enhanced v4/v5 Models:**
- :class:`JanafMultiThermoEnhanced5` — JANAF v5 with reaction enthalpy, entropy, Gibbs energy
- :class:`TabulatedTransportEnhanced4` — tabulated transport v4 with adaptive grid refinement and Pr(T)
- :class:`WilkeTransportEnhanced4` — Wilke v4 with Soret thermal diffusion and dilution correction
- :class:`ConstantTransportEnhanced4` — constant transport v4 with Barus and free-volume pressure models
- :class:`SutherlandTransportEnhanced4` — Sutherland v4 with Lennard-Jones collision integral
- :class:`SAFTVRSimplified` — Simplified SAFT-VR EOS for associating fluids
- :class:`CPAEOS` — Cubic-Plus-Association EOS
- :class:`GeneralizedAlphaEOS` — PR with selectable alpha (Soave/Twu/Mathias-Copeman)

**Phase 14 — Enhanced v5/v6 Models:**
- :class:`JanafMultiThermoEnhanced6` — JANAF v6 with Cp moments, reaction network, S_ref(T) table
- :class:`TabulatedTransportEnhanced5` — tabulated transport v5 with multi-property, error estimation, grid quality
- :class:`WilkeTransportEnhanced5` — Wilke v5 with Stockmayer collision integrals, virial correction
- :class:`ConstantTransportEnhanced5` — constant transport v5 with Ree-Eyring shear-thinning, viscosity index
- :class:`SutherlandTransportEnhanced5` — Sutherland v5 with Stockmayer potential, Sonine polynomials
- :class:`PCSAFTSimplified` — Simplified PC-SAFT EOS with dispersion contribution
- :class:`MultiFluidEOS` — Multi-fluid EOS with departure function
- :class:`ExtendedCorrespondingStatesEOS` — Extended Corresponding States EOS with shape factors

**Phase 16 — Enhanced v6/v7 Models:**
- :class:`JanafMultiThermoEnhanced7` — JANAF v7 with equilibrium constant, Cp extrapolation, mixture Cp
- :class:`TabulatedTransportEnhanced6` — tabulated transport v6 with Cp-coupled Pr, bounded extrapolation, table merging
- :class:`WilkeTransportEnhanced6` — Wilke v6 with diffusion cache, mixture validation, extreme-condition correction
- :class:`ConstantTransportEnhanced6` — constant transport v6 with Eucken kappa, thermal conductivity models
- :class:`SutherlandTransportEnhanced6` — Sutherland v6 with high-order mixing, Sutherland-LJ blending
- :class:`LatticeGasEOS` — Lattice-gas EOS for confined fluids
- :class:`CPASAFT` — CPA-SAFT hybrid EOS
- :class:`TemperatureDependentPR` — PR EOS with T-dependent binary interaction parameter
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

# Phase 12: Enhanced models v3/v4
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_4 import JanafMultiThermoEnhanced4
from pyfoam.thermophysical.tabulated_transport_enhanced_3 import TabulatedTransportEnhanced3
from pyfoam.thermophysical.wilke_transport_enhanced_3 import WilkeTransportEnhanced3
from pyfoam.thermophysical.constant_transport_enhanced_3 import ConstantTransportEnhanced3
from pyfoam.thermophysical.sutherland_transport_enhanced_3 import SutherlandTransportEnhanced3
from pyfoam.thermophysical.equation_of_state_enhanced_2 import (
    PatelTejaEOS,
    VolumeTranslatedPR,
    PatelTejaValderramaEOS,
)

# Phase 13: Enhanced models v4/v5
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_5 import JanafMultiThermoEnhanced5
from pyfoam.thermophysical.tabulated_transport_enhanced_4 import TabulatedTransportEnhanced4
from pyfoam.thermophysical.wilke_transport_enhanced_4 import WilkeTransportEnhanced4
from pyfoam.thermophysical.constant_transport_enhanced_4 import ConstantTransportEnhanced4
from pyfoam.thermophysical.sutherland_transport_enhanced_4 import SutherlandTransportEnhanced4
from pyfoam.thermophysical.equation_of_state_enhanced_3 import (
    SAFTVRSimplified,
    CPAEOS,
    GeneralizedAlphaEOS,
)

# Phase 14: Enhanced models v5/v6
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_6 import JanafMultiThermoEnhanced6
from pyfoam.thermophysical.tabulated_transport_enhanced_5 import TabulatedTransportEnhanced5
from pyfoam.thermophysical.wilke_transport_enhanced_5 import WilkeTransportEnhanced5
from pyfoam.thermophysical.constant_transport_enhanced_5 import ConstantTransportEnhanced5
from pyfoam.thermophysical.sutherland_transport_enhanced_5 import SutherlandTransportEnhanced5
from pyfoam.thermophysical.equation_of_state_enhanced_4 import (
    PCSAFTSimplified,
    MultiFluidEOS,
    ExtendedCorrespondingStatesEOS,
)

# Phase 16: Enhanced models v6/v7
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_7 import JanafMultiThermoEnhanced7
from pyfoam.thermophysical.tabulated_transport_enhanced_6 import TabulatedTransportEnhanced6
from pyfoam.thermophysical.wilke_transport_enhanced_6 import WilkeTransportEnhanced6
from pyfoam.thermophysical.constant_transport_enhanced_6 import ConstantTransportEnhanced6
from pyfoam.thermophysical.sutherland_transport_enhanced_6 import SutherlandTransportEnhanced6
from pyfoam.thermophysical.equation_of_state_enhanced_5 import (
    LatticeGasEOS,
    CPASAFT,
    TemperatureDependentPR,
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
    # Phase 12: Enhanced models v3/v4
    "JanafMultiThermoEnhanced4",
    "TabulatedTransportEnhanced3",
    "WilkeTransportEnhanced3",
    "ConstantTransportEnhanced3",
    "SutherlandTransportEnhanced3",
    "PatelTejaEOS",
    "VolumeTranslatedPR",
    "PatelTejaValderramaEOS",
    # Phase 13: Enhanced models v4/v5
    "JanafMultiThermoEnhanced5",
    "TabulatedTransportEnhanced4",
    "WilkeTransportEnhanced4",
    "ConstantTransportEnhanced4",
    "SutherlandTransportEnhanced4",
    "SAFTVRSimplified",
    "CPAEOS",
    "GeneralizedAlphaEOS",
    # Phase 14: Enhanced models v5/v6
    "JanafMultiThermoEnhanced6",
    "TabulatedTransportEnhanced5",
    "WilkeTransportEnhanced5",
    "ConstantTransportEnhanced5",
    "SutherlandTransportEnhanced5",
    "PCSAFTSimplified",
    "MultiFluidEOS",
    "ExtendedCorrespondingStatesEOS",
    # Phase 16: Enhanced models v6/v7
    "JanafMultiThermoEnhanced7",
    "TabulatedTransportEnhanced6",
    "WilkeTransportEnhanced6",
    "ConstantTransportEnhanced6",
    "SutherlandTransportEnhanced6",
    "LatticeGasEOS",
    "CPASAFT",
    "TemperatureDependentPR",
    # Chemistry models
    "ChemistryModel",
    "ODEChemistrySolver",
    "SRMChemistrySolver",
]
