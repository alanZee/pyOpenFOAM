"""
Enhanced equation of state models v3 — SAFT-VR and CPA.

Extends :class:`~pyfoam.thermophysical.equation_of_state_enhanced_2` with:

- Simplified SAFT-VR (Statistical Associating Fluid Theory - Variable Range)
- Cubic-Plus-Association (CPA) EOS combining SRK with association term
- Generalized alpha function with Twu/Soave/Mathias-Copeman selection

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_3 import SAFTVRSimplified, CPAEOS

    saft = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, m_seg=1.0)
    rho = saft.rho(p=1e6, T=400.0)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    CubicEOS,
    PengRobinsonEOS,
)
from pyfoam.thermophysical.equation_of_state_enhanced_2 import PatelTejaEOS

__all__ = [
    "SAFTVRSimplified",
    "CPAEOS",
    "GeneralizedAlphaEOS",
]

logger = logging.getLogger(__name__)


# ======================================================================
# SAFT-VR Simplified
# ======================================================================


class SAFTVRSimplified(PengRobinsonEOS):
    """Simplified SAFT-VR equation of state.

    Adds a perturbation term to the Peng-Robinson EOS to approximate
    chain and association contributions:

        p = p_PR + p_chain + p_assoc

    This is a computationally efficient approximation suitable for
    engineering calculations of associating fluids (water, alcohols).

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    m_seg : float
        Segment number (1.0 for non-chain molecules). Default 1.0.
    assoc_energy : float
        Association energy parameter (K). Default 0 (no association).
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        m_seg: float = 1.0,
        assoc_energy: float = 0.0,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._m_seg = max(m_seg, 1.0)
        self._assoc_energy = assoc_energy

    @property
    def segment_number(self) -> float:
        """Segment number m."""
        return self._m_seg

    @property
    def association_energy(self) -> float:
        """Association energy parameter (K)."""
        return self._assoc_energy

    def _chain_contribution(self, rho: torch.Tensor, T: float) -> torch.Tensor:
        """Chain contribution to pressure.

        p_chain = -rho * R * T * (m - 1) * (d ln(g_hs) / d_rho)

        Simplified: p_chain = rho * R * T * (m - 1) * phi(rho)
        """
        if self._m_seg <= 1.0:
            return torch.zeros_like(rho)

        rho_safe = rho.clamp(min=1e-10)
        # Simplified: phi(rho) = rho / (1 - eta) with packing fraction eta
        eta = rho_safe * self._Omega_b * 0.5  # Approximate packing fraction
        eta = eta.clamp(max=0.5)
        phi = rho_safe / (1.0 - eta).clamp(min=0.01)

        R = self._R
        return -rho * R * T * (self._m_seg - 1.0) * phi / rho_safe.clamp(min=1e-10)

    def _association_contribution(self, rho: torch.Tensor, T: float) -> torch.Tensor:
        """Association contribution to pressure (simplified).

        p_assoc = rho * R * T * X_assoc * (d X_assoc / d rho) / rho

        where X_assoc is the fraction of non-bonded sites.
        """
        if self._assoc_energy <= 0:
            return torch.zeros_like(rho)

        T_safe = max(T, 1.0)
        rho_safe = rho.clamp(min=1e-10)

        # Simplified association: X = 1 / (1 + K * rho * Delta)
        K_Delta = self._assoc_energy / T_safe * 1e-6  # Scaling
        X = 1.0 / (1.0 + K_Delta * rho_safe)
        X = X.clamp(min=0.01, max=1.0)

        R = self._R
        # Approximate: p_assoc ~ rho * R * T * (1 - X)
        return rho * R * T * (1.0 - X) * 0.5

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with SAFT-VR chain and association corrections.

        Iterative correction to PR density.
        """
        # Start from PR density
        rho_pr = super().rho(p, T)

        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        # Apply chain correction iteratively
        rho = rho_pr.clone()
        for _ in range(3):
            p_chain = self._chain_contribution(rho, T_val)
            p_assoc = self._association_contribution(rho, T_val)
            p_correction = (p_chain + p_assoc).abs().clamp(max=rho_pr * 0.5)
            rho = rho_pr + p_correction * 1e-6  # Small correction
            rho = rho.clamp(min=1e-10)

        return rho

    def __repr__(self) -> str:
        return (
            f"SAFTVRSimplified(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, m={self._m_seg})"
        )


# ======================================================================
# Cubic-Plus-Association (CPA)
# ======================================================================


class CPAEOS(PengRobinsonEOS):
    """Cubic-Plus-Association equation of state.

    Combines a cubic EOS (SRK) with an association term:

        p = p_cubic + p_association

    Suitable for strongly associating fluids (water, glycols, amines).

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    assoc_beta : float
        Association volume parameter. Default 0.01.
    assoc_epsilon : float
        Association energy (K). Default 2000.
    n_sites : int
        Number of association sites per molecule. Default 2.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        assoc_beta: float = 0.01,
        assoc_epsilon: float = 2000.0,
        n_sites: int = 2,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._assoc_beta = assoc_beta
        self._assoc_epsilon = assoc_epsilon
        self._n_sites = max(n_sites, 1)

    @property
    def association_beta(self) -> float:
        """Association volume parameter."""
        return self._assoc_beta

    @property
    def association_epsilon(self) -> float:
        """Association energy (K)."""
        return self._assoc_epsilon

    def _association_delta(self, T: float) -> float:
        """Association strength Delta.

        Delta = beta * (exp(epsilon / T) - 1)
        """
        T_safe = max(T, 1.0)
        exponent = min(self._assoc_epsilon / T_safe, 50.0)
        return self._assoc_beta * (math.exp(exponent) - 1.0)

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with CPA association correction."""
        rho_cubic = super().rho(p, T)

        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        if self._assoc_epsilon <= 0:
            return rho_cubic

        # Simplified: association reduces density by a factor
        Delta = self._association_delta(T_val)
        rho_safe = rho_cubic.clamp(min=1e-10)
        X_frac = 1.0 / (1.0 + self._n_sites * Delta * rho_safe * 1e-6)
        X_frac = X_frac.clamp(min=0.1, max=1.0)

        # Association tends to increase density (molecules aggregate)
        correction = 1.0 + 0.1 * (1.0 - X_frac)
        return rho_cubic * correction

    def __repr__(self) -> str:
        return (
            f"CPAEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, n_sites={self._n_sites})"
        )


# ======================================================================
# Generalized Alpha Function EOS
# ======================================================================


class GeneralizedAlphaEOS(PengRobinsonEOS):
    """Peng-Robinson with selectable generalized alpha function.

    Supports three alpha function variants:
    - "soave": alpha(T) = [1 + m*(1-sqrt(T/Tc))]^2
    - "twu": Twu et al. (1991) three-parameter alpha
    - "mathias_copeman": Mathias-Copeman (1983) three-coefficient alpha

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    alpha_type : str
        Alpha function type: "soave", "twu", or "mathias_copeman".
    mc_coeffs : tuple of 3 floats or None
        Mathias-Copeman coefficients (c1, c2, c3). If None, estimated from accentric.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        alpha_type: str = "soave",
        mc_coeffs: tuple[float, float, float] | None = None,
    ) -> None:
        if alpha_type not in ("soave", "twu", "mathias_copeman"):
            raise ValueError(
                f"alpha_type must be 'soave', 'twu', or 'mathias_copeman', "
                f"got '{alpha_type}'"
            )
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._alpha_type = alpha_type

        if alpha_type == "mathias_copeman":
            if mc_coeffs is not None:
                self._mc_c1, self._mc_c2, self._mc_c3 = mc_coeffs
            else:
                # Default estimation from accentric factor
                self._mc_c1 = 0.4207 + 1.644 * accentric - 0.5674 * accentric ** 2
                self._mc_c2 = -0.3746 + 1.527 * accentric - 0.7226 * accentric ** 2
                self._mc_c3 = 0.8751 + 0.2144 * accentric + 0.3718 * accentric ** 2

    @property
    def alpha_type(self) -> str:
        """Current alpha function type."""
        return self._alpha_type

    def alpha(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute alpha function based on selected type.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Alpha function value.
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_r = (T / self._Tc).clamp(min=1e-10)

        if self._alpha_type == "soave":
            m = 0.37464 + 1.54226 * self._accentric - 0.26992 * self._accentric ** 2
            factor = 1.0 + m * (1.0 - T_r.sqrt())
            return factor * factor

        elif self._alpha_type == "twu":
            # Twu et al. (1991) alpha
            L = 0.09846 + 0.33758 * self._accentric
            M = 0.68547 + 0.51574 * self._accentric
            N = -0.10723 + 0.47078 * self._accentric
            return T_r.pow(N * (M - 1.0)) * torch.exp(L * (1.0 - T_r.pow(M * N)))

        elif self._alpha_type == "mathias_copeman":
            # Mathias-Copeman (1983)
            sqrt_Tr = T_r.sqrt()
            return (1.0 + self._mc_c1 * (1.0 - sqrt_Tr)
                    + self._mc_c2 * (1.0 - sqrt_Tr).pow(2)
                    + self._mc_c3 * (1.0 - sqrt_Tr).pow(3)).pow(2)

        else:
            # Fallback: Soave
            m = 0.37464 + 1.54226 * self._accentric - 0.26992 * self._accentric ** 2
            factor = 1.0 + m * (1.0 - T_r.sqrt())
            return factor * factor

    def __repr__(self) -> str:
        return (
            f"GeneralizedAlphaEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"alpha={self._alpha_type}, accentric={self._accentric})"
        )
