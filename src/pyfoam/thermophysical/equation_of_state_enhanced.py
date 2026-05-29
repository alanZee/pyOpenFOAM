"""
Enhanced equation of state models.

Extends the EOS hierarchy with:

- Cubic EOS with improved alpha functions (Twu, Mathias-Copeman)
- Virial equation of state (truncated)
- Departure functions for H, S, Cp

Models:

- :class:`TwuAlphaPR` -- Peng-Robinson with Twu (1991) alpha function
- :class:`MathiasCopemanPR` -- Peng-Robinson with Mathias-Copeman alpha
- :class:`VirialEOS` -- Truncated virial EOS (B + C terms)
- :class:`SoaveRedlichKwongEOS` -- SRK cubic EOS

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced import VirialEOS, SoaveRedlichKwongEOS

    virial = VirialEOS(Mw=28.014, Cp=1040.0, B=0.0, C=0.0)
    rho = virial.rho(p=1e5, T=300.0)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    CubicEOS,
    PengRobinsonEOS,
)

__all__ = [
    "TwuAlphaPR",
    "MathiasCopemanPR",
    "VirialEOS",
    "SoaveRedlichKwongEOS",
]

logger = logging.getLogger(__name__)


# ======================================================================
# Improved alpha functions for Peng-Robinson
# ======================================================================


class TwuAlphaPR(PengRobinsonEOS):
    """Peng-Robinson EOS with Twu (1991) alpha function.

    The Twu alpha function provides better extrapolation behaviour
    at extreme temperatures compared to the standard Soave form:

        alpha(T) = T^(N*(M-1)) * exp(L*(1 - T^(N*M)))

    where T_r = T/Tc, and L, M, N are substance-specific parameters
    (correlated to the acentric factor by default).

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
    L, M, N : float or None
        Twu alpha parameters. If None, estimated from acentric factor.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        L: float | None = None,
        M: float | None = None,
        N: float | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)

        # Default Twu parameters from acentric factor correlation
        if L is None:
            L = 0.176 * accentric ** 2 + 0.664 * accentric + 0.480
        if M is None:
            M = 0.645 * accentric + 0.770
        if N is None:
            N = 2.0

        self._twu_L = L
        self._twu_M = M
        self._twu_N = N

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """Twu alpha: T_r^(N*(M-1)) * exp(L*(1 - T_r^(N*M)))."""
        T_r = T / self._Tc
        T_r = T_r.clamp(min=1e-10)
        L, M, N = self._twu_L, self._twu_M, self._twu_N
        T_r_NM = T_r.pow(N * M)
        alpha = T_r.pow(N * (M - 1.0)) * (L * (1.0 - T_r_NM)).exp()
        return alpha.clamp(min=0.01, max=10.0)

    def __repr__(self) -> str:
        return (
            f"TwuAlphaPR(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )


class MathiasCopemanPR(PengRobinsonEOS):
    """Peng-Robinson EOS with Mathias-Copeman (1983) alpha function.

    The Mathias-Copeman alpha is a three-parameter modification of the
    Soave alpha that provides improved VLE predictions:

        For T <= Tc: alpha = [1 + c1*(1 - sqrt(T_r))]^2
        For T > Tc:  alpha = [1 + c2*(1 - T_r)]^2

    By default, c1 is fitted from the acentric factor, and c2 = c3 = 0
    (which reduces to the standard Soave form for T > Tc).

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
    c1, c2, c3 : float or None
        Mathias-Copeman parameters. If None, c1 estimated from acentric factor.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        c1: float | None = None,
        c2: float | None = None,
        c3: float | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)

        if c1 is None:
            c1 = 0.37464 + 1.54226 * accentric - 0.26992 * accentric ** 2

        self._c1 = c1
        self._c2 = c2 if c2 is not None else c1 * 0.5
        self._c3 = c3 if c3 is not None else 0.0

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """Mathias-Copeman alpha with T <= Tc and T > Tc branches."""
        T_r = T / self._Tc
        T_r = T_r.clamp(min=1e-10)

        # Sub-critical: [1 + c1*(1 - sqrt(T_r))]^2
        sub_factor = 1.0 + self._c1 * (1.0 - T_r.sqrt())
        alpha_sub = sub_factor * sub_factor

        # Super-critical: [1 + c2*(1 - T_r)]^2
        sup_factor = 1.0 + self._c2 * (1.0 - T_r)
        alpha_sup = sup_factor * sup_factor

        # Blend near Tc
        mask = T_r <= 1.0
        alpha = torch.where(mask, alpha_sub, alpha_sup)

        return alpha.clamp(min=0.01, max=10.0)

    def __repr__(self) -> str:
        return (
            f"MathiasCopemanPR(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )


# ======================================================================
# Soave-Redlich-Kwong cubic EOS
# ======================================================================


class SoaveRedlichKwongEOS(CubicEOS):
    """Soave-Redlich-Kwong (SRK) cubic equation of state.

    .. math::

        p = RT/(V-b) - a*alpha(T) / (V*(V+b))

    with alpha(T) = [1 + m*(1 - sqrt(T/Tc))]^2.

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
        Acentric factor (default 0.0).
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
    ) -> None:
        Zc = 1.0 / 3.0
        omega_a = 0.42748
        omega_b = 0.08664

        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp,
            Zc=Zc, omega_a=omega_a, omega_b=omega_b,
            accentric=accentric,
        )

        self._m = 0.480 + 1.574 * accentric - 0.176 * accentric ** 2

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """SRK alpha: [1 + m*(1 - sqrt(T/Tc))]^2."""
        T_r = T / self._Tc
        T_r = T_r.clamp(min=1e-10)
        factor = 1.0 + self._m * (1.0 - T_r.sqrt())
        return factor * factor

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """SRK Z-factor coefficients.

        Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
        """
        aT = self._a(T)
        b = self._Omega_b
        R = self._R

        A = aT * p / (R * R * T * T)
        B = b * p / (R * T)

        c2 = -torch.ones_like(A)
        c1 = A - B - B * B
        c0 = -A * B

        return c2, c1, c0

    def __repr__(self) -> str:
        return (
            f"SoaveRedlichKwongEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )


# ======================================================================
# Virial equation of state
# ======================================================================


class VirialEOS(EquationOfState):
    """Truncated virial equation of state (B + C terms).

    .. math::

        Z = 1 + B/V + C/V^2 = 1 + Bp/(RT) + (C - B^2) * p^2/(RT)^2

    where B and C are the second and third virial coefficients.
    Best for low-to-moderate pressures (< 20 bar).

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Cp : float
        Specific heat at constant pressure (J/(kg*K)).
    B : float
        Second virial coefficient (m^3/mol). Can be temperature-dependent
        via the correlation: B(T) = B0 + B1/T + B2/T^2.
    C : float
        Third virial coefficient (m^6/mol^2). Default 0.
    B0, B1, B2 : float
        Temperature correlation coefficients for B(T). If B1=B2=0,
        B is constant.
    """

    def __init__(
        self,
        Mw: float,
        Cp: float,
        B: float = 0.0,
        C: float = 0.0,
        B0: float | None = None,
        B1: float = 0.0,
        B2: float = 0.0,
    ) -> None:
        if Mw <= 0:
            raise ValueError(f"Mw must be positive, got {Mw}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")

        self._Mw = Mw
        self._Cp = Cp
        self._R = 8.314462618 / (Mw * 1e-3)  # J/(kg*K)
        self._Cv = Cp - self._R
        self._gamma = Cp / self._Cv if self._Cv > 0 else float('inf')

        # Virial coefficients
        self._B = B
        self._C = C
        self._B0 = B0 if B0 is not None else B
        self._B1 = B1
        self._B2 = B2

    def _B_T(self, T: torch.Tensor) -> torch.Tensor:
        """Temperature-dependent second virial coefficient (m^3/mol).

        B(T) = B0 + B1/T + B2/T^2
        """
        T_safe = T.clamp(min=1.0)
        return self._B0 + self._B1 / T_safe + self._B2 / T_safe.pow(2)

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density from truncated virial EOS.

        Uses iterative solution of:
        Z = 1 + B*p/(RT) + (C - B^2)*p^2/(RT)^2
        rho = p*Mw / (Z * R_univ * T)

        Parameters
        ----------
        p : torch.Tensor or float
            Pressure (Pa).
        T : torch.Tensor or float
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Density (kg/m^3).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        R_univ = 8314.462618  # J/(kmol*K)

        # Compute B(T) in m^3/mol
        B_T = self._B_T(T_safe)

        # Compressibility factor: Z = 1 + B*p/(RT) + (C - B^2)*p^2/(RT)^2
        # Convert B from m^3/mol to m^3/kmol
        B_m3_kmol = B_T * 1000.0
        C_m6_kmol2 = self._C * 1e6  # m^6/mol^2 -> m^6/kmol^2

        RT = R_univ * T_safe
        Z = 1.0 + B_m3_kmol * p / RT + (C_m6_kmol2 - B_m3_kmol.pow(2)) * p.pow(2) / RT.pow(2)
        Z = Z.clamp(min=0.1)  # Prevent non-physical Z

        rho = p * self._Mw / (Z * R_univ * T_safe)  # kg/m^3

        return rho

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure from virial EOS (Newton iteration).

        Parameters
        ----------
        rho : torch.Tensor or float
            Density (kg/m^3).
        T : torch.Tensor or float
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Pressure (Pa).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(rho, torch.Tensor):
            rho = torch.tensor(rho, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        rho_safe = rho.clamp(min=1e-10)
        T_safe = T.clamp(min=1e-10)

        # Start from ideal gas estimate
        p_est = rho_safe * self._R * T_safe

        for _ in range(20):
            rho_calc = self.rho(p_est, T_safe)
            # dp/drho ~ Z*R*T/Mw (approximately)
            dp_drho = p_est / rho_calc.clamp(min=1e-10)
            correction = (rho_safe - rho_calc) * dp_drho
            p_est = p_est + 0.5 * correction
            p_est = p_est.clamp(min=1.0)

        return p_est

    def R(self) -> float:
        """Specific gas constant (J/(kg*K))."""
        return self._R

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg*K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg*K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy (ideal part): h = Cp * T."""
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy (ideal part): e = Cv * T."""
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def B_coefficient(self, T: float) -> float:
        """Second virial coefficient at temperature T.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            B(T) in m^3/mol.
        """
        T_t = torch.tensor(T, dtype=torch.float64)
        return float(self._B_T(T_t).item())

    def __repr__(self) -> str:
        return (
            f"VirialEOS(Mw={self._Mw}, Cp={self._Cp}, "
            f"B0={self._B0}, C={self._C})"
        )
