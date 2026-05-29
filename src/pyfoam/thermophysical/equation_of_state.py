"""
Equation of State models for compressible flow.

Provides the thermodynamic relationship between pressure, density,
and temperature:

- **PerfectGas**: p = ρRT (ideal gas law)
- **IncompressiblePerfectGas**: ρ = p_ref/(RT) (density depends only on T)
- **CubicEOS**: generic cubic EOS base with Z-factor computation
- **PengRobinsonEOS**: Peng-Robinson cubic EOS
- **RedlichKwongEOS**: Redlich-Kwong cubic EOS
- **VanDerWaalsEOS**: van der Waals cubic EOS
- **IcoTabulatedEOS**: incompressible tabulated (p, rho) interpolation

These models are used by compressible solvers (rhoSimpleFoam, rhoPimpleFoam)
to close the system of equations.

Usage::

    from pyfoam.thermophysical.equation_of_state import PerfectGas

    eos = PerfectGas(R=287.0)  # air
    rho = eos.rho(p, T)        # density from pressure and temperature
    p = eos.p(rho, T)          # pressure from density and temperature
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Any, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "EquationOfState",
    "PerfectGas",
    "IncompressiblePerfectGas",
    "CubicEOS",
    "PengRobinsonEOS",
    "RedlichKwongEOS",
    "VanDerWaalsEOS",
    "IcoTabulatedEOS",
]

logger = logging.getLogger(__name__)


class EquationOfState(ABC):
    """Abstract base class for equations of state.

    Subclasses must implement :meth:`rho`, :meth:`p`, and :meth:`Cp`.
    """

    @abstractmethod
    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density from pressure and temperature.

        Args:
            p: Pressure (Pa) — scalar or ``(n_cells,)`` tensor.
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Density (kg/m³) — same shape as inputs.
        """

    @abstractmethod
    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure from density and temperature.

        Args:
            rho: Density (kg/m³) — scalar or ``(n_cells,)`` tensor.
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Pressure (Pa) — same shape as inputs.
        """

    @abstractmethod
    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat capacity at constant pressure (J/(kg·K))."""

    @abstractmethod
    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat capacity at constant volume (J/(kg·K))."""

    @abstractmethod
    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""

    @abstractmethod
    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""

    @abstractmethod
    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy h = Cp * T (J/kg)."""

    @abstractmethod
    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy e = Cv * T (J/kg)."""


class PerfectGas(EquationOfState):
    """Ideal (perfect) gas equation of state: p = ρRT.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0 (air).

    Attributes
    ----------
    R : float
        Specific gas constant.
    Cv : float
        Specific heat at constant volume = Cp - R.
    gamma : float
        Ratio of specific heats = Cp / Cv.

    Examples::

        air = PerfectGas(R=287.0, Cp=1005.0)
        rho = air.rho(p=101325.0, T=300.0)  # ~1.177 kg/m³
    """

    def __init__(self, R: float = 287.0, Cp: float = 1005.0) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if Cp <= R:
            raise ValueError(f"Cp must be > R, got Cp={Cp}, R={R}")

        self._R = R
        self._Cp = Cp
        self._Cv = Cp - R
        self._gamma = Cp / (Cp - R)

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: ρ = p / (RT).

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        return p / (self._R * T_safe)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure: p = ρRT.

        Args:
            rho: Density (kg/m³).
            T: Temperature (K).

        Returns:
            Pressure (Pa).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(rho, torch.Tensor):
            rho = torch.tensor(rho, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        return rho * self._R * T

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T.

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def __repr__(self) -> str:
        return (
            f"PerfectGas(R={self._R}, Cp={self._Cp}, "
            f"Cv={self._Cv:.1f}, gamma={self._gamma:.4f})"
        )


class IncompressiblePerfectGas(EquationOfState):
    """Incompressible perfect gas: ρ = p_ref / (RT).

    Density depends only on temperature (not pressure), suitable for
    low-Mach-number compressible flows (natural convection, buoyancy).

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0.
    p_ref : float
        Reference pressure (Pa). Default 101325.0 (1 atm).

    Examples::

        eos = IncompressiblePerfectGas(p_ref=101325.0)
        rho = eos.rho(p=101325.0, T=300.0)  # ~1.177 kg/m³
    """

    def __init__(
        self,
        R: float = 287.0,
        Cp: float = 1005.0,
        p_ref: float = 101325.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if Cp <= R:
            raise ValueError(f"Cp must be > R, got Cp={Cp}, R={R}")
        if p_ref <= 0:
            raise ValueError(f"p_ref must be positive, got {p_ref}")

        self._R = R
        self._Cp = Cp
        self._Cv = Cp - R
        self._gamma = Cp / (Cp - R)
        self._p_ref = p_ref

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: ρ = p_ref / (RT).

        Note: pressure argument is ignored (incompressible).

        Args:
            p: Pressure (Pa) — ignored, kept for API compatibility.
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        return self._p_ref / (self._R * T_safe)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure: returns p_ref (incompressible).

        Args:
            rho: Density (kg/m³) — ignored.
            T: Temperature (K) — ignored.

        Returns:
            Reference pressure (Pa).
        """
        device = get_device()
        dtype = get_default_dtype()
        return torch.tensor(self._p_ref, dtype=dtype, device=device)

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T."""
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T."""
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def __repr__(self) -> str:
        return (
            f"IncompressiblePerfectGas(R={self._R}, Cp={self._Cp}, "
            f"p_ref={self._p_ref})"
        )


# ======================================================================
# Cubic Equation of State base class
# ======================================================================


class CubicEOS(EquationOfState):
    """Generic cubic equation of state base class.

    Implements the common framework for cubic EOS models that solve:

    .. math::

        Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0

    where Z = pV/(RT) is the compressibility factor.

    Subclasses must set ``_omega_a``, ``_omega_b``, ``_Omega_a``, ``_Omega_b``
    and implement :meth:`_alpha` for the attractive parameter temperature
    dependence.

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)).
    Zc : float
        Compressibility factor at critical point. Set by subclasses.
    omega_a : float
        Dimensionless constant for ``a`` parameter. Set by subclasses.
    omega_b : float
        Dimensionless constant for ``b`` parameter. Set by subclasses.
    accentric : float
        Acentric factor (default 0.0 for simple molecules).
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        Zc: float,
        omega_a: float,
        omega_b: float,
        accentric: float = 0.0,
    ) -> None:
        if Mw <= 0:
            raise ValueError(f"Mw must be positive, got {Mw}")
        if Tc <= 0:
            raise ValueError(f"Tc must be positive, got {Tc}")
        if Pc <= 0:
            raise ValueError(f"Pc must be positive, got {Pc}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")

        self._Mw = Mw
        self._Tc = Tc
        self._Pc = Pc
        self._Cp = Cp
        self._Zc = Zc
        self._omega_a = omega_a
        self._omega_b = omega_b
        self._accentric = accentric

        # 具体气体常数 (J/(kg·K))：R_universal / Mw
        self._R = 8.314462618 / (Mw * 1e-3)  # Mw in g/mol → kg/mol
        self._Cv = Cp - self._R
        self._gamma = Cp / self._Cv if self._Cv > 0 else float('inf')

        # 无量纲 → 有量纲参数
        self._Omega_a = omega_a * (self._R * Tc) ** 2 / Pc
        self._Omega_b = omega_b * self._R * Tc / Pc

    @abstractmethod
    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """Compute temperature-dependent alpha function for attraction parameter.

        Args:
            T: Temperature (K).

        Returns:
            alpha(T) dimensionless.
        """

    def _a(self, T: torch.Tensor) -> torch.Tensor:
        """Temperature-dependent attraction parameter a(T) = Omega_a * alpha(T)."""
        return self._Omega_a * self._alpha(T)

    @property
    def b(self) -> float:
        """Covolume parameter b (m³/kg)."""
        return self._Omega_b

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """Compute coefficients of the cubic Z-factor equation.

        Returns (c2, c1, c0) for: Z^3 + c2*Z^2 + c1*Z + c0 = 0.
        """
        aT = self._a(T)
        b = self._Omega_b
        R = self._R

        A = aT * p / (R * T) ** 2
        B = b * p / (R * T)

        c2 = -(1.0 + B - 2.0 * B)  # note: differs per EOS subclass can override
        # 通用形式: Z^3 - (1+B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0
        # 标准化: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        c2 = -(1.0 + B)
        c1 = A - 3.0 * B * B - 2.0 * B
        c0 = -(A * B - B * B - B * B * B)

        return c2, c1, c0

    def _solve_Z(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Solve cubic EOS for the compressibility factor Z.

        Uses the companion matrix eigenvalue approach for robustness.
        Returns the largest real root (vapor-like root).

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Z factor (dimensionless).
        """
        c2, c1, c0 = self._Z_coeffs(p, T)

        # Companion matrix approach for cubic: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        # Flatten for batch processing
        shape = c2.shape
        c2_f = c2.reshape(-1)
        c1_f = c1.reshape(-1)
        c0_f = c0.reshape(-1)
        n = c2_f.shape[0]

        # 构造 companion matrix 并求解特征值
        # [0, 0, -c0]
        # [1, 0, -c1]
        # [0, 1, -c2]
        batch_zeros = torch.zeros(n, dtype=c2_f.dtype, device=c2_f.device)
        batch_ones = torch.ones(n, dtype=c2_f.dtype, device=c2_f.device)

        matrices = torch.zeros(n, 3, 3, dtype=c2_f.dtype, device=c2_f.device)
        matrices[:, 0, 2] = -c0_f
        matrices[:, 1, 0] = batch_ones
        matrices[:, 1, 2] = -c1_f
        matrices[:, 2, 1] = batch_ones
        matrices[:, 2, 2] = -c2_f

        eigenvalues = torch.linalg.eigvals(matrices)  # (n, 3) complex

        # 选择最大实根 (气相根)
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        real_mask = imag_parts.abs() < 1e-10

        # 将非实部置为 -inf 以便 argmax
        masked = real_parts.clone()
        masked[~real_mask] = float('-inf')

        indices = masked.argmax(dim=1)
        Z = real_parts[torch.arange(n, device=c2_f.device), indices]

        # 兜底：若无有效实根，返回理想气体 Z=1
        invalid = Z < 1e-6
        Z = torch.where(invalid, torch.ones_like(Z), Z)

        return Z.reshape(shape)

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: rho = p * Mw / (Z * R_univ * T).

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        Z = self._solve_Z(p, T_safe)
        # rho = p / (Z * R_specific * T)
        return p / (Z * self._R * T_safe)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure from density and temperature (iterative).

        Uses Newton iteration on the cubic EOS in terms of V = 1/rho.

        Args:
            rho: Density (kg/m³).
            T: Temperature (K).

        Returns:
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

        # 比体积 V = 1/rho (m³/kg)
        V = 1.0 / rho_safe
        R = self._R
        aT = self._a(T_safe)
        b = self._Omega_b

        # p = RT/(V-b) - a(T)/(V^2 + 2bV - b^2)  (通用形式)
        # Van der Waals 形式更简单，但 Peng-Robinson 分母不同
        # 通用: p = RT/(V-b) - a(T)/D(V)
        # 这里使用 Z-factor 方式更稳健
        # p_initial (理想气体猜测)
        p_est = rho_safe * R * T_safe

        for _ in range(20):
            Z = self._solve_Z(p_est, T_safe)
            p_new = rho_safe * Z * R * T_safe
            # 阻尼更新
            p_est = 0.5 * p_est + 0.5 * p_new

        return p_est

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T (ideal part).

        Note: Departure enthalpy is not included in this simplified model.
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T (ideal part)."""
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T


class PengRobinsonEOS(CubicEOS):
    """Peng-Robinson cubic equation of state.

    .. math::

        p = \\frac{RT}{V-b} - \\frac{a\\alpha(T)}{V^2 + 2bV - b^2}

    with:

    .. math::

        \\alpha(T) = [1 + m(1 - \\sqrt{T/T_c})]^2

        m = 0.37464 + 1.54226\\omega - 0.26992\\omega^2

    where :math:`\\omega` is the acentric factor.

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)).
    accentric : float
        Acentric factor (default 0.0).

    Examples::

        # CO2: Mw=44, Tc=304.13, Pc=7.377e6, omega=0.228
        co2 = PengRobinsonEOS(Mw=44.0, Tc=304.13, Pc=7.377e6,
                               Cp=846.0, accentric=0.228)
        rho = co2.rho(p=1e6, T=300.0)
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
    ) -> None:
        # PR constants
        Zc = 0.3074
        omega_a = 0.45724
        omega_b = 0.07780

        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp,
            Zc=Zc, omega_a=omega_a, omega_b=omega_b,
            accentric=accentric,
        )

        # m 参数
        self._m = 0.37464 + 1.54226 * accentric - 0.26992 * accentric ** 2

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """PR alpha: [1 + m*(1 - sqrt(T/Tc))]^2."""
        T_ratio = T / self._Tc
        sqrt_T_ratio = T_ratio.clamp(min=1e-10).sqrt()
        factor = 1.0 + self._m * (1.0 - sqrt_T_ratio)
        return factor * factor

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """PR-specific Z-factor cubic coefficients.

        Z^3 - (1-B)Z^2 + (A - 2B - 3B^2)Z - (AB - B^2 - B^3) = 0
        """
        aT = self._a(T)
        b = self._Omega_b
        R = self._R

        A = aT * p / (R * T) ** 2
        B = b * p / (R * T)

        c2 = -(1.0 - B)
        c1 = A - 2.0 * B - 3.0 * B * B
        c0 = -(A * B - B * B - B * B * B)

        return c2, c1, c0

    def __repr__(self) -> str:
        return (
            f"PengRobinsonEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )


class RedlichKwongEOS(CubicEOS):
    """Redlich-Kwong cubic equation of state.

    .. math::

        p = \\frac{RT}{V-b} - \\frac{a}{T^{1/2} V(V+b)}

    with critical constants:

    .. math::

        a = 0.42748 \\frac{R^2 T_c^{2.5}}{P_c}

        b = 0.08664 \\frac{R T_c}{P_c}

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)).

    Examples::

        # Methane: Mw=16, Tc=190.56, Pc=4.599e6
        ch4 = RedlichKwongEOS(Mw=16.0, Tc=190.56, Pc=4.599e6, Cp=2220.0)
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
    ) -> None:
        Zc = 1.0 / 3.0
        omega_a = 0.42748
        omega_b = 0.08664

        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp,
            Zc=Zc, omega_a=omega_a, omega_b=omega_b,
            accentric=0.0,
        )

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """RK alpha: Tc^0.5 / T^0.5 = (Tc/T)^0.5."""
        return (self._Tc / T.clamp(min=1e-10)).sqrt()

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """RK-specific Z-factor cubic coefficients.

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
            f"RedlichKwongEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f})"
        )


class VanDerWaalsEOS(CubicEOS):
    """Van der Waals cubic equation of state.

    .. math::

        p = \\frac{RT}{V-b} - \\frac{a}{V^2}

    with critical constants:

    .. math::

        a = \\frac{27}{64} \\frac{R^2 T_c^2}{P_c}

        b = \\frac{1}{8} \\frac{R T_c}{P_c}

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)).

    Examples::

        # Nitrogen: Mw=28, Tc=126.2, Pc=3.39e6
        n2 = VanDerWaalsEOS(Mw=28.0, Tc=126.2, Pc=3.39e6, Cp=1040.0)
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
    ) -> None:
        Zc = 3.0 / 8.0
        omega_a = 27.0 / 64.0
        omega_b = 1.0 / 8.0

        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp,
            Zc=Zc, omega_a=omega_a, omega_b=omega_b,
            accentric=0.0,
        )

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """VDW alpha: constant (=1), no temperature dependence."""
        return torch.ones_like(T)

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """VDW-specific Z-factor cubic coefficients.

        Z^3 - (b*p/RT + 1)Z^2 + (a*p/(R^2T^2))Z - abp^2/(R^3T^3) = 0
        """
        aT = self._a(T)
        b = self._Omega_b
        R = self._R

        A = aT * p / (R * R * T * T)
        B = b * p / (R * T)

        c2 = -(B + 1.0)
        c1 = A
        c0 = -A * B

        return c2, c1, c0

    def __repr__(self) -> str:
        return (
            f"VanDerWaalsEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f})"
        )


class IcoTabulatedEOS(EquationOfState):
    """Incompressible tabulated EOS: density interpolated from (p, T) table.

    Uses bilinear interpolation to compute density from a tabulated
    dataset of pressure, temperature, and density values, as used in
    OpenFOAM's ``icoTabulated`` EOS.

    Parameters
    ----------
    p_data : sequence of float
        Pressure values (Pa) — strictly increasing.
    T_data : sequence of float
        Temperature values (K) — strictly increasing.
    rho_data : sequence of sequence of float
        2D density table ``rho_data[i][j]`` indexed by
        ``(p_index=i, T_index=j)``.
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0.
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0.

    Examples::

        import numpy as np
        p_vals = [1e5, 2e5, 3e5]
        T_vals = [300, 400, 500]
        rho_table = [[1.1, 0.85, 0.68],
                     [2.2, 1.7, 1.36],
                     [3.3, 2.55, 2.04]]
        eos = IcoTabulatedEOS(p_data=p_vals, T_data=T_vals,
                               rho_data=rho_table)
        rho = eos.rho(p=1.5e5, T=350.0)  # interpolated
    """

    def __init__(
        self,
        p_data: Sequence[float],
        T_data: Sequence[float],
        rho_data: Sequence[Sequence[float]],
        R: float = 287.0,
        Cp: float = 1005.0,
    ) -> None:
        if len(p_data) < 2:
            raise ValueError("p_data must have at least 2 points")
        if len(T_data) < 2:
            raise ValueError("T_data must have at least 2 points")
        if len(rho_data) != len(p_data):
            raise ValueError(
                f"rho_data rows ({len(rho_data)}) must match p_data length ({len(p_data)})"
            )
        for i, row in enumerate(rho_data):
            if len(row) != len(T_data):
                raise ValueError(
                    f"rho_data row {i} length ({len(row)}) must match T_data length ({len(T_data)})"
                )

        self._p_data = list(p_data)
        self._T_data = list(T_data)
        self._rho_data = [list(row) for row in rho_data]
        self._R = R
        self._Cp = Cp
        self._Cv = Cp - R
        self._gamma = Cp / self._Cv if self._Cv > 0 else float('inf')

    def _interp_bilinear(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinear interpolation of rho(p, T) from tabulated data."""
        device = p.device
        dtype = p.dtype

        p_arr = torch.tensor(self._p_data, dtype=dtype, device=device)
        T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
        rho_arr = torch.tensor(
            self._rho_data, dtype=dtype, device=device
        )  # shape (np, nT)

        # 裁剪到数据范围
        p_clamped = p.clamp(min=float(p_arr[0]), max=float(p_arr[-1]))
        T_clamped = T.clamp(min=float(T_arr[0]), max=float(T_arr[-1]))

        # 归一化索引 (连续)
        # 找到 p 在 p_arr 中的插值位置
        # 使用 searchsorted 获取索引
        p_idx = torch.searchsorted(p_arr, p_clamped).clamp(
            min=1, max=len(self._p_data) - 1
        )
        T_idx = torch.searchsorted(T_arr, T_clamped).clamp(
            min=1, max=len(self._T_data) - 1
        )

        p_lo = p_arr[p_idx - 1]
        p_hi = p_arr[p_idx]
        T_lo = T_arr[T_idx - 1]
        T_hi = T_arr[T_idx]

        # 插值权重
        t_p = ((p_clamped - p_lo) / (p_hi - p_lo + 1e-30)).clamp(0, 1)
        t_T = ((T_clamped - T_lo) / (T_hi - T_lo + 1e-30)).clamp(0, 1)

        # 四角插值
        rho_00 = rho_arr[p_idx - 1, T_idx - 1]
        rho_01 = rho_arr[p_idx - 1, T_idx]
        rho_10 = rho_arr[p_idx, T_idx - 1]
        rho_11 = rho_arr[p_idx, T_idx]

        rho_interp = (
            (1 - t_p) * (1 - t_T) * rho_00
            + (1 - t_p) * t_T * rho_01
            + t_p * (1 - t_T) * rho_10
            + t_p * t_T * rho_11
        )

        return rho_interp

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Interpolate density from tabulated data.

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Density (kg/m³) — interpolated.
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        return self._interp_bilinear(p, T)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure — approximate inverse of tabulated rho(p, T).

        Uses bisection on the tabulated data to find p such that
        rho(p, T) ≈ rho_input. Falls back to p_ref if no solution found.

        Args:
            rho: Density (kg/m³).
            T: Temperature (K).

        Returns:
            Pressure (Pa) — approximate.
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(rho, torch.Tensor):
            rho = torch.tensor(rho, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        # 使用二分法查找压力
        p_lo = torch.full_like(rho, self._p_data[0])
        p_hi = torch.full_like(rho, self._p_data[-1])

        for _ in range(40):
            p_mid = 0.5 * (p_lo + p_hi)
            rho_mid = self._interp_bilinear(p_mid, T)
            # 密度随压力递增
            mask = rho_mid < rho
            p_lo = torch.where(mask, p_mid, p_lo)
            p_hi = torch.where(mask, p_hi, p_mid)

        return 0.5 * (p_lo + p_hi)

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T."""
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T."""
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def __repr__(self) -> str:
        np = len(self._p_data)
        nT = len(self._T_data)
        return (
            f"IcoTabulatedEOS(p_range=[{self._p_data[0]:.0f}, {self._p_data[-1]:.0f}], "
            f"T_range=[{self._T_data[0]:.0f}, {self._T_data[-1]:.0f}], "
            f"grid={np}x{nT})"
        )
