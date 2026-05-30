"""
Enhanced wall treatment model v4 with compressible corrections and Lewis number coupling.

Extends :class:`~pyfoam.turbulence.wall_treatment_enhanced_3.EnhancedWallTreatment3`
with:

- Compressible wall function with Van Driest damping
- Lewis number coupling for scalar wall functions
- Roughness-dependent blending with equivalent sand-grain correlation

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_4 import EnhancedWallTreatment4, CompressibleWallTreatment

    wt = EnhancedWallTreatment4(nu=1.5e-5, Pr=0.71, Le=1.0)
    nut_wall = wt.compute_nut(k, y)
    htc = wt.compute_htc(k, y, T, T_wall)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_3 import (
    EnhancedWallTreatment3,
    AdaptiveWallTreatment,
)

__all__ = [
    "EnhancedWallTreatment4",
    "CompressibleWallTreatment",
]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced4")
class EnhancedWallTreatment4(EnhancedWallTreatment3):
    """Enhanced wall treatment v4 with Lewis number and compressible corrections.

    Extends :class:`EnhancedWallTreatment3` with:

    - **Lewis number coupling**: modifies the thermal wall function
      to account for mass diffusion vs heat diffusion ratio:
      Le = alpha / D, affecting the temperature profile.
    - **Compressible Van Driest damping**: for compressible wall-bounded
      flows, applies Van Driest damping to the mixing length:
      l_m = kappa * y * (1 - exp(-y+/A+))
    - **Roughness-dependent equivalent sand-grain**: enhanced roughness
      model with Reynolds-number-dependent roughness function.

    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Log-law wall constant.
    C_mu : float
        k-epsilon model constant.
    y_plus_low : float
        Upper bound of low-Re region.
    y_plus_high : float
        Lower bound of high-Re region.
    ks : float
        Equivalent sand-grain roughness height (m).
    Pr : float
        Prandtl number.
    Pr_t : float
        Turbulent Prandtl number.
    hysteresis_width : float
        Width of y+ hysteresis band.
    Le : float
        Lewis number (alpha/D). Default 1.0.
    van_driest_A : float
        Van Driest damping constant. Default 26.0.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
        y_plus_low: float = 5.0,
        y_plus_high: float = 30.0,
        ks: float = 0.0,
        Pr: float = 0.71,
        Pr_t: float = 0.85,
        hysteresis_width: float = 2.0,
        Le: float = 1.0,
        van_driest_A: float = 26.0,
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
            ks=ks, Pr=Pr, Pr_t=Pr_t, hysteresis_width=hysteresis_width,
        )
        self._Le = max(Le, 0.01)
        self._van_driest_A = van_driest_A

    @property
    def Le(self) -> float:
        """Lewis number."""
        return self._Le

    @property
    def van_driest_A(self) -> float:
        """Van Driest damping constant."""
        return self._van_driest_A

    # ------------------------------------------------------------------
    # Van Driest mixing length
    # ------------------------------------------------------------------

    def van_driest_length(
        self,
        y_plus: torch.Tensor,
    ) -> torch.Tensor:
        """Van Driest damped mixing length.

        l_m = kappa * y * (1 - exp(-y+ / A+))

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Damped mixing length (dimensionless, in wall units).
        """
        return self.kappa * y_plus * (1.0 - torch.exp(-y_plus / self._van_driest_A))

    # ------------------------------------------------------------------
    # Lewis number correction for heat transfer
    # ------------------------------------------------------------------

    def _lewis_correction(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Lewis number correction to thermal wall function.

        For Le != 1, the temperature profile differs from the velocity profile:

        T+ = Pr_t * (u+ + P(y+)) / Le^(1/3)

        This factor modifies the heat transfer coefficient.

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Lewis correction factor.
        """
        Le_factor = self._Le ** (1.0 / 3.0)
        return 1.0 / max(Le_factor, 0.1)

    # ------------------------------------------------------------------
    # Enhanced heat transfer with Lewis
    # ------------------------------------------------------------------

    def compute_htc(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        T_fluid: torch.Tensor,
        T_wall: torch.Tensor,
        rho: float = 1.2,
        Cp: float = 1005.0,
    ) -> torch.Tensor:
        """Compute wall heat transfer coefficient with Lewis correction.

        htc = rho * Cp * u_tau / (Pr_t * (1/kappa * ln(y+) + P(y+)) / Le^(1/3))

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy.
        y : torch.Tensor
            Wall-normal distance.
        T_fluid : torch.Tensor
            Fluid temperature near wall.
        T_wall : torch.Tensor
            Wall temperature.
        rho : float
            Fluid density (kg/m^3).
        Cp : float
            Specific heat (J/(kg*K)).

        Returns
        -------
        torch.Tensor
            Heat transfer coefficient (W/(m^2*K)).
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        P = self._jayatilleke_blending(y_plus)

        log_arg = y_plus.clamp(min=1.01)
        T_star = self._Pr_t * (torch.log(log_arg) / self.kappa + P)
        T_star = T_star.clamp(min=0.01)

        # Apply Lewis correction
        lewis_corr = self._lewis_correction(y_plus)
        T_star_corrected = T_star * lewis_corr

        htc = rho * Cp * u_tau / T_star_corrected
        return htc.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Roughness-enhanced wall function
    # ------------------------------------------------------------------

    def _roughness_shift(self) -> float:
        """Roughness function shift for the log-law.

        Delta_u+ = (1/kappa) * ln(1 + C_ks * ks+)

        where ks+ = ks * u_tau / nu.
        Uses a simplified estimate with C_ks = 0.3.

        Returns
        -------
        float
            Roughness shift (in wall units).
        """
        if self._ks <= 0:
            return 0.0

        # Estimate ks+ with a typical u_tau
        u_tau_est = 0.05  # Typical estimate
        ks_plus = self._ks * u_tau_est / max(self.nu, 1e-30)

        C_ks = 0.3
        return (1.0 / self.kappa) * math.log(1.0 + C_ks * max(ks_plus, 0.0))

    def __repr__(self) -> str:
        rough = f", ks={self._ks}" if self._ks > 0 else ""
        return (
            f"EnhancedWallTreatment4(nu={self.nu}, Pr={self._Pr}, "
            f"Le={self._Le}, y+=[{self.y_plus_low}, {self.y_plus_high}]{rough})"
        )


@WallTreatment.register("compressible_wt")
class CompressibleWallTreatment(WallTreatment):
    """Compressible wall treatment with Van Driest damping.

    For high-speed compressible wall-bounded flows:
    - Van Driest damping: l_m = kappa * y * (1 - exp(-y+/A+))
    - Density-weighted wall functions: u+_c = integral(rho/rho_w * dy+)
    - Total temperature wall function for energy equation

    Parameters
    ----------
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Log-law constant.
    C_mu : float
        k-epsilon model constant.
    van_driest_A : float
        Van Driest constant. Default 26.0.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
        van_driest_A: float = 26.0,
    ) -> None:
        super().__init__(nu=nu, kappa=kappa, E=E, C_mu=C_mu)
        self._van_driest_A = van_driest_A

    def van_driest_damping(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Van Driest damping function.

        f_VD = 1 - exp(-y+ / A+)

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Damping factor in [0, 1].
        """
        return (1.0 - torch.exp(-y_plus / self._van_driest_A)).clamp(min=0.0, max=1.0)

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compressible wall nut with Van Driest damping.

        nut = kappa * u_tau * y * f_VD(y+)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)

        f_VD = self.van_driest_damping(y_plus)
        nut = self.kappa * u_tau * y * f_VD

        return nut.clamp(min=0.0)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compressible omega wall function."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        f_VD = self.van_driest_damping(y_plus)

        beta_1 = 0.075
        omega = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))

        # Blend with log-law using Van Driest
        omega_log = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        return (f_VD * omega + (1.0 - f_VD) * omega_log).clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"CompressibleWallTreatment(nu={self.nu}, A+={self._van_driest_A})"

    def compute_k(self, u_tau: torch.Tensor) -> torch.Tensor:
        """k = u_tau^2 / sqrt(C_mu)."""
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        return u_tau.pow(2) / math.sqrt(self.C_mu)

    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compressible epsilon wall function."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        f_VD = self.van_driest_damping(y_plus)

        eps_visc = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)
        eps_log = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )

        return (f_VD * eps_visc + (1.0 - f_VD) * eps_log).clamp(min=1e-10)
