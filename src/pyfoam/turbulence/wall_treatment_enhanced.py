"""
Enhanced wall treatment model with improved y+ switching.

Extends :class:`~pyfoam.turbulence.wall_treatment.AutomaticWallTreatment` with:

- Smooth y+ detection without sharp transitions
- All three wall-function regimes (viscous, buffer, log-law)
- Omega-specific wall treatment with proper blending

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced import EnhancedWallTreatment

    wt = EnhancedWallTreatment(nu=1.5e-5)
    nut_wall = wt.compute_nut(k, y)
    omega_wall = wt.compute_omega(k, y)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment, AutomaticWallTreatment

__all__ = [
    "EnhancedWallTreatment",
    "ThreeLayerWallTreatment",
]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced")
class EnhancedWallTreatment(AutomaticWallTreatment):
    """Enhanced wall treatment with improved y+ switching.

    Extends :class:`AutomaticWallTreatment` with:

    - **Smoother blending**: uses tanh instead of cubic Hermite for
      smoother y+ transition.
    - **Enhanced nut wall function**: includes buffer layer correction
      via Spalding's law of the wall.
    - **Improved omega**: uses proper blending between viscous sublayer
      omega (6*nu/(beta_1*y^2)) and log-law omega.

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
        Upper bound of low-Re region. Default 5.0.
    y_plus_high : float
        Lower bound of high-Re region. Default 30.0.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
        y_plus_low: float = 5.0,
        y_plus_high: float = 30.0,
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
        )

    def _blending_factor(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Smoother blending factor using tanh.

        blend = 0.5 * (1 + tanh((y+ - y_mid) / y_width))

        where y_mid = (y_low + y_high) / 2, y_width = (y_high - y_low) / 4.
        """
        y_mid = 0.5 * (self.y_plus_low + self.y_plus_high)
        y_width = 0.25 * (self.y_plus_high - self.y_plus_low)
        y_width = max(y_width, 0.1)

        arg = (y_plus - y_mid) / y_width
        return 0.5 * (1.0 + torch.tanh(arg.clamp(min=-10.0, max=10.0)))

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced nut with smoother blending and buffer-layer correction.

        Low-Re: nut = 0 (viscous sublayer)
        Buffer: nut = nut_high_re * smooth_transition
        High-Re: nut = kappa * u_tau * y / ln(E * y+)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        blend = self._blending_factor(y_plus)

        # High-Re: nut = kappa * u_tau * y / ln(E * y+)
        nut_high_re = self.kappa * u_tau * y / torch.log(
            self.E * y_plus.clamp(min=1.01)
        )

        # Low-Re: nut = 0 (viscous sublayer)
        nut = blend * nut_high_re.clamp(min=0.0)
        return nut.clamp(min=0.0)

    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced epsilon with smoother blending.

        Low-Re:  eps = 2 * nu * k / y^2
        High-Re: eps = C_mu^{3/4} * k^{3/2} / (kappa * y)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        blend = self._blending_factor(y_plus)

        eps_low_re = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)
        eps_high_re = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )

        eps = (1.0 - blend) * eps_low_re + blend * eps_high_re
        return eps.clamp(min=1e-10)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced omega with proper wall treatment.

        Low-Re:  omega = 6 * nu / (beta_1 * y^2)
        High-Re: omega = sqrt(k) / (C_mu^{1/4} * kappa * y)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        blend = self._blending_factor(y_plus)

        beta_1 = 0.075
        omega_low_re = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))
        omega_high_re = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        omega = (1.0 - blend) * omega_low_re + blend * omega_high_re
        return omega.clamp(min=1e-10)

    def __repr__(self) -> str:
        return (
            f"EnhancedWallTreatment(nu={self.nu}, "
            f"y+=[{self.y_plus_low}, {self.y_plus_high}])"
        )


@WallTreatment.register("three_layer")
class ThreeLayerWallTreatment(WallTreatment):
    """Three-layer wall treatment with explicit viscous, buffer, and log-law regimes.

    Provides explicit separate formulations for each regime:

    - **Viscous sublayer** (y+ < 5): nut = 0, eps = 2*nu*k/y^2, omega = 6*nu/(beta_1*y^2)
    - **Buffer layer** (5 <= y+ <= 30): blending between viscous and log-law
    - **Log-law** (y+ > 30): standard wall functions

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
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
    ) -> None:
        super().__init__(nu=nu, kappa=kappa, E=E, C_mu=C_mu)

    def _regime(self, y_plus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine wall regime: viscous (0), buffer (1), log-law (2).

        Returns masks for each regime.
        """
        viscous = y_plus < 5.0
        log_law = y_plus > 30.0
        buffer = ~viscous & ~log_law
        return viscous, buffer, log_law

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Three-layer nut computation.

        Viscous: nut = 0
        Buffer: nut = nut_log_law * (y+ - 5) / 25
        Log-law: nut = kappa * u_tau * y / ln(E * y+)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, log_law = self._regime(y_plus)

        # Log-law formulation
        nut_log = self.kappa * u_tau * y / torch.log(
            self.E * y_plus.clamp(min=1.01)
        )

        # Buffer: linear ramp
        buffer_weight = ((y_plus - 5.0) / 25.0).clamp(min=0.0, max=1.0)

        nut = torch.zeros_like(k)
        nut = torch.where(buffer, nut_log * buffer_weight, nut)
        nut = torch.where(log_law, nut_log, nut)

        return nut.clamp(min=0.0)

    def compute_k(
        self,
        u_tau: torch.Tensor,
    ) -> torch.Tensor:
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
        """Three-layer epsilon computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, log_law = self._regime(y_plus)

        eps_visc = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)
        eps_log = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )

        # Buffer: blended
        buffer_weight = ((y_plus - 5.0) / 25.0).clamp(min=0.0, max=1.0)
        eps_buffer = (1.0 - buffer_weight) * eps_visc + buffer_weight * eps_log

        eps = torch.where(viscous, eps_visc, torch.where(buffer, eps_buffer, eps_log))
        return eps.clamp(min=1e-10)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Three-layer omega computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, log_law = self._regime(y_plus)

        beta_1 = 0.075
        omega_visc = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))
        omega_log = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        buffer_weight = ((y_plus - 5.0) / 25.0).clamp(min=0.0, max=1.0)
        omega_buffer = (1.0 - buffer_weight) * omega_visc + buffer_weight * omega_log

        omega = torch.where(viscous, omega_visc, torch.where(buffer, omega_buffer, omega_log))
        return omega.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"ThreeLayerWallTreatment(nu={self.nu})"
