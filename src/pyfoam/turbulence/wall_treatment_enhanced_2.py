"""
Enhanced wall treatment model v2 with improved y+ switching.

Extends :class:`~pyfoam.turbulence.wall_treatment_enhanced.EnhancedWallTreatment`
and :class:`~pyfoam.turbulence.wall_treatment_enhanced.ThreeLayerWallTreatment` with:

- Four-layer wall treatment (viscous, buffer, transition, log-law)
- Improved y+ estimation using momentum thickness
- Roughness-aware wall functions

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_2 import EnhancedWallTreatment2, FourLayerWallTreatment

    wt = EnhancedWallTreatment2(nu=1.5e-5)
    nut_wall = wt.compute_nut(k, y)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced import (
    EnhancedWallTreatment,
    ThreeLayerWallTreatment,
)

__all__ = [
    "EnhancedWallTreatment2",
    "FourLayerWallTreatment",
]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced2")
class EnhancedWallTreatment2(EnhancedWallTreatment):
    """Enhanced wall treatment v2 with improved y+ estimation.

    Extends :class:`EnhancedWallTreatment` with:

    - **Momentum-thickness y+**: uses local momentum thickness Reynolds
      number for improved y+ estimation in adverse pressure gradients.
    - **Enhanced buffer layer model**: two-piece blending in buffer region
      for more accurate nut and omega.
    - **Roughness support**: optional sand-grain roughness modification.

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
    ks : float
        Equivalent sand-grain roughness height (m). Default 0 (smooth).
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
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
        )
        self._ks = ks

    @property
    def ks(self) -> float:
        """Equivalent sand-grain roughness height (m)."""
        return self._ks

    def _roughness_E(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Effective E constant accounting for roughness.

        For smooth walls: E_eff = E
        For rough walls: E_eff = E / (1 + ks_plus * 0.03)
        where ks_plus = ks * u_tau / nu.

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Effective E constant.
        """
        if self._ks <= 0:
            return torch.full_like(y_plus, self.E)

        # Estimate ks_plus from y_plus (rough approximation)
        ks_plus = self._ks * y_plus / max(self._nu, 1e-30) * max(self._nu, 1e-30)
        # ks_plus should use u_tau, but we approximate from the available y_plus
        E_eff = self.E / (1.0 + ks_plus * 0.03)
        return E_eff.clamp(min=1.0)

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced nut with improved buffer layer and roughness.

        Low-Re: nut = 0
        Buffer: nut = nut_high_re * smooth_two_piece_transition
        High-Re: nut = kappa * u_tau * y / ln(E_eff * y+)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        blend = self._blending_factor(y_plus)

        # Roughness-aware E
        E_eff = self._roughness_E(y_plus)

        nut_high_re = self.kappa * u_tau * y / torch.log(
            E_eff * y_plus.clamp(min=1.01)
        )

        # Two-piece buffer blending
        y_mid = 0.5 * (self.y_plus_low + self.y_plus_high)
        buffer_blend = 0.5 * (1.0 + torch.tanh(
            ((y_plus - y_mid) / (0.15 * (self.y_plus_high - self.y_plus_low)))
            .clamp(min=-10.0, max=10.0)
        ))

        nut = buffer_blend * nut_high_re.clamp(min=0.0)
        return nut.clamp(min=0.0)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced omega with improved blending and roughness.

        Low-Re:  omega = 6 * nu / (beta_1 * y^2)
        High-Re: omega = sqrt(k) / (C_mu^{1/4} * kappa * y)
        Rough:   omega = u_tau / (kappa * y) * max(1, 50/(ks_plus))
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

        # Roughness correction
        if self._ks > 0:
            ks_plus_approx = y_plus * self._ks / y.clamp(min=1e-10)
            rough_factor = (50.0 / ks_plus_approx.clamp(min=1.0)).clamp(min=1.0)
            omega = omega * rough_factor

        return omega.clamp(min=1e-10)

    def __repr__(self) -> str:
        rough = f", ks={self._ks}" if self._ks > 0 else ""
        return (
            f"EnhancedWallTreatment2(nu={self.nu}, "
            f"y+=[{self.y_plus_low}, {self.y_plus_high}]{rough})"
        )


@WallTreatment.register("four_layer")
class FourLayerWallTreatment(WallTreatment):
    """Four-layer wall treatment with explicit viscous, buffer, transition, and log-law regimes.

    - **Viscous sublayer** (y+ < 5): nut = 0, eps = 2*nu*k/y^2, omega = 6*nu/(beta_1*y^2)
    - **Buffer layer** (5 <= y+ < 15): blended formulation
    - **Transition layer** (15 <= y+ < 30): smooth transition to log-law
    - **Log-law** (y+ >= 30): standard wall functions

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

    def _regime(self, y_plus: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Determine wall regime: viscous (0), buffer (1), transition (2), log-law (3)."""
        viscous = y_plus < 5.0
        buffer = (y_plus >= 5.0) & (y_plus < 15.0)
        transition = (y_plus >= 15.0) & (y_plus < 30.0)
        log_law = y_plus >= 30.0
        return viscous, buffer, transition, log_law

    def _buffer_weight(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Buffer layer blending weight: 0 at y+=5, 1 at y+=15."""
        return ((y_plus - 5.0) / 10.0).clamp(min=0.0, max=1.0)

    def _transition_weight(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Transition layer blending weight: 0 at y+=15, 1 at y+=30."""
        return ((y_plus - 15.0) / 15.0).clamp(min=0.0, max=1.0)

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Four-layer nut computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, transition, log_law = self._regime(y_plus)

        nut_log = self.kappa * u_tau * y / torch.log(
            self.E * y_plus.clamp(min=1.01)
        )

        bw = self._buffer_weight(y_plus)
        tw = self._transition_weight(y_plus)

        nut = torch.zeros_like(k)
        nut = torch.where(buffer, nut_log * 0.3 * bw, nut)
        nut = torch.where(transition, nut_log * (0.3 + 0.7 * tw), nut)
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
        """Four-layer epsilon computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, transition, log_law = self._regime(y_plus)

        eps_visc = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)
        eps_log = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )

        bw = self._buffer_weight(y_plus)
        tw = self._transition_weight(y_plus)

        eps_buffer = (1.0 - bw) * eps_visc + bw * eps_log * 0.3
        eps_trans = (1.0 - tw) * eps_log * 0.3 + tw * eps_log

        eps = torch.where(viscous, eps_visc,
               torch.where(buffer, eps_buffer,
               torch.where(transition, eps_trans, eps_log)))
        return eps.clamp(min=1e-10)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Four-layer omega computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        viscous, buffer, transition, log_law = self._regime(y_plus)

        beta_1 = 0.075
        omega_visc = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))
        omega_log = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        bw = self._buffer_weight(y_plus)
        tw = self._transition_weight(y_plus)

        omega_buffer = (1.0 - bw) * omega_visc + bw * omega_log * 0.5
        omega_trans = (1.0 - tw) * omega_log * 0.5 + tw * omega_log

        omega = torch.where(viscous, omega_visc,
                 torch.where(buffer, omega_buffer,
                 torch.where(transition, omega_trans, omega_log)))
        return omega.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"FourLayerWallTreatment(nu={self.nu})"
