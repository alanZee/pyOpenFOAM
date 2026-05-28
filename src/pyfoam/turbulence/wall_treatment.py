"""
Wall treatment models for turbulence boundary conditions.

Provides wall-treatment strategies that bridge the viscous sublayer
and log-law region. In OpenFOAM, these correspond to the wall treatment
settings in the turbulence model dictionary.

Models:

- :class:`WallTreatment` — abstract base for wall treatment strategies
- :class:`StandardWallTreatment` — standard wall function approach
  (nutkWallFunction + kqRWallFunction + epsilonWallFunction/omegaWallFunction)
- :class:`AutomaticWallTreatment` — automatic y+ switching that blends
  between low-Re and high-Re formulations

The standard wall treatment assumes the wall-adjacent cell is in the
log-law region (y+ > 30) and applies analytical wall functions.

The automatic wall treatment detects the local y+ and blends between:
- Low-Re formulation (y+ < 5): viscous sublayer resolved, nut=0
- Blended region (5 < y+ < 30): smooth blending
- High-Re formulation (y+ > 30): standard wall functions

Usage::

    from pyfoam.turbulence.wall_treatment import (
        StandardWallTreatment,
        AutomaticWallTreatment,
    )

    wt = StandardWallTreatment(nu=1.5e-5)
    nut_wall = wt.compute_nut(k, y)
    k_wall = wt.compute_k(u_tau)
    epsilon_wall = wt.compute_epsilon(k, y)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "WallTreatment",
    "StandardWallTreatment",
    "AutomaticWallTreatment",
]

logger = logging.getLogger(__name__)

# Physical constants
_KAPPA: float = 0.41  # Von Karman constant
_E: float = 9.8       # Log-law constant
_C_MU: float = 0.09   # k-epsilon model constant


class WallTreatment(ABC):
    """Abstract base class for turbulence wall treatment.

    Subclasses provide methods to compute wall values for nut, k,
    epsilon, and omega based on the local y+ and near-wall quantities.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @WallTreatment.register("standard")
        class StandardWallTreatment(WallTreatment):
            ...

        wt = WallTreatment.create("standard", nu=1.5e-5)
    """

    _registry: ClassVar[dict[str, Type["WallTreatment"]]] = {}

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
    ) -> None:
        """
        Parameters
        ----------
        nu : float
            Molecular kinematic viscosity (m²/s).
        kappa : float
            Von Karman constant.
        E : float
            Log-law wall constant.
        C_mu : float
            k-epsilon model constant.
        """
        self.nu = nu
        self.kappa = kappa
        self.E = E
        self.C_mu = C_mu

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a wall treatment class under *name*."""

        def decorator(treatment_cls: Type[WallTreatment]) -> Type[WallTreatment]:
            if name in cls._registry:
                raise ValueError(
                    f"Wall treatment '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = treatment_cls
            return treatment_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "WallTreatment":
        """Factory: create a wall treatment by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown wall treatment '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered wall treatment type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent viscosity at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy at wall-adjacent cells ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance from cell centre to wall face ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            nu_t at each wall face ``(n_faces,)``.
        """

    @abstractmethod
    def compute_k(
        self,
        u_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent kinetic energy at wall faces.

        Parameters
        ----------
        u_tau : torch.Tensor
            Friction velocity at wall faces ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            k at each wall face ``(n_faces,)``.
        """

    @abstractmethod
    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dissipation rate at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            epsilon at each wall face ``(n_faces,)``.
        """

    @abstractmethod
    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute specific dissipation rate at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            omega at each wall face ``(n_faces,)``.
        """

    def compute_u_tau(
        self,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute friction velocity from k.

        u_tau = C_mu^{1/4} * sqrt(k)

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            Friction velocity ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        return self.C_mu ** 0.25 * torch.sqrt(k.clamp(min=1e-16))

    def compute_y_plus(
        self,
        u_tau: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute y+ = u_tau * y / nu.

        Parameters
        ----------
        u_tau : torch.Tensor
            Friction velocity ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            y+ at each wall face ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        return (u_tau * y / max(self.nu, 1e-30)).clamp(min=1e-4)


@WallTreatment.register("standard")
class StandardWallTreatment(WallTreatment):
    """Standard wall function treatment (high-Re).

    Assumes the wall-adjacent cell is in the log-law region (y+ > 30)
    and applies standard analytical wall functions:

    - nut = kappa * u_tau * y / ln(E * y+)
    - k = u_tau^2 / sqrt(C_mu)
    - epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)
    - omega = sqrt(k) / (C_mu^{1/4} * kappa * y)

    This is the classical approach used in most industrial CFD.
    """

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """nut = kappa * u_tau * y / ln(E * y+)."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)

        nut = self.kappa * u_tau * y / torch.log(self.E * y_plus)
        return nut.clamp(min=0.0)

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
        """epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        eps = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )
        return eps.clamp(min=1e-10)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """omega = sqrt(k) / (C_mu^{1/4} * kappa * y)."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        omega = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )
        return omega.clamp(min=1e-10)


@WallTreatment.register("automatic")
class AutomaticWallTreatment(WallTreatment):
    """Automatic y+ wall treatment with blending.

    Detects the local y+ and blends between:

    - Low-Re (y+ < 5): viscous sublayer resolved, nut -> 0
    - Blended (5 <= y+ <= 30): smooth transition
    - High-Re (y+ > 30): standard wall functions

    The blending ensures smooth behavior across the y+ transition,
    which is essential for hybrid meshes and flows with varying y+.

    Parameters
    ----------
    y_plus_low : float
        Upper bound of low-Re region (default 5).
    y_plus_high : float
        Lower bound of high-Re region (default 30).
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
        super().__init__(nu=nu, kappa=kappa, E=E, C_mu=C_mu)
        self.y_plus_low = y_plus_low
        self.y_plus_high = y_plus_high

    def _blending_factor(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Compute blending factor: 0 = low-Re, 1 = high-Re.

        Uses a smooth cubic Hermite interpolation:
            t = (y+ - y_low) / (y_high - y_low)   clamped to [0, 1]
            blend = 3t² - 2t³
        """
        t = (y_plus - self.y_plus_low) / max(
            self.y_plus_high - self.y_plus_low, 1e-10
        )
        t = t.clamp(0.0, 1.0)
        return 3.0 * t.pow(2) - 2.0 * t.pow(3)

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Blended nut: low-Re (nut=0) and high-Re (wall function).

        nut = blend * nut_high_re
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

        # Low-Re: nut = 0 (resolved viscous sublayer)
        nut = blend * nut_high_re.clamp(min=0.0)
        return nut.clamp(min=0.0)

    def compute_k(self, u_tau: torch.Tensor) -> torch.Tensor:
        """k = u_tau^2 / sqrt(C_mu) (same for all y+ regions)."""
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        return u_tau.pow(2) / math.sqrt(self.C_mu)

    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Blended epsilon: low-Re and high-Re formulations.

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

        # Low-Re: eps = 2 * nu * k / y^2
        eps_low_re = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)

        # High-Re: eps = C_mu^{3/4} * k^{3/2} / (kappa * y)
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
        """Blended omega: low-Re and high-Re formulations.

        Low-Re:  omega = 6 * nu / (beta_1 * y^2)   with beta_1 = 0.075
        High-Re: omega = sqrt(k) / (C_mu^{1/4} * kappa * y)
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        blend = self._blending_factor(y_plus)

        # Low-Re: omega = 6 * nu / (beta_1 * y^2)
        beta_1 = 0.075
        omega_low_re = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))

        # High-Re: omega = sqrt(k) / (C_mu^{1/4} * kappa * y)
        omega_high_re = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        omega = (1.0 - blend) * omega_low_re + blend * omega_high_re
        return omega.clamp(min=1e-10)
