"""
Compressibility corrections for turbulence models.

Provides corrections to turbulence model equations that account for
the effects of compressibility (variable density) on turbulent
transport.  These are critical for high-speed (transonic/supersonic)
flows where density fluctuations are significant.

Models:
- :class:`CompressibilityCorrection` — abstract base with RTS registry
- :class:`SarkarModel` — Sarkar compressibility correction
- :class:`ZemanModel` — Zeman compressibility correction

Both models modify the turbulence dissipation rate to account for
the dilatational dissipation that becomes important at high
turbulent Mach numbers.

References:
- Sarkar, S., Erlebacher, G., Hussaini, M.Y., & Kreiss, H.O.
  (1991). "The analysis and modelling of dilatational terms in
  compressible turbulence." J. Fluid Mech., 227, 473-493.
- Zeman, O. (1990). "Dilatation dissipation: The concept and
  application in modeling compressible mixing layers." Phys. Fluids
  A, 2(2), 178-188.

Usage::

    from pyfoam.turbulence.compressibility_corrections import CompressibilityCorrection

    model = CompressibilityCorrection.create("Sarkar")
    eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "CompressibilityCorrection",
    "SarkarModel",
    "ZemanModel",
]


class CompressibilityCorrection(ABC):
    """Abstract base class for compressibility corrections to turbulence.

    Subclasses must implement :meth:`correct_dissipation`.

    RTS (Run-Time Selection) registry::

        model = CompressibilityCorrection.create("Sarkar")
    """

    _registry: ClassVar[dict[str, Type[CompressibilityCorrection]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a compressibility correction under *name*."""

        def decorator(
            model_cls: Type[CompressibilityCorrection],
        ) -> Type[CompressibilityCorrection]:
            if name in cls._registry:
                raise ValueError(
                    f"Compressibility correction '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> CompressibilityCorrection:
        """Create a compressibility correction by name.

        Parameters
        ----------
        name : str
            Registered model name.
        **kwargs
            Arguments forwarded to the model constructor.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown compressibility correction '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def correct_dissipation(
        self,
        epsilon: torch.Tensor,
        k: torch.Tensor,
        nut: torch.Tensor,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute compressibility-corrected dissipation rate.

        Parameters
        ----------
        epsilon : torch.Tensor
            ``(n_cells,)`` turbulent dissipation rate.
        k : torch.Tensor
            ``(n_cells,)`` turbulent kinetic energy.
        nut : torch.Tensor
            ``(n_cells,)`` turbulent viscosity.
        rho : torch.Tensor or float
            ``(n_cells,)`` density field or scalar.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` corrected dissipation rate.
        """

    @abstractmethod
    def turbulent_mach_number(
        self,
        k: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent Mach number.

        M_t = sqrt(2k) / a

        Parameters
        ----------
        k : torch.Tensor
            ``(n_cells,)`` turbulent kinetic energy.
        a : torch.Tensor
            ``(n_cells,)`` speed of sound.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent Mach number.
        """


# ======================================================================
# Sarkar compressibility correction
# ======================================================================


@CompressibilityCorrection.register("Sarkar")
class SarkarModel(CompressibilityCorrection):
    """Sarkar compressibility correction model.

    Adds a dilatational dissipation term to the turbulence equations:

        epsilon_total = epsilon_solenoidal + epsilon_dilatational

    where:
        epsilon_dilatational = alpha_1 * epsilon * M_t^2
                             + alpha_2 * epsilon * M_t^4

    with model constants alpha_1 = 1.0 and alpha_2 = 0.5 (Sarkar 1991).

    This correction increases the effective dissipation at high
    turbulent Mach numbers, reducing the predicted turbulent kinetic
    energy — physically consistent with the observed reduction of
    turbulence intensity in compressible mixing layers.

    Parameters
    ----------
    alpha_1 : float
        Linear correction coefficient.  Default: 1.0.
    alpha_2 : float
        Quartic correction coefficient.  Default: 0.5.
    """

    def __init__(
        self,
        alpha_1: float = 1.0,
        alpha_2: float = 0.5,
    ) -> None:
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def turbulent_mach_number(
        self,
        k: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent Mach number.

        M_t = sqrt(2k) / a

        Parameters
        ----------
        k : torch.Tensor
            ``(n_cells,)`` turbulent kinetic energy.
        a : torch.Tensor
            ``(n_cells,)`` speed of sound.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent Mach number.
        """
        a_safe = a.abs().clamp(min=1e-10)
        return (2.0 * k.clamp(min=0.0)).sqrt() / a_safe

    def correct_dissipation(
        self,
        epsilon: torch.Tensor,
        k: torch.Tensor,
        nut: torch.Tensor,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Apply Sarkar compressibility correction.

        Parameters
        ----------
        epsilon : torch.Tensor
            ``(n_cells,)`` solenoidal dissipation rate.
        k : torch.Tensor
            ``(n_cells,)`` turbulent kinetic energy.
        nut : torch.Tensor
            ``(n_cells,)`` turbulent viscosity (unused directly).
        rho : torch.Tensor or float
            ``(n_cells,)`` density (unused directly).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` corrected total dissipation rate.
        """
        # Compute turbulent Mach number from k and a reference speed of sound
        # a = sqrt(gamma * R * T) ~ 340 m/s for air at STP
        # For a generic correction, use a = sqrt(k / nut) * scale
        # or accept the dilatational contribution directly
        k_safe = k.clamp(min=0.0)

        # Estimate M_t from turbulence intensity
        # M_t = sqrt(2k) / a; use a = 340 m/s (air) as default
        a_ref = 340.0
        M_t = (2.0 * k_safe).sqrt() / a_ref

        # Dilatational dissipation
        eps_dilat = (
            self.alpha_1 * epsilon * M_t * M_t
            + self.alpha_2 * epsilon * M_t.pow(4)
        )

        return epsilon + eps_dilat

    def __repr__(self) -> str:
        return f"SarkarModel(alpha_1={self.alpha_1}, alpha_2={self.alpha_2})"


# ======================================================================
# Zeman compressibility correction
# ======================================================================


@CompressibilityCorrection.register("Zeman")
class ZemanModel(CompressibilityCorrection):
    """Zeman compressibility correction model.

    Based on Zeman (1990), this model introduces a threshold
    turbulent Mach number below which compressibility effects are
    negligible:

        epsilon_dilatational = alpha_3 * epsilon * (M_t - M_t0)^2
                               * H(M_t - M_t0)

    where H is the Heaviside step function and M_t0 is the threshold
    (typically 0.1-0.2).  Above the threshold, the correction grows
    quadratically.

    Parameters
    ----------
    alpha_3 : float
        Correction intensity.  Default: 0.75.
    M_t0 : float
        Threshold turbulent Mach number.  Default: 0.1.
    """

    def __init__(
        self,
        alpha_3: float = 0.75,
        M_t0: float = 0.1,
    ) -> None:
        self.alpha_3 = alpha_3
        self.M_t0 = M_t0

    def turbulent_mach_number(
        self,
        k: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent Mach number.

        M_t = sqrt(2k) / a
        """
        a_safe = a.abs().clamp(min=1e-10)
        return (2.0 * k.clamp(min=0.0)).sqrt() / a_safe

    def correct_dissipation(
        self,
        epsilon: torch.Tensor,
        k: torch.Tensor,
        nut: torch.Tensor,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Apply Zeman compressibility correction.

        Parameters
        ----------
        epsilon : torch.Tensor
            ``(n_cells,)`` solenoidal dissipation rate.
        k : torch.Tensor
            ``(n_cells,)`` turbulent kinetic energy.
        nut : torch.Tensor
            ``(n_cells,)`` turbulent viscosity (unused directly).
        rho : torch.Tensor or float
            ``(n_cells,)`` density (unused directly).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` corrected total dissipation rate.
        """
        k_safe = k.clamp(min=0.0)

        # Turbulent Mach number
        a_ref = 340.0
        M_t = (2.0 * k_safe).sqrt() / a_ref

        # Heaviside-like threshold (smooth approximation)
        excess = (M_t - self.M_t0).clamp(min=0.0)

        # Dilatational dissipation (quadratic above threshold)
        eps_dilat = self.alpha_3 * epsilon * excess.pow(2)

        return epsilon + eps_dilat

    def __repr__(self) -> str:
        return f"ZemanModel(alpha_3={self.alpha_3}, M_t0={self.M_t0})"
