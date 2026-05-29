"""
Kato-Launder damping for multiphase turbulence models.

Reduces excessive turbulence production at phase interfaces by
modifying the production term using the Kato-Launder rotation
production limiter.

The standard turbulence production term is:

    P_k = 2 ν_t |S|²

which overpredicts production in stagnation regions and at
multiphase interfaces where strain rates are artificially high.

The Kato-Launder modification replaces |S|² with |S| * |Ω|
(vorticity magnitude), yielding:

    P_k = 2 ν_t |S| |Ω|

This naturally reduces production in regions where strain is high
but vorticity is low (stagnation and interface regions), since
|Ω| ≈ 0 in irrotational strain.

For multiphase flows, an additional alpha-dependent damping factor
is applied to suppress production at the interface:

    P_k = 2 ν_t |S| |Ω| * f(alpha)

where f(alpha) = 1 - 4 * alpha * (1 - alpha) * damping_strength
smoothly reduces production near the interface (alpha ≈ 0.5).

References
----------
Kato, M. & Launder, B.E. (1993). The modelling of turbulent flow
around stationary and vibrating square cylinders. Proc. 9th
Symposium on Turbulent Shear Flows, Kyoto, Japan, 10-4.

Usage::

    from pyfoam.multiphase.turbulence_kato_launder import KatoLaunderDamping

    model = KatoLaunderDamping(damping_strength=0.9)
    P_damped = model.damp_production(alpha, S_mag, Omega_mag, nu_t)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["KatoLaunderDamping"]

logger = logging.getLogger(__name__)


class KatoLaunderDamping:
    """Kato-Launder production damping for multiphase turbulence.

    Modifies the turbulence production term to prevent overprediction
    at phase interfaces and stagnation regions.  The production is
    replaced by the Kato-Launder rotation-based formulation with
    additional multiphase interface damping.

    Parameters
    ----------
    damping_strength : float
        Strength of the alpha-based interface damping.
        0.0 = no alpha damping, 1.0 = full suppression at interface.
        Default: 0.9.
    alpha_min : float
        Lower alpha threshold for damping region.  Default: 0.01.
    alpha_max : float
        Upper alpha threshold for damping region.  Default: 0.99.
    use_rotation : bool
        If True, replace |S|² with |S| * |Ω| (Kato-Launder).
        If False, only apply alpha-based damping.  Default: True.

    Examples::

        >>> model = KatoLaunderDamping(damping_strength=0.9)
        >>> alpha = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        >>> S_mag = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
        >>> Omega_mag = torch.tensor([5.0, 5.0, 0.1, 5.0, 5.0])
        >>> nu_t = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> P = model.damp_production(alpha, S_mag, Omega_mag, nu_t)
    """

    # Class-level registry for alternative damping formulations
    _registry: ClassVar[dict[str, Type["KatoLaunderDamping"]]] = {}

    def __init__(
        self,
        damping_strength: float = 0.9,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        use_rotation: bool = True,
    ) -> None:
        self.damping_strength = damping_strength
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.use_rotation = use_rotation

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping variant under *name*."""
        def decorator(model_cls: Type[KatoLaunderDamping]) -> Type[KatoLaunderDamping]:
            if name in cls._registry:
                raise ValueError(
                    f"Kato-Launder variant '{name}' is already registered"
                )
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "KatoLaunderDamping":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown Kato-Launder variant '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered variant names."""
        return sorted(cls._registry.keys())

    def compute_interface_factor(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the alpha-based interface damping factor.

        Uses the interface indicator: f = 4 * alpha * (1 - alpha),
        which peaks at 1.0 when alpha = 0.5 (the interface).

        The damping factor is:
            d = 1 - damping_strength * f

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)`` in [0, 1].
            1 = no damping, 0 = full suppression.
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        # Interface indicator: peaks at alpha = 0.5
        indicator = 4.0 * alpha_c * (1.0 - alpha_c)
        # Only apply in the interface region
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        indicator = indicator * in_interface.to(indicator.dtype)
        return (1.0 - self.damping_strength * indicator).clamp(min=0.0, max=1.0)

    def damp_production(
        self,
        alpha: torch.Tensor,
        S_mag: torch.Tensor,
        Omega_mag: torch.Tensor,
        nu_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the damped turbulence production term.

        Standard production: P_k = 2 * nu_t * S²
        Kato-Launder:       P_k = 2 * nu_t * S * Omega
        With interface:     P_k = 2 * nu_t * S * Omega * f(alpha)

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].
        S_mag : torch.Tensor
            Strain rate magnitude |S| ``(n_cells,)``.
        Omega_mag : torch.Tensor
            Vorticity magnitude |Ω| ``(n_cells,)``.
        nu_t : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Damped production term ``(n_cells,)``.
        """
        if self.use_rotation:
            # Kato-Launder: replace |S|² with |S| * |Ω|
            production = 2.0 * nu_t * S_mag * Omega_mag
        else:
            # Standard production (only alpha damping applied)
            production = 2.0 * nu_t * S_mag.pow(2)

        # Apply interface damping
        d = self.compute_interface_factor(alpha)
        return production * d

    def damp_k_source(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        S_mag: torch.Tensor,
        Omega_mag: torch.Tensor,
        nu_t: torch.Tensor,
        epsilon_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the net k source term with damping.

        Source = P_damped - epsilon

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        S_mag : torch.Tensor
            Strain rate magnitude ``(n_cells,)``.
        Omega_mag : torch.Tensor
            Vorticity magnitude ``(n_cells,)``.
        nu_t : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        epsilon_k : torch.Tensor | None
            Dissipation rate of k.  If None, estimated as
            C_mu * k^(3/2) / delta.

        Returns
        -------
        torch.Tensor
            Net source term for k transport equation ``(n_cells,)``.
        """
        P_damped = self.damp_production(alpha, S_mag, Omega_mag, nu_t)

        if epsilon_k is not None:
            dissipation = epsilon_k
        else:
            # Estimate: epsilon ~ C_mu * k^1.5 / delta
            # With C_mu = 0.09 as standard k-epsilon constant
            C_mu = 0.09
            k_pos = k.clamp(min=1e-30)
            dissipation = C_mu * k_pos.pow(1.5)

        return P_damped - dissipation

    def __repr__(self) -> str:
        return (
            f"KatoLaunderDamping(strength={self.damping_strength}, "
            f"use_rotation={self.use_rotation})"
        )
