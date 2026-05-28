"""
Turbulence production limiters for RANS k-epsilon type models.

In high-strain regions (stagnation points, impingement zones) the
standard production term P_k = 2 * nu_t * |S|^2 can become
unrealistically large, leading to over-production of turbulent kinetic
energy.  Production limiters cap P_k to a physically reasonable value.

Models:

- :class:`ProductionLimiter` — abstract base class
- :class:`StandardLimiter` — standard P_k < C_lim * epsilon limiter
- :class:`KatoLimiter` — Kato-Launder limiter (replaces |S| with sqrt(|S|*|Omega|))

Usage::

    from pyfoam.turbulence.production_limiter import ProductionLimiter

    limiter = ProductionLimiter.create("standard", C_lim=20.0)
    P_k_limited = limiter.limit(P_k, epsilon)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

__all__ = [
    "ProductionLimiter",
    "StandardLimiter",
    "KatoLimiter",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ProductionLimiter(ABC):
    """Abstract base class for turbulence production limiters.

    Subclasses must implement :meth:`limit` which caps the production
    term P_k to a physically reasonable value.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @ProductionLimiter.register("standard")
        class StandardLimiter(ProductionLimiter):
            ...

        limiter = ProductionLimiter.create("standard", C_lim=20.0)
    """

    _registry: ClassVar[dict[str, Type[ProductionLimiter]]] = {}

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a limiter under *name*."""

        def decorator(lim_cls: Type[ProductionLimiter]) -> Type[ProductionLimiter]:
            if name in cls._registry:
                raise ValueError(
                    f"Production limiter '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = lim_cls
            return lim_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ProductionLimiter:
        """Factory: create a limiter instance by registered *name*.

        Parameters
        ----------
        name : str
            Registered limiter type name.
        **kwargs
            Constructor arguments forwarded to the limiter class.

        Returns
        -------
        ProductionLimiter
            Instantiated production limiter.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown production limiter '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered limiter type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def limit(
        self,
        P_k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Limit the turbulence production term P_k.

        Parameters
        ----------
        P_k : torch.Tensor
            Un-limited production rate of k ``(n_cells,)`` [m^2/s^3].
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)`` [m^2/s^3].

        Returns
        -------
        torch.Tensor
            Limited production rate ``(n_cells,)`` [m^2/s^3].
        """


# ---------------------------------------------------------------------------
# Concrete limiters
# ---------------------------------------------------------------------------


@ProductionLimiter.register("standard")
class StandardLimiter(ProductionLimiter):
    """Standard production limiter: P_k < C_lim * epsilon.

    Caps the production term to a multiple of the dissipation rate::

        P_k_limited = min(P_k, C_lim * epsilon)

    The default C_lim = 20.0 is the value recommended by Menter (1994)
    for the k-omega SST model and is widely used in industrial CFD.

    Parameters
    ----------
    C_lim : float
        Limiter coefficient (default: 20.0).
    """

    def __init__(self, C_lim: float = 20.0) -> None:
        self.C_lim = C_lim

    def limit(
        self,
        P_k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Apply standard production limiter.

        P_k_limited = min(P_k, C_lim * epsilon)

        Parameters
        ----------
        P_k : torch.Tensor
            Un-limited production rate of k ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Limited production rate ``(n_cells,)``.
        """
        P_k_limit = self.C_lim * epsilon
        return torch.min(P_k, P_k_limit)


@ProductionLimiter.register("kato")
class KatoLimiter(ProductionLimiter):
    """Kato-Launder production limiter.

    Instead of capping P_k after computation, the Kato-Launder
    approach modifies the production term itself to prevent
    over-production at stagnation points::

        P_k_KL = mu_t * |S| * |Omega|

    This replaces the standard P_k = mu_t * |S|^2 with a product of
    strain-rate magnitude and vorticity magnitude.  At stagnation
    points |Omega| ~ 0 while |S| is large, so the product remains
    bounded.  In shear layers both |S| and |Omega| are comparable,
    so the result is close to the standard production.

    Parameters
    ----------
    nu_t : float or None
        Kinematic eddy viscosity.  If provided, ``P_k`` is assumed to
        be the strain production (2 * nu_t * |S|^2) and the limiter
        extracts |S| from it.  If ``None``, the caller must supply
        ``strain_magnitude`` and ``vorticity_magnitude`` separately
        via the ``limit()`` method (not supported in this simple
        interface — use ``limit_with_strain`` instead).
    """

    def __init__(self, nu_t: float | None = None) -> None:
        self.nu_t = nu_t

    def limit(
        self,
        P_k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Kato-Launder limiter (simplified form).

        When ``nu_t`` is provided the strain magnitude is extracted
        from ``P_k = 2 * nu_t * |S|^2``::

            |S| = sqrt(P_k / (2 * nu_t))
            P_k_KL = 2 * nu_t * |S| * sqrt(epsilon / (2 * nu_t * C_mu))

        where C_mu = 0.09 is the standard k-epsilon constant.  The
        vorticity magnitude is approximated from epsilon and nu_t.

        Parameters
        ----------
        P_k : torch.Tensor
            Strain production rate of k ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Kato-Launder limited production rate ``(n_cells,)``.
        """
        C_mu = 0.09

        if self.nu_t is not None and self.nu_t > 1e-30:
            nu_t_tensor = torch.tensor(
                self.nu_t, dtype=P_k.dtype, device=P_k.device,
            )
            # Extract |S| from P_k = 2 * nu_t * |S|^2
            strain_sq = P_k.clamp(min=0.0) / (2.0 * nu_t_tensor)
            strain_mag = torch.sqrt(strain_sq)

            # Estimate |Omega| from epsilon: epsilon ~ C_mu * k * omega
            # omega ~ epsilon / (C_mu * k) ~ epsilon / (C_mu * P_k / epsilon)
            # Simplified: use |Omega| ~ sqrt(epsilon / nu_t) as scale
            omega_mag = torch.sqrt(epsilon.clamp(min=1e-30) / nu_t_tensor)

            # Kato-Launder: P_k_KL = 2 * nu_t * |S| * |Omega|
            # But we need to scale so that in equilibrium it matches
            # P_k = epsilon.  Use: P_k_KL = nu_t * |S| * |Omega|
            return nu_t_tensor * strain_mag * omega_mag
        else:
            # 无 nu_t 信息时回退到标准限幅器 (C_lim * epsilon)
            return torch.min(P_k, 20.0 * epsilon)

    @staticmethod
    def limit_with_strain(
        P_k: torch.Tensor,
        strain_magnitude: torch.Tensor,
        vorticity_magnitude: torch.Tensor,
        nu_t: torch.Tensor,
    ) -> torch.Tensor:
        """Kato-Launder limiter with explicit strain and vorticity.

        Computes the Kato-Launder production directly from the
        strain-rate and vorticity magnitudes::

            P_k_KL = 2 * nu_t * |S| * |Omega|

        Parameters
        ----------
        P_k : torch.Tensor
            Original production rate ``(n_cells,)``.
        strain_magnitude : torch.Tensor
            |S| = sqrt(2 * S_ij * S_ij) ``(n_cells,)``.
        vorticity_magnitude : torch.Tensor
            |Omega| = sqrt(2 * Omega_ij * Omega_ij) ``(n_cells,)``.
        nu_t : torch.Tensor
            Kinematic eddy viscosity ``(n_cells,)`` or scalar.

        Returns
        -------
        torch.Tensor
            Kato-Launder production rate ``(n_cells,)``.
        """
        return 2.0 * nu_t * strain_magnitude * vorticity_magnitude
