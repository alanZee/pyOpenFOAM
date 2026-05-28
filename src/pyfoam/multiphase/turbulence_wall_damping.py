"""
Wall damping models for multiphase VOF (Volume of Fluid) simulations.

Near the wall, the standard CSF (Continuum Surface Force) model for
surface tension can produce parasitic currents.  Brackbill et al. (1992)
proposed a wall-damping approach that modifies the turbulence production
near the wall to account for the presence of the interface.

Models:

- :class:`TurbulenceWallDampingModel` — abstract base with RTS registry
- :class:`BrackbillDamping` — Brackbill near-wall damping for VOF

The Brackbill model applies a wall-distance-dependent damping to the
turbulence quantities in cells where the volume fraction is between
thresholds (i.e., near the interface) and the cell is close to a wall:

    f_wall = 1 - exp(-y+ / A_wall)
    f_interface = 4 * alpha * (1 - alpha)
    damping = f_interface * (1 - f_wall)

    k_damped = k * (1 - damping_coeff * damping)
    epsilon_damped = epsilon * (1 - damping_coeff * damping)

where:
- y+ is the non-dimensional wall distance
- A_wall is the wall damping constant (default: 25)
- damping_coeff controls overall damping strength (0-1 range recommended)

Reference:
    Brackbill, J.U., Kothe, D.B., Zemach, C. (1992).
    "A continuum method for modeling surface tension."
    Journal of Computational Physics, 100(2), 335-354.

Usage::

    from pyfoam.multiphase.turbulence_wall_damping import BrackbillDamping

    model = BrackbillDamping(damping_coeff=0.5, A_wall=25.0)
    k_damped = model.damp_k(alpha, k, y_plus=y_plus_field)
    eps_damped = model.damp_epsilon(alpha, epsilon, y_plus=y_plus_field)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceWallDampingModel",
    "BrackbillDamping",
]

logger = logging.getLogger(__name__)


class TurbulenceWallDampingModel(ABC):
    """Abstract base class for near-wall turbulence damping in VOF simulations.

    Subclasses implement :meth:`damp_k` and :meth:`damp_epsilon` (and
    optionally :meth:`damp_omega`) to suppress turbulence near walls
    where the phase interface is present.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceWallDampingModel"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 0.5,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        damping_coeff : float
            Damping strength coefficient in [0, 1]. Default: 0.5.
        alpha_min : float
            Lower alpha threshold for interface region. Default: 0.01.
        alpha_max : float
            Upper alpha threshold for interface region. Default: 0.99.
        """
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a wall damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceWallDampingModel]) -> Type[TurbulenceWallDampingModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence wall damping model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceWallDampingModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence wall damping model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    def compute_interface_indicator(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute interface indicator: 4 * alpha * (1 - alpha).

        Peaks at 1.0 when alpha = 0.5, zero at alpha = 0 or 1.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].

        Returns
        -------
        torch.Tensor
            Interface indicator ``(n_cells,)``.
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        return 4.0 * alpha_c * (1.0 - alpha_c)

    @abstractmethod
    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply near-wall damping to turbulent kinetic energy.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        y_plus : torch.Tensor, optional
            Non-dimensional wall distance ``(n_cells,)``.
        """

    @abstractmethod
    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply near-wall damping to turbulent dissipation rate.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.
        y_plus : torch.Tensor, optional
            Non-dimensional wall distance ``(n_cells,)``.
        """


@TurbulenceWallDampingModel.register("brackbillDamping")
class BrackbillDamping(TurbulenceWallDampingModel):
    """Brackbill near-wall damping model for VOF simulations.

    Damps turbulence near the wall where the phase interface is present,
    reducing parasitic currents from the CSF surface tension model:

        f_interface = 4 * alpha * (1 - alpha)
        f_wall = 1 - exp(-y+ / A_wall)
        total_factor = damping_coeff * f_interface * (1 - f_wall)
        k_damped = k * max(0, 1 - total_factor)

    The ``(1 - f_wall)`` term ensures maximum damping at the wall (y+ = 0)
    and a smooth transition to no damping away from the wall.

    Parameters
    ----------
    damping_coeff : float
        Damping strength in [0, 1]. Default: 0.5.
    A_wall : float
        Wall damping constant controlling the damping range. Default: 25.0.
    alpha_min : float
        Lower alpha threshold. Default: 0.01.
    alpha_max : float
        Upper alpha threshold. Default: 0.99.
    """

    def __init__(
        self,
        damping_coeff: float = 0.5,
        A_wall: float = 25.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self.A_wall = A_wall

    def _compute_wall_factor(self, y_plus: torch.Tensor | None, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Compute wall damping factor f_wall.

        f_wall = 1 - exp(-y+ / A_wall)

        When y_plus is None, returns zero (no wall damping).
        """
        if y_plus is not None:
            y_p = y_plus.to(device=ref_tensor.device, dtype=ref_tensor.dtype).clamp(min=0.0)
            return 1.0 - torch.exp(-y_p / max(self.A_wall, 1e-6))
        else:
            return torch.zeros_like(ref_tensor)

    def _compute_damping(self, alpha: torch.Tensor, y_plus: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        """Compute the total damping factor.

        total = damping_coeff * f_interface * (1 - f_wall)
        Only applied in the interface region.
        """
        f_interface = self.compute_interface_indicator(alpha)
        f_wall = self._compute_wall_factor(y_plus, ref)

        # Interface region filter
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        total = self.damping_coeff * f_interface * (1.0 - f_wall)
        return total * in_interface.to(ref.dtype)

    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp k using Brackbill near-wall model.

        k_damped = k * max(0, 1 - total_factor)
        """
        total = self._compute_damping(alpha, y_plus, k)
        return k * (1.0 - total).clamp(min=0.0)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp epsilon using Brackbill near-wall model.

        epsilon_damped = epsilon * max(0, 1 - total_factor)
        """
        total = self._compute_damping(alpha, y_plus, epsilon)
        return epsilon * (1.0 - total).clamp(min=0.0)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp omega using Brackbill near-wall model.

        omega_damped = omega * max(0, 1 - total_factor)
        """
        total = self._compute_damping(alpha, y_plus, omega)
        return omega * (1.0 - total).clamp(min=0.0)
