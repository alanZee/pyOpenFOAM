"""
Radiation models for buoyant solvers.

Implements simplified radiation models for energy equation coupling:

- :class:`RadiationModel` — abstract base for radiation models.
- :class:`P1Radiation` — P1 spherical harmonics radiation model.

The P1 model is the simplest non-trivial radiation model. It solves a
single transport equation for the incident radiation G:

    ∇·(Γ ∇G) = a(G − 4σT⁴)

where:
    Γ = 1 / (3(a + σ_s))     (diffusion coefficient)
    a = absorption coefficient
    σ_s = scattering coefficient
    σ = Stefan-Boltzmann constant (5.670374419e-8 W/(m²·K⁴))

The radiation heat source in the energy equation is:
    S_rad = a(G − 4σT⁴)

For a simplified implementation (without solving the full G equation),
we use the optically thin approximation:
    S_rad = 4aσ(T⁴ − T_ref⁴)

Reference:
    OpenFOAM ``radiationModels::P1``
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["RadiationModel", "P1Radiation"]

logger = logging.getLogger(__name__)

# Stefan-Boltzmann constant (W/(m²·K⁴))
STEFAN_BOLTZMANN = 5.670374419e-8


class RadiationModel(ABC):
    """Abstract base class for radiation models.

    Subclasses implement :meth:`Sh` which returns the radiation heat
    source term for the energy equation.
    """

    @abstractmethod
    def Sh(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the radiation heat source for the energy equation.

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radiation source term (W/m³).
        """
        ...

    @abstractmethod
    def correct(self) -> None:
        """Update the radiation model (called after energy solve)."""
        ...


class P1Radiation(RadiationModel):
    """P1 radiation model (simplified).

    Uses the optically thin approximation for the radiation source:
        S_rad = 4 * a * σ * (T⁴ − T_ref⁴)

    This is appropriate for:
    - Gases with moderate optical thickness
    - Enclosure problems (rooms, furnaces)
    - Cases where radiation is a correction to convection

    Parameters
    ----------
    absorption_coeff : float
        Absorption coefficient ``a`` (1/m). Default 0.1 for air.
    T_ref : float
        Reference temperature for radiation exchange (K). Default 300 K.
    sigma : float
        Stefan-Boltzmann constant. Default 5.670374419e-8.

    Examples::

        radiation = P1Radiation(absorption_coeff=0.5, T_ref=300)
        S_rad = radiation.Sh(T)  # source term for energy equation
    """

    def __init__(
        self,
        absorption_coeff: float = 0.1,
        T_ref: float = 300.0,
        sigma: float = STEFAN_BOLTZMANN,
    ) -> None:
        self._a = absorption_coeff
        self._T_ref = T_ref
        self._sigma = sigma

    def Sh(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute P1 radiation source: S = 4aσ(T⁴ − T_ref⁴).

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radiation source (W/m³).
            Positive = heat gain (radiation absorbed).
            Negative = heat loss (radiation emitted).
        """
        T_safe = T.clamp(min=1e-10)
        T4 = T_safe ** 4
        T_ref4 = self._T_ref ** 4
        return 4.0 * self._a * self._sigma * (T4 - T_ref4)

    def correct(self) -> None:
        """Update radiation model (no-op for P1)."""
        pass

    @property
    def absorption_coeff(self) -> float:
        """Absorption coefficient (1/m)."""
        return self._a

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    def __repr__(self) -> str:
        return (
            f"P1Radiation(a={self._a}, T_ref={self._T_ref}, "
            f"sigma={self._sigma})"
        )
