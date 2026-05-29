"""
Kato-Launder production limiter for turbulence models.

Implements the Kato-Launder (1976) modification to the turbulence
production term that prevents over-production at stagnation points.
The standard production term in k-epsilon models is::

    P_k = 2 * nu_t * |S|^2

where |S| = sqrt(2 * S_ij * S_ij) is the strain-rate magnitude.
At stagnation points, |S| is large but vorticity |Omega| is near
zero, leading to unphysical over-production of turbulent kinetic
energy.

The Kato-Launder modification replaces |S|^2 with |S| * |Omega|::

    P_k_KL = 2 * nu_t * |S| * |Omega|

This ensures P_k -> 0 at stagnation points (where |Omega| -> 0)
while remaining close to the standard value in shear layers
(where |S| ~ |Omega|).

Reference:
    Kato, M. and Launder, B.E., 1996.
    "The modelling of turbulent flow around stationary and vibrating
    square cylinders."
    Proc. 9th Symposium on Turbulent Shear Flows, Kyoto, Japan.

Usage::

    from pyfoam.turbulence.turbulence_kato_launder import KatoLaunderDamping

    damping = KatoLaunderDamping(nu_t=1e-3)
    P_k_limited = damping.damp(P_k, strain_mag, vorticity_mag)
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["KatoLaunderDamping"]

logger = logging.getLogger(__name__)


class KatoLaunderDamping:
    """Kato-Launder turbulence production limiter.

    Modifies the production term to use |S| * |Omega| instead of |S|^2,
    preventing over-production at stagnation points while maintaining
    correct behaviour in shear layers.

    The damped production is::

        P_k_KL = 2 * nu_t * |S| * |Omega|

    or equivalently, given the standard production P_k = 2 * nu_t * |S|^2::

        P_k_KL = P_k * (|Omega| / |S|)

    This ratio equals 1 in pure shear flow and approaches 0 at
    stagnation points.

    Parameters
    ----------
    nu_t : float or torch.Tensor
        Kinematic eddy viscosity. Can be a scalar (uniform) or a
        per-cell tensor ``(n_cells,)``.
    """

    def __init__(
        self,
        nu_t: float | torch.Tensor = 1e-3,
    ) -> None:
        if isinstance(nu_t, torch.Tensor):
            self._nu_t = nu_t
        else:
            self._nu_t = float(nu_t)

    @property
    def nu_t(self) -> float | torch.Tensor:
        """Kinematic eddy viscosity."""
        return self._nu_t

    def damp(
        self,
        P_k: torch.Tensor,
        strain_mag: torch.Tensor,
        vorticity_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Kato-Launder damping to the production term.

        Computes::

            P_k_KL = 2 * nu_t * |S| * |Omega|

        If ``nu_t`` is scalar, it is broadcast to match ``P_k``.
        If ``nu_t`` is a tensor, it must have the same shape as ``P_k``.

        Parameters
        ----------
        P_k : torch.Tensor
            Standard production rate of k ``(n_cells,)`` [m^2/s^3].
        strain_mag : torch.Tensor
            Strain-rate magnitude |S| = sqrt(2 * S_ij * S_ij)
            ``(n_cells,)`` [1/s].
        vorticity_mag : torch.Tensor
            Vorticity magnitude |Omega| = sqrt(2 * Omega_ij * Omega_ij)
            ``(n_cells,)`` [1/s].

        Returns
        -------
        torch.Tensor
            Kato-Launder damped production rate ``(n_cells,)`` [m^2/s^3].
        """
        device = P_k.device
        dtype = P_k.dtype

        if isinstance(self._nu_t, torch.Tensor):
            nu_t = self._nu_t.to(device=device, dtype=dtype)
        else:
            nu_t = torch.tensor(self._nu_t, dtype=dtype, device=device)

        # Clamp to avoid division by zero
        strain_safe = strain_mag.clamp(min=1e-30)
        vort_safe = vorticity_mag.clamp(min=1e-30)

        # P_k_KL = 2 * nu_t * |S| * |Omega|
        return 2.0 * nu_t * strain_safe * vort_safe

    def damp_from_P_k(
        self,
        P_k: torch.Tensor,
        strain_mag: torch.Tensor,
        vorticity_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Kato-Launder damping by scaling existing P_k.

        Given P_k = 2 * nu_t * |S|^2, compute::

            P_k_KL = P_k * |Omega| / |S|

        This is useful when P_k is already computed and we just need
        to scale it.  Avoids recomputing nu_t * |S|.

        Parameters
        ----------
        P_k : torch.Tensor
            Standard production rate ``(n_cells,)`` [m^2/s^3].
        strain_mag : torch.Tensor
            Strain-rate magnitude |S| ``(n_cells,)`` [1/s].
        vorticity_mag : torch.Tensor
            Vorticity magnitude |Omega| ``(n_cells,)`` [1/s].

        Returns
        -------
        torch.Tensor
            Damped production rate ``(n_cells,)`` [m^2/s^3].
        """
        strain_safe = strain_mag.clamp(min=1e-30)
        vort_safe = vorticity_mag.clamp(min=1e-30)
        ratio = vort_safe / strain_safe
        return P_k * ratio

    def damping_factor(
        self,
        strain_mag: torch.Tensor,
        vorticity_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Kato-Launder damping factor.

        factor = |Omega| / |S|

        This ratio is:
        - ~1.0 in pure shear layers
        - ~0.0 at stagnation points

        Parameters
        ----------
        strain_mag : torch.Tensor
            Strain-rate magnitude ``(n_cells,)`` [1/s].
        vorticity_mag : torch.Tensor
            Vorticity magnitude ``(n_cells,)`` [1/s].

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)`` in [0, 1].
        """
        strain_safe = strain_mag.clamp(min=1e-30)
        vort_safe = vorticity_mag.clamp(min=1e-30)
        return (vort_safe / strain_safe).clamp(max=1.0)
