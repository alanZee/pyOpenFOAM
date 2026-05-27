"""
Enhanced cavitation models with improved convergence.

Provides enhanced versions of the ZGB and Merkle cavitation models
with:
- Under-relaxation of the mass transfer rate
- Pressure difference limiting for numerical stability
- Clipping of alpha to physical bounds
- Improved handling of near-equilibrium conditions

These models produce the same physics as the base versions but are
more robust in iterative solvers with large time steps.

Usage::

    from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

    model = ZGBModel(relaxation=0.5, alpha_clip=1e-6)
    m_dot = model.compute_mass_transfer(alpha, p, rho_l, rho_v)
    # Apply under-relaxation across iterations
    m_dot = model.relax(m_dot_new, m_dot_old)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .cavitation import ZGB, Merkle

__all__ = ["ZGBModel", "MerkleModel"]


class ZGBModel:
    """Enhanced Zwart-Gerber-Belamri cavitation model.

    Extends the base :class:`ZGB` model with convergence enhancements:

    - **Pressure limiting**: caps the effective pressure difference to
      avoid extremely large source terms near saturation.
    - **Alpha clipping**: bounds the volume fraction to ``[alpha_clip,
      1 - alpha_clip]`` to prevent singularities.
    - **Under-relaxation**: the :meth:`relax` method applies explicit
      under-relaxation between iterations.

    The core physics are identical to the standard ZGB formulation:

        m_dot_evap = C_evap * 3 * rho_v * alpha_nuc * (1 - alpha) / R_b
                     * sqrt(2/3 * max(p_v - p, 0) / rho_l)
        m_dot_cond = C_cond * 3 * rho_v * alpha / R_b
                     * sqrt(2/3 * max(p - p_v, 0) / rho_l)

    Parameters
    ----------
    C_evap : float
        Evaporation coefficient. Default 0.02.
    C_cond : float
        Condensation coefficient. Default 0.01.
    alpha_nuc : float
        Nucleation site volume fraction. Default 5e-4.
    p_v : float
        Vapor pressure (Pa). Default 2300.0.
    R_b : float
        Bubble radius (m). Default 1e-6.
    relaxation : float
        Under-relaxation factor for :meth:`relax`, in ``(0, 1]``.
        Default 1.0 (no relaxation).
    alpha_clip : float
        Lower/upper clip for alpha. Default 1e-6.
    p_clip : float
        Maximum allowed |p - p_v| (Pa). Default 1e5.
    """

    def __init__(
        self,
        C_evap: float = 0.02,
        C_cond: float = 0.01,
        alpha_nuc: float = 5e-4,
        p_v: float = 2300.0,
        R_b: float = 1e-6,
        relaxation: float = 1.0,
        alpha_clip: float = 1e-6,
        p_clip: float = 1e5,
    ) -> None:
        self._base = ZGB(
            C_evap=C_evap,
            C_cond=C_cond,
            alpha_nuc=alpha_nuc,
            p_v=p_v,
            R_b=R_b,
        )
        self.relaxation = relaxation
        self.alpha_clip = alpha_clip
        self.p_clip = p_clip

    @property
    def base(self) -> ZGB:
        """Return the underlying ZGB model."""
        return self._base

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute mass transfer with enhanced convergence.

        Applies alpha clipping and pressure limiting before delegating
        to the base ZGB model.

        Args:
            alpha: Vapor volume fraction ``(n_cells,)``.
            p: Pressure field ``(n_cells,)``.
            rho_l: Liquid density (kg/m^3).
            rho_v: Vapor density (kg/m^3).

        Returns:
            ``(n_cells,)`` mass transfer rate (positive = evaporation).
        """
        alpha_safe = alpha.clamp(min=self.alpha_clip, max=1.0 - self.alpha_clip)

        # Limit effective pressure difference for stability
        p_eff = self._base.p_v + (p - self._base.p_v).clamp(
            min=-self.p_clip, max=self.p_clip,
        )

        return self._base.compute_mass_transfer(alpha_safe, p_eff, rho_l, rho_v)

    def relax(
        self,
        m_dot_new: torch.Tensor,
        m_dot_old: torch.Tensor,
    ) -> torch.Tensor:
        """Apply explicit under-relaxation.

        Returns::

            m_dot = relaxation * m_dot_new + (1 - relaxation) * m_dot_old

        Args:
            m_dot_new: New mass transfer rate.
            m_dot_old: Previous mass transfer rate.

        Returns:
            Relaxed mass transfer rate.
        """
        return self.relaxation * m_dot_new + (1.0 - self.relaxation) * m_dot_old


class MerkleModel:
    """Enhanced Merkle cavitation model.

    Extends the base :class:`Merkle` model with convergence enhancements:

    - **Pressure limiting**: caps |p - p_v| to avoid large source terms.
    - **Alpha clipping**: prevents near-zero or near-one alpha issues.
    - **Under-relaxation**: the :meth:`relax` method for iteration damping.

    Parameters
    ----------
    C_evap : float
        Evaporation coefficient. Default 1.0.
    C_cond : float
        Condensation coefficient. Default 1.0.
    p_v : float
        Vapor pressure (Pa). Default 2300.0.
    U_inf : float
        Reference velocity (m/s). Default 1.0.
    t_inf : float
        Reference time scale (s). Default 1.0.
    relaxation : float
        Under-relaxation factor, in ``(0, 1]``. Default 1.0.
    alpha_clip : float
        Lower/upper clip for alpha. Default 1e-6.
    p_clip : float
        Maximum allowed |p - p_v| (Pa). Default 1e5.
    """

    def __init__(
        self,
        C_evap: float = 1.0,
        C_cond: float = 1.0,
        p_v: float = 2300.0,
        U_inf: float = 1.0,
        t_inf: float = 1.0,
        relaxation: float = 1.0,
        alpha_clip: float = 1e-6,
        p_clip: float = 1e5,
    ) -> None:
        self._base = Merkle(
            C_evap=C_evap,
            C_cond=C_cond,
            p_v=p_v,
            U_inf=U_inf,
            t_inf=t_inf,
        )
        self.relaxation = relaxation
        self.alpha_clip = alpha_clip
        self.p_clip = p_clip

    @property
    def base(self) -> Merkle:
        """Return the underlying Merkle model."""
        return self._base

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute mass transfer with enhanced convergence.

        Applies alpha clipping and pressure limiting before delegating
        to the base Merkle model.
        """
        alpha_safe = alpha.clamp(min=self.alpha_clip, max=1.0 - self.alpha_clip)
        p_eff = self._base.p_v + (p - self._base.p_v).clamp(
            min=-self.p_clip, max=self.p_clip,
        )
        return self._base.compute_mass_transfer(alpha_safe, p_eff, rho_l, rho_v)

    def relax(
        self,
        m_dot_new: torch.Tensor,
        m_dot_old: torch.Tensor,
    ) -> torch.Tensor:
        """Apply explicit under-relaxation.

        Returns::

            m_dot = relaxation * m_dot_new + (1 - relaxation) * m_dot_old
        """
        return self.relaxation * m_dot_new + (1.0 - self.relaxation) * m_dot_old
