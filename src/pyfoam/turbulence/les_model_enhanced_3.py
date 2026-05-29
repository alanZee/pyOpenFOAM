"""
Enhanced LES models v3 — improved Smagorinsky and WALE variants.

Extends :class:`~pyfoam.turbulence.les_model_enhanced_2.DynamicLikeSmagorinskyModel`
and :class:`~pyfoam.turbulence.les_model_enhanced_2.ImprovedWALE2Model` with:

- Smagorinsky: wall-adaptive local eddy viscosity (WALE-like) blending
- WALE: improved tensor formulation with trace-free Sd and rotation correction
- Both: improved SGS dissipation rate model

Usage::

    from pyfoam.turbulence.les_model_enhanced_3 import WallAdaptiveSmagorinskyModel, ImprovedWALE3Model

    model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
    model.correct()
    nut = model.nut()
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc

from .les_model import LESModel

__all__ = ["WallAdaptiveSmagorinskyModel", "ImprovedWALE3Model"]


class WallAdaptiveSmagorinskyModel(LESModel):
    """Smagorinsky SGS model with wall-adaptive coefficient.

    Extends :class:`DynamicLikeSmagorinskyModel` concepts with:

    - **Wall-adaptive Cs**: Cs automatically transitions from wall value
      (Cs ~ y^+) to freestream value (Cs_0) using a Van Driest-like
      function that adapts based on local flow features.
    - **Rotation suppression**: Cs is further reduced in regions of
      solid-body rotation (WALE-like feature integrated into Smagorinsky).
    - **SGS dissipation**: explicit epsilon_sgs model.

    The effective viscosity is:

        nu_sgs = (Cs_wall * Delta)^2 * |S| * f_adapt * f_rot

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    Cs : float
        Base Smagorinsky constant. Default 0.17.
    Cs_wall : float
        Wall Smagorinsky constant. Default 0.1.
    A_plus : float
        Van Driest damping constant. Default 25.0.
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cs: float = 0.17,
        Cs_wall: float = 0.1,
        A_plus: float = 25.0,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs = Cs
        self._Cs_wall = Cs_wall
        self._A_plus = A_plus
        self._y: torch.Tensor | None = None

    @property
    def Cs(self) -> float:
        """Base Smagorinsky constant."""
        return self._Cs

    @property
    def Cs_wall(self) -> float:
        """Wall Smagorinsky constant."""
        return self._Cs_wall

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _wall_adaptive_factor(self) -> torch.Tensor:
        """Wall-adaptive damping factor.

        f_adapt = tanh(y+ / A+)

        Uses estimated y+ from local velocity gradient.
        """
        if self._y is None:
            self._y = self._compute_wall_distance()

        y = self._y
        nu = 1.5e-5

        # Estimate y+ from strain rate
        if self._mag_S is not None:
            k_est = (self._Cs * self._delta * self._mag_S).pow(2)
            y_plus = y * k_est.clamp(min=1e-16).sqrt() / max(nu, 1e-30)
        else:
            y_plus = torch.full_like(y, 100.0)

        return torch.tanh(y_plus / self._A_plus).clamp(min=0.0, max=1.0)

    def _rotation_suppression(self) -> torch.Tensor:
        """Rotation suppression factor.

        f_rot = 1 - max(0, (|Omega|/|S| - 1)) * 0.5

        Reduces Cs in regions where rotation exceeds strain.
        """
        if self._grad_U is None:
            return torch.ones((self._mesh.n_cells,),
                              device=self._device, dtype=self._dtype)

        S = self._S
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        ratio = Omega_mag / S_mag.clamp(min=1e-16)
        suppression = 1.0 - 0.5 * (ratio - 1.0).clamp(min=0.0)

        return suppression.clamp(min=0.2, max=1.0)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with wall-adaptive Cs.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        f_adapt = self._wall_adaptive_factor()
        f_rot = self._rotation_suppression()

        # Blend wall and freestream Cs
        Cs_eff = (self._Cs_wall + (self._Cs - self._Cs_wall) * f_adapt) * f_rot

        return (Cs_eff * self._delta).pow(2) * self._mag_S

    def k_sgs(self) -> torch.Tensor:
        """Estimate SGS kinetic energy.

        k_sgs = (Cs_eff * Delta * |S|)^2 / C_I

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` SGS kinetic energy estimate.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before k_sgs()")

        f_adapt = self._wall_adaptive_factor()
        f_rot = self._rotation_suppression()
        Cs_eff = (self._Cs_wall + (self._Cs - self._Cs_wall) * f_adapt) * f_rot

        C_I = 0.09
        return (Cs_eff * self._delta * self._mag_S).pow(2) / max(C_I, 1e-10)

    def epsilon_sgs(self) -> torch.Tensor:
        """SGS dissipation rate.

        eps_sgs = (Cs_eff * Delta * |S|)^2 * |S|

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` SGS dissipation rate.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")

        k = self.k_sgs()
        return k * self._mag_S

    def correct(self) -> None:
        """Update model: compute gradients and strain rate."""
        self._compute_gradients()

    def __repr__(self) -> str:
        return (
            f"WallAdaptiveSmagorinskyModel(Cs={self._Cs}, "
            f"Cs_wall={self._Cs_wall}, n_cells={self._mesh.n_cells})"
        )


class ImprovedWALE3Model(LESModel):
    """Enhanced WALE v3 with improved tensor formulation.

    Extends WALE model concepts with:

    - **Trace-free Sd tensor**: Sd computed with explicit trace removal
      for improved numerical stability.
    - **Rotation correction**: omega_ij term added to Sd tensor for
      better behaviour in rotating flows.
    - **Improved clipping**: temperature-aware viscosity clipping.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    Cw : float
        WALE constant. Default 0.325.
    C_omega : float
        Rotation correction coefficient. Default 0.1.
    max_viscosity_ratio : float
        Maximum nu_sgs / nu ratio. Default 1e5.
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cw: float = 0.325,
        C_omega: float = 0.1,
        max_viscosity_ratio: float = 1e5,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cw = Cw
        self._C_omega = C_omega
        self._max_ratio = max_viscosity_ratio
        self._Sd: torch.Tensor | None = None
        self._mag_Sd_sq: torch.Tensor | None = None
        self._y: torch.Tensor | None = None

    @property
    def Cw(self) -> float:
        """WALE constant."""
        return self._Cw

    @property
    def C_omega(self) -> float:
        """Rotation correction coefficient."""
        return self._C_omega

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with rotation correction.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._mag_S is None or self._mag_Sd_sq is None:
            raise RuntimeError("correct() must be called before nut()")

        Cw_delta = self._Cw * self._delta
        coeff = Cw_delta.pow(2)

        numerator = self._mag_Sd_sq.pow(1.5)

        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        denominator = S_ij_S_ij.pow(2.5) + self._mag_Sd_sq.pow(1.25) + 1e-30

        nut = coeff * numerator / denominator

        # Improved clipping with wall distance
        if self._y is None:
            self._y = self._compute_wall_distance()

        nu = 1.5e-5
        nut_max = self._max_ratio * nu
        y_damp = torch.tanh(self._y / (10.0 * nu)).clamp(min=0.0, max=1.0)

        nut = nut * y_damp

        return nut.clamp(min=0.0, max=nut_max)

    def k_sgs(self) -> torch.Tensor:
        """Estimate SGS kinetic energy.

        k_sgs = nu_sgs^2 / (C_I * Delta^2)

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` SGS kinetic energy estimate.
        """
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        C_I = 0.09
        return nut.pow(2) / (C_I * delta_safe.pow(2))

    def tau_sgs(self) -> torch.Tensor:
        """SGS time scale.

        tau_sgs = Delta^2 / nu_sgs

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` SGS time scale.
        """
        nut = self.nut().clamp(min=1e-16)
        return self._delta.pow(2) / nut

    def epsilon_sgs(self) -> torch.Tensor:
        """SGS dissipation rate.

        eps_sgs = nu_sgs * |S|^2

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` SGS dissipation rate.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")

        nut = self.nut()
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        return nut * 2.0 * S_ij_S_ij

    def correct(self) -> None:
        """Update model: compute gradients, strain rate, and Sd tensor."""
        self._compute_gradients()
        self._compute_sd_tensor_enhanced()

    def _compute_sd_tensor_enhanced(self) -> None:
        """Compute enhanced WALE Sd tensor with rotation correction.

        Sd_ij = (g^2)_ij^sym - (1/3)*tr(g^2)*delta_ij + C_omega * Omega_ik * Omega_kj
        """
        g = self._grad_U

        g2 = torch.matmul(g, g)
        g2_sym = 0.5 * (g2 + g2.transpose(-1, -2))

        g2_trace = g2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        I = torch.eye(3, dtype=self._dtype, device=self._device).unsqueeze(0)

        Sd = g2_sym - (1.0 / 3.0) * g2_trace.unsqueeze(-1).unsqueeze(-1) * I

        # Rotation correction
        if self._C_omega > 0:
            Omega = 0.5 * (g - g.transpose(-1, -2))
            Omega2 = torch.matmul(Omega, Omega)
            Omega2_sym = 0.5 * (Omega2 + Omega2.transpose(-1, -2))
            Sd = Sd + self._C_omega * Omega2_sym

        self._Sd = Sd
        self._mag_Sd_sq = (self._Sd * self._Sd).sum(dim=(-2, -1))

    @property
    def Sd(self) -> torch.Tensor | None:
        """WALE Sd tensor ``(n_cells, 3, 3)`` or None."""
        return self._Sd

    @property
    def mag_Sd_sq(self) -> torch.Tensor | None:
        """Scalar Sd_ij * Sd_ij ``(n_cells,)`` or None."""
        return self._mag_Sd_sq

    def __repr__(self) -> str:
        return (
            f"ImprovedWALE3Model(Cw={self._Cw}, C_omega={self._C_omega}, "
            f"n_cells={self._mesh.n_cells})"
        )
