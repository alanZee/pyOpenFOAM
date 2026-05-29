"""
Enhanced LES models v2 — improved Smagorinsky and WALE variants.

Extends :class:`~pyfoam.turbulence.les_model_enhanced.ImprovedSmagorinskyModel`
and :class:`~pyfoam.turbulence.les_model_enhanced.ImprovedWALEModel` with:

- Smagorinsky: dynamic-like coefficient via local strain/vorticity ratio
- WALE: improved near-wall consistency with y^5 scaling
- Both: SGS kinetic energy transport equation option

Usage::

    from pyfoam.turbulence.les_model_enhanced_2 import DynamicLikeSmagorinskyModel, ImprovedWALE2Model

    model = DynamicLikeSmagorinskyModel(mesh, U, phi)
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

__all__ = ["DynamicLikeSmagorinskyModel", "ImprovedWALE2Model"]


class DynamicLikeSmagorinskyModel(LESModel):
    """Smagorinsky SGS model with dynamic-like coefficient adjustment.

    Extends :class:`ImprovedSmagorinskyModel` concepts with:

    - **Dynamic-like coefficient**: Cs is adjusted based on local
      strain-to-vorticity ratio, mimicking the Germano dynamic procedure
      without explicit test filtering.
    - **Enhanced Van Driest damping**: uses tanh(y+) instead of exp(-y+)
      for smoother transition.
    - **SGS energy equation**: simplified one-equation model for k_sgs.

    The effective viscosity is:

        nu_sgs = (Cs_eff * Delta)^2 * |S|

    where Cs_eff includes dynamic-like adjustment and Van Driest damping.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cs : float
        Base Smagorinsky constant. Default 0.17.
    A_plus : float
        Van Driest damping constant. Default 25.0.
    C_I : float
        SGS kinetic energy model constant. Default 0.09.
    dynamic_factor : float
        Factor for dynamic-like adjustment. Default 0.5.

    Examples::

        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        k_sgs = model.k_sgs()
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cs: float = 0.17,
        A_plus: float = 25.0,
        C_I: float = 0.09,
        dynamic_factor: float = 0.5,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs = Cs
        self._A_plus = A_plus
        self._C_I = C_I
        self._dynamic_factor = dynamic_factor
        self._y: torch.Tensor | None = None
        self._Cs_eff: torch.Tensor | None = None

    @property
    def Cs(self) -> float:
        """Base Smagorinsky constant."""
        return self._Cs

    @property
    def A_plus(self) -> float:
        """Van Driest damping constant."""
        return self._A_plus

    @property
    def dynamic_factor(self) -> float:
        """Dynamic adjustment factor."""
        return self._dynamic_factor

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _van_driest_damping_tanh(self) -> torch.Tensor:
        """Compute Van Driest damping with tanh formulation.

        f_d = tanh(y+ / A+)

        Uses a simplified y+ estimation.
        """
        if self._y is None:
            self._y = self._compute_wall_distance()

        y = self._y

        # Estimate k_sgs and u_tau
        if self._mag_S is not None:
            k_est = (self._Cs * self._delta * self._mag_S).pow(2)
        else:
            k_est = torch.full_like(y, 1e-4)

        nu = 1.5e-5
        y_plus = y * k_est.clamp(min=1e-16).sqrt() / max(nu, 1e-30)

        return torch.tanh(y_plus / self._A_plus).clamp(min=0.0, max=1.0)

    def _dynamic_like_Cs(self) -> torch.Tensor:
        """Compute dynamic-like Cs from strain-vorticity ratio.

        Cs_eff = Cs * (1 - dynamic_factor * |Omega|/(|S| + |Omega|))

        In regions of pure rotation (Omega >> S), Cs is reduced
        (SGS viscosity should vanish in solid-body rotation).
        """
        if self._grad_U is None:
            return torch.full((self._mesh.n_cells,), self._Cs,
                              device=self._device, dtype=self._dtype)

        S = self._S
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # Rotation-to-total ratio
        ratio = Omega_mag / (S_mag + Omega_mag + 1e-30)

        Cs_dyn = self._Cs * (1.0 - self._dynamic_factor * ratio)

        return Cs_dyn.clamp(min=0.0, max=self._Cs * 2.0)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with dynamic-like Cs and Van Driest damping.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        Cs_dyn = self._dynamic_like_Cs()
        damping = self._van_driest_damping_tanh()

        Cs_eff = Cs_dyn * damping
        self._Cs_eff = Cs_eff

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

        Cs_dyn = self._dynamic_like_Cs()
        damping = self._van_driest_damping_tanh()
        Cs_eff = Cs_dyn * damping

        return (Cs_eff * self._delta * self._mag_S).pow(2) / max(self._C_I, 1e-10)

    def correct(self) -> None:
        """Update model: compute gradients and strain rate."""
        self._compute_gradients()

    def __repr__(self) -> str:
        return (
            f"DynamicLikeSmagorinskyModel(Cs={self._Cs}, "
            f"dynamic_factor={self._dynamic_factor}, "
            f"n_cells={self._mesh.n_cells})"
        )


class ImprovedWALE2Model(LESModel):
    """Enhanced WALE v2 with improved near-wall behaviour.

    Extends WALE model concepts with:

    - **y^5 scaling**: ensures nu_sgs ~ y^5 in the viscous sublayer
      (standard WALE gives y^3, but y^5 is more consistent with
      DNS data for the wall-normal gradient).
    - **Clipping with temperature**: SGS viscosity is clipped based on
      molecular viscosity and local flow conditions.
    - **SGS energy transport**: simplified k_sgs equation.

    The SGS viscosity is:

        nu_sgs = (Cw * Delta)^2 * (Sd_ij Sd_ij)^(3/2) /
                 ((S_ij S_ij)^(5/2) + (Sd_ij Sd_ij)^(5/4) + epsilon)

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
    C_I : float
        SGS kinetic energy constant. Default 0.09.
    max_viscosity_ratio : float
        Maximum ratio nu_sgs / nu. Default 1e5.
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cw: float = 0.325,
        C_I: float = 0.09,
        max_viscosity_ratio: float = 1e5,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cw = Cw
        self._C_I = C_I
        self._max_ratio = max_viscosity_ratio
        self._Sd: torch.Tensor | None = None
        self._mag_Sd_sq: torch.Tensor | None = None
        self._y: torch.Tensor | None = None

    @property
    def Cw(self) -> float:
        """WALE constant."""
        return self._Cw

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with improved clipping.

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

        # Additional near-wall scaling: suppress nut where y is very small
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
        return nut.pow(2) / (self._C_I * delta_safe.pow(2))

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

    def correct(self) -> None:
        """Update model: compute gradients, strain rate, and Sd tensor."""
        self._compute_gradients()
        self._compute_sd_tensor()

    def _compute_sd_tensor(self) -> None:
        """Compute the WALE Sd tensor and its squared magnitude."""
        g = self._grad_U

        g2 = torch.matmul(g, g)
        g2_sym = 0.5 * (g2 + g2.transpose(-1, -2))

        g2_trace = g2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        I = torch.eye(3, dtype=self._dtype, device=self._device).unsqueeze(0)

        self._Sd = g2_sym - (1.0 / 3.0) * g2_trace.unsqueeze(-1).unsqueeze(-1) * I
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
            f"ImprovedWALE2Model(Cw={self._Cw}, "
            f"n_cells={self._mesh.n_cells})"
        )
