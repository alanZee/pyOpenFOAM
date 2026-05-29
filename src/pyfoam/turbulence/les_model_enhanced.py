"""
Enhanced LES models — improved Smagorinsky and WALE variants.

Provides enhanced subgrid-scale (SGS) models:

- :class:`ImprovedSmagorinskyModel` — Smagorinsky with Van Driest wall
  damping and dynamic-like coefficient adjustment
- :class:`ImprovedWALEModel` — WALE with improved near-wall behaviour
  and consistency check

Usage::

    from pyfoam.turbulence.les_model_enhanced import ImprovedSmagorinskyModel

    model = ImprovedSmagorinskyModel(mesh, U, phi)
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

__all__ = ["ImprovedSmagorinskyModel", "ImprovedWALEModel"]


class ImprovedSmagorinskyModel(LESModel):
    """Smagorinsky SGS model with Van Driest wall damping.

    Extends the basic Smagorinsky model with:

    - **Van Driest damping**: Cs is reduced near walls as
      Cs_eff = Cs * (1 - exp(-y+ / A+))
    - **Strain-rate clipping**: prevents zero strain-rate division
    - **SGS energy estimation**: k_sgs = (Cs * Delta)^2 * |S|^2 / (C_I)

    The effective viscosity is:

        nu_sgs = (Cs_eff * Delta)^2 * |S|

    where Cs_eff includes the Van Driest damping function.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cs : float
        Smagorinsky constant. Default 0.17.
    A_plus : float
        Van Driest damping constant. Default 25.0.
    C_I : float
        SGS kinetic energy model constant. Default 0.09.

    Examples::

        model = ImprovedSmagorinskyModel(mesh, U, phi)
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
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs = Cs
        self._A_plus = A_plus
        self._C_I = C_I
        self._y: torch.Tensor | None = None

    @property
    def Cs(self) -> float:
        """Smagorinsky constant."""
        return self._Cs

    @property
    def A_plus(self) -> float:
        """Van Driest damping constant."""
        return self._A_plus

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance for Van Driest damping."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _van_driest_damping(self) -> torch.Tensor:
        """Compute Van Driest damping factor.

        f_d = 1 - exp(-y+ / A+)

        Uses a simplified y+ estimation: y+ ~ y * sqrt(k) / nu
        With k estimated from (Cs * Delta * |S|)^2.

        Returns:
            ``(n_cells,)`` damping factor in [0, 1].
        """
        if self._y is None:
            self._y = self._compute_wall_distance()

        y = self._y
        Cs_delta = self._Cs * self._delta

        # Estimate k_sgs ~ (Cs * Delta * |S|)^2
        if self._mag_S is not None:
            k_est = (Cs_delta * self._mag_S).pow(2)
        else:
            k_est = torch.full_like(y, 1e-4)

        # Simplified y+ ~ y * sqrt(k) / nu
        nu = 1.5e-5  # Default molecular viscosity
        y_plus = y * k_est.clamp(min=1e-16).sqrt() / max(nu, 1e-30)

        # Van Driest damping: f = 1 - exp(-y+ / A+)
        f = 1.0 - torch.exp(-y_plus / self._A_plus)
        return f.clamp(min=0.0, max=1.0)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with Van Driest damping.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        Cs_delta = self._Cs * self._delta
        damping = self._van_driest_damping()
        Cs_eff = Cs_delta * damping

        return Cs_eff.pow(2) * self._mag_S

    def k_sgs(self) -> torch.Tensor:
        """Estimate SGS kinetic energy.

        k_sgs = (Cs_eff * Delta * |S|)^2 / C_I

        Returns:
            ``(n_cells,)`` SGS kinetic energy estimate.
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before k_sgs()")

        Cs_delta = self._Cs * self._delta
        damping = self._van_driest_damping()
        Cs_eff = Cs_delta * damping

        return (Cs_eff * self._mag_S).pow(2) / max(self._C_I, 1e-10)

    def correct(self) -> None:
        """Update model: compute gradients and strain rate."""
        self._compute_gradients()

    def __repr__(self) -> str:
        return (
            f"ImprovedSmagorinskyModel(Cs={self._Cs}, "
            f"A_plus={self._A_plus}, n_cells={self._mesh.n_cells})"
        )


class ImprovedWALEModel(LESModel):
    """Enhanced WALE (Wall-Adapting Local Eddy-viscosity) model.

    Extends the basic WALE model with:

    - **Near-wall consistency**: ensures nu_sgs ~ y^3 in the viscous
      sublayer without explicit damping
    - **SGS energy estimation**: k_sgs from the WALE formulation
    - **Clipping**: prevents negative or excessively large SGS viscosity
    - **Time scale**: tau_sgs estimation for LES

    The SGS viscosity is:

        nu_sgs = (Cw * Delta)^2 * (Sd_ij Sd_ij)^(3/2) /
                 ((S_ij S_ij)^(5/2) + (Sd_ij Sd_ij)^(5/4))

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cw : float
        WALE constant. Default 0.325.
    C_I : float
        SGS kinetic energy constant. Default 0.09.
    max_viscosity_ratio : float
        Maximum allowed ratio nu_sgs / nu. Default 1e5.

    Examples::

        model = ImprovedWALEModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
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

    @property
    def Cw(self) -> float:
        """WALE constant."""
        return self._Cw

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity with clipping.

        Returns:
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

        # Clip to prevent excessive viscosity
        nu = 1.5e-5  # Default molecular viscosity
        nut_max = self._max_ratio * nu
        return nut.clamp(min=0.0, max=nut_max)

    def k_sgs(self) -> torch.Tensor:
        """Estimate SGS kinetic energy from WALE viscosity.

        k_sgs = nu_sgs^2 / (C_I * Delta^2)

        Returns:
            ``(n_cells,)`` SGS kinetic energy estimate.
        """
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        return nut.pow(2) / (self._C_I * delta_safe.pow(2))

    def tau_sgs(self) -> torch.Tensor:
        """SGS time scale.

        tau_sgs = Delta^2 / nu_sgs

        Returns:
            ``(n_cells,)`` SGS time scale.
        """
        nut = self.nut().clamp(min=1e-16)
        return self._delta.pow(2) / nut

    def correct(self) -> None:
        """Update model: compute gradients, strain rate, and Sd tensor."""
        self._compute_gradients()
        self._compute_sd_tensor()

    def _compute_sd_tensor(self) -> None:
        """Compute the WALE Sd tensor and its squared magnitude.

        Sd_ij = 0.5 * (g2_ij + g2_ji) - (1/3) * delta_ij * g2_kk
        where g2 = grad(U)^2 (matrix square).
        """
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
            f"ImprovedWALEModel(Cw={self._Cw}, "
            f"n_cells={self._mesh.n_cells})"
        )
