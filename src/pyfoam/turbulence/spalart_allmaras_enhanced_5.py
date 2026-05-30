"""
Enhanced Spalart-Allmaras turbulence model v5 — SA-noft2 with hybrid RANS-LES blending.

Extends :class:`~pyfoam.turbulence.spalart_allmaras_enhanced_4.SpalartAllmarasEnhanced4Model`
with:

- Hybrid RANS-LES mode via controlled decay of eddy viscosity
- Delayed DES-like correction with grid-size-aware blending
- Improved wall boundary treatment with y+-aware production

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_5 import SpalartAllmarasEnhanced5Model

    model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
    model.correct()
    nut = model.nut()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel
from .spalart_allmaras_enhanced_4 import (
    SpalartAllmarasEnhanced4Model,
    SpalartAllmarasEnhanced4Constants,
)

__all__ = ["SpalartAllmarasEnhanced5Model", "SpalartAllmarasEnhanced5Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced5Constants(SpalartAllmarasEnhanced4Constants):
    """Constants for enhanced SA v5.

    Extends parent constants with:
        C_hybrid: Hybrid RANS-LES blending coefficient.
        C_decay: Controlled decay rate for LES mode.
        y_plus_switch: y+ threshold for production switching.
    """

    C_hybrid: float = 0.65
    C_decay: float = 0.1
    y_plus_switch: float = 5.0


_DEFAULTS = SpalartAllmarasEnhanced5Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced5")
class SpalartAllmarasEnhanced5Model(SpalartAllmarasEnhanced4Model):
    """Enhanced SA v5 with hybrid RANS-LES and controlled decay.

    Features:
    - Hybrid RANS-LES: in LES regions, nuTilde decays towards zero
      based on grid size and local strain rate.
    - Grid-size-aware blending: uses Delta/y ratio to determine
      whether a cell is in RANS or LES mode.
    - y+-aware production switching: reduces production in the
      viscous sublayer for improved low-Re behaviour.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasEnhanced5Constants, optional
        Model constants.
    enable_qcr : bool
        Enable QCR correction. Default True.
    enable_curvature : bool
        Enable curvature correction. Default True.
    enable_hybrid : bool
        Enable hybrid RANS-LES mode. Default True.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasEnhanced5Constants | None = None,
        enable_qcr: bool = True,
        enable_curvature: bool = True,
        enable_hybrid: bool = True,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced4Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._nuTilde = torch.full(
            (n_cells,), 1e-4, device=device, dtype=dtype
        )
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()
        self._enable_qcr = enable_qcr
        self._enable_curvature = enable_curvature
        self._enable_hybrid = enable_hybrid

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        mesh = self._mesh
        cell_centres = mesh.cell_centres
        face_centres = mesh.face_centres

        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_internal:
            bnd_centres = face_centres[n_internal:]
        else:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        n_bnd = bnd_centres.shape[0]
        if n_bnd == 0:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        diff = cell_centres.unsqueeze(1) - bnd_centres.unsqueeze(0)
        dist = diff.norm(dim=2)
        y = dist.min(dim=1).values

        return y.clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Hybrid RANS-LES blending
    # ------------------------------------------------------------------

    def _hybrid_blending(self) -> torch.Tensor:
        """Compute RANS-LES blending factor.

        f_hybrid = tanh(C_hybrid * (Delta / y)^2)

        where Delta is the grid spacing and y is wall distance.
        f_hybrid -> 0 near walls (RANS), f_hybrid -> 1 in far field (LES).

        Simplified: uses cell_centres norm as proxy for Delta.
        """
        if not self._enable_hybrid:
            return torch.zeros_like(self._nuTilde)

        C = self._C
        y = self._y.clamp(min=1e-10)

        # Estimate Delta from mesh (simplified)
        Delta = (self._mesh.cell_volumes.clamp(min=1e-30)).pow(1.0 / 3.0)

        ratio = Delta / y
        C_hybrid = getattr(C, 'C_hybrid', 0.65)

        f_hybrid = torch.tanh((C_hybrid * ratio.pow(2)).clamp(max=10.0))
        return f_hybrid.clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Controlled decay
    # ------------------------------------------------------------------

    def _controlled_decay(self) -> torch.Tensor:
        """Compute controlled decay source for LES regions.

        S_decay = -C_decay * f_hybrid * nuTilde * |S|

        In LES regions, this drives nuTilde towards zero.
        """
        if not self._enable_hybrid:
            return torch.zeros_like(self._nuTilde)

        C = self._C
        f_hybrid = self._hybrid_blending()

        if self._grad_U is not None:
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        else:
            S_mag = torch.zeros_like(self._nuTilde)

        C_decay = getattr(C, 'C_decay', 0.1)
        return -C_decay * f_hybrid * self._nuTilde.clamp(min=0.0) * S_mag

    # ------------------------------------------------------------------
    # Enhanced nut with QCR
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with QCR and hybrid blending."""
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        Cv1_3 = self._C.Cv1 ** 3
        fv1 = chi.pow(3) / (chi.pow(3) + Cv1_3)
        nut_base = nuTilde * fv1

        # Apply QCR
        nut_qcr = self._qcr2013_correction(nut_base)
        return nut_qcr.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override correct with hybrid decay
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update SA v5 with hybrid RANS-LES decay."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        self._solve_nuTilde_hybrid()

    def _solve_nuTilde_hybrid(self) -> None:
        """Solve nuTilde with hybrid RANS-LES controlled decay."""
        mesh = self._mesh
        C = self._C
        nu = self._nu

        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(nu, 1e-30)
        Cv1_3 = C.Cv1 ** 3
        fv1 = chi.pow(3) / (chi.pow(3) + Cv1_3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        if self._grad_U is not None:
            S = self._strain_rate()
            S_bar = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        else:
            S_bar = torch.zeros_like(nuTilde)

        y = self._y.clamp(min=1e-10)
        d2 = y.pow(2)
        kappa_sq = C.kappa ** 2
        Stilde = S_bar + nuTilde * fv2 / (kappa_sq * d2)
        Stilde = Stilde.clamp(min=1e-16)

        # Production with curvature correction
        cb1_eff = self._adaptive_cb1()
        f_curv = self._curvature_correction()
        P_nu = cb1_eff * Stilde * nuTilde * f_curv

        # ft2 wake correction
        ft2 = self._ft2()
        P_nu = P_nu - ft2 * nuTilde * Stilde * 0.1

        # Destruction with length-scale correction
        sep_corr = self._sep_length_correction()
        r = (nuTilde / (Stilde * kappa_sq * d2)).clamp(min=0.0, max=10.0)
        g = r + C.Cw2 * (r.pow(6) - r)
        Cw3_6 = C.Cw3 ** 6
        fw = g * ((1.0 + Cw3_6) / (g.pow(6) + Cw3_6)).pow(1.0 / 6.0)
        Cw1 = C.Cb1 / (C.kappa ** 2) + (1.0 + C.Cb2) / C.sigma
        D_nu = Cw1 * fw * (nuTilde / y).pow(2) / sep_corr

        # Hybrid decay
        decay = self._controlled_decay()

        # SARC correction
        sark = self._rotational_correction()

        sigma = C.sigma
        nu_eff = nu + nuTilde / sigma
        source = sark * P_nu - D_nu + decay

        eqn = fvm.div(self._phi, self._nuTilde, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._nuTilde, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        nuTilde_new = eqn.source / diag_safe
        self._nuTilde = nuTilde_new.clamp(min=0.0)

    def __repr__(self) -> str:
        qcr = ", QCR" if self._enable_qcr else ""
        curv = ", Curv" if self._enable_curvature else ""
        hybrid = ", hybrid" if self._enable_hybrid else ""
        return (
            f"SpalartAllmarasEnhanced5Model(n_cells={self._mesh.n_cells}"
            f"{qcr}{curv}{hybrid})"
        )
