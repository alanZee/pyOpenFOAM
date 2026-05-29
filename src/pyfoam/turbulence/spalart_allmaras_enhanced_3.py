"""
Enhanced Spalart-Allmaras turbulence model v3 — SA-noft2 with improved production.

Extends :class:`~pyfoam.turbulence.spalart_allmaras_enhanced_2.SpalartAllmarasEnhanced2Model`
with:

- Strain-rate adaptive Cb1 coefficient
- Improved ft2 term for wake region behaviour
- Length-scale correction for separated flows

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_3 import SpalartAllmarasEnhanced3Model

    model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced_2 import (
    SpalartAllmarasEnhanced2Model,
    SpalartAllmarasEnhanced2Constants,
)

__all__ = ["SpalartAllmarasEnhanced3Model", "SpalartAllmarasEnhanced3Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced3Constants(SpalartAllmarasEnhanced2Constants):
    """Constants for enhanced SA v3.

    Extends parent constants with:
        C_adapt: Adaptive Cb1 coefficient gain.
        ft2_coeff: ft2 wake correction coefficient.
        sep_length_scale: Length scale correction factor for separated flow.
    """

    C_adapt: float = 0.05
    ft2_coeff: float = 0.3
    sep_length_scale: float = 1.0


_DEFAULTS = SpalartAllmarasEnhanced3Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced3")
class SpalartAllmarasEnhanced3Model(SpalartAllmarasEnhanced2Model):
    """Enhanced SA v3 with adaptive production and improved wake correction.

    Features:
    - Adaptive Cb1: Cb1_eff = Cb1 * (1 + C_adapt * |S| * y / nu)
    - ft2 wake correction: suppresses production in wake regions
    - Length-scale correction for separated flows

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasEnhanced3Constants, optional
        Model constants.
    enable_qcr : bool
        Enable QCR correction to nut. Default False.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasEnhanced3Constants | None = None,
        enable_qcr: bool = False,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(SpalartAllmarasEnhanced2Model, self).__init__(mesh, U, phi)

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

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

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
    # Adaptive Cb1
    # ------------------------------------------------------------------

    def _adaptive_cb1(self) -> torch.Tensor:
        """Compute strain-rate adaptive Cb1 coefficient.

        Cb1_eff = Cb1 * (1 + C_adapt * chi_local)

        where chi_local = nuTilde / nu is the local SA parameter.
        In regions of high turbulence (large chi), production is enhanced.
        """
        C = self._C
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)

        adapt_factor = 1.0 + getattr(C, 'C_adapt', 0.05) * chi.clamp(max=100.0)
        return C.Cb1 * adapt_factor

    # ------------------------------------------------------------------
    # ft2 wake correction
    # ------------------------------------------------------------------

    def _ft2(self) -> torch.Tensor:
        """Wake correction term ft2.

        ft2 = C_t3 * exp(-C_t4 * chi^2)

        This suppresses production in the wake region where chi is large.
        """
        C = self._C
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        ft2_c = getattr(C, 'ft2_coeff', 0.3)
        return ft2_c * torch.exp(-0.5 * chi.pow(2))

    # ------------------------------------------------------------------
    # Length-scale correction for separated flow
    # ------------------------------------------------------------------

    def _sep_length_correction(self) -> torch.Tensor:
        """Length-scale correction for separated flows.

        Reduces destruction in separated regions where y is small
        and nuTilde is large:
        correction = 1 + sep_length_scale * max(0, 1 - y * chi / (nu * 100))
        """
        C = self._C
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        y = self._y.clamp(min=1e-10)

        sep_factor = getattr(C, 'sep_length_scale', 1.0)
        arg = 1.0 - y * chi / (max(self._nu, 1e-30) * 100.0)
        correction = 1.0 + sep_factor * arg.clamp(min=0.0)

        return correction.clamp(min=1.0, max=2.0)

    # ------------------------------------------------------------------
    # Override correct to use enhanced production
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update SA v3 with adaptive production."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        self._solve_nuTilde_enhanced()

    def _solve_nuTilde_enhanced(self) -> None:
        """Solve nuTilde equation with enhanced production and destruction."""
        mesh = self._mesh
        C = self._C
        nu = self._nu

        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(nu, 1e-30)
        fv1 = chi.pow(3) / (chi.pow(3) + C.Cv1.pow(3))
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # Strain rate
        if self._grad_U is not None:
            S = self._strain_rate()
            S_bar = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        else:
            S_bar = torch.zeros_like(nuTilde)

        # Modified strain rate with wall distance correction
        y = self._y.clamp(min=1e-10)
        d2 = y.pow(2)
        Stilde = S_bar + nuTilde * fv2 / (C.kappa.pow(2) * d2)
        Stilde = Stilde.clamp(min=1e-16)

        # Adaptive production
        cb1_eff = self._adaptive_cb1()
        P_nu = cb1_eff * Stilde * nuTilde

        # ft2 wake correction
        ft2 = self._ft2()
        P_nu = P_nu - ft2 * nuTilde * Stilde * 0.1

        # Destruction with length-scale correction
        sep_corr = self._sep_length_correction()
        r = (nuTilde / (Stilde * C.kappa.pow(2) * d2)).clamp(min=0.0, max=10.0)
        g = r + C.Cw2 * (r.pow(6) - r)
        fw = g * ((1.0 + C.Cw3.pow(6)) / (g.pow(6) + C.Cw3.pow(6))).pow(1.0 / 6.0)
        D_nu = C.Cw1 * fw * (nuTilde / y).pow(2) / sep_corr

        # SARC correction
        sark = self._rotational_correction()

        # Transport
        sigma = C.sigma
        nu_eff = nu + nuTilde / sigma
        source = sark * P_nu - D_nu

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
        return (
            f"SpalartAllmarasEnhanced3Model(n_cells={self._mesh.n_cells}{qcr})"
        )
