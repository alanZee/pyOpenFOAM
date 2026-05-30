"""
Enhanced Spalart-Allmaras turbulence model v4 — SA-noft2 with trip-free corrections.

Extends :class:`~pyfoam.turbulence.spalart_allmaras_enhanced_3.SpalartAllmarasEnhanced3Model`
with:

- Quadratic constitutive relation (QCR2013) for improved Reynolds stress anisotropy
- Curvature correction (Dacles-Mariani et al., 1995)
- Improved ft2 model for separated flow predictions

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_4 import SpalartAllmarasEnhanced4Model

    model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced_3 import (
    SpalartAllmarasEnhanced3Model,
    SpalartAllmarasEnhanced3Constants,
)

__all__ = ["SpalartAllmarasEnhanced4Model", "SpalartAllmarasEnhanced4Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced4Constants(SpalartAllmarasEnhanced3Constants):
    """Constants for enhanced SA v4.

    Extends parent constants with:
        C_qcr: QCR correction coefficient.
        C_curv: Curvature correction coefficient.
        ft2_exp: Exponent for improved ft2 model.
    """

    C_qcr: float = 0.3
    C_curv: float = 0.5
    ft2_exp: float = 1.5


_DEFAULTS = SpalartAllmarasEnhanced4Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced4")
class SpalartAllmarasEnhanced4Model(SpalartAllmarasEnhanced3Model):
    """Enhanced SA v4 with QCR2013 and curvature correction.

    Features:
    - QCR2013: quadratic constitutive relation for Reynolds stress
    - Dacles-Mariani curvature correction
    - Improved ft2 with configurable exponent

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasEnhanced4Constants, optional
        Model constants.
    enable_qcr : bool
        Enable QCR correction. Default True.
    enable_curvature : bool
        Enable curvature correction. Default True.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasEnhanced4Constants | None = None,
        enable_qcr: bool = True,
        enable_curvature: bool = True,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced3Model, self).__init__(mesh, U, phi)

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
    # QCR2013 correction
    # ------------------------------------------------------------------

    def _qcr2013_correction(self, nut: torch.Tensor) -> torch.Tensor:
        """QCR2013 quadratic constitutive relation correction.

        tau_ij_QCR = tau_ij_linear + C_qcr * (O_ik * tau_kj - tau_ik * O_kj)

        This is applied as a scalar correction to nut for the eddy viscosity
        model: nut_QCR = nut * (1 + C_qcr * |Omega| / |S|)
        """
        if not self._enable_qcr:
            return nut

        if self._grad_U is None:
            return nut

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        C_qcr = getattr(C, 'C_qcr', 0.3)
        ratio = Omega_mag / S_mag.clamp(min=1e-16)
        factor = 1.0 + C_qcr * ratio

        return nut * factor.clamp(min=1.0, max=2.0)

    # ------------------------------------------------------------------
    # Curvature correction
    # ------------------------------------------------------------------

    def _curvature_correction(self) -> torch.Tensor:
        """Dacles-Mariani curvature correction for SA.

        f_curv = 1 / (1 + C_curv * (r_tilde)^2)

        where r_tilde = Omega / |S|  (rotation-to-strain ratio).
        Reduces production in regions of strong rotation.
        """
        if not self._enable_curvature or self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        r_tilde = Omega_mag / S_mag.clamp(min=1e-16)
        C_curv = getattr(C, 'C_curv', 0.5)

        f_curv = 1.0 / (1.0 + C_curv * r_tilde.pow(2))
        return f_curv.clamp(min=0.5, max=1.0)

    # ------------------------------------------------------------------
    # Improved ft2
    # ------------------------------------------------------------------

    def _ft2(self) -> torch.Tensor:
        """Improved ft2 wake correction with configurable exponent.

        ft2 = C_t3 * exp(-C_t4 * chi^ft2_exp)
        """
        C = self._C
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        ft2_c = getattr(C, 'ft2_coeff', 0.3)
        ft2_exp = getattr(C, 'ft2_exp', 1.5)
        return ft2_c * torch.exp(-0.5 * chi.pow(ft2_exp))

    # ------------------------------------------------------------------
    # Override nut to apply QCR
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with QCR correction."""
        # Compute base nut from parent chain
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        Cv1_3 = self._C.Cv1 ** 3
        fv1 = chi.pow(3) / (chi.pow(3) + Cv1_3)
        nut_base = nuTilde * fv1

        # Apply QCR
        nut_qcr = self._qcr2013_correction(nut_base)
        return nut_qcr.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override correct with curvature correction
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update SA v4 with curvature correction."""
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
        """Solve nuTilde with enhanced production including curvature correction."""
        mesh = self._mesh
        C = self._C
        nu = self._nu

        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(nu, 1e-30)
        Cv1_3 = C.Cv1 ** 3
        fv1 = chi.pow(3) / (chi.pow(3) + Cv1_3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # Strain rate
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

        # Adaptive production with curvature correction
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
        curv = ", Curv" if self._enable_curvature else ""
        return (
            f"SpalartAllmarasEnhanced4Model(n_cells={self._mesh.n_cells}{qcr}{curv})"
        )
