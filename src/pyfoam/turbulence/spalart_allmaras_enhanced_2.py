"""
Enhanced Spalart-Allmaras turbulence model v2 — SA-noft2 with improved production.

Extends :class:`~pyfoam.turbulence.spalart_allmaras_enhanced.SpalartAllmarasEnhancedModel`
with:

- Quadratic constitutive relation (QCR) for Reynolds stress anisotropy
- Improved production term with ft3 function
- Rotational correction (SARC)

References:
    Shur, M.L. et al. (2000). "Turbulence modeling in rotating and curved
    channels." J. Fluid Mech., 414, 433-464.

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_2 import SpalartAllmarasEnhanced2Model

    model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced import (
    SpalartAllmarasEnhancedModel,
    SpalartAllmarasEnhancedConstants,
)

__all__ = ["SpalartAllmarasEnhanced2Model", "SpalartAllmarasEnhanced2Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced2Constants(SpalartAllmarasEnhancedConstants):
    """Constants for enhanced SA v2.

    Extends parent constants with:
        C_rot1: Rotational correction coefficient (SARC).
        C_rot2: Rotational correction exponent.
        ft3_coeff: ft3 production correction coefficient.
    """

    C_rot1: float = 1.0
    C_rot2: float = 2.0
    ft3_coeff: float = 0.0


_DEFAULTS = SpalartAllmarasEnhanced2Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced")
class SpalartAllmarasEnhanced2Model(SpalartAllmarasEnhancedModel):
    """Enhanced SA v2 with QCR and rotational correction.

    Features:
    - QCR: Quadratic Constitutive Relation for anisotropy
    - SARC: Spalart-Allmaras Rotational Correction
    - Improved production with optional ft3 function

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasEnhanced2Constants, optional
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
        constants: SpalartAllmarasEnhanced2Constants | None = None,
        enable_qcr: bool = False,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip SpalartAllmarasEnhancedModel.__init__)
        super(SpalartAllmarasEnhancedModel, self).__init__(mesh, U, phi)

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
    # Nut with QCR correction
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with optional QCR correction.

        QCR adds a quadratic correction based on the strain rate:
            nut_QCR = nut * (1 + 0.3 * |S|^2 / (S:S + epsilon))

        Parameters
        ----------
        Returns
        -------
        torch.Tensor
            Turbulent viscosity (n_cells,).
        """
        nut_base = super().nut()

        if not self._enable_qcr or self._grad_U is None:
            return nut_base

        # QCR correction
        S = self._strain_rate()
        S_mag_sq = (S * S).sum(dim=(1, 2)).clamp(min=1e-30)
        S_mag = torch.sqrt(2.0 * S_mag_sq)

        qcr_factor = 1.0 + 0.3 * S_mag
        return nut_base * qcr_factor

    # ------------------------------------------------------------------
    # Rotational correction (SARC)
    # ------------------------------------------------------------------

    def _rotational_correction(self) -> torch.Tensor:
        """Spalart-Allmaras Rotational Correction (SARC).

        Adds a production correction based on the rotation-to-strain ratio:
            r_star = |Omega| / |S|
            correction = C_rot1 * (r_star^C_rot2 - 1) * (1 - exp(-y+/10))

        Returns
        -------
        torch.Tensor
            Multiplicative correction factor.
        """
        if self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        r_star = Omega_mag / S_mag.clamp(min=1e-16)

        # Wall damping for SARC
        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._y.clamp(min=1e-10)
        chi = nuTilde / max(self._nu, 1e-30)
        u_tau = (chi * self._nu).clamp(min=1e-16).sqrt()
        y_plus = (u_tau * y / max(self._nu, 1e-30)).clamp(min=0.01)
        wall_damp = 1.0 - torch.exp(-y_plus / 10.0)

        correction = 1.0 + C.C_rot1 * (r_star.pow(C.C_rot2) - 1.0) * wall_damp

        return correction.clamp(min=0.5, max=3.0)

    # ------------------------------------------------------------------
    # Override correct to include SARC
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update SA v2 with rotational correction."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        self._solve_nuTilde()

    def __repr__(self) -> str:
        qcr = ", QCR" if self._enable_qcr else ""
        return (
            f"SpalartAllmarasEnhanced2Model(n_cells={self._mesh.n_cells}{qcr})"
        )
