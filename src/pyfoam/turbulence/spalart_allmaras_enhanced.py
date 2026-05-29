"""
Enhanced Spalart-Allmaras turbulence model — SA-noft2 variant.

Implements the SA model following Spalart & Allmaras (1994) with
the noft2 (no ft2 trip term) variant commonly used in OpenFOAM:

- Trip term ft2 removed (ft2 = 0)
- Simplified production/destruction balance
- Improved numerical stability

The SA-noft2 variant is the default in OpenFOAM and many production
codes. It removes the transition-related trip term, making the model
fully turbulent.

References:
    Spalart, P.R. & Allmaras, S.R. (1994). "A one-equation turbulence
    model for aerodynamic flows." La Recherche Aerospatiale, 1, 5-21.

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced import SpalartAllmarasEnhancedModel

    model = SpalartAllmarasEnhancedModel(mesh, U, phi)
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

__all__ = ["SpalartAllmarasEnhancedModel", "SpalartAllmarasEnhancedConstants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhancedConstants:
    """Constants for the SA-noft2 model.

    Attributes:
        sigma: Turbulent Prandtl number for nu_tilde (sigma = 2/3).
        kappa: Von Karman constant (kappa = 0.41).
        Cb1: Production coefficient (Cb1 = 0.1355).
        Cb2: Secondary diffusion coefficient (Cb2 = 0.622).
        Cw2: Destruction coefficient (Cw2 = 0.3).
        Cw3: Destruction coefficient (Cw3 = 2.0).
        Cv1: Wall-damping constant (Cv1 = 7.1).
    """

    sigma: float = 2.0 / 3.0
    kappa: float = 0.41
    Cb1: float = 0.1355
    Cb2: float = 0.622
    Cw2: float = 0.3
    Cw3: float = 2.0
    Cv1: float = 7.1


_DEFAULTS = SpalartAllmarasEnhancedConstants()


@TurbulenceModel.register("SpalartAllmarasNoft2")
class SpalartAllmarasEnhancedModel(TurbulenceModel):
    """SA-noft2 one-equation turbulence model.

    Solves a transport equation for nu_tilde (SA working variable)
    with ft2 = 0 (no trip term). This is the standard variant used
    in most production CFD codes.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasEnhancedConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasEnhancedConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._nuTilde = torch.full(
            (n_cells,), 1e-4, device=device, dtype=dtype
        )
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nuTilde_field(self) -> torch.Tensor:
        """Working variable nu_tilde ``(n_cells,)``."""
        return self._nuTilde

    @nuTilde_field.setter
    def nuTilde_field(self, value: torch.Tensor) -> None:
        self._nuTilde = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity: mu_t = rho * nu_tilde * fv1.

        fv1 = chi^3 / (chi^3 + Cv1^3) where chi = nu_tilde / nu
        """
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        fv1 = chi**3 / (chi**3 + self._C.Cv1**3)
        return nuTilde * fv1

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy (approximated from nu_tilde).

        k ~ nu_tilde * |S| / sqrt(C_mu)
        Returns zero if velocity gradient not yet computed.
        """
        if self._grad_U is None:
            return torch.zeros_like(self._nuTilde)
        C_mu = 0.09
        S_mag = self._strain_magnitude()
        return self._nuTilde.clamp(min=0.0) * S_mag / C_mu**0.5

    def correct(self) -> None:
        """Update the SA-noft2 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        self._solve_nuTilde()

    # ------------------------------------------------------------------
    # Transport equation (SA-noft2: ft2 = 0)
    # ------------------------------------------------------------------

    def _solve_nuTilde(self) -> None:
        """Solve the nu_tilde transport equation.

        With ft2 = 0 (noft2 variant):
        dnu_tilde/dt + div(U nu_tilde) = Cb1 * Shat * nu_tilde
            + 1/sigma [div((nu + nu_tilde) grad(nu_tilde)) + Cb2 (grad(nu_tilde))^2]
            - Cw1 * fw * (nu_tilde / d)^2
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._y.clamp(min=1e-10)

        # chi = nu_tilde / nu
        chi = nuTilde / max(self._nu, 1e-30)

        # fv1 = chi^3 / (chi^3 + Cv1^3)
        fv1 = chi**3 / (chi**3 + C.Cv1**3)

        # fv2 = 1 - chi / (1 + chi * fv1)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # Strain rate magnitude
        S_mag = self._strain_magnitude()

        # Shat = S + nu_tilde / (kappa^2 d^2) fv2 (modified vorticity)
        S_hat = S_mag + nuTilde / (C.kappa**2 * y**2) * fv2
        S_hat = S_hat.clamp(min=1e-10)

        # Production: Cb1 * Shat * nu_tilde (no ft2 term)
        production = C.Cb1 * S_hat * nuTilde

        # Destruction
        # Cw1 = Cb1/kappa^2 + (1 + Cb2)/sigma
        Cw1 = C.Cb1 / C.kappa**2 + (1.0 + C.Cb2) / C.sigma

        # r = nu_tilde / (Shat * kappa^2 * d^2)
        r = (nuTilde / (S_hat * C.kappa**2 * y**2)).clamp(max=10.0)

        # g = r + Cw2 (r^6 - r)
        g = r + C.Cw2 * (r**6 - r)

        # fw = g [(1 + Cw3^6) / (g^6 + Cw3^6)]^(1/6)
        fw = g * ((1.0 + C.Cw3**6) / (g**6 + C.Cw3**6)) ** (1.0 / 6.0)

        destruction = Cw1 * fw * (nuTilde / y) ** 2

        # Effective diffusivity: (nu + nu_tilde) / sigma
        nu_eff = (self._nu + nuTilde) / C.sigma

        # Build equation
        eqn = fvm.div(self._phi, nuTilde, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, nuTilde, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: production - destruction
        source = production - destruction
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        nuTilde_new = eqn.source / diag_safe
        self._nuTilde = nuTilde_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Strain rate tensor S = 0.5 (grad(U) + grad(U)^T)."""
        return 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))

    def _strain_magnitude(self) -> torch.Tensor:
        """Magnitude of strain rate |S| = sqrt(2 S:S)."""
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

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

    def __repr__(self) -> str:
        return f"SpalartAllmarasEnhancedModel(n_cells={self._mesh.n_cells})"
