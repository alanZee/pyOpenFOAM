"""
Spalart-Allmaras one-equation turbulence model (Spalart & Allmaras 1992).

Implements a one-equation model solving for the working variable ν̃
(Spalart-Allmaras variable).  The turbulent viscosity is computed as::

    μ_t = ρ ν̃ fv1

where fv1 is a wall-damping function.  This model is particularly
well-suited for external aerodynamic flows with attached or mildly
separated boundary layers.

Constants (OpenFOAM defaults):
    σ = 2/3, κ = 0.41, Cb1 = 0.1355, Cb2 = 0.622,
    Cw2 = 0.3, Cw3 = 2.0, Cv1 = 7.1, Ct1 = 1.0,
    Ct2 = 2.0, Ct3 = 1.1, Ct4 = 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["SpalartAllmarasModel", "SpalartAllmarasConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpalartAllmarasConstants:
    """Constants for the Spalart-Allmaras turbulence model.

    Attributes:
        sigma: Turbulent Prandtl number for ν̃ (σ = 2/3).
        kappa: Von Karman constant (κ = 0.41).
        Cb1: Production coefficient (Cb1 = 0.1355).
        Cb2: Secondary diffusion coefficient (Cb2 = 0.622).
        Cw2: Destruction coefficient (Cw2 = 0.3).
        Cw3: Destruction coefficient (Cw3 = 2.0).
        Cv1: Wall-damping constant (Cv1 = 7.1).
        Ct1: Trip term constant (Ct1 = 1.0).
        Ct2: Trip term constant (Ct2 = 2.0).
        Ct3: Trip term constant (Ct3 = 1.1).
        Ct4: Trip term constant (Ct4 = 0.5).
    """

    sigma: float = 2.0 / 3.0
    kappa: float = 0.41
    Cb1: float = 0.1355
    Cb2: float = 0.622
    Cw2: float = 0.3
    Cw3: float = 2.0
    Cv1: float = 7.1
    Ct1: float = 1.0
    Ct2: float = 2.0
    Ct3: float = 1.1
    Ct4: float = 0.5


_DEFAULT_CONSTANTS = SpalartAllmarasConstants()


# ---------------------------------------------------------------------------
# Spalart-Allmaras model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("SpalartAllmaras")
class SpalartAllmarasModel(TurbulenceModel):
    """Spalart-Allmaras one-equation turbulence model.

    Solves a transport equation for the working variable ν̃ and computes
    turbulent viscosity as μ_t = ρ ν̃ fv1.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasConstants, optional
        Model constants.  Defaults to OpenFOAM standard values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Working variable ν̃ (initialised to small value)
        self._nuTilde = torch.full(
            (n_cells,), 1e-4, device=device, dtype=dtype
        )

        # Velocity gradient tensor
        self._grad_U: torch.Tensor | None = None

        # Wall distance
        self._y = self._compute_wall_distance()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nuTilde_field(self) -> torch.Tensor:
        """Working variable ν̃ ``(n_cells,)``."""
        return self._nuTilde

    @nuTilde_field.setter
    def nuTilde_field(self, value: torch.Tensor) -> None:
        self._nuTilde = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity: μ_t = ν̃ fv1.

        fv1 = χ³ / (χ³ + Cv1³) where χ = ν̃ / ν

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        fv1 = chi**3 / (chi**3 + self._C.Cv1**3)
        return nuTilde * fv1

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy (approximated).

        k ≈ ν̃ √(2Ω:Ω) / √(Cμ)  (simplified estimate)
        """
        # For S-A model, k is not directly solved.
        # Use a simplified estimate: k ≈ ν̃ * |S| / sqrt(C_mu)
        # where C_mu ≈ 0.09 and |S| is strain rate magnitude
        C_mu = 0.09
        S_mag = self._strain_magnitude()
        return self._nuTilde.clamp(min=0.0) * S_mag / C_mu**0.5

    def correct(self) -> None:
        """Update the S-A model: compute nut, solve ν̃ equation."""
        # Compute velocity gradient tensor (n_cells, 3, 3)
        # fvc.grad only works with scalar fields, so compute component by component
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        # Solve ν̃ transport equation
        self._solve_nuTilde()

    # ------------------------------------------------------------------
    # Internal: transport equation
    # ------------------------------------------------------------------

    def _solve_nuTilde(self) -> None:
        """Solve the ν̃ transport equation.

        ∂ν̃/∂t + ∇·(U ν̃) = Cb1 (1 - ft2) Ŝ ν̃
                          + 1/σ [∇·((ν + ν̃) ∇ν̃) + Cb2 (∇ν̃)²]
                          - [Cw1 fw - Cb1 ft2 / κ²] (ν̃ / d)²
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._y.clamp(min=1e-10)

        # Compute χ = ν̃ / ν
        chi = nuTilde / max(self._nu, 1e-30)

        # fv1 = χ³ / (χ³ + Cv1³)
        fv1 = chi**3 / (chi**3 + C.Cv1**3)

        # fv2 = 1 - χ / (1 + χ fv1)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # ft2 = Ct3 exp(-Ct4 χ²) (trip term, often set to 0)
        ft2 = C.Ct3 * torch.exp(-C.Ct4 * chi**2)

        # Strain rate magnitude
        S_mag = self._strain_magnitude()

        # Ŝ = S + ν̃ / (κ² d²) fv2 (modified vorticity)
        S_hat = S_mag + nuTilde / (C.kappa**2 * y**2) * fv2

        # Production: Cb1 (1 - ft2) Ŝ ν̃
        production = C.Cb1 * (1.0 - ft2) * S_hat * nuTilde

        # Destruction: [Cw1 fw - Cb1 ft2 / κ²] (ν̃ / d)²
        # Cw1 = Cb1 / κ² + (1 + Cb2) / σ
        Cw1 = C.Cb1 / C.kappa**2 + (1.0 + C.Cb2) / C.sigma

        # r = ν̃ / (Ŝ κ² d²)
        r = (nuTilde / (S_hat * C.kappa**2 * y**2)).clamp(max=10.0)

        # g = r + Cw2 (r⁶ - r)
        g = r + C.Cw2 * (r**6 - r)

        # fw = g [(1 + Cw3⁶) / (g⁶ + Cw3⁶)]^(1/6)
        fw = g * ((1.0 + C.Cw3**6) / (g**6 + C.Cw3**6)) ** (1.0 / 6.0)

        destruction = (Cw1 * fw - C.Cb1 * ft2 / C.kappa**2) * (nuTilde / y) ** 2

        # Effective diffusivity: (ν + ν̃) / σ
        nu_eff = (self._nu + nuTilde) / C.sigma

        # Build equation: convection + diffusion = source
        eqn = fvm.div(self._phi, nuTilde, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, nuTilde, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: production - destruction
        # Note: Cb2 term (∇ν̃)² is implicit in the Laplacian treatment
        source = production - destruction
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        nuTilde_new = eqn.source / diag_safe
        self._nuTilde = nuTilde_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Internal: helper computations
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Compute strain rate tensor S = 0.5 (∇U + ∇U^T).

        Returns:
            ``(n_cells, 3, 3)`` strain rate tensor.
        """
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _strain_magnitude(self) -> torch.Tensor:
        """Compute strain rate magnitude |S| = √(2 S:S).

        Returns:
            ``(n_cells,)`` strain rate magnitude.
        """
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute approximate wall distance for each cell.

        Uses distance from cell centre to the nearest boundary face centre.

        Returns:
            ``(n_cells,)`` wall distance.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        cell_centres = mesh.cell_centres  # (n_cells, 3)
        face_centres = mesh.face_centres  # (n_faces, 3)

        # Get boundary face centres
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_internal:
            bnd_centres = face_centres[n_internal:]
        else:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        n_bnd = bnd_centres.shape[0]
        if n_bnd == 0:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        # For each cell, find distance to nearest boundary face centre
        diff = cell_centres.unsqueeze(1) - bnd_centres.unsqueeze(0)
        dist = diff.norm(dim=2)
        y = dist.min(dim=1).values

        return y.clamp(min=1e-6)
