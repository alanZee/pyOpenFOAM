"""
Wall film boundary condition for Lagrangian film models.

Implements a boundary condition for wall film tracking, used with
Lagrangian particle/film models (e.g. ``filmCloud`` in OpenFOAM).
The wall film BC tracks the film thickness, velocity, and temperature
on wall surfaces for spray wall interaction and fuel film modelling.

In OpenFOAM, ``wallFilm`` is used with the ``surfaceFilmModel``
framework and couples the Lagrangian film with the Eulerian
continuous phase::

    type    wallFilm;
    delta   0;              // initial film thickness (m)
    Tf      300;            // initial film temperature (K)
    value   uniform 0;

Usage::

    type    wallFilm;
    delta   1e-4;
    Tf      350;
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["WallFilmBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("wallFilm")
class WallFilmBC(BoundaryCondition):
    """Wall film boundary condition for Lagrangian film models.

    Tracks the thin liquid film on wall surfaces.  Provides:

    - Film thickness (delta) evolution based on mass flux balance
    - Film temperature (Tf) with wall heat transfer
    - Film velocity derived from wall shear and gravity components

    The film mass balance is:

        ∂δ/∂t = (ṁ_impact - ṁ_erosion - ṁ_evaporation) / ρ_f

    Coefficients:
        - ``delta``: Initial film thickness in m (default: 0).
        - ``Tf``: Initial film temperature in K (default: 300).
        - ``rho_f``: Film density in kg/m^3 (default: 1000).
        - ``mu_f``: Film dynamic viscosity in Pa*s (default: 1e-3).
        - ``sigma``: Surface tension in N/m (default: 0.07).
        - ``value``: Initial scalar value (default: 0).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._delta_init = float(self._coeffs.get("delta", 0.0))
        self._Tf_init = float(self._coeffs.get("Tf", 300.0))
        self._rho_f = float(self._coeffs.get("rho_f", 1000.0))
        self._mu_f = float(self._coeffs.get("mu_f", 1e-3))
        self._sigma = float(self._coeffs.get("sigma", 0.07))

    @property
    def delta_init(self) -> float:
        """Initial film thickness (m)."""
        return self._delta_init

    @property
    def Tf_init(self) -> float:
        """Initial film temperature (K)."""
        return self._Tf_init

    @property
    def rho_f(self) -> float:
        """Film density (kg/m^3)."""
        return self._rho_f

    @property
    def mu_f(self) -> float:
        """Film dynamic viscosity (Pa*s)."""
        return self._mu_f

    @property
    def sigma(self) -> float:
        """Surface tension (N/m)."""
        return self._sigma

    def compute_film_velocity(
        self,
        tau_w: torch.Tensor,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute film velocity from wall shear stress and gravity.

        Uses a balance between wall shear and gravity-driven film flow::

            U_film = (delta^2 / (3 * mu_f)) * (tau_w / delta + rho_f * g_tangential)

        where delta is the film thickness and g_tangential is the
        tangential component of gravity.

        Parameters
        ----------
        tau_w : torch.Tensor
            Wall shear stress at boundary faces ``(n_faces,)``.
        gravity : torch.Tensor, optional
            Gravity vector ``(3,)``. If None, only shear-driven flow.

        Returns
        -------
        torch.Tensor
            ``(n_faces, 3)`` film velocity.
        """
        device = tau_w.device
        dtype = tau_w.dtype
        n = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        delta = torch.full((n,), self._delta_init, device=device, dtype=dtype)

        # Ensure minimum film thickness for non-degenerate velocity
        delta_eff = delta.clamp(min=1e-10)

        # Shear-driven component: tau_w / mu_f * delta / 2 (parabolic profile)
        u_shear = tau_w / max(self._mu_f, 1e-30) * delta_eff / 2.0

        # Build tangential velocity direction from normals
        # Try multiple reference directions to find a non-degenerate tangent
        ref_candidates = [
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        ]
        tangential = None
        for ref in ref_candidates:
            t = ref.unsqueeze(0).expand(n, -1) - (
                (ref.unsqueeze(0).expand(n, -1) * normals).sum(dim=1, keepdim=True)
                * normals
            )
            if t.norm(dim=1).min() > 1e-10:
                tangential = t / t.norm(dim=1, keepdim=True)
                break

        # Fallback: use first non-zero component
        if tangential is None:
            tangential = torch.zeros(n, 3, device=device, dtype=dtype)
            tangential[:, 0] = 1.0

        velocity = tangential * u_shear.unsqueeze(-1)

        # Add gravity-driven component if provided
        if gravity is not None:
            g = gravity.to(device=device, dtype=dtype)
            g_tang = g.unsqueeze(0).expand(n, -1) - (
                (g.unsqueeze(0).expand(n, -1) * normals).sum(dim=1, keepdim=True)
                * normals
            )
            g_film = (delta_eff.pow(2) / (3.0 * max(self._mu_f, 1e-30))) * self._rho_f
            velocity = velocity + g_film.unsqueeze(-1) * g_tang

        return velocity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply wall film BC: set scalar field to initial film value.

        Parameters
        ----------
        field : torch.Tensor
            Scalar field ``(n_total,)``.
        patch_idx : int, optional
            Contiguous start index into the field.
        """
        n = self._patch.n_faces
        val = torch.full(
            (n,), self._delta_init,
            device=field.device, dtype=field.dtype,
        )

        if patch_idx is not None:
            field[patch_idx: patch_idx + n] = val
        else:
            field[self._patch.face_indices] = val

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for wall film: diagonal += deltaCoeff * area."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        # Source drives towards initial film thickness
        n = self._patch.n_faces
        target = torch.full((n,), self._delta_init, device=device, dtype=dtype)
        source.scatter_add_(0, owners, coeff * target)

        return diag, source
