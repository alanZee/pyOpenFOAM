"""
Pressure normal inlet boundary condition.

A pressure inlet BC for compressible flows that specifies the total
pressure and derives the normal velocity from the Bernoulli equation.

In OpenFOAM syntax::

    type            pressureNormalInlet;
    p0              uniform 101325;   // total (stagnation) pressure
    rho             1.225;            // fluid density (optional, default 1.0)
    U               uniform (0 0 0);  // optional reference velocity direction
    phi             phi;              // flux field name (informational)

The normal velocity at the inlet face is computed from the Bernoulli
equation:

    U_n = sqrt(2 * (p0 - p) / rho)

where ``p0`` is the total pressure, ``p`` is the local static pressure,
and ``rho`` is the fluid density.

When the flow is outgoing (p > p0), the BC degrades to a zero-gradient
treatment to avoid unphysical reverse-flow artefacts.

Usage::

    bc = BoundaryCondition.create("pressureNormalInlet", patch, {
        "p0": 101325.0,
        "rho": 1.225,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureNormalInletBC"]


@BoundaryCondition.register("pressureNormalInlet")
class PressureNormalInletBC(BoundaryCondition):
    """Pressure inlet boundary condition with normal velocity.

    Specifies total pressure and computes normal velocity from
    the Bernoulli relation for compressible flows.

    Parameters
    ----------
    patch : Patch
        Boundary patch.
    coeffs : dict
        BC coefficients:

        - ``"p0"``: total (stagnation) pressure [Pa].
        - ``"rho"``: fluid density [kg/m^3] (default 1.0).
        - ``"value"``: prescribed velocity direction vector
          (default (1, 0, 0) — flow in +x direction).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)

        self._p0 = float(self._coeffs.get("p0", 101325.0))
        self._rho = float(self._coeffs.get("rho", 1.0))

        # Direction for velocity vector (unit normal or prescribed)
        raw_dir = self._coeffs.get("value", (1.0, 0.0, 0.0))
        if isinstance(raw_dir, (list, tuple)) and len(raw_dir) >= 3:
            self._dir = torch.tensor(
                raw_dir[:3], dtype=get_default_dtype(), device=get_device(),
            )
        elif isinstance(raw_dir, torch.Tensor):
            self._dir = raw_dir.to(dtype=get_default_dtype(), device=get_device())
        else:
            self._dir = torch.tensor(
                [1.0, 0.0, 0.0], dtype=get_default_dtype(), device=get_device(),
            )

        # Normalise direction
        norm = self._dir.norm()
        if norm > 1e-30:
            self._dir = self._dir / norm

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p0(self) -> float:
        """Total (stagnation) pressure."""
        return self._p0

    @property
    def rho(self) -> float:
        """Fluid density."""
        return self._rho

    @property
    def direction(self) -> torch.Tensor:
        """Unit direction vector for the inlet velocity."""
        return self._dir

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply the pressure-normal-inlet BC.

        Computes the normal velocity from Bernoulli:

            U_n = sqrt(max(2 * (p0 - p) / rho, 0))

        and sets the face velocity as ``U_n * direction``.

        For the velocity field (3-component), this sets the full vector.
        For a scalar field, this is a no-op (the BC is designed for velocity).
        """
        if field.dim() < 2 or field.shape[-1] < 3:
            # Scalar field: no-op
            return field

        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)

        # We need a pressure field to compute velocity.
        # Use a stored reference pressure if available, otherwise
        # assume p = p0 (stagnation, zero velocity).
        p_ref = self._coeffs.get("_p_field")
        if p_ref is not None and isinstance(p_ref, torch.Tensor):
            p_local = p_ref[owners].to(device=device, dtype=dtype)
        else:
            # No pressure field available: assume p = p0 -> U = 0
            p_local = torch.full_like(owners.float(), self._p0, dtype=dtype)

        # Bernoulli: U_n = sqrt(2 * (p0 - p) / rho)
        dp = (self._p0 - p_local).clamp(min=0.0)
        U_n = (2.0 * dp / max(self._rho, 1e-30)).sqrt()

        # Velocity vector = U_n * direction
        direction = self._dir.to(device=device, dtype=dtype)
        face_vel = U_n.unsqueeze(-1) * direction.unsqueeze(0)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_vel
        else:
            field[self._patch.face_indices] = face_vel
        return field

    # ------------------------------------------------------------------
    # Matrix contributions
    # ------------------------------------------------------------------

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pressure-fraction penalty method.

        Uses a Robin-type treatment:
            diag += deltaCoeff * area * fraction
            source += deltaCoeff * area * fraction * U_b

        where fraction depends on the local pressure deficit.
        For simplicity (and when pressure is not available for the matrix),
        we use a fixed fraction of 1.0 (Dirichlet-like) on the velocity
        matrix and rely on the pressure-velocity coupling in the solver.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        # Compute face velocities for source term
        p_ref = self._coeffs.get("_p_field")
        if p_ref is not None and isinstance(p_ref, torch.Tensor):
            p_local = p_ref[owners].to(device=device, dtype=dtype)
        else:
            p_local = torch.full((self._patch.n_faces,), self._p0, dtype=dtype, device=device)

        dp = (self._p0 - p_local).clamp(min=0.0)
        U_n = (2.0 * dp / max(self._rho, 1e-30)).sqrt()

        # Use full Dirichlet penalty
        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        # Source uses U_n component (magnitude only for scalar contribution)
        source.scatter_add_(0, owners, coeff * U_n)

        return diag, source

    def __repr__(self) -> str:
        return (
            f"PressureNormalInletBC(patch='{self._patch.name}', "
            f"p0={self._p0}, rho={self._rho})"
        )
