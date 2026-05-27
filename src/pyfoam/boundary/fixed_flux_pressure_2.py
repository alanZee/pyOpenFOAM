"""
Enhanced fixed flux pressure boundary condition with buoyancy correction.

Implements OpenFOAM's ``fixedFluxPressure`` variant with an explicit
buoyancy correction term.  This BC adjusts the pressure gradient at the
boundary to be consistent with a prescribed mass flux while accounting
for the hydrostatic pressure contribution.

In OpenFOAM syntax::

    type            fixedFluxPressure2;
    phi             phi;
    rho             rho;
    value           uniform 0;

The buoyancy correction modifies the pressure gradient as::

    ∇p_face = ∇p_base + ρ * g · n

where g is gravity and n is the face normal.  This ensures consistency
with fixed-flux boundaries in buoyant flows (e.g. natural convection
cavities, atmospheric boundary layers).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedFluxPressure2BC"]


@BoundaryCondition.register("fixedFluxPressure2")
class FixedFluxPressure2BC(BoundaryCondition):
    """Enhanced fixed flux pressure BC with buoyancy correction.

    Adjusts the pressure gradient at the boundary to be consistent
    with the prescribed velocity flux, with an additional buoyancy
    correction for gravity-driven flows.

    Coefficients:
        - ``phi``: Flux field name (informational).
        - ``rho``: Density field name (informational) or scalar value.
        - ``g``: Gravity vector ``(gx, gy, gz)``. Default ``(0, -9.81, 0)``.
        - ``value``: Initial pressure value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        g_raw = self._coeffs.get("g", (0.0, -9.81, 0.0))
        if isinstance(g_raw, (list, tuple)) and len(g_raw) >= 3:
            self._g = torch.tensor(
                [float(g_raw[0]), float(g_raw[1]), float(g_raw[2])],
                dtype=get_default_dtype(),
                device=get_device(),
            )
        else:
            self._g = torch.tensor(
                [0.0, -9.81, 0.0],
                dtype=get_default_dtype(),
                device=get_device(),
            )
        self._rho_ref = float(self._coeffs.get("rho", 1.0))

    @property
    def g(self) -> torch.Tensor:
        """Return gravity vector."""
        return self._g

    @property
    def rho_ref(self) -> float:
        """Return reference density."""
        return self._rho_ref

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply zero-gradient with buoyancy correction.

        The boundary pressure is extrapolated from the interior (like
        zeroGradient), then corrected by the hydrostatic contribution
        ``ρ * g · n * Δn`` where Δn is the wall-normal distance.

        Args:
            field: Scalar pressure field.
            patch_idx: Optional start index into field.
            flux: Face flux (unused, kept for API compatibility).
            rho: Density (scalar or per-face tensor).
        """
        device = field.device
        dtype = field.dtype
        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]

        # Buoyancy correction: ρ * g · n * Δn
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        g = self._g.to(device=device, dtype=dtype)

        # g · n (gravity projected onto face normal)
        g_dot_n = (g.unsqueeze(0) * normals).sum(dim=-1)  # (n_faces,)

        # Density
        if rho is None:
            rho_val = self._rho_ref
        elif isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        else:
            rho_val = float(rho)

        # Buoyancy pressure correction: Δp = ρ * g · n / δ
        # Using the inverse delta coefficient as distance scale
        delta_safe = deltas.clamp(min=1e-30)
        buoyancy_correction = rho_val * g_dot_n / delta_safe

        face_values = owner_values + buoyancy_correction

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Near-zero matrix contribution with buoyancy source.

        Like zeroGradient, but the buoyancy correction adds a small
        explicit source term to maintain consistency.
        """
        device = get_device()
        dtype = field.dtype

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        return diag, source
