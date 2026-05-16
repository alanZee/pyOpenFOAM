"""
Pressure boundary conditions.

Implements OpenFOAM pressure boundary conditions:
- totalPressure: Total pressure BC (p_total = p + 0.5*rho*|U|²)
- fixedFluxPressure: Pressure BC for fixed flux
- prghPressure: Pressure BC for buoyancy (p_rgh = p - rho*g*h)
- waveTransmissive: Non-reflective pressure BC

In OpenFOAM syntax::

    // totalPressure
    type        totalPressure;
    p0          uniform 101325;  // total pressure
    gamma       1.4;             // ratio of specific heats (for compressible)
    phi         phi;             // flux field name
    rho         rho;             // density field name
    psi         psi;             // compressibility field name
    value       uniform 101325;

    // fixedFluxPressure
    type        fixedFluxPressure;
    phi         phi;
    value       uniform 0;

    // prghPressure
    type        prghPressure;
    p0          uniform 101325;  // reference pressure
    rho         rho;
    value       uniform 0;

    // waveTransmissive
    type        waveTransmissive;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;          // far-field value
    lInf        1;               // characteristic length
    value       uniform 101325;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "TotalPressureBC",
    "FixedFluxPressureBC",
    "PrghPressureBC",
    "WaveTransmissiveBC",
]


@BoundaryCondition.register("totalPressure")
class TotalPressureBC(BoundaryCondition):
    """Total pressure boundary condition.

    Prescribes total pressure p₀ at the boundary.  The static pressure
    is computed from::

        p = p₀ - 0.5 * rho * |U|²

    For incompressible flow (rho = const), this simplifies to::

        p = p₀ - 0.5 * rho * |U|²

    For compressible flow, the relationship involves gamma::

        p = p₀ * (1 - 0.5*(gamma-1)*|U|²/(gamma*R*T))^(gamma/(gamma-1))

    Coefficients:
        - ``p0``: Total pressure (Pa).
        - ``gamma``: Ratio of specific heats (default: 1.4).
        - ``phi``: Flux field name (informational).
        - ``rho``: Density field name (informational).
        - ``psi``: Compressibility field name (informational).
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))

    @property
    def p0(self) -> float:
        """Return total pressure."""
        return self._p0

    @property
    def gamma(self) -> float:
        """Return ratio of specific heats."""
        return self._gamma

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face pressure from total pressure.

        p = p₀ - 0.5 * rho * |U|²

        Args:
            field: Pressure field.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
            rho: Density (scalar or per-face tensor).
        """
        device = field.device
        dtype = field.dtype

        if velocity is None:
            # No velocity info → use total pressure directly
            p = torch.full(
                (self._patch.n_faces,),
                self._p0,
                dtype=dtype,
                device=device,
            )
        else:
            # Compute dynamic pressure: 0.5 * rho * |U|²
            u_mag_sq = (velocity * velocity).sum(dim=-1)
            if rho is None:
                rho_val = 1.0
            elif isinstance(rho, torch.Tensor):
                rho_val = rho.to(device=device, dtype=dtype)
            else:
                rho_val = float(rho)

            # p = p₀ - 0.5 * rho * |U|²
            p = self._p0 - 0.5 * rho_val * u_mag_sq

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p
        else:
            field[self._patch.face_indices] = p
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for total pressure BC."""
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
        source.scatter_add_(0, owners, coeff * self._p0)

        return diag, source


@BoundaryCondition.register("fixedFluxPressure")
class FixedFluxPressureBC(BoundaryCondition):
    """Fixed flux pressure boundary condition.

    Adjusts the pressure gradient at the boundary to produce a specified
    flux.  This is used when the velocity is fixed and the pressure
    must be adjusted to be consistent.

    In practice, this BC modifies the pressure gradient to match the
    prescribed velocity flux, ensuring consistency between the
    velocity and pressure fields.

    Coefficients:
        - ``phi``: Flux field name (informational).
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply zero-gradient behaviour.

        The pressure is extrapolated from the interior, consistent
        with the fixed flux constraint.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero matrix contribution (like zeroGradient).

        The pressure gradient is implicitly adjusted by the
        velocity-pressure coupling in the SIMPLE/PISO algorithm.
        """
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("prghPressure")
class PrghPressureBC(BoundaryCondition):
    """Buoyancy pressure (p_rgh) boundary condition.

    For buoyancy-driven flows, the pressure is decomposed as::

        p = p_rgh + rho*g*h

    where p_rgh is the reduced pressure (excluding the hydrostatic
    component), g is gravity, and h is the height.

    This BC prescribes p_rgh at the boundary, typically equal to
    the reference pressure p₀.

    Coefficients:
        - ``p0``: Reference pressure (Pa).
        - ``rho``: Density field name (informational).
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))

    @property
    def p0(self) -> float:
        """Return reference pressure."""
        return self._p0

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face p_rgh to reference pressure."""
        device = field.device
        dtype = field.dtype

        p = torch.full(
            (self._patch.n_faces,),
            self._p0,
            dtype=dtype,
            device=device,
        )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p
        else:
            field[self._patch.face_indices] = p
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for p_rgh BC."""
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
        source.scatter_add_(0, owners, coeff * self._p0)

        return diag, source


@BoundaryCondition.register("waveTransmissive")
class WaveTransmissiveBC(BoundaryCondition):
    """Wave transmissive (non-reflective) pressure boundary condition.

    Implements a non-reflective boundary condition for pressure that
    allows waves to pass through without reflection.  This is based
    on the characteristic wave decomposition.

    The boundary condition updates the pressure based on the
    characteristic wave speed and the difference between the
    boundary and far-field values::

        p_boundary = p_interior + rho * c * (U · n) * (p - p_inf) / (rho * c + l_inf)

    where:
        - c is the speed of sound
        - n is the face normal
        - p_inf is the far-field pressure
        - l_inf is a characteristic length scale

    Coefficients:
        - ``phi``: Flux field name (informational).
        - ``rho``: Density field name (informational).
        - ``psi``: Compressibility field name (informational).
        - ``gamma``: Ratio of specific heats (default: 1.4).
        - ``fieldInf``: Far-field value.
        - ``lInf``: Characteristic length scale.
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._field_inf = float(self._coeffs.get("fieldInf", 101325.0))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))

    @property
    def field_inf(self) -> float:
        """Return far-field value."""
        return self._field_inf

    @property
    def l_inf(self) -> float:
        """Return characteristic length scale."""
        return self._l_inf

    @property
    def gamma(self) -> float:
        """Return ratio of specific heats."""
        return self._gamma

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
        psi: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply non-reflective pressure BC.

        For incompressible flow, this is essentially a zero-gradient BC
        with a correction based on the wave speed.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]

        if velocity is not None and rho is not None:
            # Compute normal velocity component
            normals = self._patch.face_normals.to(device=device, dtype=dtype)
            u_n = (velocity * normals).sum(dim=-1)

            # Get density
            if isinstance(rho, torch.Tensor):
                rho_val = rho.to(device=device, dtype=dtype)
            else:
                rho_val = float(rho)

            # Simple wave transmissive correction
            # p_boundary = p_interior + rho * c * u_n * (p - p_inf) / (rho * c + l_inf)
            # For simplicity, use a relaxation approach
            p_interior = owner_values
            dp = p_interior - self._field_inf
            correction = rho_val * u_n * dp / (rho_val * abs(u_n) + self._l_inf)
            p_boundary = p_interior + correction
        else:
            # No velocity/density info → zero gradient
            p_boundary = owner_values

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p_boundary
        else:
            field[self._patch.face_indices] = p_boundary
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero matrix contribution (like zeroGradient).

        The non-reflective BC is applied as a correction to the
        boundary value, not as a matrix modification.
        """
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
