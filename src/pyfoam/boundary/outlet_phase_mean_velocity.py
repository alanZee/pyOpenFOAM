"""
outletPhaseMeanVelocity -- outlet phase mean velocity BC for Euler-Euler models.

Prescribes the mean velocity for a specific phase at an outlet boundary
in multiphase Euler-Euler simulations.  Unlike the inlet variant
(``phaseMeanVelocity``), this BC applies a modified treatment suitable
for outflow boundaries where the velocity is dominated by the
convective flux rather than being strictly prescribed::

    type              outletPhaseMeanVelocity;
    Umean             uniform (1 0 0);   // desired phase mean velocity
    alphaField        alpha.gas;         // volume fraction field name
    phaseName         gas;               // phase name
    value             uniform (0 0 0);

The boundary velocity adjusts based on the local volume fraction::

    U_phase = Umean / alpha_phase

For outlet faces where alpha approaches zero, the velocity is clamped
to avoid singularities.  The BC also contributes an outlet-type
zero-gradient treatment for the matrix (allowing information to leave
the domain).

Usage::

    bc = BoundaryCondition.create("outletPhaseMeanVelocity", patch, coeffs={
        "Umean": [1.0, 0.0, 0.0],
        "phaseName": "gas",
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocityBC"]


@BoundaryCondition.register("outletPhaseMeanVelocity")
class OutletPhaseMeanVelocityBC(BoundaryCondition):
    """Outlet phase mean velocity BC for Euler-Euler multiphase.

    Prescribes the mean velocity for a specific phase at an outlet
    boundary.  The boundary-face velocity is computed so that the
    phase-averaged velocity matches the prescribed mean, accounting
    for the local volume fraction::

        U_phase = Umean / alpha_phase

    A minimum alpha threshold prevents singularities when the phase
    is depleted at the outlet.

    Coefficients
    ------------
    Umean : list[float] | torch.Tensor
        Desired phase mean velocity vector (m/s).
    alphaField : str
        Name of the volume fraction field (informational).
    phaseName : str
        Phase name (informational).
    alphaMin : float
        Minimum volume fraction to avoid division by zero.  Default: 1e-4.
    value : list[float] | torch.Tensor
        Initial velocity shape (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Umean = self._parse_vector("Umean", [0.0, 0.0, 0.0])
        self._phase_name = str(self._coeffs.get("phaseName", ""))
        self._alpha_field = str(self._coeffs.get("alphaField", ""))
        self._alpha_min = float(self._coeffs.get("alphaMin", 1e-4))

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def Umean(self) -> torch.Tensor:
        """Prescribed phase mean velocity."""
        return self._Umean

    @property
    def phase_name(self) -> str:
        """Phase name."""
        return self._phase_name

    @property
    def alpha_field(self) -> str:
        """Volume fraction field name."""
        return self._alpha_field

    @property
    def alpha_min(self) -> float:
        """Minimum volume fraction threshold."""
        return self._alpha_min

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply outlet phase-mean velocity BC.

        Sets boundary-face velocity to Umean / alpha_phase to ensure the
        phase-averaged velocity equals the prescribed mean.  Uses a
        higher alpha threshold than the inlet variant to provide more
        robust behaviour when the phase fraction is low at outlets.

        Args:
            field: Velocity field ``(n_cells_or_faces, 3)``.
            patch_idx: Optional start index into *field*.
            alpha: Phase volume fraction ``(n_faces,)``.  If ``None``,
                assumes alpha = 1 (single-phase limit).
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        Umean = self._Umean.to(device=device, dtype=dtype)

        if alpha is not None:
            alpha_safe = alpha.to(device=device, dtype=dtype).clamp(min=self._alpha_min)
            # U_phase = Umean / alpha to get desired phase velocity
            velocity = Umean.unsqueeze(0) / alpha_safe.unsqueeze(-1)
        else:
            velocity = Umean.unsqueeze(0).expand(n_faces, -1).clone()

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = velocity
        else:
            field[self._patch.face_indices] = velocity
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for fixed-value BC (outlet phase mean velocity).

        Uses the same penalty approach as the inlet variant but with
        reduced diagonal contribution to allow some outflow freedom.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        Umean = self._Umean.to(device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        # Reduced penalty factor for outlet (0.5 * full penalty)
        coeff = 0.5 * deltas * areas

        diag.scatter_add_(0, owners, coeff)
        # Project onto x-component for scalar matrix
        source.scatter_add_(0, owners, coeff * Umean[0])

        return diag, source
