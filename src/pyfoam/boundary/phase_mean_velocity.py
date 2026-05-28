"""
phaseMeanVelocity — phase-mean velocity boundary condition for Euler-Euler models.

Prescribes a mean velocity for a specific phase in multiphase Euler-Euler
simulations.  The velocity is adjusted so that the phase-averaged velocity
equals the prescribed value:

    U_phase = U_mean / alpha_phase

where alpha_phase is the local volume fraction of the phase.

This is used in Euler-Euler two-phase and multiphase solvers to set inlet
conditions where a specific phase velocity is desired (rather than the
mixture velocity).

In OpenFOAM syntax::

    type            phaseMeanVelocity;
    Umean           uniform (1 0 0);   // desired phase velocity
    alphaField      alpha.phaseName;   // volume fraction field (informational)
    phaseName       gas;               // phase name (informational)
    value           uniform (0 0 0);

Usage::

    bc = BoundaryCondition.create("phaseMeanVelocity", patch, coeffs={
        "Umean": [1.0, 0.0, 0.0],
        "phaseName": "gas",
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PhaseMeanVelocityBC"]


@BoundaryCondition.register("phaseMeanVelocity")
class PhaseMeanVelocityBC(BoundaryCondition):
    """Phase-mean velocity boundary condition for Euler-Euler multiphase.

    Prescribes the mean velocity for a specific phase.  The boundary-face
    velocity is computed so that the phase-averaged velocity matches the
    prescribed value, accounting for the local volume fraction.

    Coefficients
    ------------
    Umean : list[float] | torch.Tensor
        Desired phase mean velocity vector (m/s).
    alphaField : str
        Name of the volume fraction field (informational).
    phaseName : str
        Phase name (informational).
    value : list[float] | torch.Tensor
        Initial velocity shape (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Umean = self._parse_vector("Umean", [0.0, 0.0, 0.0])
        self._phase_name = str(self._coeffs.get("phaseName", ""))
        self._alpha_field = str(self._coeffs.get("alphaField", ""))

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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply phase-mean velocity BC.

        Sets boundary-face velocity to Umean / alpha_phase to ensure the
        phase-averaged velocity equals the prescribed mean.

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
            alpha_safe = alpha.to(device=device, dtype=dtype).clamp(min=1e-6)
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
        """Penalty method for fixed-value BC (phase mean velocity)."""
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

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        # Project onto x-component for scalar matrix
        source.scatter_add_(0, owners, coeff * Umean[0])

        return diag, source
