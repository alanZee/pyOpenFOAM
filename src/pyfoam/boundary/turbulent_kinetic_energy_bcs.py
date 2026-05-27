"""
Turbulent kinetic energy boundary conditions.

Implements OpenFOAM turbulence kinetic energy boundary conditions:
- ``turbulentIntensityKE``: Computes k from turbulence intensity and
  mean velocity magnitude.
- ``fixedTurbulentKE``: Prescribed turbulent kinetic energy value.

In OpenFOAM syntax::

    // turbulentIntensityKE
    type        turbulentIntensityKE;
    intensity   0.05;           // turbulence intensity (I)
    U           U;              // velocity field reference
    value       uniform 0.01;

    // fixedTurbulentKE
    type        fixedTurbulentKE;
    k           0.01;           // prescribed k value
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "TurbulentIntensityKEBC",
    "FixedTurbulentKEBC",
]


@BoundaryCondition.register("turbulentIntensityKE")
class TurbulentIntensityKEBC(BoundaryCondition):
    """Turbulent intensity-based kinetic energy BC.

    Computes turbulent kinetic energy from the turbulence intensity
    and the local velocity magnitude::

        k = 1.5 * (I * |U|)^2

    where:
        - I is the turbulence intensity (typically 0.01 to 0.10)
        - |U| is the magnitude of the local velocity vector

    This is a general-purpose inlet BC for the k-equation in
    RANS turbulence models (k-epsilon, k-omega, etc.).

    Coefficients:
        - ``intensity``: Turbulence intensity (default: 0.05).
        - ``U``: Velocity field name (informational, for naming only).
        - ``value``: Initial k value (shape hint, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k from turbulence intensity.

        k = 1.5 * (I * |U|)^2

        Parameters
        ----------
        field : torch.Tensor
            Turbulent kinetic energy field.
        patch_idx : int, optional
            Start index into *field*.
        velocity : torch.Tensor, optional
            ``(n_faces, 3)`` velocity at boundary faces.
        """
        device = field.device
        dtype = field.dtype

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k = 1.5 * (self._intensity * u_mag) ** 2
        else:
            k = torch.full(
                (self._patch.n_faces,), 0.01,
                dtype=dtype, device=device,
            )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: implicit source for k inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        k_default = 0.01
        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


@BoundaryCondition.register("fixedTurbulentKE")
class FixedTurbulentKEBC(BoundaryCondition):
    """Fixed turbulent kinetic energy boundary condition.

    Prescribes a constant (or per-face) turbulent kinetic energy value
    at the boundary.  This is useful for setting a known k at inlets
    or far-field boundaries.

    The prescribed value can be:
    - A scalar (``"k"`` or ``"value"`` coefficient) applied uniformly
    - A per-face tensor passed directly to ``apply()``

    Coefficients:
        - ``k``: Fixed turbulent kinetic energy value (default: 0.01).
        - ``value``: Alias for ``k`` (OpenFOAM convention).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        # "k" coefficient takes precedence over "value"
        raw = self._coeffs.get("k", self._coeffs.get("value", 0.01))
        if isinstance(raw, torch.Tensor):
            self._k_value = raw.clone()
        else:
            self._k_value = float(raw)

    @property
    def k_value(self) -> float | torch.Tensor:
        """Fixed turbulent kinetic energy value."""
        return self._k_value

    @k_value.setter
    def k_value(self, val: float | torch.Tensor) -> None:
        self._k_value = val

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k to the prescribed value.

        Parameters
        ----------
        field : torch.Tensor
            Turbulent kinetic energy field.
        patch_idx : int, optional
            Start index into *field*.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if isinstance(self._k_value, torch.Tensor):
            k = self._k_value.to(device=device, dtype=dtype)
        else:
            k = torch.full((n,), float(self._k_value), dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx: patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: implicit source for fixed k BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        if isinstance(self._k_value, torch.Tensor):
            k_val = float(self._k_value.mean().item())
        else:
            k_val = float(self._k_value)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_val)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
