"""
Variable height boundary condition for shallow water equations.

Prescribes water depth at boundary faces based on bathymetry and a
reference free-surface height.  In shallow-water modelling the water
depth ``h`` at each face is::

    h = max(z_surface - z_bathymetry, h_min)

where ``z_surface`` is the free-surface elevation (constant for a
flat-surface assumption) and ``z_bathymetry`` is the local bottom
elevation.

In OpenFOAM syntax::

    type            variableHeight;
    z_surface       2.0;           // free-surface height (m)
    h_min           1e-4;          // minimum depth to avoid dry cells
    value           uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["VariableHeightBC"]


@BoundaryCondition.register("variableHeight")
class VariableHeightBC(BoundaryCondition):
    """Variable height boundary condition for shallow water flows.

    Prescribes the water depth at each boundary face from bathymetry
    data and a (uniform) free-surface elevation.

    Coefficients:
        - ``z_surface``: Free-surface elevation (m).  Default: 1.0.
        - ``h_min``: Minimum water depth to prevent dry cells (m).
          Default: 1e-4.
        - ``value``: Initial field value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._z_surface = float(self._coeffs.get("z_surface", 1.0))
        self._h_min = float(self._coeffs.get("h_min", 1e-4))

    @property
    def z_surface(self) -> float:
        """Free-surface elevation (m)."""
        return self._z_surface

    @property
    def h_min(self) -> float:
        """Minimum water depth (m)."""
        return self._h_min

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        face_centres: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply variable-height water depth at boundary faces.

        h = max(z_surface - z_bathymetry, h_min)

        Args:
            field: Water-depth field to modify.
            patch_idx: Optional start index into *field*.
            face_centres: ``(n_faces, 3)`` tensor of face centre positions.
                The z-component (index 2) is used as bathymetry elevation.

        Returns:
            Modified field tensor.
        """
        dtype = field.dtype
        n = self._patch.n_faces

        if face_centres is not None:
            z_bathy = face_centres[:, 2].to(dtype=dtype, device=field.device)
            h = (self._z_surface - z_bathy).clamp(min=self._h_min)
        else:
            # 无面心坐标时回退到均匀水深 z_surface
            h = torch.full(
                (n,), self._z_surface, dtype=dtype, device=field.device,
            )

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = h
        else:
            field[self._patch.face_indices] = h
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        face_centres: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit matrix contributions (fixedValue penalty method).

        Args:
            field: Current depth field.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.
            face_centres: ``(n_faces, 3)`` face centre positions.

        Returns:
            ``(diag, source)`` tuple.
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
        coeff = deltas * areas

        # 计算目标水深
        n_faces = self._patch.n_faces
        if face_centres is not None:
            z_bathy = face_centres[:, 2].to(device=device, dtype=dtype)
            h_target = (self._z_surface - z_bathy).clamp(min=self._h_min)
        else:
            h_target = torch.full(
                (n_faces,), self._z_surface, device=device, dtype=dtype,
            )

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * h_target)

        return diag, source
