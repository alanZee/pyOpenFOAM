"""
Hydrostatic pressure boundary condition.

Implements the ``hydrostaticPressure`` BC which sets the boundary pressure
based on the hydrostatic equation::

    p = p_ref + rho * g * (z_ref - z_face)

where ``g`` is the gravitational acceleration magnitude, ``z_ref`` is the
reference height, and ``z_face`` is the z-coordinate of the boundary face
centre.

In OpenFOAM syntax::

    type        hydrostaticPressure;
    p_ref       101325;          // reference pressure (Pa)
    rho         1.225;           // density (kg/m³)
    g           9.81;            // gravitational acceleration (m/s²)
    z_ref       0.0;             // reference height (m)
    value       uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["HydrostaticPressureBC"]


@BoundaryCondition.register("hydrostaticPressure")
class HydrostaticPressureBC(BoundaryCondition):
    """Hydrostatic pressure boundary condition.

    Sets boundary face pressure to the hydrostatic value based on the
    vertical distance from a reference height::

        p = p_ref + rho * g * (z_ref - z_face)

    This is useful for open boundaries in buoyant flows where the pressure
    varies with depth.

    Coefficients:
        - ``p_ref``: Reference pressure (Pa). Default: 0.
        - ``rho``: Fluid density (kg/m³). Default: 1.0.
        - ``g``: Gravitational acceleration (m/s²). Default: 9.81.
        - ``z_ref``: Reference height (m). Default: 0.0.
        - ``value``: Initial field value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p_ref = float(self._coeffs.get("p_ref", 0.0))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._g = float(self._coeffs.get("g", 9.81))
        self._z_ref = torch.tensor(
            float(self._coeffs.get("z_ref", 0.0)),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def p_ref(self) -> float:
        """Reference pressure (Pa)."""
        return self._p_ref

    @property
    def rho(self) -> float:
        """Fluid density (kg/m³)."""
        return self._rho

    @property
    def gravity(self) -> float:
        """Gravitational acceleration (m/s²)."""
        return self._g

    @property
    def z_ref(self) -> torch.Tensor:
        """Reference height (m)."""
        return self._z_ref

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        face_centres: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply hydrostatic pressure at boundary faces.

        p = p_ref + rho * g * (z_ref - z_face)

        Args:
            field: Pressure field to modify.
            patch_idx: Optional start index into field.
            face_centres: ``(n_faces, 3)`` tensor of face centre positions.
                The z-component (index 2) is used for height.

        Returns:
            Modified field tensor.
        """
        dtype = field.dtype

        if face_centres is None:
            # 无面心坐标时回退到零梯度（拷贝 owner 值）
            owners = self._patch.owner_cells.to(device=field.device)
            owner_values = field[owners]
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = owner_values
            else:
                field[self._patch.face_indices] = owner_values
            return field

        # z 分量（面心坐标的第 3 列）
        z_face = face_centres[:, 2].to(dtype=dtype, device=field.device)
        z_ref = self._z_ref.to(dtype=dtype, device=field.device)

        p_hydro = self._p_ref + self._rho * self._g * (z_ref - z_face)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p_hydro
        else:
            field[self._patch.face_indices] = p_hydro
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        face_centres: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit matrix contributions for hydrostatic pressure.

        Uses the penalty method (same as fixedValue)::

            diag[c]   += deltaCoeff * area
            source[c] += deltaCoeff * area * p_hydro

        Args:
            field: Current pressure field.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.
            face_centres: ``(n_faces, 3)`` tensor of face centre positions.

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

        # 计算目标静水压力值
        if face_centres is not None:
            z_face = face_centres[:, 2].to(device=device, dtype=dtype)
            z_ref = self._z_ref.to(device=device, dtype=dtype)
            p_hydro = self._p_ref + self._rho * self._g * (z_ref - z_face)
        else:
            # 回退到 p_ref
            n_faces = self._patch.n_faces
            p_hydro = torch.full(
                (n_faces,), self._p_ref, device=device, dtype=dtype
            )

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * p_hydro)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
