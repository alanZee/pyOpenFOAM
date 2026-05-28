"""
Enhanced variable height boundary condition with momentum-consistent
velocity correction.

Extends :class:`VariableHeightBC` (``variableHeight``) by applying a
momentum-consistent velocity correction at the boundary.  After the
water depth ``h`` is computed, the velocity magnitude at the boundary
is adjusted to conserve mass flux:

    u_corrected = u_internal * (h_internal / h_boundary)

This prevents spurious mass flux errors when the free-surface height
varies between the interior and the boundary.

In OpenFOAM syntax::

    type            variableHeight2;
    z_surface       2.0;
    h_min           1e-4;
    velocityCorrection  true;
    value           uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["VariableHeight2BC"]


@BoundaryCondition.register("variableHeight2")
class VariableHeight2BC(BoundaryCondition):
    """Enhanced variable height boundary condition.

    Prescribes water depth at boundary faces with optional
    momentum-consistent velocity correction.

    Coefficients:
        - ``z_surface``: Free-surface elevation (m). Default: 1.0.
        - ``h_min``: Minimum water depth to prevent dry cells (m).
          Default: 1e-4.
        - ``velocityCorrection``: bool, apply momentum correction.
          Default: True.
        - ``value``: Initial field value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._z_surface = float(self._coeffs.get("z_surface", 1.0))
        self._h_min = float(self._coeffs.get("h_min", 1e-4))
        self._velocity_correction = bool(
            self._coeffs.get("velocityCorrection", True)
        )

    @property
    def z_surface(self) -> float:
        """Free-surface elevation (m)."""
        return self._z_surface

    @property
    def h_min(self) -> float:
        """Minimum water depth (m)."""
        return self._h_min

    @property
    def velocity_correction(self) -> bool:
        """Whether momentum-consistent velocity correction is enabled."""
        return self._velocity_correction

    def compute_depth(
        self,
        n_faces: int,
        face_centres: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute the water depth at boundary faces.

        h = max(z_surface - z_bathymetry, h_min)

        Args:
            n_faces: Number of boundary faces.
            face_centres: ``(n_faces, 3)`` tensor of face centre positions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Water depth ``(n_faces,)``.
        """
        if device is None:
            device = get_device()
        if dtype is None:
            dtype = get_default_dtype()

        if face_centres is not None:
            z_bathy = face_centres[:, 2].to(dtype=dtype, device=device)
            h = (self._z_surface - z_bathy).clamp(min=self._h_min)
        else:
            h = torch.full(
                (n_faces,), self._z_surface, dtype=dtype, device=device,
            )
        return h

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        face_centres: torch.Tensor | None = None,
        internal_depth: float | None = None,
        internal_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply variable-height water depth at boundary faces.

        Optionally applies momentum-consistent velocity correction:

            u_corrected = u_internal * (h_internal / h_boundary)

        For a scalar depth field, only the depth is set.
        For a vector velocity field, the correction is applied.

        Args:
            field: Field tensor to modify.
            patch_idx: Optional start index into *field*.
            face_centres: ``(n_faces, 3)`` face centre positions.
            internal_depth: Water depth in the interior cell.
            internal_velocity: ``(3,)`` or ``(n_faces, 3)`` interior velocity.

        Returns:
            Modified field tensor.
        """
        dtype = field.dtype
        device = field.device
        n = self._patch.n_faces

        h = self.compute_depth(n, face_centres, device, dtype)

        if field.dim() == 1:
            # Scalar depth field
            if patch_idx is not None:
                field[patch_idx : patch_idx + n] = h
            else:
                field[self._patch.face_indices] = h
        elif field.dim() == 2:
            # Vector velocity field — apply depth-based correction
            if self._velocity_correction and internal_depth is not None:
                h_int = torch.tensor(
                    max(internal_depth, self._h_min), dtype=dtype, device=device
                )
                scale = h_int / h.clamp(min=self._h_min)
                scale = scale.clamp(max=10.0)  # 防止极端放大

                if internal_velocity is not None:
                    u_int = internal_velocity.to(device=device, dtype=dtype)
                    if u_int.dim() == 1:
                        # (3,) -> (n_faces, 3)
                        u_int = u_int.unsqueeze(0).expand(n, -1)
                    values = u_int * scale.unsqueeze(-1)
                else:
                    # 无内部速度信息，设为零
                    values = torch.zeros(n, field.shape[1], dtype=dtype, device=device)
            else:
                # 无速度修正：零值
                values = torch.zeros(n, field.shape[1], dtype=dtype, device=device)

            if patch_idx is not None:
                field[patch_idx : patch_idx + n] = values
            else:
                field[self._patch.face_indices] = values

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
            field: Current field.
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
        h_target = self.compute_depth(n_faces, face_centres, device, dtype)

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * h_target)

        return diag, source
