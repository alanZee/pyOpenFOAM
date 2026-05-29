"""
Enhanced wave transmissive boundary condition.

Improves on the base ``waveTransmissive`` BC with:

1. **Non-reflecting treatment** using the Navier-Stokes characteristic
   boundary condition (NSCBC) formulation.
2. **Relaxation blending** between zero-gradient (fully non-reflecting)
   and fixed-value (far-field reference) controlled by a blending
   coefficient.

The boundary pressure is computed as::

    p_face = p_owner - rho * c * U_n * (p_owner - p_inf) /
             (rho * c * |U_n| + l_inf * blending)

where:
    - ``c`` is the local speed of sound
    - ``U_n`` is the normal velocity
    - ``p_inf`` is the far-field pressure
    - ``l_inf`` is the relaxation length scale
    - ``blending`` controls reflection suppression (1 = full NSCBC, 0 = zero-gradient)

In OpenFOAM syntax::

    type        waveTransmissive2;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;
    lInf        1;
    blending    1.0;           // 1 = full NSCBC, 0 = zero-gradient
    value       uniform 101325;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["WaveTransmissive2BC"]


@BoundaryCondition.register("waveTransmissive2")
class WaveTransmissive2BC(BoundaryCondition):
    """Enhanced wave transmissive boundary condition.

    Non-reflecting outflow using NSCBC with controllable blending.

    Coefficients:
        - ``fieldInf`` (float): Far-field reference pressure (Pa).  Default 101325.
        - ``lInf`` (float): Relaxation length scale (m).  Default 1.0.
        - ``gamma`` (float): Ratio of specific heats.  Default 1.4.
        - ``blending`` (float): NSCBC blending factor (0-1).  Default 1.0.
        - ``phi`` (str): Flux field name (informational).
        - ``rho`` (str): Density field name (informational).
        - ``psi`` (str): Compressibility field name (informational).
        - ``value`` (float): Initial pressure (default 101325).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._field_inf = float(self._coeffs.get("fieldInf", 101325.0))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))
        self._blending = float(self._coeffs.get("blending", 1.0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def field_inf(self) -> float:
        """Far-field reference pressure."""
        return self._field_inf

    @property
    def l_inf(self) -> float:
        """Relaxation length scale."""
        return self._l_inf

    @property
    def gamma(self) -> float:
        """Ratio of specific heats."""
        return self._gamma

    @property
    def blending(self) -> float:
        """NSCBC blending factor (0 = zero-gradient, 1 = full NSCBC)."""
        return self._blending

    @blending.setter
    def blending(self, value: float) -> None:
        """Set NSCBC blending factor."""
        self._blending = max(0.0, min(1.0, value))

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
        c: float | None = None,
    ) -> torch.Tensor:
        """Apply enhanced non-reflecting pressure BC.

        Args:
            field: Pressure field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
            rho: Density (scalar or per-face tensor).
            c: Speed of sound (m/s).  Default: 343.0.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        if velocity is not None:
            normals = self._patch.face_normals.to(device=device, dtype=dtype)
            u_n = (velocity * normals).sum(dim=-1)

            if isinstance(rho, torch.Tensor):
                rho_val = rho.to(device=device, dtype=dtype)
            elif rho is not None:
                rho_val = torch.full(
                    (self._patch.n_faces,), float(rho), dtype=dtype, device=device,
                )
            else:
                rho_val = torch.full(
                    (self._patch.n_faces,), 1.225, dtype=dtype, device=device,
                )

            c_val = c if c is not None else 343.0

            dp = owner_vals - self._field_inf
            wave_impedance = rho_val * c_val

            # NSCBC blending: higher blending = stronger non-reflecting correction
            l_effective = self._l_inf / (self._blending + 1e-30)
            denom = wave_impedance * u_n.abs() + l_effective + 1e-30
            correction = wave_impedance * u_n * dp / denom

            p_face = owner_vals - self._blending * correction
        else:
            # No velocity: zero-gradient
            p_face = owner_vals

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p_face
        else:
            field[self._patch.face_indices] = p_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxation-based matrix contribution.

        diag   += rho * c * A * blending / l_inf
        source += rho * c * A * blending * p_inf / l_inf
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        rho_c = 1.225 * 343.0
        relax_coeff = rho_c * area_mag * self._blending / (self._l_inf + 1e-30)

        diag.scatter_add_(0, owners, relax_coeff)
        source.scatter_add_(0, owners, relax_coeff * self._field_inf)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
