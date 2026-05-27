"""
Buoyant pressure boundary condition for buoyancy-driven flows.

Implements the ``buoyantPressure`` BC which corrects the pressure at
open boundaries for the hydrostatic contribution in buoyant flows::

    p_boundary = p_interior - rho * g · (x_face - x_ref)

where ``g`` is the gravity vector, ``x_face`` the face-centre position,
and ``x_ref`` a reference height.

In OpenFOAM syntax::

    type    buoyantPressure;
    rho     rho;              // density field name (informational)
    value   uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["BuoyantPressureBC"]


@BoundaryCondition.register("buoyantPressure")
class BuoyantPressureBC(BoundaryCondition):
    """Hydrostatic pressure correction for buoyant flows.

    At open boundaries the pressure is corrected by removing (or adding)
    the hydrostatic component so that the boundary value represents
    p_rgh (reduced pressure) rather than absolute pressure::

        p_rgh = p - rho * g · x

    The BC sets::

        p_rgh_boundary = p_interior

    which is equivalent to a zero-gradient condition on p_rgh, but the
    matrix contribution accounts for the rho*g correction when coupling
    back to the absolute pressure field.

    Coefficients:
        - ``rho``: Density field name (informational, not used directly).
        - ``g``: Gravity vector ``[gx, gy, gz]`` (default: ``[0, -9.81, 0]``).
        - ``value``: Initial pressure (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        g_raw = self._coeffs.get("g", [0.0, -9.81, 0.0])
        self._g = torch.tensor(g_raw, dtype=get_default_dtype(), device=get_device())
        # rho field name (informational)
        self._rho_name = self._coeffs.get("rho", "rho")

    @property
    def gravity(self) -> torch.Tensor:
        """Return the gravity vector."""
        return self._g

    @property
    def rho_name(self) -> str:
        """Return the density field name."""
        return self._rho_name

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply buoyant pressure BC (zero-gradient on p_rgh).

        For the reduced-pressure formulation the boundary value equals
        the adjacent cell value (zero-gradient).  The hydrostatic
        correction is implicit in the p_rgh definition.

        Args:
            field: Pressure field (p_rgh).
            patch_idx: Optional start index into field.
            rho: Density — unused here but accepted for API consistency.
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
        rho: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hydrostatic correction matrix contribution.

        Adds a source correction for the hydrostatic pressure gradient::

            source[c] += rho * (g · n) * area

        where ``n`` is the outward face normal and ``area`` is the face area.

        Args:
            field: Current pressure field.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.
            rho: Density (scalar or per-face tensor).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        g = self._g.to(device=device, dtype=dtype)

        # g · n for each face
        g_dot_n = (normals * g).sum(dim=-1)

        if rho is None:
            rho_val = 1.0
        elif isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        else:
            rho_val = float(rho)

        # source[c] += rho * g_n * area
        hydro_source = rho_val * g_dot_n * areas
        source.scatter_add_(0, owners, hydro_source)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
