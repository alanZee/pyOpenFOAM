"""
AMG-aware pressure interpolation boundary condition.

A pressure BC designed for use with Algebraic Multigrid (AMG) solvers.
Interpolates the boundary pressure from the owner cell while ensuring
the resulting linear system has good AMG convergence properties::

    p_face = p_owner + (grad_p . d)

where ``d`` is the wall-normal distance vector from cell-centre to
face-centre, and ``grad_p`` is an optional pressure gradient.

The key feature is that this BC adds a stabilised diagonal contribution
to the matrix that preserves the M-matrix property required for AMG::

    diag += A / delta * (1 + epsilon)

In OpenFOAM syntax::

    type                pressureInterpolationAMG;
    interpolationScheme cell;     // cell | linear | corrected
    nCorrectors         2;        // number of pressure correction iterations
    value               uniform 0;

Usage::

    bc = BoundaryCondition.create("pressureInterpolationAMG", patch, coeffs={})
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureInterpolationAMGBC"]


@BoundaryCondition.register("pressureInterpolationAMG")
class PressureInterpolationAMGBC(BoundaryCondition):
    """AMG-aware pressure interpolation boundary condition.

    Interpolates boundary pressure from owner cells with stabilised
    matrix contributions that preserve AMG convergence.

    The boundary value is set to the owner cell value (zero-gradient),
    while the matrix contribution uses an AMG-friendly penalty
    formulation.

    Coefficients:
        - ``interpolationScheme`` (str): Interpolation method.
          ``"cell"`` (default), ``"linear"``, or ``"corrected"``.
        - ``nCorrectors`` (int): Number of correction iterations.
          Default 2.
        - ``epsilon`` (float): AMG stabilisation parameter.
          Default 0.1.
        - ``value`` (float): Initial pressure (default 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._interpolation_scheme = self._coeffs.get("interpolationScheme", "cell")
        self._n_correctors = int(self._coeffs.get("nCorrectors", 2))
        self._epsilon = float(self._coeffs.get("epsilon", 0.1))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def interpolation_scheme(self) -> str:
        """Interpolation scheme name."""
        return self._interpolation_scheme

    @property
    def n_correctors(self) -> int:
        """Number of correction iterations."""
        return self._n_correctors

    @property
    def epsilon(self) -> float:
        """AMG stabilisation parameter."""
        return self._epsilon

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        grad_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary pressure via interpolation from owner cells.

        For the ``"cell"`` scheme, uses zero-gradient (owner value).
        For ``"linear"``, adds a first-order gradient correction.

        Args:
            field: Pressure field.
            patch_idx: Optional start index.
            grad_p: ``(n_faces, 3)`` pressure gradient at boundary
                faces.  Only used with ``"linear"`` or ``"corrected"``
                interpolation.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        if grad_p is not None and self._interpolation_scheme in ("linear", "corrected"):
            # Gradient-based correction: p_face = p_owner + grad_p . d
            # d = face_centre - cell_centre (approximated via normals/delta)
            normals = self._patch.face_normals.to(device=device, dtype=dtype)
            deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
            dist = 1.0 / deltas.clamp(min=1e-30)
            d_vec = normals * dist.unsqueeze(-1)  # (n_faces, 3)
            grad = grad_p.to(device=device, dtype=dtype)
            correction = (grad * d_vec).sum(dim=-1)

            if self._interpolation_scheme == "corrected":
                # Additional correction for non-orthogonal meshes
                correction = correction * 1.5

            face_vals = owner_vals + correction
        else:
            face_vals = owner_vals

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_vals
        else:
            field[self._patch.face_indices] = face_vals
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AMG-stabilised matrix contribution.

        Adds a penalty-based coupling that preserves the M-matrix
        property::

            diag   += (1 + epsilon) * A * delta
            source += (1 + epsilon) * A * delta * p_owner

        The ``epsilon`` term ensures strict diagonal dominance for AMG.
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

        # AMG-stabilised coefficient: (1 + epsilon) * A * delta
        coeff = (1.0 + self._epsilon) * areas * deltas
        owner_vals = field[owners].to(device=device, dtype=dtype)

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * owner_vals)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
