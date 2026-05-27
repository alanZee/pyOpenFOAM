"""
Turbulent inlet boundary condition.

Generates a fluctuating turbulent inlet velocity profile by
superimposing random perturbations on a mean profile.  This is
useful for synthesising turbulence at inlets without resolving
an upstream domain.

In OpenFOAM syntax::

    type        turbulentInlet;
    referenceField uniform (1 0 0);  // mean velocity
    fluctuationScale (0.1 0.05 0.05); // RMS fluctuation per component
    alpha       0.1;                  // relaxation factor [0,1]
    value       uniform (1 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentInletBC"]


@BoundaryCondition.register("turbulentInlet")
class TurbulentInletBC(BoundaryCondition):
    """Fluctuating turbulent inlet boundary condition.

    Generates inlet velocity with random fluctuations::

        U_face = (1 - alpha) * U_face_prev + alpha * (U_mean + epsilon)

    where ``epsilon`` is drawn from a normal distribution scaled by
    ``fluctuationScale``.

    Coefficients:
        - ``referenceField``: Mean velocity ``[Ux, Uy, Uz]``
          (default: ``[1, 0, 0]``).
        - ``fluctuationScale``: RMS fluctuation per component
          ``[sigma_x, sigma_y, sigma_z]`` (default: ``[0.1, 0.1, 0.1]``).
        - ``alpha``: Relaxation factor in [0, 1] (default: 1.0 —
          fully random each call).
        - ``value``: Initial velocity (used for shape).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        ref_raw = self._coeffs.get("referenceField", [1.0, 0.0, 0.0])
        self._reference = torch.tensor(
            ref_raw, dtype=get_default_dtype(), device=get_device()
        )

        scale_raw = self._coeffs.get("fluctuationScale", [0.1, 0.1, 0.1])
        self._fluctuation_scale = torch.tensor(
            scale_raw, dtype=get_default_dtype(), device=get_device()
        )

        self._alpha = float(self._coeffs.get("alpha", 1.0))

        # Previous face values (initialised to mean)
        self._prev_values: torch.Tensor | None = None

    @property
    def reference_field(self) -> torch.Tensor:
        """Return the mean velocity vector."""
        return self._reference

    @property
    def fluctuation_scale(self) -> torch.Tensor:
        """Return the fluctuation scale per component."""
        return self._fluctuation_scale

    @property
    def alpha(self) -> float:
        """Return the relaxation factor."""
        return self._alpha

    def _generate_fluctuations(self, n_faces: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate random velocity fluctuations.

        Returns:
            ``(n_faces, 3)`` tensor of fluctuation velocities.
        """
        # Standard normal scaled by fluctuationScale
        noise = torch.randn(n_faces, 3, device=device, dtype=dtype)
        scale = self._fluctuation_scale.to(device=device, dtype=dtype)
        return noise * scale

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocities with turbulent fluctuations.

        U = (1 - alpha) * U_prev + alpha * (U_mean + epsilon)
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        ref = self._reference.to(device=device, dtype=dtype)
        # Mean velocity broadcast to all faces: (n_faces, 3)
        mean = ref.unsqueeze(0).expand(n_faces, -1)

        # Generate fluctuations
        epsilon = self._generate_fluctuations(n_faces, device, dtype)

        # Blend with previous values if available
        if self._prev_values is not None and self._alpha < 1.0:
            prev = self._prev_values.to(device=device, dtype=dtype)
            face_values = (1.0 - self._alpha) * prev + self._alpha * (mean + epsilon)
        else:
            face_values = mean + epsilon

        # Store for next call
        self._prev_values = face_values.detach().clone()

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method with mean reference value.

        Uses the reference (mean) velocity for the implicit matrix
        contribution.  Fluctuations appear only in the explicit source
        to preserve diagonal dominance.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * refValue
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

        ref = self._reference.to(device=device, dtype=dtype)
        # Use x-component of reference for scalar matrix contribution
        ref_scalar = ref[0] if ref.dim() > 0 else ref

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * ref_scalar)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
