"""
Enhanced pressure inlet/outlet velocity boundary condition — version 2.

An improved version of ``pressureInletOutletVelocity`` that adds:

1. **Turbulence-aware treatment**: At inflow, the velocity direction
   accounts for turbulent fluctuations by blending the zero-gradient
   (interior) velocity with a prescribed inlet direction.

2. **Blending zone**: A smooth transition between inlet and outlet
   treatment using the flux direction, preventing sharp switches.

In OpenFOAM syntax::

    type              pressureInletOutletVelocity2;
    phi               phi;
    inletDir          (1 0 0);     // inlet direction (will be normalised)
    blending          0.1;         // blending width (default: 0.1)
    value             uniform (0 0 0);

Behaviour:
- **Inflow** (phi < 0): blends prescribed direction with zero-gradient.
- **Outflow** (phi >= 0): zero-gradient (copies owner values).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureInletOutletVelocity2BC"]


@BoundaryCondition.register("pressureInletOutletVelocity2")
class PressureInletOutletVelocity2BC(BoundaryCondition):
    """Enhanced pressure inlet/outlet velocity BC (version 2).

    Adds direction blending at inflow to the basic zero-gradient
    treatment.  At inflow, the velocity is a blend between the
    interior (zero-gradient) value and a prescribed inlet direction,
    scaled by the interior velocity magnitude.

    Coefficients:
        - ``inletDir``: Inlet flow direction (default [1, 0, 0]).
        - ``blending``: Blending width for smooth transition (default 0.1).
        - ``phi``: Flux field name (informational).
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._inlet_dir = self._parse_vector("inletDir", [1.0, 0.0, 0.0])
        # Normalise
        norm = torch.norm(self._inlet_dir)
        if norm > 0:
            self._inlet_dir = self._inlet_dir / norm
        self._blending = float(self._coeffs.get("blending", 0.1))

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def inlet_dir(self) -> torch.Tensor:
        """Return normalised inlet direction."""
        return self._inlet_dir

    @property
    def blending(self) -> float:
        """Return blending width."""
        return self._blending

    def _flow_direction(
        self,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Determine flow direction: positive = inflow fraction in [0, 1]."""
        device = get_device()
        n = self._patch.n_faces

        if flux is not None:
            f = flux.to(device=device, dtype=get_default_dtype())
            # Smooth step: 1 for strong inflow, 0 for strong outflow
            return torch.sigmoid(-f / max(self._blending, 1e-10))

        if velocity is not None:
            normals = self._patch.face_normals.to(
                device=velocity.device, dtype=velocity.dtype
            )
            vn = (velocity * normals).sum(dim=-1)
            return torch.sigmoid(-vn / max(self._blending, 1e-10))

        return torch.zeros(n, dtype=get_default_dtype(), device=device)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply enhanced pressure inlet/outlet velocity.

        At inflow: blends zero-gradient velocity with prescribed direction.
        At outflow: pure zero-gradient.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners].to(dtype=dtype)

        # Inflow weight (1 for inflow, 0 for outflow)
        alpha = self._flow_direction(flux, velocity).to(device=device, dtype=dtype)

        # Interior velocity magnitude
        u_mag = owner_values.norm(dim=-1)

        # Prescribed inlet velocity = direction * magnitude
        inlet_vel = self._inlet_dir.to(device=device, dtype=dtype).unsqueeze(0) * u_mag.unsqueeze(-1)

        # Blend: inflow * inlet_vel + (1 - inflow) * zero_gradient
        face_values = alpha.unsqueeze(-1) * inlet_vel + (1.0 - alpha.unsqueeze(-1)) * owner_values

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions with inflow/outflow blending."""
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
        alpha = self._flow_direction(flux, velocity).to(device=device, dtype=dtype)

        # Only inflow faces contribute to matrix
        masked_coeff = coeff * alpha

        # Use inlet direction x-component for source projection
        inlet_u_x = self._inlet_dir[0].to(dtype=dtype)

        diag.scatter_add_(0, owners, masked_coeff)
        source.scatter_add_(0, owners, masked_coeff * inlet_u_x)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
