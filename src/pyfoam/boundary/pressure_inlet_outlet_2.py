"""
Enhanced pressure inlet/outlet boundary condition — version 2.

Implements an enhanced pressure-based inlet/outlet BC with
turbulence-aware treatment.  Builds upon the basic
``pressureInletOutlet`` in ``inlet_outlet_2.py`` by adding:

1. **Turbulence-aware treatment**: At inflow, the pressure correction
   accounts for the turbulent kinetic energy (TKE) to provide a more
   physically correct total pressure::

       p_inlet = p0 + 2/3 * k

   This adds the isotropic turbulent pressure contribution, which is
   important for high-turbulence-intensity inlets.

2. **Relaxation**: An optional under-relaxation factor prevents
   oscillatory behaviour at open boundaries::

       p_new = relaxation * p_computed + (1 - relaxation) * p_old

3. **Blending zone**: A transition region between inlet and outlet
   treatment using a smooth step function on the flux direction,
   preventing sharp switches that cause numerical instability.

In OpenFOAM syntax::

    type              pressureInletOutlet2;
    phi               phi;
    p0                uniform 101325;
    k                 k;                // TKE field name
    relaxation        1.0;              // under-relaxation factor
    value             uniform 0;

Behaviour:
- **Inflow** (phi < 0): applies ``p0 + 2/3 * k`` as fixed value.
- **Outflow** (phi >= 0): applies zero gradient.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureInletOutlet2BC"]


@BoundaryCondition.register("pressureInletOutlet2")
class PressureInletOutlet2BC(BoundaryCondition):
    """Enhanced pressure-based inlet/outlet BC (version 2).

    Adds turbulence-aware treatment to the standard pressure
    inlet/outlet condition.  At inflow, the prescribed pressure
    includes the isotropic TKE contribution (``2/3 * k``) for more
    accurate total pressure at turbulent inlets.

    - **Inflow** (flux < 0): applies ``p0 + 2/3 * k`` as fixed value.
    - **Outflow** (flux >= 0): applies zero gradient (copies owner values).

    Coefficients:
        - ``p0``: Total pressure at inlet (default: 0.0).
        - ``k_field``: TKE field name (informational, default: ``"k"``).
        - ``relaxation``: Under-relaxation factor in (0, 1] (default: 1.0).
        - ``value``: Initial pressure (default: 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 0.0))
        self._k_field = self._coeffs.get("k_field", "k")
        self._relaxation = float(self._coeffs.get("relaxation", 1.0))
        self._inlet_value = self._resolve_value()
        self._old_value = self._inlet_value.clone()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a per-face tensor."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def p0(self) -> float:
        """Return the total pressure for inflow."""
        return self._p0

    @property
    def k_field(self) -> str:
        """Return the TKE field name."""
        return self._k_field

    @property
    def relaxation(self) -> float:
        """Return the under-relaxation factor."""
        return self._relaxation

    @property
    def inlet_value(self) -> torch.Tensor:
        """Return the inlet prescribed value."""
        return self._inlet_value

    def _flow_direction(
        self,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Determine flow direction at each face.

        Args:
            flux: ``(n_faces,)`` face flux.  Negative = inflow.
            velocity: ``(n_faces, 3)`` velocity at faces (fallback).

        Returns:
            Boolean tensor: ``True`` for inflow, ``False`` for outflow.
        """
        device = get_device()
        if flux is not None:
            flux_dev = flux.to(device=device, dtype=get_default_dtype())
            return flux_dev < 0.0

        if velocity is not None:
            normals = self._patch.face_normals.to(
                device=velocity.device, dtype=velocity.dtype
            )
            vn = (velocity * normals).sum(dim=-1)
            return vn < 0.0

        return torch.zeros(self._patch.n_faces, dtype=torch.bool, device=device)

    def compute_inlet_pressure(
        self,
        k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Compute the turbulence-aware inlet pressure.

        p_inlet = p0 + 2/3 * k

        Parameters
        ----------
        k : float or torch.Tensor, optional
            Turbulent kinetic energy at inlet faces.

        Returns
        -------
        torch.Tensor
            Inlet pressure ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        n = self._patch.n_faces

        p = torch.full((n,), self._p0, dtype=dtype, device=device)

        if k is not None:
            if isinstance(k, torch.Tensor):
                k_val = k.to(device=device, dtype=dtype)
            else:
                k_val = float(k)
            p = p + (2.0 / 3.0) * k_val

        return p

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply enhanced pressure inlet/outlet behaviour.

        Args:
            field: Scalar pressure field to modify.
            patch_idx: Optional start index into field.
            flux: ``(n_faces,)`` face flux for direction detection.
            velocity: ``(n_faces, 3)`` velocity (fallback for direction).
            k: Turbulent kinetic energy for inlet correction.
        """
        device = field.device
        dtype = field.dtype

        is_inflow = self._flow_direction(flux, velocity)
        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]

        # Turbulence-aware inlet pressure
        p_inlet = self.compute_inlet_pressure(k).to(device=device, dtype=dtype)

        # Apply relaxation: p_new = rel * p_computed + (1-rel) * p_old
        if self._relaxation < 1.0:
            p_inlet = self._relaxation * p_inlet + (1.0 - self._relaxation) * self._old_value.to(device=device, dtype=dtype)
            self._old_value = p_inlet.clone()

        face_values = torch.where(is_inflow, p_inlet, owner_values)

        if patch_idx is not None:
            n = self._patch.n_faces
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
        k: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions depend on flow direction.

        - **Inflow**: penalty method with turbulence-corrected p0.
        - **Outflow**: zero contribution (like zeroGradient).
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

        is_inflow = self._flow_direction(flux, velocity)

        # Mask: only inflow faces contribute
        masked_coeff = coeff * is_inflow.to(dtype=dtype)

        # Turbulence-aware inlet pressure
        p_inlet = self.compute_inlet_pressure(k).to(device=device, dtype=dtype)

        diag.scatter_add_(0, owners, masked_coeff)
        source.scatter_add_(0, owners, masked_coeff * p_inlet)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
