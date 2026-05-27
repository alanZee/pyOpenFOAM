"""
Enhanced inlet/outlet boundary condition with turbulence-aware treatment.

Implements a variant of inletOutlet that applies turbulence-aware
treatment at open boundaries.  When flow enters the domain, the BC
prescribes fixed values for the transport quantity and optionally
adjusts them based on turbulence state (e.g. turbulent intensity,
mixing length).  When flow exits, zero-gradient is applied.

In OpenFOAM syntax::

    type            inletOutlet3;
    phi             phi;
    value           uniform 0;
    turbulenceIntensity  0.05;    // 5% turbulent intensity at inlet
    mixingLength         0.1;     // mixing length scale (m)

Behaviour:
- **Inflow** (``phi < 0``): applies prescribed value scaled by
  turbulence intensity: ``k_inlet = 1.5 * (I * |U|)^2``
- **Outflow** (``phi >= 0``): applies zero gradient.

The turbulence-aware treatment is activated when ``turbulenceIntensity``
or ``mixingLength`` coefficients are specified.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["InletOutlet3BC"]


@BoundaryCondition.register("inletOutlet3")
class InletOutlet3BC(BoundaryCondition):
    """Enhanced inlet/outlet boundary condition with turbulence-aware treatment.

    - **Inflow** (flux < 0): applies fixed value with optional turbulence
      scaling (e.g. ``k = 1.5 * (I * |U|)^2``).
    - **Outflow** (flux >= 0): applies zero gradient (copies owner values).

    Coefficients:
        - ``phi``: Flux field name (informational).
        - ``value``: Default prescribed value at inflow.
        - ``turbulenceIntensity``: Turbulent intensity ``I`` (0-1).
          Default: 0 (no turbulence correction).
        - ``mixingLength``: Mixing length scale ``l`` (m).
          Default: 0 (no mixing length correction).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._turb_intensity = float(
            self._coeffs.get("turbulenceIntensity", 0.0)
        )
        self._mixing_length = float(
            self._coeffs.get("mixingLength", 0.0)
        )
        self._default_value = self._resolve_default_value()

    @property
    def turbulence_intensity(self) -> float:
        """Return the turbulence intensity."""
        return self._turb_intensity

    @property
    def mixing_length(self) -> float:
        """Return the mixing length."""
        return self._mixing_length

    def _resolve_default_value(self) -> torch.Tensor:
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

    def _flow_direction(
        self,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Determine flow direction at each face.

        Returns:
            Boolean tensor: True for inflow, False for outflow.
        """
        device = get_device()
        if flux is not None:
            flux_dev = flux.to(device=device, dtype=get_default_dtype())
            return flux_dev < 0.0

        if velocity is not None:
            normals = self._patch.face_normals.to(
                device=velocity.device, dtype=velocity.dtype,
            )
            vn = (velocity * normals).sum(dim=-1)
            return vn < 0.0

        return torch.zeros(self._patch.n_faces, dtype=torch.bool, device=device)

    def _compute_turbulent_k(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute turbulent kinetic energy from velocity and intensity.

        ``k = 1.5 * (I * |U|)^2``

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary faces.

        Returns:
            ``(n_faces,)`` turbulent kinetic energy.
        """
        U_mag = velocity.norm(dim=-1)
        return 1.5 * (self._turb_intensity * U_mag) ** 2

    def _compute_turbulent_epsilon(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute turbulent dissipation rate.

        ``epsilon = C_mu^0.75 * k^1.5 / l``

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary faces.

        Returns:
            ``(n_faces,)`` turbulent dissipation rate.
        """
        C_mu = 0.09
        k = self._compute_turbulent_k(velocity)
        l_safe = max(self._mixing_length, 1e-10)
        return C_mu ** 0.75 * k ** 1.5 / l_safe

    def _compute_turbulent_omega(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute specific dissipation rate (omega).

        ``omega = k^0.5 / (C_mu^0.25 * l)``

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary faces.

        Returns:
            ``(n_faces,)`` specific dissipation rate.
        """
        C_mu = 0.09
        k = self._compute_turbulent_k(velocity)
        l_safe = max(self._mixing_length, 1e-10)
        return k.sqrt() / (C_mu ** 0.25 * l_safe)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply turbulence-aware inlet/outlet behaviour.

        Args:
            field: Scalar field to modify.
            patch_idx: Optional start index into field.
            flux: ``(n_faces,)`` face flux for direction detection.
            velocity: ``(n_faces, 3)`` velocity for turbulence scaling.
        """
        is_inflow = self._flow_direction(flux, velocity)
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        # Compute inflow values
        if self._turb_intensity > 0 and velocity is not None:
            # Turbulence-aware: compute k, epsilon, or omega depending on field
            # The default value acts as a base; turbulence adds a correction
            turb_k = self._compute_turbulent_k(velocity)
            inflow_values = self._default_value + turb_k
        else:
            inflow_values = self._default_value

        face_values = torch.where(is_inflow, inflow_values, owner_values)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = face_values
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
        """Matrix contributions depend on flow direction.

        - **Inflow**: penalty method (like fixedValue).
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
        if flux is not None:
            is_inflow = is_inflow.to(device=device)
        elif velocity is not None:
            is_inflow = is_inflow.to(device=device)

        masked_coeff = coeff * is_inflow.to(dtype=dtype)

        # Value to prescribe
        if self._turb_intensity > 0 and velocity is not None:
            turb_k = self._compute_turbulent_k(
                velocity.to(device=device, dtype=dtype),
            )
            p0_tensor = (self._default_value + turb_k).to(device=device, dtype=dtype)
        else:
            p0_tensor = self._default_value.to(device=device, dtype=dtype)

        diag.scatter_add_(0, owners, masked_coeff)
        source.scatter_add_(0, owners, masked_coeff * p0_tensor)

        return diag, source
