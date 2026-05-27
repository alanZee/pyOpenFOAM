"""
fixedShearStress boundary condition.

Wall BC with a prescribed shear stress vector.  Adjusts the velocity
gradient at the wall to produce the desired wall shear stress.

In OpenFOAM syntax::

    type          fixedShearStress;
    tau0          uniform (0.1 0 0);   // prescribed shear stress (Pa)
    rho           1.0;                  // reference density (kg/m³)
    value         uniform (0 0 0);

The wall velocity is adjusted each iteration so that::

    tau_wall = mu * (du/dy)|_wall = tau0

In the discrete formulation the face value is set to produce the
target shear stress via the penalty method.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedShearStressBC"]


@BoundaryCondition.register("fixedShearStress")
class FixedShearStressBC(BoundaryCondition):
    """Fixed shear stress wall boundary condition.

    Prescribes the wall shear stress vector.  The boundary face velocity
    is computed so that the viscous stress at the wall equals the
    prescribed value.

    For a simple wall with no penetration (u_n = 0) and a tangential
    shear stress tau, the equivalent wall velocity is::

        u_t = tau * delta / mu

    where delta is the wall-normal distance (1/deltaCoeff) and mu is
    the dynamic viscosity.  When viscosity is not provided, the BC
    applies tau directly as a force source (momentum equation source
    contribution).

    Coefficients:
        - ``tau0``: Prescribed shear stress vector (Pa).
            Can be a scalar (applied in x-direction) or 3-component vector.
        - ``rho``: Reference density (default: 1.0 kg/m³).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._tau0 = self._parse_tau()
        self._rho = float(self._coeffs.get("rho", 1.0))

    def _parse_tau(self) -> torch.Tensor:
        """Parse the shear stress coefficient into a (3,) tensor."""
        device = get_device()
        dtype = get_default_dtype()
        raw = self._coeffs.get("tau0", [0.0, 0.0, 0.0])

        if isinstance(raw, (int, float)):
            # Scalar → apply in x-direction
            return torch.tensor(
                [float(raw), 0.0, 0.0], dtype=dtype, device=device
            )
        if isinstance(raw, torch.Tensor):
            t = raw.to(dtype=dtype, device=device).flatten()
            if t.numel() == 1:
                return torch.tensor(
                    [t.item(), 0.0, 0.0], dtype=dtype, device=device
                )
            if t.numel() == 3:
                return t
            raise ValueError(
                f"tau0 must be a scalar or 3-element vector, got {t.numel()} elements"
            )
        # Sequence
        t = torch.tensor(raw, dtype=dtype, device=device).flatten()
        if t.numel() == 1:
            return torch.tensor(
                [t.item(), 0.0, 0.0], dtype=dtype, device=device
            )
        if t.numel() != 3:
            raise ValueError(
                f"tau0 must be a scalar or 3-element vector, got {t.numel()} elements"
            )
        return t

    @property
    def tau0(self) -> torch.Tensor:
        """Return the prescribed shear stress vector (3,)."""
        return self._tau0

    @property
    def rho(self) -> float:
        """Return the reference density."""
        return self._rho

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values for prescribed shear stress.

        The face velocity is set to produce the target wall shear stress
        via the penalty method.  The tangential velocity is::

            u_t = tau0 / (rho * deltaCoeff)

        where deltaCoeff = 1/delta is the inverse wall-normal distance.
        """
        device = field.device
        dtype = field.dtype

        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        tau = self._tau0.to(device=device, dtype=dtype)

        # u_wall = tau / (rho * deltaCoeff)
        # Safe division: clamp deltaCoeff
        safe_deltas = torch.where(
            deltas > 1e-30, deltas, torch.ones_like(deltas) * 1e-30
        )
        inv_rho_delta = 1.0 / (self._rho * safe_deltas)

        # Build (n_faces, 3) velocity: tau * inv_rho_delta broadcast
        velocity = tau.unsqueeze(0) * inv_rho_delta.unsqueeze(1)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = velocity
        else:
            field[self._patch.face_indices] = velocity
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for fixed shear stress BC.

        Uses fixedValue-style penalty.  The source term includes the
        prescribed shear stress projected onto the x-component for the
        scalar matrix system.
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

        # Compute velocity values
        safe_deltas = torch.where(
            deltas > 1e-30, deltas, torch.ones_like(deltas) * 1e-30
        )
        inv_rho_delta = 1.0 / (self._rho * safe_deltas)
        tau = self._tau0.to(device=device, dtype=dtype)
        velocity_x = tau[0] * inv_rho_delta

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity_x)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
