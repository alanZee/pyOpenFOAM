"""
Turbulence boundary conditions.

Implements OpenFOAM turbulence boundary conditions:
- turbulentIntensityKineticEnergyInlet: k from turbulence intensity
- turbulentMixingLengthDissipationRateInlet: epsilon from mixing length
- turbulentMixingLengthFrequencyInlet: omega from mixing length

In OpenFOAM syntax::

    // turbulentIntensityKineticEnergyInlet
    type        turbulentIntensityKineticEnergyInlet;
    intensity   0.05;           // turbulence intensity (5%)
    U           U;              // velocity field name
    value       uniform 0.01;

    // turbulentMixingLengthDissipationRateInlet
    type        turbulentMixingLengthDissipationRateInlet;
    mixingLength 0.01;          // mixing length (m)
    Cmu         0.09;           // k-epsilon model constant
    value       uniform 0.01;

    // turbulentMixingLengthFrequencyInlet
    type        turbulentMixingLengthFrequencyInlet;
    mixingLength 0.01;          // mixing length (m)
    Cmu         0.09;           // k-omega model constant
    beta        0.075;          // k-omega model constant
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "TurbulentIntensityKineticEnergyInletBC",
    "TurbulentMixingLengthDissipationRateInletBC",
    "TurbulentMixingLengthFrequencyInletBC",
]


@BoundaryCondition.register("turbulentIntensityKineticEnergyInlet")
class TurbulentIntensityKineticEnergyInletBC(BoundaryCondition):
    """Turbulent kinetic energy inlet boundary condition.

    Computes turbulent kinetic energy k from the turbulence intensity I
    and the mean velocity U::

        k = 1.5 * (I * |U|)²

    where:
        - I is the turbulence intensity (typically 0.01 to 0.10)
        - |U| is the magnitude of the mean velocity

    Coefficients:
        - ``intensity``: Turbulence intensity (default: 0.05).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))

    @property
    def intensity(self) -> float:
        """Return turbulence intensity."""
        return self._intensity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k from turbulence intensity.

        k = 1.5 * (I * |U|)²

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
        """
        device = field.device
        dtype = field.dtype

        if velocity is not None:
            # Compute velocity magnitude
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            # k = 1.5 * (I * |U|)²
            k = 1.5 * (self._intensity * u_mag) ** 2
        else:
            # No velocity info → use a default value
            k = torch.full(
                (self._patch.n_faces,),
                0.01,  # default k
                dtype=dtype,
                device=device,
            )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for k inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Use a default k value for matrix contributions
        k_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


@BoundaryCondition.register("turbulentMixingLengthDissipationRateInlet")
class TurbulentMixingLengthDissipationRateInletBC(BoundaryCondition):
    """Turbulent dissipation rate inlet boundary condition.

    Computes turbulent dissipation rate epsilon from the mixing length l
    and turbulent kinetic energy k::

        epsilon = C_mu^0.75 * k^1.5 / l

    where:
        - C_mu is a model constant (typically 0.09)
        - k is the turbulent kinetic energy
        - l is the mixing length

    If k is not provided, it is estimated from the velocity as::

        k = 1.5 * (I * |U|)²  with I = 0.1

    Coefficients:
        - ``mixingLength``: Mixing length (m).
        - ``Cmu``: k-epsilon model constant (default: 0.09).
        - ``value``: Initial epsilon value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))

    @property
    def mixing_length(self) -> float:
        """Return mixing length."""
        return self._mixing_length

    @property
    def C_mu(self) -> float:
        """Return C_mu constant."""
        return self._C_mu

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon from mixing length.

        epsilon = C_mu^0.75 * k^1.5 / l

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index into field.
            k: ``(n_faces,)`` turbulent kinetic energy at boundary faces.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
        """
        device = field.device
        dtype = field.dtype

        if k is not None:
            # epsilon = C_mu^0.75 * k^1.5 / l
            epsilon = (self._C_mu ** 0.75) * (k ** 1.5) / self._mixing_length
        elif velocity is not None:
            # Estimate k from velocity (assuming I = 0.1)
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (0.1 * u_mag) ** 2
            epsilon = (self._C_mu ** 0.75) * (k_est ** 1.5) / self._mixing_length
        else:
            # No info → use default
            epsilon = torch.full(
                (self._patch.n_faces,),
                0.01,  # default epsilon
                dtype=dtype,
                device=device,
            )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = epsilon
        else:
            field[self._patch.face_indices] = epsilon
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for epsilon inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Use a default epsilon value for matrix contributions
        epsilon_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * epsilon_default)

        return diag, source


@BoundaryCondition.register("turbulentMixingLengthFrequencyInlet")
class TurbulentMixingLengthFrequencyInletBC(BoundaryCondition):
    """Turbulent specific dissipation rate (omega) inlet boundary condition.

    Computes turbulent specific dissipation rate omega from the mixing
    length l and turbulent kinetic energy k::

        omega = k^0.5 / (C_mu^0.25 * l)

    where:
        - C_mu is a model constant (typically 0.09)
        - k is the turbulent kinetic energy
        - l is the mixing length

    If k is not provided, it is estimated from the velocity as::

        k = 1.5 * (I * |U|)²  with I = 0.1

    Coefficients:
        - ``mixingLength``: Mixing length (m).
        - ``Cmu``: k-omega model constant (default: 0.09).
        - ``beta``: k-omega model constant (default: 0.075).
        - ``value``: Initial omega value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._beta = float(self._coeffs.get("beta", 0.075))

    @property
    def mixing_length(self) -> float:
        """Return mixing length."""
        return self._mixing_length

    @property
    def C_mu(self) -> float:
        """Return C_mu constant."""
        return self._C_mu

    @property
    def beta(self) -> float:
        """Return beta constant."""
        return self._beta

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face omega from mixing length.

        omega = k^0.5 / (C_mu^0.25 * l)

        Args:
            field: Turbulent specific dissipation rate field.
            patch_idx: Optional start index into field.
            k: ``(n_faces,)`` turbulent kinetic energy at boundary faces.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
        """
        device = field.device
        dtype = field.dtype

        if k is not None:
            # omega = k^0.5 / (C_mu^0.25 * l)
            omega = torch.sqrt(k) / (self._C_mu ** 0.25 * self._mixing_length)
        elif velocity is not None:
            # Estimate k from velocity (assuming I = 0.1)
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (0.1 * u_mag) ** 2
            omega = torch.sqrt(k_est) / (self._C_mu ** 0.25 * self._mixing_length)
        else:
            # No info → use default
            omega = torch.full(
                (self._patch.n_faces,),
                0.01,  # default omega
                dtype=dtype,
                device=device,
            )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = omega
        else:
            field[self._patch.face_indices] = omega
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for omega inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Use a default omega value for matrix contributions
        omega_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * omega_default)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
