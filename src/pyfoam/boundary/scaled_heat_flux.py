"""
scaledHeatFlux -- scaled heat flux boundary condition.

Prescribes a heat flux that is a scaled version of a reference heat flux::

    type    scaledHeatFlux;
    scale   2.0;           // scaling factor
    q_ref   500.0;         // reference heat flux (W/m2)
    k       0.025;         // thermal conductivity (W/(m K))
    value   uniform 300;   // reference temperature (K)

The effective heat flux is::

    q = scale * q_ref

This is useful for parametric studies or control applications where
the heat flux is proportional to a known reference value.

Usage::

    bc = BoundaryCondition.create("scaledHeatFlux", patch, coeffs={
        "scale": 2.0, "q_ref": 500.0, "k": 0.025
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFluxBC"]


@BoundaryCondition.register("scaledHeatFlux")
class ScaledHeatFluxBC(BoundaryCondition):
    """Scaled heat flux boundary condition.

    Prescribes a wall heat flux as a scaled version of a reference value::

        q = scale * q_ref

    The temperature gradient is derived from Fourier's law::

        dT/dn = -q / k

    and applied through matrix contributions for implicit coupling
    with the energy equation.

    Coefficients
    ------------
    scale : float
        Flux scaling factor.  Default: 1.0.
    q_ref : float
        Reference heat flux (W/m^2).  Default: 0.0.
    k : float
        Thermal conductivity (W/(m K)).  Default: 0.025 (air).
    value : float
        Reference temperature (K).  Default: 300.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._scale = float(self._coeffs.get("scale", 1.0))
        self._q_ref = float(self._coeffs.get("q_ref", 0.0))
        self._k = float(self._coeffs.get("k", 0.025))
        self._T_ref = float(self._coeffs.get("value", 300.0))

    @property
    def scale(self) -> float:
        """Flux scaling factor."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    def q_ref(self) -> float:
        """Reference heat flux (W/m^2)."""
        return self._q_ref

    @q_ref.setter
    def q_ref(self, value: float) -> None:
        self._q_ref = value

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = value

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def q(self) -> float:
        """Effective heat flux: scale * q_ref (W/m^2)."""
        return self._scale * self._q_ref

    @property
    def gradient(self) -> float:
        """Wall-normal temperature gradient (K/m).

        dT/dn = -q / k = -(scale * q_ref) / k
        """
        if abs(self._k) < 1e-30:
            return 0.0
        return -self.q / self._k

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply scaled heat flux BC to temperature field.

        Sets the boundary temperature based on the reference temperature
        and the prescribed flux gradient.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        delta = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        delta_safe = delta.abs().clamp(min=1e-10)

        # T_face = T_ref + gradient / delta
        T_face = torch.full(
            (n_faces,), self._T_ref, dtype=dtype, device=device,
        )
        T_face = T_face + self.gradient / delta_safe

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = T_face
        else:
            field[self._patch.face_indices] = T_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions for scaled heat flux BC.

        Uses fixedGradient treatment:
            - Source += q * area  (heat flux contribution)
            - Diagonal remains unchanged (implicit zero-gradient part)
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

        # Heat flux contribution to source: q * A
        flux_contrib = self.q * area_mag
        source.scatter_add_(0, owners, flux_contrib)

        return diag, source
