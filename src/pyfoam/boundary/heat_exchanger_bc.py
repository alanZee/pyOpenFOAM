"""
heatExchanger — heat exchanger boundary condition.

Implements a multi-zone heat exchanger boundary condition using the
effectiveness-NTU method for modelling heat exchange between the fluid
and a thermal reservoir.

The effectiveness-NTU model:

    epsilon = 1 - exp(-NTU)       (for C_min/C_max = 0, single-phase)
    Q = epsilon * C_min * (T_hot - T_cold)
    q = Q / A                     (heat flux per unit area)

The BC applies a Robin-type condition:

    -k * dT/dn = h * (T_HEX - T_wall)

where T_HEX is the heat exchanger outlet temperature determined by the
effectiveness-NTU model.

In OpenFOAM syntax::

    type            heatExchanger;
    h               100.0;         // heat transfer coefficient (W/(m^2 K))
    Treservoir      350.0;         // hot-side reservoir temperature (K)
    effectiveness   0.8;           // heat exchanger effectiveness [-]
    Cmin            500.0;         // minimum capacity rate (W/K)
    nZones          2;             // number of zones
    value           uniform 300;   // initial temperature

Usage::

    bc = BoundaryCondition.create("heatExchanger", patch, coeffs={
        "h": 100.0,
        "Treservoir": 350.0,
        "effectiveness": 0.8,
        "Cmin": 500.0,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["HeatExchangerBC"]


@BoundaryCondition.register("heatExchanger")
class HeatExchangerBC(BoundaryCondition):
    """Heat exchanger boundary condition using effectiveness-NTU model.

    Models a multi-zone heat exchanger attached to the boundary.
    Each zone tracks its own fluid outlet temperature, which is
    updated every time step based on the effectiveness-NTU method.

    The BC acts as a Robin condition:

        -k * dT/dn = h * (T_HEX_zone - T_wall)

    Coefficients
    ------------
    h : float
        Convective heat transfer coefficient (W/(m^2 K)). Default: 100.0.
    Treservoir : float
        Hot-side reservoir temperature (K). Default: 350.0.
    effectiveness : float
        Heat exchanger effectiveness in [0, 1]. Default: 0.8.
    Cmin : float
        Minimum capacity rate (W/K). Default: 500.0.
    nZones : int
        Number of thermal zones on the patch. Default: 1.
    value : float
        Initial wall temperature (K). Default: 300.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._h = float(self._coeffs.get("h", 100.0))
        self._T_reservoir = float(self._coeffs.get("Treservoir", 350.0))
        self._effectiveness = float(self._coeffs.get("effectiveness", 0.8))
        self._Cmin = float(self._coeffs.get("Cmin", 500.0))
        self._n_zones = int(self._coeffs.get("nZones", 1))
        self._T_ref = float(self._coeffs.get("value", 300.0))

        # Clamp effectiveness to [0, 1]
        self._effectiveness = max(0.0, min(1.0, self._effectiveness))

        # Per-zone outlet temperatures (initialised to reservoir inlet)
        self._zone_T_out = torch.full(
            (self._n_zones,), self._T_reservoir,
            dtype=get_default_dtype(), device=get_device(),
        )

        # Per-zone face count (distribute faces evenly)
        n_faces = self._patch.n_faces
        base = n_faces // self._n_zones
        remainder = n_faces % self._n_zones
        self._zone_face_counts = torch.tensor(
            [base + (1 if i < remainder else 0) for i in range(self._n_zones)],
            dtype=torch.long,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def h(self) -> float:
        """Heat transfer coefficient (W/(m^2 K))."""
        return self._h

    @property
    def T_reservoir(self) -> float:
        """Hot-side reservoir temperature (K)."""
        return self._T_reservoir

    @property
    def effectiveness(self) -> float:
        """Heat exchanger effectiveness."""
        return self._effectiveness

    @property
    def Cmin(self) -> float:
        """Minimum capacity rate (W/K)."""
        return self._Cmin

    @property
    def n_zones(self) -> int:
        """Number of thermal zones."""
        return self._n_zones

    @property
    def zone_T_out(self) -> torch.Tensor:
        """Per-zone outlet temperatures."""
        return self._zone_T_out

    # ------------------------------------------------------------------
    # Zone operations
    # ------------------------------------------------------------------

    def _get_zone_face_range(self, zone_idx: int) -> tuple[int, int]:
        """Return (start, end) face indices for zone *zone_idx*."""
        start = int(self._zone_face_counts[:zone_idx].sum().item())
        end = start + int(self._zone_face_counts[zone_idx].item())
        return start, end

    def update_zone_temperatures(self, wall_temperatures: torch.Tensor) -> None:
        """Update zone outlet temperatures using effectiveness-NTU.

        For each zone, compute the heat transfer from the reservoir to
        the wall and update the zone's fluid outlet temperature.

        Parameters
        ----------
        wall_temperatures : torch.Tensor
            Wall face temperatures ``(n_faces,)``.
        """
        device = wall_temperatures.device
        dtype = wall_temperatures.dtype
        T_res = self._T_reservoir
        eff = self._effectiveness
        Cmin = self._Cmin

        for z in range(self._n_zones):
            start, end = self._get_zone_face_range(z)
            if start >= end:
                continue

            T_wall_zone = wall_temperatures[start:end]
            T_wall_avg = float(T_wall_zone.mean().item())

            # Effectiveness-NTU: Q = eff * Cmin * (T_res - T_wall_avg)
            Q = eff * Cmin * (T_res - T_wall_avg)

            # Zone outlet: T_out = T_res - Q / Cmin (if Cmin > 0)
            if Cmin > 1e-30:
                T_out = T_res - Q / Cmin
            else:
                T_out = T_wall_avg

            self._zone_T_out[z] = T_out

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply heat exchanger BC to temperature field.

        Sets each zone's face temperatures to the zone outlet temperature.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        T_face = torch.empty(n_faces, dtype=dtype, device=device)

        for z in range(self._n_zones):
            start, end = self._get_zone_face_range(z)
            if start >= end:
                continue
            T_face[start:end] = self._zone_T_out[z].to(device=device, dtype=dtype)

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
        """Matrix contributions for heat exchanger Robin BC.

        For each zone:
            diag[c]   += h * A
            source[c] += h * A * T_HEX_zone
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

        for z in range(self._n_zones):
            start, end = self._get_zone_face_range(z)
            if start >= end:
                continue

            z_owners = owners[start:end]
            z_areas = area_mag[start:end]
            T_hex = self._zone_T_out[z].to(device=device, dtype=dtype)

            h_A = self._h * z_areas
            diag.scatter_add_(0, z_owners, h_A)
            source.scatter_add_(0, z_owners, h_A * T_hex)

        return diag, source
