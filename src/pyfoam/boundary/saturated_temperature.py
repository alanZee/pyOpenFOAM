"""
Saturated temperature boundary condition — saturation temperature from pressure.

Computes the saturation temperature at each boundary face using the
Clausius-Clapeyron relation:

    T_sat = T_ref * exp((p - p_ref) * h_fg / (R * T_ref^2))

where:
    - T_ref  : reference temperature (K)
    - p_ref  : reference pressure corresponding to T_ref (Pa)
    - h_fg   : latent heat of vaporisation (J/kg)
    - R      : specific gas constant for the vapour (J/(kg K))
    - p      : local pressure at the face (Pa)

This BC is used in boiling/condensation simulations where the interface
temperature is governed by thermodynamic equilibrium.

In OpenFOAM syntax::

    wall
    {
        type    saturatedTemperature;
        T_ref   uniform 373.15;    // reference temperature (K)
        p_ref   uniform 101325;    // reference pressure (Pa)
        h_fg    uniform 2.257e6;   // latent heat (J/kg)
        R_v     uniform 461.5;     // gas constant for water vapour
        value   uniform 373.15;
    }

Usage::

    from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SaturatedTemperatureBC"]


@BoundaryCondition.register("saturatedTemperature")
class SaturatedTemperatureBC(BoundaryCondition):
    """Saturation temperature boundary condition.

    Computes T_sat at each face from the local pressure using the
    linearised Clausius-Clapeyron equation.

    Coefficients:
        - ``T_ref``  : Reference temperature (default: 373.15 K).
        - ``p_ref``  : Reference pressure (default: 101325 Pa).
        - ``h_fg``   : Latent heat of vaporisation (default: 2.257e6 J/kg).
        - ``R_v``    : Specific gas constant for vapour (default: 461.5 J/(kg K)).
        - ``value``  : Initial/fallback temperature (default: 373.15 K).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._T_ref = self._parse_scalar("T_ref", 373.15)
        self._p_ref = self._parse_scalar("p_ref", 101325.0)
        self._h_fg = self._parse_scalar("h_fg", 2.257e6)
        self._R_v = self._parse_scalar("R_v", 461.5)
        self._value = self._parse_scalar("value", 373.15)

    def _parse_scalar(self, key: str, default: float) -> torch.Tensor:
        """Parse a coefficient into a per-face tensor."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def T_ref(self) -> torch.Tensor:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def p_ref(self) -> torch.Tensor:
        """Reference pressure (Pa)."""
        return self._p_ref

    @property
    def h_fg(self) -> torch.Tensor:
        """Latent heat of vaporisation (J/kg)."""
        return self._h_fg

    @property
    def R_v(self) -> torch.Tensor:
        """Specific gas constant for vapour (J/(kg K))."""
        return self._R_v

    @property
    def value(self) -> torch.Tensor:
        """Current saturation temperature values at faces."""
        return self._value

    def compute_T_sat(self, p_face: torch.Tensor) -> torch.Tensor:
        """Compute saturation temperature from local pressure.

        Uses the Clausius-Clapeyron relation:
            T_sat = T_ref * exp((p - p_ref) * h_fg / (R * T_ref^2))

        Parameters
        ----------
        p_face : torch.Tensor
            ``(n_faces,)`` pressure at each boundary face.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` saturation temperature.
        """
        exponent = (
            (p_face - self._p_ref) * self._h_fg
            / (self._R_v * self._T_ref * self._T_ref)
        )
        # Clip exponent for numerical stability
        exponent = exponent.clamp(min=-20.0, max=20.0)
        return self._T_ref * torch.exp(exponent)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values to current saturation temperature.

        If a pressure field has been provided via ``update_pressure``,
        the saturation temperature is recomputed; otherwise the stored
        ``value`` is used.
        """
        face_vals = self._value
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = face_vals
        else:
            field[self._patch.face_indices] = face_vals
        return field

    def update_pressure(self, p_face: torch.Tensor) -> None:
        """Update the saturation temperature from a new pressure field.

        Parameters
        ----------
        p_face : torch.Tensor
            ``(n_faces,)`` pressure at each boundary face.
        """
        self._value = self.compute_T_sat(p_face)

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: large diagonal + matching source.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * T_sat
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
        values = self._value.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source
