"""
Subcooling temperature boundary condition.

Computes the subcooling temperature at each boundary face:

    T_sub = T_sat - T

where:
    - T_sat : saturation temperature at local pressure (K)
    - T     : local temperature (K)

A positive value indicates subcooled liquid (T < T_sat), which drives
condensation.  A negative value indicates superheated liquid (T > T_sat),
which drives evaporation.

This BC is commonly used in boiling and condensation simulations for
wall heat transfer models.

In OpenFOAM syntax::

    wall
    {
        type    subcooling;
        T_ref   uniform 373.15;
        p_ref   uniform 101325;
        h_fg    uniform 2.257e6;
        R_v     uniform 461.5;
        value   uniform 0;
    }

Usage::

    from pyfoam.boundary.subcooling import SubcoolingBC
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SubcoolingBC"]


@BoundaryCondition.register("subcooling")
class SubcoolingBC(BoundaryCondition):
    """Subcooling temperature boundary condition.

    Computes the temperature difference between the local saturation
    temperature and the actual fluid temperature at the wall:

        T_sub = T_sat(p) - T_wall

    This quantity drives wall boiling / condensation heat transfer
    models (e.g. Rensselaer, Kurul-Podowski).

    Coefficients:
        - ``T_ref``  : Reference temperature for Clausius-Clapeyron (default: 373.15 K).
        - ``p_ref``  : Reference pressure (default: 101325 Pa).
        - ``h_fg``   : Latent heat of vaporisation (default: 2.257e6 J/kg).
        - ``R_v``    : Specific gas constant for vapour (default: 461.5 J/(kg K)).
        - ``value``  : Initial subcooling temperature (default: 0 K).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._T_ref = self._parse_scalar("T_ref", 373.15)
        self._p_ref = self._parse_scalar("p_ref", 101325.0)
        self._h_fg = self._parse_scalar("h_fg", 2.257e6)
        self._R_v = self._parse_scalar("R_v", 461.5)
        self._value = self._parse_scalar("value", 0.0)

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
        """Current subcooling temperature at faces."""
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
        exponent = exponent.clamp(min=-20.0, max=20.0)
        return self._T_ref * torch.exp(exponent)

    def compute_subcooling(
        self,
        p_face: torch.Tensor,
        T_face: torch.Tensor,
    ) -> torch.Tensor:
        """Compute subcooling temperature.

        T_sub = T_sat(p) - T

        Parameters
        ----------
        p_face : torch.Tensor
            ``(n_faces,)`` pressure at each boundary face.
        T_face : torch.Tensor
            ``(n_faces,)`` temperature at each boundary face.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` subcooling temperature.
            Positive = subcooled (condensation).
            Negative = superheated (evaporation).
        """
        T_sat = self.compute_T_sat(p_face)
        return T_sat - T_face

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> None:
        """Set boundary-face values to current subcooling temperature."""
        face_vals = self._value
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = face_vals
        else:
            field[self._patch.face_indices] = face_vals

    def update(
        self,
        p_face: torch.Tensor,
        T_face: torch.Tensor,
    ) -> None:
        """Update subcooling from new pressure and temperature fields.

        Parameters
        ----------
        p_face : torch.Tensor
            ``(n_faces,)`` pressure at each boundary face.
        T_face : torch.Tensor
            ``(n_faces,)`` temperature at each boundary face.
        """
        self._value = self.compute_subcooling(p_face, T_face)

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions (subcooling is typically used as a
        post-processing quantity, but for implicit coupling it acts
        as a mixed BC with T_sat driving the boundary value).

        diag[c]   += f * deltaCoeff * faceArea
        source[c] += f * deltaCoeff * faceArea * T_sat
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

        # T_sat is the driving temperature for subcooling BC
        # For penalty method, use T_sat as the prescribed value
        T_sat = self._T_ref  # fallback
        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * T_sat)

        return diag, source
