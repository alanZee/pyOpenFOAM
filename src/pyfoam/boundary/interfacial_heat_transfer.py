"""
Interfacial heat transfer boundary condition for boiling/condensation.

Implements the ``interfacialHeatTransferBC`` which models heat transfer
at the interface between two phases in Euler-Euler multiphase simulations.
The heat flux at the interface is::

    q_i = h_i * A_i * (T_sat - T)

where:
    - ``h_i`` is the interfacial heat transfer coefficient (W/(m^2 K))
    - ``A_i`` is the interfacial area density (1/m)
    - ``T_sat`` is the saturation temperature (K)
    - ``T`` is the local temperature (K)

The interfacial heat transfer coefficient depends on the flow regime:

    For evaporation (T >= T_sat):
        h_i = h_fg * m_dot / (T_sat - T_ref)

    For condensation (T < T_sat):
        h_i = Nu * lambda_l / D_h

In OpenFOAM syntax::

    type              interfacialHeatTransfer;
    alpha             alpha.vapor;
    hi                10000;       // interfacial HT coefficient (W/m^2-K)
    Tsat              373.15;      // saturation temperature (K)
    L                 2.26e6;      // latent heat of vaporisation (J/kg)
    alphaMax          0.8;         // maximum dispersed fraction for HT
    value             uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["InterfacialHeatTransferBC"]


@BoundaryCondition.register("interfacialHeatTransfer")
class InterfacialHeatTransferBC(BoundaryCondition):
    """Interfacial heat transfer BC for boiling/condensation.

    Models heat exchange at the phase interface in Euler-Euler
    multiphase simulations.  The BC contributes a temperature-dependent
    source term to the energy equation for cells adjacent to the
    interface.

    The heat transfer rate is::

        Q_i = h_i * A_i * alpha * (T_sat - T) * V_cell

    where ``alpha`` is the dispersed phase volume fraction (proxy for
    interfacial area density), ``V_cell`` is the cell volume (approximated
    from patch face areas), and the sign convention is:
        - Q > 0: heating (condensation, T < T_sat)
        - Q < 0: cooling (evaporation, T > T_sat)

    Coefficients:
        - ``hi``: Interfacial heat transfer coefficient in W/(m^2 K)
          (default: 10000.0).
        - ``Tsat``: Saturation temperature in K (default: 373.15).
        - ``L``: Latent heat of vaporisation in J/kg (default: 2.26e6).
        - ``alphaMax``: Maximum dispersed fraction for heat transfer
          (default: 0.8). Beyond this, interfacial area is reduced.
        - ``alpha``: Dispersed-phase volume fraction field name
          (default: ``"alpha.d"``).
        - ``value``: Initial temperature (default: ``uniform 0``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._hi = float(self._coeffs.get("hi", 10000.0))
        self._Tsat = float(self._coeffs.get("Tsat", 373.15))
        self._L = float(self._coeffs.get("L", 2.26e6))
        self._alphaMax = float(self._coeffs.get("alphaMax", 0.8))
        self._alpha_name = self._coeffs.get("alpha", "alpha.d")

    @property
    def hi(self) -> float:
        """Interfacial heat transfer coefficient (W/(m^2 K))."""
        return self._hi

    @property
    def Tsat(self) -> float:
        """Saturation temperature (K)."""
        return self._Tsat

    @property
    def L(self) -> float:
        """Latent heat of vaporisation (J/kg)."""
        return self._L

    @property
    def alphaMax(self) -> float:
        """Maximum dispersed fraction for HT."""
        return self._alphaMax

    @property
    def alpha_name(self) -> str:
        """Dispersed-phase volume fraction field name."""
        return self._alpha_name

    def interfacial_area_factor(self, alpha: torch.Tensor | float) -> torch.Tensor:
        """Compute interfacial area density factor from volume fraction.

        Uses a simple linear model that saturates at alphaMax::

            A_i = alpha          if alpha < alphaMax
            A_i = alphaMax       otherwise

        Parameters
        ----------
        alpha : torch.Tensor or float
            Dispersed-phase volume fraction.

        Returns
        -------
        torch.Tensor
            Interfacial area density factor.
        """
        if isinstance(alpha, (int, float)):
            return torch.tensor(
                min(float(alpha), self._alphaMax),
                dtype=get_default_dtype(),
            )
        return alpha.clamp(max=self._alphaMax)

    def heat_transfer_rate(
        self,
        T: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interfacial heat transfer rate per unit volume.

        Q = hi * A_i(alpha) * (Tsat - T)

        Parameters
        ----------
        T : torch.Tensor
            Temperature field ``(n_cells,)``.
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Heat transfer rate (W/m^3). Positive = heating.
        """
        A_i = self.interfacial_area_factor(alpha)
        return self._hi * A_i * (self._Tsat - T)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        T: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply interfacial heat transfer BC.

        Sets the patch face temperature to a Robin-type condition that
        blends between the internal value and the saturation temperature.

        Parameters
        ----------
        field : torch.Tensor
            Temperature field ``(n_cells,)``.
        patch_idx : int, optional
            Start index into *field*.
        alpha : float or torch.Tensor, optional
            Dispersed-phase volume fraction.
        T : float or torch.Tensor, optional
            Unused (field contains current temperature).
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        alpha: torch.Tensor | float | None = None,
        T: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interfacial heat transfer source contribution.

        Adds a temperature-dependent source term for cells adjacent to
        the interface::

            diag[c]   += hi * A_i * area  (implicit part)
            source[c] += hi * A_i * Tsat * area  (explicit part)

        This linearisation ensures boundedness: as T -> Tsat, the source
        vanishes.

        Parameters
        ----------
        field : torch.Tensor
            Current temperature field.
        n_cells : int
            Total number of cells.
        diag : torch.Tensor, optional
            Pre-existing diagonal tensor.
        source : torch.Tensor, optional
            Pre-existing source tensor.
        alpha : float or torch.Tensor, optional
            Dispersed-phase volume fraction.
        T : float or torch.Tensor, optional
            Current temperature (unused; uses field).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        # Alpha (dispersed-phase volume fraction)
        if alpha is None:
            alpha_val = 0.1
        elif isinstance(alpha, torch.Tensor):
            alpha_val = alpha[owners].to(device=device, dtype=dtype)
        else:
            alpha_val = float(alpha)

        # Interfacial area factor
        A_i = self.interfacial_area_factor(alpha_val)

        # Robin-type linearisation: diag += hi * A_i * area, source += hi * A_i * Tsat * area
        coeff = self._hi * A_i * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * self._Tsat)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
