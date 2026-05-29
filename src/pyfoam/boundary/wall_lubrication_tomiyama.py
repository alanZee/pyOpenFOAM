"""
Enhanced Tomiyama wall lubrication force BC — version 2.

Implements an enhanced Tomiyama et al. (2002) wall lubrication model
with both Eötvös-number-dependent coefficient AND distance-dependent
scaling.  This combines the Eo-dependent ``f(Eo)`` correlation from
``wall_lubrication_3.py`` with an Antal-style distance-dependent
coefficient to provide a more physically complete model::

    F_wl = C_w_eff * rho_c * alpha_d * |V_slip|^2 / D_p * n_w

where the effective coefficient is::

    C_w_eff(Eo, y) = Cw0 * f(Eo) * Dp / y_w

with ``f(Eo)`` being the piecewise-continuous Tomiyama correlation and
``y_w`` the wall distance.  The coefficient is capped at ``CwMax`` to
prevent singularities near the wall.

In OpenFOAM syntax::

    type              tomiyamaWallLubrication2;
    alpha             alpha.water;
    Cw                0.05;        // base wall lubrication coefficient
    Dp                0.003;       // bubble diameter (m)
    CwMax             10.0;        // maximum coefficient cap
    rhoContinuous     rho.liquid;  // continuous phase density
    rhoDispersed      rho.air;     // dispersed phase density
    sigma             0.072;       // surface tension (N/m)
    value             uniform (0 0 0);

Reference:
    Tomiyama, A., Tamai, H., Zun, I., Hosokawa, S., 2002.
    "Transverse migration of single bubbles in simple shear flows."
    Chem. Eng. Sci. 57(11), 1849–1858.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TomiyamaWallLubrication2BC"]


@BoundaryCondition.register("tomiyamaWallLubrication2")
class TomiyamaWallLubrication2BC(BoundaryCondition):
    """Enhanced Tomiyama wall lubrication force BC (version 2).

    Combines Eötvös-number-dependent coefficient with distance-dependent
    scaling for a physically complete wall lubrication model.

    The effective wall lubrication coefficient is::

        C_w_eff(Eo, y) = min(Cw0 * f(Eo) * Dp / y_w, CwMax)

    where:
        - ``Cw0`` is the base coefficient
        - ``f(Eo)`` is the Tomiyama Eo-dependent correlation
        - ``Dp`` is the bubble diameter
        - ``y_w`` is the wall distance
        - ``CwMax`` is the coefficient cap

    Coefficients:
        - ``Cw``: Base wall lubrication coefficient (default: 0.05).
        - ``Dp``: Bubble diameter in metres (default: 0.003).
        - ``CwMax``: Maximum coefficient cap (default: 10.0).
        - ``sigma``: Surface tension in N/m (default: 0.072).
        - ``rho_c``: Continuous-phase density (default: 1000.0).
        - ``rho_d``: Dispersed-phase density (default: 1.225).
        - ``alpha``: Dispersed-phase volume fraction field name.
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Cw0 = float(self._coeffs.get("Cw", 0.05))
        self._Dp = float(self._coeffs.get("Dp", 0.003))
        self._CwMax = float(self._coeffs.get("CwMax", 10.0))
        self._sigma = float(self._coeffs.get("sigma", 0.072))
        self._rho_c = float(self._coeffs.get("rho_c", 1000.0))
        self._rho_d = float(self._coeffs.get("rho_d", 1.225))
        self._alpha_name = self._coeffs.get("alpha", "alpha.d")

    @property
    def Cw0(self) -> float:
        """Base wall lubrication coefficient."""
        return self._Cw0

    @property
    def Dp(self) -> float:
        """Bubble diameter."""
        return self._Dp

    @property
    def CwMax(self) -> float:
        """Maximum coefficient cap."""
        return self._CwMax

    @property
    def sigma(self) -> float:
        """Surface tension (N/m)."""
        return self._sigma

    @property
    def rho_c(self) -> float:
        """Continuous-phase density (kg/m^3)."""
        return self._rho_c

    @property
    def rho_d(self) -> float:
        """Dispersed-phase density (kg/m^3)."""
        return self._rho_d

    @property
    def alpha_name(self) -> str:
        """Dispersed-phase volume fraction field name."""
        return self._alpha_name

    def eotvos_number(self) -> float:
        """Compute Eotvos number: Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        g = 9.81
        return g * abs(self._rho_c - self._rho_d) * self._Dp ** 2 / self._sigma

    def f_eotvos(self, Eo: float) -> float:
        """Tomiyama's Eotvos-dependent wall lubrication correlation.

        Parameters
        ----------
        Eo : float
            Eotvos number.

        Returns
        -------
        float
            f(Eo) value.
        """
        if Eo < 1.0:
            return 0.0
        elif Eo < 5.0:
            return 0.474 * (1.0 - math.exp(-0.0183 * Eo)) * math.exp(1.48 * Eo)
        elif Eo < 33.0:
            return 0.0219 * Eo
        else:
            return 0.474

    def effective_coefficient(self, wall_distance: torch.Tensor) -> torch.Tensor:
        """Compute the enhanced Tomiyama wall lubrication coefficient.

        C_w_eff = min(Cw0 * f(Eo) * Dp / y_w, CwMax)

        Parameters
        ----------
        wall_distance : torch.Tensor
            Distance from wall for each boundary face.

        Returns
        -------
        torch.Tensor
            Effective wall lubrication coefficient per face.
        """
        Eo = self.eotvos_number()
        f_eo = self.f_eotvos(Eo)
        safe_dist = wall_distance.clamp(min=1e-10)
        Cw_raw = self._Cw0 * f_eo * self._Dp / safe_dist
        return Cw_raw.clamp(max=self._CwMax)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply wall lubrication velocity correction.

        Sets the wall face velocity to the owner cell value (zero-gradient
        treatment).  The actual force is handled by ``matrix_contributions``.

        Parameters
        ----------
        field : torch.Tensor
            Velocity field ``(n_cells, 3)`` or ``(n_cells,)``.
        patch_idx : int, optional
            Start index into *field*.
        alpha : float or torch.Tensor, optional
            Dispersed-phase volume fraction (unused here).
        rho : float or torch.Tensor, optional
            Dispersed-phase density (unused here).
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
        rho: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhanced Tomiyama wall lubrication force source contribution.

        Adds a Eotvos-dependent AND distance-dependent source term::

            source[c] += C_w_eff * rho_c * alpha * area / Dp

        Parameters
        ----------
        field : torch.Tensor
            Current velocity field.
        n_cells : int
            Total number of cells.
        diag : torch.Tensor, optional
            Pre-existing diagonal tensor.
        source : torch.Tensor, optional
            Pre-existing source tensor.
        alpha : float or torch.Tensor, optional
            Dispersed-phase volume fraction.
        rho : float or torch.Tensor, optional
            Dispersed-phase density (unused; uses rho_c).
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
        wall_dist = 1.0 / deltas.clamp(min=1e-10)

        # Enhanced coefficient: Eo-dependent + distance-dependent
        Cw_eff = self.effective_coefficient(wall_dist)

        # Alpha
        if alpha is None:
            alpha_val = 0.1
        elif isinstance(alpha, torch.Tensor):
            alpha_val = alpha[owners].to(device=device, dtype=dtype)
        else:
            alpha_val = float(alpha)

        # Force coefficient
        force_coeff = Cw_eff * self._rho_c * alpha_val * areas / self._Dp

        source.scatter_add_(0, owners, force_coeff)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
