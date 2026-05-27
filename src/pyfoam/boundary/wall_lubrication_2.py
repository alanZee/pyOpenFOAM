"""
Enhanced wall lubrication force boundary condition — Antal model.

Implements the Antal et al. (1991) wall lubrication force model with
distance-dependent coefficient.  Unlike the basic :class:`WallLubricationBC`
in ``wall_lubrication.py``, this model uses the Antal formulation with
explicit wall-distance dependence::

    F_wl = C_w * rho_d * alpha_d * V_slip^2 / y_w * n_w

where:
    - ``C_w`` is the wall lubrication coefficient (function of y/D)
    - ``rho_d`` is the dispersed-phase density
    - ``alpha_d`` is the dispersed-phase volume fraction
    - ``V_slip`` is the relative (slip) velocity magnitude
    - ``y_w`` is the distance from the wall
    - ``n_w`` is the wall-normal direction

The Antal model uses a distance-dependent coefficient::

    C_w(y/D) = max(C_w0 * D / y_w, C_w_max)

where ``C_w0`` is the base coefficient, ``D`` is the particle diameter,
and ``C_w_max`` caps the force at very small wall distances.

In OpenFOAM syntax::

    type              antalWallLubrication;
    alpha             alpha.water;
    Cw                0.05;        // base wall lubrication coefficient
    Dp                0.003;       // bubble/particle diameter (m)
    CwMax             10.0;        // maximum coefficient cap
    rho               rho.water;
    value             uniform (0 0 0);

Reference:
    Antal, S.P., Lahey, R.T., Flaherty, J.E., 1991.
    "Analysis of phase distribution in fully developed
    laminar bubbly two-phase flow."
    Int. J. Multiphase Flow 17(5), 635–652.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["AntalWallLubricationBC"]


@BoundaryCondition.register("antalWallLubrication")
class AntalWallLubricationBC(BoundaryCondition):
    """Antal wall lubrication force BC for multiphase Euler-Euler models.

    Applies the Antal et al. (1991) wall lubrication force model with
    distance-dependent coefficient.  The force pushes the dispersed
    phase away from walls.

    The model coefficient is::

        C_w_eff = max(Cw0 * Dp / y_w, CwMax)

    where ``y_w`` is the wall distance (approximated from ``delta_coeffs``).

    Coefficients:
        - ``Cw``: Base wall lubrication coefficient (default: 0.05).
        - ``Dp``: Particle/bubble diameter in metres (default: 0.003).
        - ``CwMax``: Maximum coefficient cap (default: 10.0).
        - ``alpha``: Dispersed-phase volume fraction field name
          (informational, default: ``"alpha.d"``).
        - ``rho``: Dispersed-phase density field name
          (informational, default: ``"rho.d"``).
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Cw0 = float(self._coeffs.get("Cw", 0.05))
        self._Dp = float(self._coeffs.get("Dp", 0.003))
        self._CwMax = float(self._coeffs.get("CwMax", 10.0))
        self._alpha_name = self._coeffs.get("alpha", "alpha.d")
        self._rho_name = self._coeffs.get("rho", "rho.d")

    @property
    def Cw0(self) -> float:
        """Base wall lubrication coefficient."""
        return self._Cw0

    @property
    def Dp(self) -> float:
        """Particle/bubble diameter."""
        return self._Dp

    @property
    def CwMax(self) -> float:
        """Maximum coefficient cap."""
        return self._CwMax

    @property
    def alpha_name(self) -> str:
        """Dispersed-phase volume fraction field name."""
        return self._alpha_name

    @property
    def rho_name(self) -> str:
        """Dispersed-phase density field name."""
        return self._rho_name

    def effective_coefficient(self, wall_distance: torch.Tensor) -> torch.Tensor:
        """Compute the distance-dependent Antal coefficient.

        C_w_eff = max(Cw0 * Dp / y_w, CwMax)

        Parameters
        ----------
        wall_distance : torch.Tensor
            Distance from wall for each boundary face.

        Returns
        -------
        torch.Tensor
            Effective wall lubrication coefficient per face.
        """
        safe_dist = wall_distance.clamp(min=1e-10)
        Cw_raw = self._Cw0 * self._Dp / safe_dist
        return Cw_raw.clamp(max=self._CwMax)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply Antal wall lubrication velocity correction.

        The wall face velocity is set to a zero-gradient condition
        (owner cell value).  The actual force is handled by
        ``matrix_contributions``.

        Parameters
        ----------
        field : torch.Tensor
            Velocity field ``(n_cells, 3)`` or ``(n_cells,)``.
        patch_idx : int, optional
            Start index into *field*.
        alpha : float or torch.Tensor, optional
            Dispersed-phase volume fraction.
        rho : float or torch.Tensor, optional
            Dispersed-phase density.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx: patch_idx + n] = owner_values
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
        """Antal wall lubrication force source contribution.

        Adds a distance-dependent source term for wall-adjacent cells::

            source[c] += C_w_eff * rho * alpha * area * v_n / Dp

        where ``C_w_eff = max(Cw0 * Dp / y_w, CwMax)``.

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
            Dispersed-phase density.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        # Wall distance ≈ 1 / delta_coeffs
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        wall_dist = 1.0 / deltas.clamp(min=1e-10)

        # Distance-dependent Antal coefficient
        Cw_eff = self.effective_coefficient(wall_dist)

        # Alpha
        if alpha is None:
            alpha_val = 0.1
        elif isinstance(alpha, torch.Tensor):
            alpha_val = alpha[owners].to(device=device, dtype=dtype)
        else:
            alpha_val = float(alpha)

        # Density
        if rho is None:
            rho_val = 1000.0
        elif isinstance(rho, torch.Tensor):
            rho_val = rho[owners].to(device=device, dtype=dtype)
        else:
            rho_val = float(rho)

        # Force coefficient: Cw_eff * rho * alpha * area / Dp
        force_coeff = Cw_eff * rho_val * alpha_val * areas / self._Dp

        # Add source contribution
        source.scatter_add_(0, owners, force_coeff)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
