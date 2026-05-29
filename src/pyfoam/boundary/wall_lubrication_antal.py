"""
Enhanced Antal wall lubrication force BC — version 2.

Implements an enhanced Antal et al. (1991) wall lubrication force model
with distance-dependent coefficient, adjustable distance exponent, and
optional parabolic damping near the interface.  Builds upon the basic
Antal model in ``wall_lubrication_2.py`` by adding:

1. **Distance exponent**: The coefficient scales as ``Dp^exponent / y_w``
   instead of the fixed linear relation, allowing calibration for different
   bubble sizes and flow regimes.
2. **Parabolic interface damping**: An optional alpha-dependent damping
   factor that reduces the wall lubrication force near the interface
   (``alpha ≈ 0.5``), preventing overcorrection in VOF/Euler-Euler
   simulations::

       f_d(alpha) = 1 - damping * 4 * alpha * (1 - alpha)

3. **CwMax cap**: Prevents singularities at very small wall distances.

The effective wall lubrication coefficient is::

    C_w_eff = min(Cw0 * Dp^exponent / y_w, CwMax) * f_d(alpha)

In OpenFOAM syntax::

    type              antalWallLubrication2;
    alpha             alpha.water;
    Cw                0.05;        // base wall lubrication coefficient
    Dp                0.003;       // bubble/particle diameter (m)
    CwMax             10.0;        // maximum coefficient cap
    exponent          1.0;         // distance exponent (1.0 = standard Antal)
    damping           0.0;         // interface damping strength [0,1]
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

__all__ = ["AntalWallLubrication2BC"]


@BoundaryCondition.register("antalWallLubrication2")
class AntalWallLubrication2BC(BoundaryCondition):
    """Enhanced Antal wall lubrication force BC (version 2).

    Extends the standard Antal model with a configurable distance
    exponent and optional interface damping for multiphase simulations.

    The effective coefficient is::

        C_w_eff = min(Cw0 * Dp^exponent / y_w, CwMax) * f_d(alpha)

    where ``f_d(alpha) = 1 - damping * 4 * alpha * (1 - alpha)``.

    Coefficients:
        - ``Cw``: Base wall lubrication coefficient (default: 0.05).
        - ``Dp``: Particle/bubble diameter in metres (default: 0.003).
        - ``CwMax``: Maximum coefficient cap (default: 10.0).
        - ``exponent``: Distance exponent (default: 1.0).
        - ``damping``: Interface damping strength in [0, 1] (default: 0.0).
        - ``alpha``: Dispersed-phase volume fraction field name
          (default: ``"alpha.d"``).
        - ``rho``: Dispersed-phase density field name
          (default: ``"rho.d"``).
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Cw0 = float(self._coeffs.get("Cw", 0.05))
        self._Dp = float(self._coeffs.get("Dp", 0.003))
        self._CwMax = float(self._coeffs.get("CwMax", 10.0))
        self._exponent = float(self._coeffs.get("exponent", 1.0))
        self._damping = float(self._coeffs.get("damping", 0.0))
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
    def exponent(self) -> float:
        """Distance exponent."""
        return self._exponent

    @property
    def damping(self) -> float:
        """Interface damping strength."""
        return self._damping

    @property
    def alpha_name(self) -> str:
        """Dispersed-phase volume fraction field name."""
        return self._alpha_name

    @property
    def rho_name(self) -> str:
        """Dispersed-phase density field name."""
        return self._rho_name

    def interface_damping_factor(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the alpha-dependent interface damping factor.

        Uses the parabolic interface indicator::

            f_d = 1 - damping * 4 * alpha * (1 - alpha)

        which smoothly reduces the force near the interface (alpha = 0.5).

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)`` in [0, 1].
            1 = no damping, 0 = full suppression.
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        indicator = 4.0 * alpha_c * (1.0 - alpha_c)
        return (1.0 - self._damping * indicator).clamp(min=0.0, max=1.0)

    def effective_coefficient(
        self,
        wall_distance: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the enhanced Antal wall lubrication coefficient.

        C_w_eff = min(Cw0 * Dp^exponent / y_w, CwMax) * f_d(alpha)

        Parameters
        ----------
        wall_distance : torch.Tensor
            Distance from wall for each boundary face.
        alpha : torch.Tensor, optional
            Volume fraction per face for interface damping.

        Returns
        -------
        torch.Tensor
            Effective wall lubrication coefficient per face.
        """
        safe_dist = wall_distance.clamp(min=1e-10)
        Dp_exp = self._Dp ** self._exponent
        Cw_raw = self._Cw0 * Dp_exp / safe_dist
        Cw_eff = Cw_raw.clamp(max=self._CwMax)

        if self._damping > 0.0 and alpha is not None:
            f_d = self.interface_damping_factor(alpha)
            Cw_eff = Cw_eff * f_d

        return Cw_eff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply enhanced Antal wall lubrication velocity correction.

        Sets wall face velocity to owner cell value (zero-gradient).
        The actual force is handled by ``matrix_contributions``.

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
        """Enhanced Antal wall lubrication force source contribution.

        Adds a distance-dependent source term with optional interface
        damping for wall-adjacent cells::

            source[c] += C_w_eff * rho * alpha * area / Dp

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
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        wall_dist = 1.0 / deltas.clamp(min=1e-10)

        # Alpha for interface damping
        alpha_tensor = None
        if alpha is None:
            alpha_val = 0.1
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha[owners].to(device=device, dtype=dtype)
            alpha_val = alpha_tensor
        else:
            alpha_val = float(alpha)

        # Enhanced coefficient: distance-dependent with exponent + interface damping
        Cw_eff = self.effective_coefficient(wall_dist, alpha_tensor)

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
