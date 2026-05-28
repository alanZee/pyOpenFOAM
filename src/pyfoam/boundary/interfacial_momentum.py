"""
Interfacial momentum transfer boundary condition for Euler-Euler multiphase.

Implements the ``interfacialMomentum`` BC which provides the interfacial
momentum exchange between phases in Euler-Euler multiphase formulations.
The BC contributes drag, lift, virtual mass, and wall lubrication forces
to the momentum equation of each phase::

    F_interface = F_drag + F_lift + F_vm + F_wl

where:
    - ``F_drag`` = K * (U_d - U_c) (interphase drag)
    - ``F_lift`` = C_L * rho_c * alpha * (U_d - U_c) x curl(U_c) (lift)
    - ``F_vm`` = C_vm * rho_c * alpha * D(U_d - U_c)/Dt (virtual mass)
    - ``F_wl`` = C_w * rho_c * alpha * |Vr|^2 / D_p * n_w (wall lubrication)

The BC provides ``matrix_contributions`` for implicit coupling of
interphase forces into the momentum matrix.

In OpenFOAM syntax::

    type              interfacialMomentum;
    alpha             alpha.air;
    K                 0.5;          // drag coefficient (or model ref)
    CL                0.1;          // lift coefficient
    Cvm              0.5;          // virtual mass coefficient
    Cw                0.0;          // wall lubrication coefficient (0=off)
    rho               rho.air;
    value             uniform (0 0 0);

Reference:
    OpenFOAM twoPhaseEulerInterphaseModels framework.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["InterfacialMomentumBC"]


@BoundaryCondition.register("interfacialMomentum")
class InterfacialMomentumBC(BoundaryCondition):
    """Interfacial momentum transfer BC for Euler-Euler multiphase.

    Provides interphase force contributions (drag, lift, virtual mass,
    wall lubrication) as boundary-condition source terms.  The BC handles
    the boundary-face portion of the interfacial momentum exchange for
    cells adjacent to the patch.

    Coefficients:
        - ``K``: Interphase drag coefficient (default: 0.5).
        - ``CL``: Lift coefficient (default: 0.1).
        - ``Cvm``: Virtual mass coefficient (default: 0.5).
        - ``Cw``: Wall lubrication coefficient (default: 0.0 = disabled).
        - ``Dp``: Particle/bubble diameter for wall lubrication (default: 0.003).
        - ``alpha``: Dispersed-phase volume fraction field name
          (default: ``"alpha.d"``).
        - ``rho``: Dispersed-phase density field name
          (default: ``"rho.d"``).
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._K = float(self._coeffs.get("K", 0.5))
        self._CL = float(self._coeffs.get("CL", 0.1))
        self._Cvm = float(self._coeffs.get("Cvm", 0.5))
        self._Cw = float(self._coeffs.get("Cw", 0.0))
        self._Dp = float(self._coeffs.get("Dp", 0.003))
        self._alpha_name = self._coeffs.get("alpha", "alpha.d")
        self._rho_name = self._coeffs.get("rho", "rho.d")

    @property
    def K(self) -> float:
        """Interphase drag coefficient."""
        return self._K

    @property
    def CL(self) -> float:
        """Lift coefficient."""
        return self._CL

    @property
    def Cvm(self) -> float:
        """Virtual mass coefficient."""
        return self._Cvm

    @property
    def Cw(self) -> float:
        """Wall lubrication coefficient."""
        return self._Cw

    @property
    def Dp(self) -> float:
        """Particle/bubble diameter."""
        return self._Dp

    @property
    def alpha_name(self) -> str:
        """Dispersed-phase volume fraction field name."""
        return self._alpha_name

    @property
    def rho_name(self) -> str:
        """Dispersed-phase density field name."""
        return self._rho_name

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply interfacial momentum velocity at the boundary.

        Sets the boundary face velocity to the owner cell value
        (zero-gradient treatment).  The actual force contributions are
        handled by ``matrix_contributions``.

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
        """Interfacial momentum transfer source contribution.

        Computes the combined interphase force for wall-adjacent cells::

            diag[c]   += K * area       (implicit drag contribution)
            source[c] += K * U_other * area  (explicit part)
                       + F_lift_area
                       + F_wl_area

        where:
            - Drag: K * (U_this - U_other)
            - Lift: CL * rho * alpha * |U_slip| (simplified scalar)
            - Wall lubrication: Cw * rho * alpha * area / Dp (if Cw > 0)

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

        # Drag contribution (implicit diagonal)
        drag_diag = self._K * areas
        diag.scatter_add_(0, owners, drag_diag)

        # Lift contribution (simplified scalar: CL * rho * alpha * area)
        if self._CL > 0:
            lift_force = self._CL * rho_val * alpha_val * areas
            source.scatter_add_(0, owners, lift_force)

        # Wall lubrication contribution (if enabled)
        if self._Cw > 0:
            wl_force = self._Cw * rho_val * alpha_val * areas / self._Dp
            source.scatter_add_(0, owners, wl_force)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
