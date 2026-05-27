"""
ActuationDiskModel — actuation disk source for wind turbine modelling.

Applies a thrust force as a volumetric source term in cells belonging
to the actuation disk region.  This is the standard "thin disk" or
"actuator disk" approach used in wind energy CFD, where the rotor is
replaced by a momentum sink distributed over the disk volume.

The thrust force per unit volume is::

    F_T = 0.5 * Ct * rho * A_disk * U_inf^2 / V_disk

where:

- ``Ct`` — thrust coefficient [-]
- ``rho`` — fluid density [kg/m^3]
- ``A_disk`` — disk frontal area [m^2]
- ``U_inf`` — freestream velocity magnitude [m/s]
- ``V_disk`` — total volume of the disk region [m^3]

The force is distributed uniformly across the specified cells.
For simplicity, the current implementation uses a constant Ct and
user-supplied freestream velocity; coupling with local velocity
fields is delegated to the caller or a higher-level interface.

Corresponds to OpenFOAM's ``actuationDisk`` fvModel.

Usage::

    from pyfoam.fv.actuation_disk import ActuationDiskModel

    model = ActuationDiskModel(
        Ct=0.8, rho=1.225, A_disk=50.0,
        U_inf=10.0, V_disk=10.0, cells=disk_cells,
    )
    model.apply(momentum_matrix, U_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = ["ActuationDiskModel"]


@FvModel.register("actuationDisk")
class ActuationDiskModel(FvModel):
    """Actuation disk momentum source for wind turbine simulation.

    Models the rotor as a thin disk that extracts momentum from the
    flow.  The total thrust force is distributed uniformly across the
    specified cells as a volumetric force [N/m^3].

    The thrust is linearised semi-implicitly for stability::

        F_T = 0.5 * Ct * rho * A * U_inf^2 / V  (per cell)

    This force is applied purely as an explicit source (Su) since the
    freestream velocity is treated as a fixed parameter.  An optional
    implicit coefficient ``Sp`` can be supplied to improve convergence
    when the disk velocity is coupled with the local flow field.

    Parameters
    ----------
    Ct : float
        Thrust coefficient [-].  Typically 0.6-0.9 for wind turbines.
    rho : float
        Fluid density [kg/m^3].  Default ``1.225`` (air at sea level).
    A_disk : float
        Rotor frontal area [m^2].
    U_inf : float
        Freestream (reference) velocity magnitude [m/s].
    V_disk : float
        Total volume of the actuation disk cells [m^3].
        Used to convert total thrust to volumetric force.
    cells : list[int] | torch.Tensor | None
        Cell indices in the disk region.  ``None`` means all cells
        (unusual for an actuation disk, but supported).
    Sp : float
        Optional implicit linearisation coefficient [1/s].
        Default ``0.0`` (purely explicit).  Setting ``Sp < 0``
        improves diagonal dominance when the disk force is coupled
        to the local velocity.

    Examples::

        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0,
            U_inf=10.0, V_disk=10.0, cells=[42, 43, 44],
        )
        model.apply(matrix, U_field)
    """

    def __init__(
        self,
        *,
        Ct: float = 0.8,
        rho: float = 1.225,
        A_disk: float = 1.0,
        U_inf: float = 10.0,
        V_disk: float = 1.0,
        cells: list[int] | torch.Tensor | None = None,
        Sp: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            Ct=Ct, rho=rho, A_disk=A_disk, U_inf=U_inf,
            V_disk=V_disk, cells=cells, Sp=Sp, **kwargs,
        )
        if Ct < 0.0:
            raise ValueError(f"Ct must be >= 0, got {Ct}")
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if A_disk < 0.0:
            raise ValueError(f"A_disk must be >= 0, got {A_disk}")
        if V_disk <= 0.0:
            raise ValueError(f"V_disk must be > 0, got {V_disk}")

        self._Ct = Ct
        self._rho = rho
        self._A_disk = A_disk
        self._U_inf = U_inf
        self._V_disk = V_disk
        self._Sp = Sp
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def Ct(self) -> float:
        """Thrust coefficient [-]."""
        return self._Ct

    @property
    def rho(self) -> float:
        """Fluid density [kg/m^3]."""
        return self._rho

    @property
    def A_disk(self) -> float:
        """Rotor frontal area [m^2]."""
        return self._A_disk

    @property
    def U_inf(self) -> float:
        """Freestream velocity [m/s]."""
        return self._U_inf

    @property
    def V_disk(self) -> float:
        """Total disk cell volume [m^3]."""
        return self._V_disk

    @property
    def thrust_force(self) -> float:
        """Total thrust force [N] = 0.5 * Ct * rho * A * U_inf^2."""
        return 0.5 * self._Ct * self._rho * self._A_disk * self._U_inf ** 2

    @property
    def volumetric_force(self) -> float:
        """Volumetric force [N/m^3] = thrust / V_disk."""
        return self.thrust_force / self._V_disk

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Apply the actuation disk thrust force to the momentum matrix.

        The total thrust is computed from the disk parameters and
        distributed uniformly across the specified cells.  The force
        acts as a momentum sink (negative Su in the flow direction).

        The ``field`` parameter (velocity magnitude or component) is
        used only when ``Sp != 0`` for implicit linearisation.

        Args:
            matrix: The momentum :class:`FvMatrix` to modify.
            field: Current velocity field ``(n_cells,)`` or
                ``(n_cells, 3)`` — only the norm is used for Sp.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # Volumetric thrust per cell [N/m^3]
        F_vol = self.volumetric_force

        # Number of disk cells
        if self._cells is not None:
            n_disk = len(self._cells)
        else:
            n_disk = n

        # Per-cell force: total thrust / n_disk_cells
        # Negative sign: thrust opposes the flow (momentum sink)
        f_cell = -F_vol / n_disk if n_disk > 0 else 0.0

        su = torch.zeros(n, device=device, dtype=dtype)
        sp = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, f_cell)
            if self._Sp != 0.0:
                sp.scatter_(0, idx, self._Sp)
        else:
            su[:] = f_cell
            if self._Sp != 0.0:
                sp[:] = self._Sp

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"ActuationDiskModel(Ct={self._Ct}, rho={self._rho}, "
            f"A_disk={self._A_disk}, U_inf={self._U_inf}, "
            f"V_disk={self._V_disk})"
        )
