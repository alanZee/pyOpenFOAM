"""
HeatExchangerModel — heat exchanger source for thermal simulations.

Applies a temperature-dependent volumetric heat source that models
heat transfer in a heat exchanger region.  The heat transfer rate
follows a Newton's cooling law formulation::

    Q = h_vol * (T_fluid - T_coolant)

where:

- ``h_vol`` — volumetric heat transfer coefficient [W/(m^3 K)]
- ``T_fluid`` — local fluid temperature [K]
- ``T_coolant`` — coolant (secondary fluid) temperature [K]

For stability, the source is linearised semi-implicitly::

    Su = h_vol * (T_coolant)        (explicit, drives toward coolant)
    Sp = -h_vol                      (implicit, removes energy)

This ensures diagonal dominance since ``Sp <= 0``.

Corresponds to OpenFOAM's ``heatExchanger`` fvModel.

Usage::

    from pyfoam.fv.heat_exchanger import HeatExchangerModel

    model = HeatExchangerModel(
        h_vol=5000.0, T_coolant=300.0, cells=he_cells,
    )
    model.apply(energy_matrix, T_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = ["HeatExchangerModel"]


@FvModel.register("heatExchanger")
class HeatExchangerModel(FvModel):
    """Volumetric heat exchanger source model.

    Models heat transfer between the primary fluid and a secondary
    (coolant) fluid at a fixed reference temperature.  The heat
    exchange rate is proportional to the temperature difference and
    a volumetric heat transfer coefficient.

    The source is linearised in semi-implicit form for stability::

        Q_total = h_vol * (T_coolant - T_fluid)
                = h_vol * T_coolant  -  h_vol * T_fluid
                = Su                  +  Sp * T

        Su = h_vol * T_coolant     (explicit)
        Sp = -h_vol                (implicit, negative => stable)

    This formulation drives the fluid temperature toward the coolant
    temperature, with the implicit term providing diagonal dominance.

    Parameters
    ----------
    h_vol : float
        Volumetric heat transfer coefficient [W/(m^3 K)].
        Positive values model cooling (fluid hotter than coolant);
        negative values model heating.
    T_coolant : float
        Coolant (secondary fluid) reference temperature [K].
        Default ``300.0``.
    cells : list[int] | torch.Tensor | None
        Cell indices in the heat exchanger region.
        ``None`` means all cells.

    Examples::

        # Air-cooled heat exchanger
        model = HeatExchangerModel(h_vol=5000.0, T_coolant=300.0)
        model.apply(energy_matrix, T_field)

        # Localised heat exchanger region
        model = HeatExchangerModel(
            h_vol=1e4, T_coolant=293.15, cells=[10, 11, 12, 13],
        )
    """

    def __init__(
        self,
        *,
        h_vol: float = 0.0,
        T_coolant: float = 300.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            h_vol=h_vol, T_coolant=T_coolant, cells=cells, **kwargs,
        )
        self._h_vol = h_vol
        self._T_coolant = T_coolant
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def h_vol(self) -> float:
        """Volumetric heat transfer coefficient [W/(m^3 K)]."""
        return self._h_vol

    @property
    def T_coolant(self) -> float:
        """Coolant reference temperature [K]."""
        return self._T_coolant

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Apply the heat exchanger source to the energy matrix.

        Adds the semi-implicit heat exchange terms::

            Su = h_vol * T_coolant   (per cell)
            Sp = -h_vol              (per cell)

        The source drives the fluid temperature toward the coolant
        temperature, with the implicit ``Sp`` term ensuring diagonal
        dominance for numerical stability.

        Args:
            matrix: The energy :class:`FvMatrix` to modify.
            field: Current temperature field ``(n_cells,)``.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # Semi-implicit split
        su_val = self._h_vol * self._T_coolant   # explicit source
        sp_val = -self._h_vol                      # implicit coefficient

        su = torch.zeros(n, device=device, dtype=dtype)
        sp = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
            sp.scatter_(0, idx, sp_val)
        else:
            su[:] = su_val
            sp[:] = sp_val

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"HeatExchangerModel(h_vol={self._h_vol}, "
            f"T_coolant={self._T_coolant})"
        )
