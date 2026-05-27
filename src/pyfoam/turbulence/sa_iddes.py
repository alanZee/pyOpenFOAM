"""
Spalart-Allmaras IDDES model (Shur et al. 2008).

Implements Improved Delayed Detached Eddy Simulation (IDDES) based on
the Spalart-Allmaras RANS model.  IDDES extends DDES by introducing a
wall-modeled LES (WMLES) capability through an improved blending
function that accounts for both geometric and grid-based length scales.

The key modification over DDES is the IDDES modified distance:

    d̃_IDDES = f_d * d + (1 - f_d) * h_wm

where h_wm is the wall-modeled length scale:

    h_wm = min( max(C_IDDES * d_w, C_IDDES * h_max, h_min), h_max )

and the DDES delay function is replaced by an IDDES variant that
considers the ratio of time scales:

    f_d_IDDES = 1 - tanh((C_IDDES * r_d)^3)

with additional blending through the grid-based length scale to
ensure smooth transition between RANS and WMLES modes.

Additional constants:
    C_IDDES: Blending constant (default: 0.65, same as C_DES).
    C_dt: Damping function constant for temporal blending (default: 20.0).

References
----------
Shur, M.L., Spalart, P.R., Strelets, M.Kh. & Travin, A.K. (2008).
A hybrid RANS-LES approach with delayed-DES and wall-modelled LES
capabilities. International Journal of Heat and Fluid Flow, 29(6),
1638–1649.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel
from .sa_ddes import SpalartAllmarasDDESModel, SpalartAllmarasDDESConstants

__all__ = ["SpalartAllmarasIDDESModel", "SpalartAllmarasIDDESConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpalartAllmarasIDDESConstants(SpalartAllmarasDDESConstants):
    """Constants for the SA IDDES model.

    Extends DDES constants with IDDES-specific parameters.

    Attributes:
        C_IDDES: IDDES blending constant (default: 0.65).
        C_dt: Damping function constant for temporal term (default: 20.0).
    """

    C_IDDES: float = 0.65
    C_dt: float = 20.0


_DEFAULT_CONSTANTS = SpalartAllmarasIDDESConstants()


# ---------------------------------------------------------------------------
# SA IDDES model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("SpalartAllmarasIDDES")
class SpalartAllmarasIDDESModel(SpalartAllmarasDDESModel):
    """Spalart-Allmaras IDDES model.

    Improved DES variant that adds wall-modelled LES capability through
    an improved blending function combining RANS wall distance with
    grid-based length scales.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasIDDESConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasIDDESConstants | None = None,
        **kwargs: Any,
    ) -> None:
        iddes_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=iddes_constants, **kwargs)

        self._C_IDDES = constants.C_IDDES if constants else _DEFAULT_CONSTANTS.C_IDDES
        self._C_dt = constants.C_dt if constants else _DEFAULT_CONSTANTS.C_dt

        # Pre-compute h_min (minimum grid spacing per cell)
        self._h_min = self._compute_h_min()

    def _compute_h_min(self) -> torch.Tensor:
        """Compute minimum grid spacing per cell.

        h_min = min(Δx, Δy, Δz) approximated from cell volume and
        face area information.

        Returns:
            ``(n_cells,)`` minimum grid spacing.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
        # Approximate h_min from cube root of volume (isotropic cells)
        # For anisotropic grids, h_min would be the minimum edge length
        h_min = volumes.pow(1.0 / 3.0)

        return h_min.clamp(min=1e-10)

    def _h_wm(self) -> torch.Tensor:
        """Compute wall-modelled length scale h_wm.

        h_wm = min( max(C_IDDES * d_w, C_IDDES * h_max, h_min), h_max )

        where:
            d_w = wall distance
            h_max = max grid spacing (delta_max)
            h_min = min grid spacing

        Returns:
            ``(n_cells,)`` wall-modelled length scale.
        """
        d_w = self._y
        h_max = self._delta_max
        h_min = self._h_min
        C = self._C_IDDES

        # Inner max: max(C*d_w, C*h_max, h_min)
        inner = torch.max(
            torch.max(C * d_w, C * h_max),
            h_min,
        )

        # Outer min: min(inner, h_max)
        h_wm = torch.min(inner, h_max)

        return h_wm.clamp(min=1e-10)

    def _modified_distance(self) -> torch.Tensor:
        """Compute IDDES-modified wall distance.

        d̃ = f_d * d + (1 - f_d) * h_wm

        This replaces the DDES formulation with an improved blending
        that smoothly transitions between RANS (f_d → 1, d̃ → d) and
        WMLES (f_d → 0, d̃ → h_wm) modes.

        Returns:
            ``(n_cells,)`` modified wall distance.
        """
        d = self._y
        f_d = self._f_d()
        h_wm = self._h_wm()

        # IDDES blending: RANS distance + WMLES length scale
        d_tilde = f_d * d + (1.0 - f_d) * h_wm

        return d_tilde.clamp(min=1e-10)
