"""
Spalart-Allmaras DES model (Spalart et al. 1997).

Implements Detached Eddy Simulation (DES) based on the
Spalart-Allmaras RANS model.  DES replaces the wall distance d
in the SA destruction term with a modified distance:

    d_tilde = min(d, C_DES * delta_max)

where C_DES = 0.65 and delta_max = V^(1/3) is the cube root of
the cell volume (max grid spacing estimate).

In boundary layers (where d < C_DES * delta_max) the model reduces
to pure SA RANS.  In separated / free-shear regions (where
C_DES * delta_max < d) the model acts as a subgrid-scale LES model.

References
----------
Spalart, P.R., Jou, W.H., Strelets, M. & Allmaras, S.R. (1997).
Comments on the feasibility of LES for wings, and on a hybrid
RANS/LES approach. 1st AFOSR Int. Conf. on DNS/LES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel
from .spalart_allmaras import SpalartAllmarasModel, SpalartAllmarasConstants

__all__ = ["SpalartAllmarasDESModel", "SpalartAllmarasDESConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpalartAllmarasDESConstants(SpalartAllmarasConstants):
    """Constants for the SA DES model.

    Extends SA constants with the DES constant.

    Attributes:
        C_DES: DES constant (default: 0.65).
    """

    C_DES: float = 0.65


_DEFAULT_CONSTANTS = SpalartAllmarasDESConstants()


# ---------------------------------------------------------------------------
# SA DES model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("SpalartAllmarasDES")
class SpalartAllmarasDESModel(SpalartAllmarasModel):
    """Spalart-Allmaras DES model.

    Classic DES97 variant that blends SA RANS in the boundary layer
    with LES-like behaviour in detached / free-shear regions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasDESConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasDESConstants | None = None,
        **kwargs: Any,
    ) -> None:
        sa_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=sa_constants, **kwargs)

        self._C_DES = constants.C_DES if constants else _DEFAULT_CONSTANTS.C_DES

        # DES filter width (cube root of cell volume)
        self._delta_max = self._compute_delta_max()

    def _compute_delta_max(self) -> torch.Tensor:
        """Compute DES filter width: Δ = V^(1/3).

        Returns:
            ``(n_cells,)`` cube root of cell volumes.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
        delta = volumes.pow(1.0 / 3.0)

        return delta.clamp(min=1e-10)

    def _modified_distance(self) -> torch.Tensor:
        """Compute DES-modified wall distance.

        d_tilde = min(d, C_DES * delta_max)

        Returns:
            ``(n_cells,)`` modified wall distance.
        """
        d = self._y
        delta = self._C_DES * self._delta_max

        d_tilde = torch.min(d, delta)

        return d_tilde.clamp(min=1e-10)

    def _solve_nuTilde(self) -> None:
        """Solve nu_tilde equation with DES-modified distance.

        Uses the modified wall distance d_tilde instead of the geometric
        distance d for the destruction term.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._modified_distance().clamp(min=1e-10)  # Use DES distance

        # Compute chi = nu_tilde / nu
        chi = nuTilde / max(self._nu, 1e-30)

        # fv1 = chi^3 / (chi^3 + Cv1^3)
        fv1 = chi**3 / (chi**3 + C.Cv1**3)

        # fv2 = 1 - chi / (1 + chi fv1)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # ft2 = Ct3 exp(-Ct4 chi^2)
        ft2 = C.Ct3 * torch.exp(-C.Ct4 * chi**2)

        # Strain rate magnitude
        S_mag = self._strain_magnitude()

        # S_hat = S + nu_tilde / (kappa^2 d^2) fv2
        S_hat = S_mag + nuTilde / (C.kappa**2 * y**2) * fv2

        # Production: Cb1 (1 - ft2) S_hat nu_tilde
        production = C.Cb1 * (1.0 - ft2) * S_hat * nuTilde

        # Destruction
        Cw1 = C.Cb1 / C.kappa**2 + (1.0 + C.Cb2) / C.sigma
        r = (nuTilde / (S_hat * C.kappa**2 * y**2)).clamp(max=10.0)
        g = r + C.Cw2 * (r**6 - r)
        fw = g * ((1.0 + C.Cw3**6) / (g**6 + C.Cw3**6)) ** (1.0 / 6.0)
        destruction = (Cw1 * fw - C.Cb1 * ft2 / C.kappa**2) * (nuTilde / y) ** 2

        # Effective diffusivity
        nu_eff = (self._nu + nuTilde) / C.sigma

        # Build equation
        eqn = fvm.div(self._phi, nuTilde, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, nuTilde, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        source = production - destruction
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        nuTilde_new = eqn.source / diag_safe
        self._nuTilde = nuTilde_new.clamp(min=1e-10)
