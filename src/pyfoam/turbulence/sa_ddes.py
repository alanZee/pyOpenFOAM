"""
Spalart-Allmaras DDES model (Spalart et al. 2006).

Implements Delayed Detached Eddy Simulation (DDES) based on the
Spalart-Allmaras RANS model.  DDES improves upon DES by introducing
a delay function that prevents the model from switching to LES mode
inside the boundary layer, even when the grid is fine enough.

The key modification is the introduction of r_d parameter:

    r_d = ν_t / (κ² d² √(2 S_ij S_ij))

and the delay function:

    f_d = 1 - tanh((C_d r_d)^3)

The modified distance is:

    d̃ = d - f_d max(0, d - C_DES Δ)

where d is the wall distance, Δ is the grid filter width, and C_DES ≈ 0.65.

References
----------
Spalart, P.R., Deck, S., Shur, M.L., Squires, K.D., Strelets, M.Kh.
& Travin, A. (2006). A new version of detached-eddy simulation,
resistant to ambiguous grid densities. Theoretical and Computational
Fluid Dynamics, 20(3), 181–195.
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

__all__ = ["SpalartAllmarasDDESModel", "SpalartAllmarasDDESConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpalartAllmarasDDESConstants(SpalartAllmarasConstants):
    """Constants for the SA DDES model.

    Extends SA constants with DDES-specific parameters.

    Attributes:
        C_DES: DES constant (default: 0.65).
        C_d: Delay function constant (default: 8.0).
    """

    C_DES: float = 0.65
    C_d: float = 8.0


_DEFAULT_CONSTANTS = SpalartAllmarasDDESConstants()


# ---------------------------------------------------------------------------
# SA DDES model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("SpalartAllmarasDDES")
class SpalartAllmarasDDESModel(SpalartAllmarasModel):
    """Spalart-Allmaras DDES model.

    Delayed DES variant that preserves RANS behaviour inside the
    boundary layer regardless of grid refinement.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SpalartAllmarasDDESConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SpalartAllmarasDDESConstants | None = None,
        **kwargs: Any,
    ) -> None:
        sa_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=sa_constants, **kwargs)

        self._C_DES = constants.C_DES if constants else _DEFAULT_CONSTANTS.C_DES
        self._C_d = constants.C_d if constants else _DEFAULT_CONSTANTS.C_d

        # DES filter width (max dimension per cell)
        self._delta_max = self._compute_delta_max()

    def _compute_delta_max(self) -> torch.Tensor:
        """Compute DES filter width: Δ = max(Δx, Δy, Δz).

        Returns:
            ``(n_cells,)`` maximum grid spacing.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
        delta = volumes.pow(1.0 / 3.0)

        return delta.clamp(min=1e-10)

    def _r_d(self) -> torch.Tensor:
        """Compute DDES parameter r_d.

        r_d = ν_t / (κ² d² √(2 S_ij S_ij))

        Returns:
            ``(n_cells,)`` r_d values.
        """
        nu_t = self.nut().clamp(min=0.0)
        y = self._y.clamp(min=1e-10)
        kappa = self._C.kappa

        # Need strain rate magnitude
        if self._grad_U is not None:
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        else:
            S_mag = torch.ones_like(y)

        r_d = nu_t / (kappa**2 * y**2 * S_mag.clamp(min=1e-10))
        return r_d

    def _f_d(self) -> torch.Tensor:
        """Compute DDES delay function f_d.

        f_d = 1 - tanh((C_d r_d)^3)

        Returns:
            ``(n_cells,)`` delay function values in [0, 1].
            f_d ≈ 1: RANS mode (boundary layer)
            f_d ≈ 0: LES mode (detached region)
        """
        r_d = self._r_d()
        return 1.0 - torch.tanh((self._C_d * r_d) ** 3)

    def _modified_distance(self) -> torch.Tensor:
        """Compute DDES-modified wall distance.

        d̃ = d - f_d max(0, d - C_DES Δ)

        Returns:
            ``(n_cells,)`` modified wall distance.
        """
        d = self._y
        f_d = self._f_d()
        delta = self._C_DES * self._delta_max

        # d̃ = d - f_d * max(0, d - delta)
        d_tilde = d - f_d * torch.clamp(d - delta, min=0.0)

        return d_tilde.clamp(min=1e-10)

    def _solve_nuTilde(self) -> None:
        """Solve ν̃ equation with DDES-modified distance.

        Uses the modified wall distance d̃ instead of the geometric
        distance d for the destruction term.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._modified_distance().clamp(min=1e-10)  # Use DDES distance

        # Compute χ = ν̃ / ν
        chi = nuTilde / max(self._nu, 1e-30)

        # fv1 = χ³ / (χ³ + Cv1³)
        fv1 = chi**3 / (chi**3 + C.Cv1**3)

        # fv2 = 1 - χ / (1 + χ fv1)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # ft2 = Ct3 exp(-Ct4 χ²)
        ft2 = C.Ct3 * torch.exp(-C.Ct4 * chi**2)

        # Strain rate magnitude
        S_mag = self._strain_magnitude()

        # Ŝ = S + ν̃ / (κ² d²) fv2
        S_hat = S_mag + nuTilde / (C.kappa**2 * y**2) * fv2

        # Production: Cb1 (1 - ft2) Ŝ ν̃
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
