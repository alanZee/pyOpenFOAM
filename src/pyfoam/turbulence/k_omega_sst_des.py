"""
k-ω SST DES model (Strelets 2001).

Implements Detached Eddy Simulation (DES) based on the k-ω SST RANS
model.  The model uses RANS in the boundary layer and switches to
LES-like behaviour in separated/free-shear regions by limiting the
turbulent length scale.

DES length scale:
    l_des = min(l_rans, C_DES Δ)

where:
    l_rans = √k / (β* ω) for SST (or √k / (β* ε) for k-ε region)
    C_DES = 0.65 (default)
    Δ = max(Δx, Δy, Δz) or V^{1/3}

The model blends between k-ω (near wall) and k-ε (freestream) using
the SST blending functions F1 and F2, with the DES limiter applied
separately in each region.

References
----------
Strelets, M. (2001). Detached eddy simulation of massively separated
flows. AIAA Paper 2001-0879.

Menter, F.R. & Kuntz, M. (2004). Adaptation of eddy-viscosity
turbulence models to unsteady separated flow behind vehicles. In:
McCallen, R., Browand, F. & Ross, J. (eds) The Aerodynamics of
Heavy Vehicles: Trucks, Buses, and Trains. Springer, 339–352.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel
from .k_omega_sst import KOmegaSSTModel, KOmegaSSTConstants

__all__ = ["KOmegaSSTDESModel", "KOmegaSSTDESConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaSSTDESConstants(KOmegaSSTConstants):
    """Constants for the k-ω SST DES model.

    Extends SST constants with DES-specific parameters.

    Attributes:
        C_DES: DES constant (default: 0.65).
    """

    C_DES: float = 0.65


_DEFAULT_CONSTANTS = KOmegaSSTDESConstants()


# ---------------------------------------------------------------------------
# k-ω SST DES model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmegaSSTDES")
class KOmegaSSTDESModel(KOmegaSSTModel):
    """k-ω SST DES model.

    Uses SST in the boundary layer and switches to LES-like behaviour
    in detached/free-shear regions by limiting the turbulent length scale.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTDESConstants, optional
        Model constants.  Defaults to SST + C_DES=0.65.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTDESConstants | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize as SST model
        sst_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=sst_constants, **kwargs)

        self._C_DES = constants.C_DES if constants else _DEFAULT_CONSTANTS.C_DES

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

        cell_centres = mesh.cell_centres  # (n_cells, 3)

        # Approximate max dimension from cell volume
        # Δ ≈ V^{1/3} * 1.0 (cube root approximation)
        # More accurate: use actual cell dimensions
        volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
        delta = volumes.pow(1.0 / 3.0)

        return delta.clamp(min=1e-10)

    def _F1(self) -> torch.Tensor:
        """Override F1 with DES limiter.

        In DES, F1 is modified to activate the limiter in the
        freestream region while preserving RANS near walls.
        """
        # Get base SST F1
        F1_sst = super()._F1()

        # DES limiter: switch to LES when l_sst > C_DES * delta_max
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        beta_star = self._C.beta_star

        l_rans = torch.sqrt(k) / (beta_star * omega)
        l_des = self._C_DES * self._delta_max

        # F1_DES = F1_SST when l_rans < l_des (RANS region)
        # F1_DES → 0 when l_rans > l_des (LES region)
        ratio = l_rans / l_des.clamp(min=1e-10)
        F1_des = F1_sst * torch.exp(-torch.clamp(ratio - 1.0, min=0.0) ** 2)

        return F1_des

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve k equation with DES length scale limiter.

        The destruction term uses the DES-limited length scale:
            ε = β* k^{3/2} / l_des  (instead of β* ω k)
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Blended diffusivity
        F1 = self._F1()
        sigma_k = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        nut = self.nut()
        nu_eff = self._nu + sigma_k * nut

        # Build equation
        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # DES destruction: β* k^{3/2} / l_des
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        l_rans = torch.sqrt(k_safe) / (C.beta_star * omega_safe)
        l_des = self._C_DES * self._delta_max
        l_eff = torch.min(l_rans, l_des).clamp(min=1e-10)

        destruction = C.beta_star * k_safe.pow(1.5) / l_eff

        source = P_k - destruction
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)
