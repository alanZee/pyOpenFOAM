"""
k-ω SST SAS model (Menter & Egorov 2010).

Implements the Scale-Adaptive Simulation (SAS) based on the k-ω SST
RANS model.  The SAS concept introduces a source term in the ω equation
that activates when the resolved flow exhibits unsteady structures,
enabling the model to dynamically adapt its length scale to the
resolved turbulence.

SAS source term in ω equation:
    Q_SAS = max(ζ₂ σ k |∇ω|² / ω³ - C_SAS 2k / (σ L_VK²), 0)

where L_VK is the von Karman length scale:
    L_VK = √k / (ω κ |d²U/dy²| / |∇U|)

and is bounded to avoid division by zero.

References
----------
Menter, F.R. & Egorov, Y. (2010). The scale-adaptive simulation method
for unsteady turbulent flow predictions.  Theory, implementation and
validation.  Flow, Turbulence and Combustion, 85, 113–138.

Menter, F.R. & Egorov, Y. (2005). A scale-adaptive simulation model
using two-equation models.  AIAA Paper 2005-1095.
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

__all__ = ["KOmegaSSTSASModel", "KOmegaSSTSASConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaSSTSASConstants(KOmegaSSTConstants):
    """Constants for the k-ω SST SAS model.

    Extends SST constants with SAS-specific parameters.

    Attributes:
        zeta2: SAS constant ζ₂ (default: 3.51).
        sigma: SAS constant σ (default: 2/3).
        C_SAS: SAS constant C_SAS (default: 1.5).
    """

    zeta2: float = 3.51
    sigma: float = 2.0 / 3.0
    C_SAS: float = 1.5


_DEFAULT_CONSTANTS = KOmegaSSTSASConstants()


# ---------------------------------------------------------------------------
# k-ω SST SAS model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmegaSSTSAS")
class KOmegaSSTSASModel(KOmegaSSTModel):
    """k-ω SST SAS model.

    Extends the SST model with a SAS source term in the ω equation that
    activates in unsteady / separated flow regions, enabling the model to
    adapt its length scale to the resolved turbulence.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTSASConstants, optional
        Model constants.  Defaults to SST + SAS defaults.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTSASConstants | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize as SST model
        sas_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=sas_constants, **kwargs)

        self._C_SAS = constants.C_SAS if constants else _DEFAULT_CONSTANTS.C_SAS
        self._zeta2 = constants.zeta2 if constants else _DEFAULT_CONSTANTS.zeta2
        self._sigma_sas = constants.sigma if constants else _DEFAULT_CONSTANTS.sigma

    # ------------------------------------------------------------------
    # Von Karman length scale
    # ------------------------------------------------------------------

    def _von_karman_length_scale(self) -> torch.Tensor:
        """Compute von Karman length scale L_VK.

        L_VK = √k / (ω κ |d²U/dn²| / |∇U|)

        Uses the velocity gradient tensor already stored in _grad_U.
        When _grad_U is None (before first correct()), returns a large
        value that effectively disables the SAS term.

        Returns:
            ``(n_cells,)`` von Karman length scale.
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        C = self._C
        kappa = C.kappa

        # If no velocity gradient computed yet, return large L_VK
        if self._grad_U is None:
            return torch.full_like(k, 1e10)

        # |∇U| = Frobenius norm of the velocity gradient tensor
        grad_U = self._grad_U  # (n_cells, 3, 3)
        grad_U_mag = torch.sqrt(
            (grad_U * grad_U).sum(dim=(1, 2)).clamp(min=1e-30)
        )  # (n_cells,)

        # Approximate |d²U/dy²| from the gradient of the velocity gradient.
        # For a simplified implementation, use |∇(∇U)| via the spatial
        # derivative of grad_U.  However, computing second derivatives
        # requires mesh connectivity (gradient of a tensor field).
        #
        # Simplified approach: use |S|·|∇U| as a proxy for the second
        # derivative magnitude.  This captures the essential SAS behaviour:
        # the term activates when there is strong velocity gradient variation.
        #
        # |d²U/dy²| ≈ |∇|U||  (magnitude of gradient of velocity magnitude)
        U_mag = torch.sqrt(
            (self._U * self._U).sum(dim=1).clamp(min=1e-30)
        )  # (n_cells,)
        grad_U_mag_field = fvc.grad(U_mag, "Gauss linear", mesh=self._mesh)
        d2U_dy2 = torch.sqrt(
            (grad_U_mag_field * grad_U_mag_field).sum(dim=1).clamp(min=1e-30)
        )  # (n_cells,)

        # L_VK = √k / (ω κ |d²U/dy²| / |∇U|)
        #      = √k |∇U| / (ω κ |d²U/dy²|)
        L_VK = (
            torch.sqrt(k) * grad_U_mag.clamp(min=1e-10)
            / (omega * kappa * d2U_dy2.clamp(min=1e-10))
        )

        # Bound L_VK to avoid numerical issues
        return L_VK.clamp(min=1e-10, max=1e10)

    # ------------------------------------------------------------------
    # SAS source term
    # ------------------------------------------------------------------

    def _sas_source(self) -> torch.Tensor:
        """Compute the SAS source term for the ω equation.

        Q_SAS = max(ζ₂ σ k |∇ω|² / ω³ - C_SAS 2k / (σ L_VK²), 0)

        Returns:
            ``(n_cells,)`` SAS source term (non-negative).
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        C = self._C

        # |∇ω|² — need gradient of omega
        # After first correct(), _grad_U is available; compute grad(omega)
        # using the same discretisation
        grad_omega = fvc.grad(self._omega, "Gauss linear", mesh=self._mesh)
        grad_omega_sq = (grad_omega * grad_omega).sum(dim=1)  # (n_cells,)

        # First term: ζ₂ σ k |∇ω|² / ω³
        term1 = (
            self._zeta2
            * self._sigma_sas
            * k
            * grad_omega_sq
            / omega.pow(3)
        )

        # Second term: C_SAS 2k / (σ L_VK²)
        L_VK = self._von_karman_length_scale()
        term2 = self._C_SAS * 2.0 * k / (self._sigma_sas * L_VK.pow(2))

        # Q_SAS = max(term1 - term2, 0)
        return torch.clamp(term1 - term2, min=0.0)

    # ------------------------------------------------------------------
    # Override omega solver
    # ------------------------------------------------------------------

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve ω transport equation with SAS source term.

        Extends the base SST ω equation with:
            + Q_SAS   (SAS source term)
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Blended coefficients
        F1 = self._F1()
        sigma_omega = F1 * C.sigma_omega1 + (1.0 - F1) * C.sigma_omega2
        beta = F1 * C.beta1 + (1.0 - F1) * C.beta2
        gamma = F1 * C.gamma1 + (1.0 - F1) * C.gamma2

        nut = self.nut()
        nu_eff = self._nu + sigma_omega * nut

        # Build equation
        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._omega, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: γ P_k/ν_t - β ω² + Q_SAS
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nut_safe = nut.clamp(min=1e-16)

        source = gamma * P_k / nut_safe - beta * omega_safe**2

        # SAS source term
        if self._grad_U is not None:
            source = source + self._sas_source()

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)
