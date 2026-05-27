"""
k-omega SST Langtry-Menter transition model.

Implements the gamma-Re_theta transition model (Langtry & Menter 2009)
on top of the k-omega SST RANS model.  Adds two additional transport
equations for intermittency gamma and transition momentum thickness
Reynolds number Re_thetat.

The intermittency gamma modifies the production and destruction terms
in the k equation via an effective intermittency gamma_eff, which
triggers transition from laminar to turbulent flow.

Key correlation functions:
- F_length: Controls the length of the transition region
- F_onset: Triggers the transition onset
- Re_theta_c: Critical momentum thickness Reynolds number
- gamma_sep: Separation-induced intermittency

References
----------
Langtry, R.B. & Menter, F.R. (2009). Correlation-based transition
modelling for unstructured parallelized CFD solvers. AIAA Journal,
47(12), 2894-2906.

Langtry, R.B. (2006). A correlation-based transition model using local
variables for unstructured parallelized CFD solvers. Ph.D. thesis,
University of Stuttgart.
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

__all__ = ["KOmegaSSTLMModel", "KOmegaSSTLMConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaSSTLMConstants(KOmegaSSTConstants):
    """Constants for the k-omega SST Langtry-Menter transition model.

    Extends SST constants with transition-specific parameters.

    Attributes:
        ca1: Transition onset correlation constant 1.
        ca2: Transition onset correlation constant 2.
        ce1: Separation transition constant 1.
        ce2: Separation transition constant 2.
        cThetat: Momentum thickness transport equation constant.
        sigmaf: Diffusion constant for gamma equation.
        sigmat: Diffusion constant for Re_thetat equation.
    """

    ca1: float = 2.0
    ca2: float = 0.06
    ce1: float = 1.0
    ce2: float = 50.0
    cThetat: float = 0.03
    sigmaf: float = 1.0
    sigmat: float = 2.0


_DEFAULT_CONSTANTS = KOmegaSSTLMConstants()


# ---------------------------------------------------------------------------
# k-omega SST Langtry-Menter transition model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmegaSSTLM")
class KOmegaSSTLMModel(KOmegaSSTModel):
    """k-omega SST Langtry-Menter transition model.

    Extends the k-omega SST model with two additional transport equations
    for intermittency gamma and transition momentum thickness Reynolds
    number Re_thetat.  The effective intermittency modifies production
    and destruction in the k equation to model laminar-turbulent transition.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTLMConstants, optional
        Model constants.  Defaults to Langtry-Menter (2009) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTLMConstants | None = None,
        **kwargs: Any,
    ) -> None:
        # Pass SST-compatible constants to parent
        lm_constants = constants or _DEFAULT_CONSTANTS
        super().__init__(mesh, U, phi, constants=lm_constants, **kwargs)

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Transition-specific fields
        # Intermittency: gamma = 1 means fully turbulent
        self._gamma = torch.ones(n_cells, device=device, dtype=dtype)
        # Transition momentum thickness Reynolds number
        self._Re_thetat = torch.zeros(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Properties for transition fields
    # ------------------------------------------------------------------

    @property
    def gamma_field(self) -> torch.Tensor:
        """Intermittency field ``(n_cells,)``."""
        return self._gamma

    @gamma_field.setter
    def gamma_field(self, value: torch.Tensor) -> None:
        self._gamma = value.to(device=self._device, dtype=self._dtype)

    @property
    def Re_thetat_field(self) -> torch.Tensor:
        """Transition momentum thickness Reynolds number ``(n_cells,)``."""
        return self._Re_thetat

    @Re_thetat_field.setter
    def Re_thetat_field(self, value: torch.Tensor) -> None:
        self._Re_thetat = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Transition correlation functions
    # ------------------------------------------------------------------

    def _Re_theta_c(self) -> torch.Tensor:
        """Critical momentum thickness Reynolds number.

        Re_theta_c = f(Re_thetat) — correlates the transition onset
        location based on the local value of Re_thetat.

        Uses a simplified correlation:
            Re_theta_c = Re_thetat  (when Re_thetat > 0)
            Re_theta_c = 0          (otherwise)

        Returns:
            ``(n_cells,)`` critical Re_theta.
        """
        return self._Re_thetat.clamp(min=0.0)

    def _Re_theta_t(self) -> torch.Tensor:
        """Compute local momentum thickness Reynolds number from flow field.

        Re_theta_t = max(Re_theta_t_local, Re_thetat_transport)

        Simplified estimation based on strain rate and freestream conditions.

        Returns:
            ``(n_cells,)`` momentum thickness Reynolds number.
        """
        # Local estimate from strain rate
        if self._grad_U is not None:
            S_mag = self._strain_magnitude()
        else:
            S_mag = torch.zeros(
                self._mesh.n_cells, device=self._device, dtype=self._dtype
            )

        # Simplified: Re_theta_t_local based on flat-plate correlation
        # T_u (turbulence intensity) assumed small for this estimate
        # Re_theta_t_local ~ 1173 (zero pressure gradient, low Tu)
        Re_theta_t_local = torch.full_like(S_mag, 1173.0)

        return torch.max(Re_theta_t_local, self._Re_thetat)

    def _F_onset(self) -> torch.Tensor:
        """Transition onset function.

        F_onset controls when the intermittency starts growing.
        It triggers when the momentum thickness Reynolds number
        exceeds the critical value.

        F_onset = max(0, Re_omega - c_omega * Re_theta_c)
                  / ((1 - gamma) * nu_t * omega * y^2)  (simplified)

        Simplified version:
            F_onset = min(max(Re_omega / (Re_theta_c * 2.193) - 1, 0), 1) * 2

        Returns:
            ``(n_cells,)`` onset function values.
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)

        # Re_omega = rho * omega * y^2 / mu (dimensionless wall distance in omega)
        Re_omega = omega * y**2 / self._nu

        # Critical Re_theta from transport equation
        Re_theta_c = self._Re_theta_c().clamp(min=1.0)

        # F_onset1: ratio-based trigger
        F_onset1 = (Re_omega / (2.193 * Re_theta_c) - 1.0).clamp(min=0.0)
        F_onset1 = torch.min(F_onset1, torch.tensor(2.0, device=self._device))

        return F_onset1

    def _F_length(self) -> torch.Tensor:
        """Transition length function.

        F_length controls the length of the transition region.
        Based on Re_thetat correlation.

        Simplified: F_length = 33.8 * 0.5 (constant approximation
        for moderate Re_thetat)

        Returns:
            ``(n_cells,)`` transition length values.
        """
        # Simplified correlation: F_length varies with Re_thetat
        # For Re_thetat ~ 200-900, F_length ~ 30-40
        # Use a piecewise-linear approximation
        Re_thetat = self._Re_thetat.clamp(min=0.0, max=2000.0)
        F_length = 33.8 * (Re_thetat / (Re_thetat + 100.0)).clamp(min=0.5)
        return F_length

    def _gamma_sep(self) -> torch.Tensor:
        """Separation-induced intermittency.

        gamma_sep activates transition in separated flow regions.

        gamma_sep = min(ce2 * max(0, Re_omega / (3.235 * Re_theta_c) - 1)
                        * F_onset, ce2) * F_onset

        Returns:
            ``(n_cells,)`` separation intermittency.
        """
        C = self._C
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-10)
        y = self._y.clamp(min=1e-10)

        Re_omega = omega * y**2 / self._nu
        Re_theta_c = self._Re_theta_c().clamp(min=1.0)

        # Separation criterion
        sep_ratio = (Re_omega / (3.235 * Re_theta_c) - 1.0).clamp(min=0.0)
        gamma_sep = C.ce2 * sep_ratio

        # Clip to physically meaningful range
        return gamma_sep.clamp(min=0.0, max=C.ce2)

    def _gamma_eff(self) -> torch.Tensor:
        """Effective intermittency for k equation modification.

        gamma_eff = max(gamma, gamma_sep)

        This is the intermittency that modifies production and
        destruction in the k transport equation.

        Returns:
            ``(n_cells,)`` effective intermittency.
        """
        gamma_sep = self._gamma_sep()
        return torch.max(self._gamma, gamma_sep)

    # ------------------------------------------------------------------
    # Override: solve gamma transport equation
    # ------------------------------------------------------------------

    def _solve_gamma(self) -> None:
        """Solve the intermittency transport equation.

        D(gamma)/Dt = P_gamma - E_gamma + diffusion

        Production: P_gamma = ca1 * F_length * sqrt(gamma * F_onset)
                              * (1 - gamma) * S * rho * nu_t
        Destruction: E_gamma = ca2 * rho * F_turb * gamma * Omega

        Simplified here using implicit source terms.
        """
        mesh = self._mesh
        C = self._C
        n_cells = mesh.n_cells

        gamma_safe = self._gamma.clamp(min=1e-16, max=1.0 - 1e-10)

        F_onset = self._F_onset()
        F_length = self._F_length()

        if self._grad_U is not None:
            S_mag = self._strain_magnitude()
        else:
            S_mag = torch.zeros(n_cells, device=self._device, dtype=self._dtype)

        omega_safe = self._omega.clamp(min=1e-16)

        # Production of intermittency
        P_gamma = (
            C.ca1
            * F_length
            * torch.sqrt(gamma_safe * F_onset.clamp(min=0.0))
            * (1.0 - gamma_safe)
            * S_mag
        )

        # Destruction of intermittency
        # F_turb: turbulent intermittency function (simplified)
        # F_turb = exp(-(Re_omega / 200)^4)
        y = self._y.clamp(min=1e-10)
        Re_omega = omega_safe * y**2 / self._nu
        F_turb = torch.exp(-(Re_omega / 200.0) ** 4)

        E_gamma = C.ca2 * F_turb * gamma_safe * omega_safe

        # Source = production - destruction
        source = P_gamma - E_gamma

        # Implicit diffusion term (simplified)
        nut = self.nut()
        diff_coeff = self._nu + C.sigmaf * nut

        # Build advection-diffusion equation
        eqn = fvm.div(self._phi, self._gamma, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            diff_coeff, self._gamma, "Gauss linear corrected", mesh=mesh
        )
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        gamma_new = eqn.source / diag_safe

        # Clamp to physical range [0, 1] — intermittency cannot be negative
        # and is 1 for fully turbulent
        self._gamma = gamma_new.clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Override: solve Re_thetat transport equation
    # ------------------------------------------------------------------

    def _solve_Re_thetat(self) -> None:
        """Solve the momentum thickness Reynolds number transport equation.

        D(Re_thetat)/Dt = cThetat * (Re_thetat_local - Re_thetat) * rho * U^2
                          + diffusion

        This equation is primarily a convection-diffusion equation
        that transports the onset criterion from the freestream into
        the boundary layer.
        """
        mesh = self._mesh
        C = self._C

        Re_thetat_local = self._Re_theta_t()

        # Convection by mean flow + diffusion
        nut = self.nut()
        diff_coeff = self._nu + C.sigmat * nut

        eqn = fvm.div(self._phi, self._Re_thetat, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            diff_coeff, self._Re_thetat, "Gauss linear corrected", mesh=mesh
        )
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: drive Re_thetat towards Re_thetat_local
        # Source = cThetat * (Re_thetat_local - Re_thetat) * |U|^2
        U_mag_sq = (self._U * self._U).sum(dim=1)
        source = C.cThetat * (Re_thetat_local - self._Re_thetat) * U_mag_sq

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        Re_new = eqn.source / diag_safe
        self._Re_thetat = Re_new.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override: k equation with gamma_eff modification
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve k equation with intermittency modification.

        The production and destruction terms are multiplied by gamma_eff
        to model the laminar-turbulent transition:

            Dk/Dt = P_k' - D_k'
            P_k' = gamma_eff * P_k
            D_k' = gamma_eff * beta* * omega * k

        Parameters
        ----------
        P_k : torch.Tensor
            Uncorrected production rate ``(n_cells,)``.
        """
        mesh = self._mesh
        C = self._C

        gamma_eff = self._gamma_eff()

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

        # Modified source: gamma_eff * (P_k - beta* omega k)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        source = gamma_eff * (P_k - C.beta_star * omega_safe * k_safe)
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Override: correct() to include transition equations
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the k-omega SST transition model.

        Solves transport equations in order:
        1. Velocity gradient tensor (from base class)
        2. Re_thetat transport equation
        3. gamma transport equation
        4. k equation (with gamma_eff modification)
        5. omega equation (from base class)
        """
        # Compute velocity gradient tensor
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        # Production rate (for omega equation)
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Solve transition equations first
        self._solve_Re_thetat()
        self._solve_gamma()

        # Solve turbulence equations with intermittency modification
        self._solve_k(P_k)

        # Omega equation — use base class implementation
        # The omega equation is not directly modified by gamma in the
        # Langtry-Menter formulation
        super()._solve_omega(P_k)
