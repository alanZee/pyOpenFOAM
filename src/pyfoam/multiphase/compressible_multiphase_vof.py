"""
N-phase Volume of Fluid (VOF) for compressible multiphase flows.

Extends the incompressible N-phase VOF to compressible flows where
density varies with pressure and temperature.  Each phase has its
own equation of state relating density to thermodynamic variables.

Governing equations for each phase i (1..N-1):

    d(alpha_i)/dt + div(U * alpha_i) + div(U_r * alpha_i * (1 - alpha_i)) = 0

Mixture density:

    rho_m = sum(alpha_i * rho_i(p, T))

Mixture pressure (from Dalton's law for ideal-gas mixtures):

    1/p = sum(alpha_i / p_i(rho_i, T))

Mixture viscosity:

    mu_m = sum(alpha_i * mu_i)

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

    model = CompressibleMultiphaseVoF(
        phase_names=["gas", "liquid"],
        eos_type=["perfectGas", "incompressible"],
        rho_ref=[1.225, 998.0],
        mu=[1.8e-5, 1.002e-3],
        R=[287.0, None],
        gamma=[1.4, None],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["CompressibleMultiphaseVoF"]

logger = logging.getLogger(__name__)


class CompressibleMultiphaseVoF:
    """N-phase VOF model for compressible multiphase flows.

    Each phase has a simple equation of state:
    - ``"perfectGas"``: rho = p / (R * T), needs R and gamma
    - ``"incompressible"``: rho = rho_ref (constant), needs rho_ref

    Mixture properties are computed as volume-fraction-weighted sums.

    Parameters
    ----------
    phase_names : sequence of str
        Names of the N phases (N >= 2).
    eos_type : sequence of str
        Equation of state type for each phase: ``"perfectGas"`` or
        ``"incompressible"``.
    rho_ref : sequence of float
        Reference density for each phase. For perfect gas, this is the
        density at reference conditions (used only for initialisation).
    mu : sequence of float
        Dynamic viscosity for each phase (Pa·s).
    R : sequence of float or None
        Specific gas constant (J/(kg·K)) for perfect-gas phases.
        None for incompressible phases.
    gamma : sequence of float or None
        Ratio of specific heats for perfect-gas phases.
        None for incompressible phases.
    p_ref : float
        Reference pressure (Pa). Default: 101325.
    T_ref : float
        Reference temperature (K). Default: 300.
    C_alpha : float
        Compression coefficient. Default: 1.0.

    Examples::

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1.002e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
    """

    def __init__(
        self,
        phase_names: Sequence[str],
        eos_type: Sequence[str],
        rho_ref: Sequence[float],
        mu: Sequence[float],
        R: Sequence[float | None] | None = None,
        gamma: Sequence[float | None] | None = None,
        p_ref: float = 101325.0,
        T_ref: float = 300.0,
        C_alpha: float = 1.0,
    ) -> None:
        n_phases = len(phase_names)
        if n_phases < 2:
            raise ValueError("Need at least 2 phases")
        if len(eos_type) != n_phases:
            raise ValueError(f"eos_type length != n_phases ({n_phases})")
        if len(rho_ref) != n_phases:
            raise ValueError(f"rho_ref length != n_phases ({n_phases})")
        if len(mu) != n_phases:
            raise ValueError(f"mu length != n_phases ({n_phases})")

        for et in eos_type:
            if et not in ("perfectGas", "incompressible"):
                raise ValueError(f"Unknown EOS type '{et}'")

        if R is None:
            R = [None] * n_phases
        if gamma is None:
            gamma = [None] * n_phases

        # Validate perfect-gas phases have R and gamma
        for i, et in enumerate(eos_type):
            if et == "perfectGas":
                if R[i] is None or R[i] <= 0:
                    raise ValueError(f"Phase {i} (perfectGas) needs R > 0")
                if gamma[i] is None or gamma[i] <= 1.0:
                    raise ValueError(f"Phase {i} (perfectGas) needs gamma > 1")

        self._n_phases = n_phases
        self._phase_names = list(phase_names)
        self._eos_type = list(eos_type)
        self._rho_ref = list(rho_ref)
        self._mu = list(mu)
        self._R = list(R)
        self._gamma = list(gamma)
        self._p_ref = p_ref
        self._T_ref = T_ref
        self._C_alpha = C_alpha

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_phases(self) -> int:
        return self._n_phases

    @property
    def phase_names(self) -> list[str]:
        return self._phase_names.copy()

    @property
    def eos_type(self) -> list[str]:
        return self._eos_type.copy()

    @property
    def C_alpha(self) -> float:
        return self._C_alpha

    # ------------------------------------------------------------------
    # Volume fraction constraint
    # ------------------------------------------------------------------

    def compute_last_alpha(self, alphas: torch.Tensor) -> torch.Tensor:
        """N-th phase volume fraction from summation constraint."""
        alpha_N = 1.0 - alphas.sum(dim=-1)
        return alpha_N.clamp(0.0, 1.0)

    def validate_alphas(self, alphas: torch.Tensor) -> torch.Tensor:
        """Validate, clamp, and renormalise volume fractions."""
        alphas = alphas.clamp(0.0, 1.0)
        total = alphas.sum(dim=-1) + 1e-30
        scale = torch.where(total > 1.0, 1.0 / total, torch.ones_like(total))
        return alphas * scale.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Phase density from EOS
    # ------------------------------------------------------------------

    def phase_density(
        self, phase_idx: int, p: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute density for a single phase.

        - perfectGas: rho = p / (R * T)
        - incompressible: rho = rho_ref

        Parameters
        ----------
        phase_idx : int
            Phase index.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` phase density (kg/m^3).
        """
        et = self._eos_type[phase_idx]
        if et == "perfectGas":
            R = self._R[phase_idx]
            T_safe = T.clamp(min=1.0)
            return p / (R * T_safe)
        else:
            # incompressible
            device = p.device
            dtype = p.dtype
            return torch.full_like(p, self._rho_ref[phase_idx])

    # ------------------------------------------------------------------
    # Mixture properties
    # ------------------------------------------------------------------

    def mixture_density(
        self, alphas: torch.Tensor, p: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture density: rho_m = sum(alpha_i * rho_i(p, T)).

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture density (kg/m^3).
        """
        alphas = self.validate_alphas(alphas)
        rho_m = torch.zeros_like(p)
        for i in range(self._n_phases - 1):
            rho_m = rho_m + alphas[:, i] * self.phase_density(i, p, T)
        alpha_N = self.compute_last_alpha(alphas)
        rho_m = rho_m + alpha_N * self.phase_density(self._n_phases - 1, p, T)
        return rho_m

    def mixture_viscosity(self, alphas: torch.Tensor) -> torch.Tensor:
        """Mixture viscosity: mu_m = sum(alpha_i * mu_i).

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture dynamic viscosity (Pa·s).
        """
        alphas = self.validate_alphas(alphas)
        device = alphas.device
        dtype = alphas.dtype
        mu_m = torch.zeros(alphas.shape[0], device=device, dtype=dtype)
        for i in range(self._n_phases - 1):
            mu_m = mu_m + alphas[:, i] * self._mu[i]
        alpha_N = self.compute_last_alpha(alphas)
        mu_m = mu_m + alpha_N * self._mu[-1]
        return mu_m

    def mixture_pressure(
        self, alphas: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture pressure for gas-in-liquid systems.

        Uses Dalton's law:
            p_m = sum(alpha_i * p_i)   for gas phases
            p_m is dominated by the liquid pressure otherwise.

        Simplification: for perfect-gas phases, p_i = rho_i * R * T;
        for incompressible phases, p_i = p_ref.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture pressure (Pa).
        """
        alphas = self.validate_alphas(alphas)
        T_safe = T.clamp(min=1.0)
        p_m = torch.zeros_like(T)

        for i in range(self._n_phases - 1):
            if self._eos_type[i] == "perfectGas":
                # p_i = rho_ref * R * T (approximate)
                p_i = self._rho_ref[i] * self._R[i] * T_safe
            else:
                p_i = torch.full_like(T, self._p_ref)
            p_m = p_m + alphas[:, i] * p_i

        # Last phase
        alpha_N = self.compute_last_alpha(alphas)
        idx_last = self._n_phases - 1
        if self._eos_type[idx_last] == "perfectGas":
            p_last = self._rho_ref[idx_last] * self._R[idx_last] * T_safe
        else:
            p_last = torch.full_like(T, self._p_ref)
        p_m = p_m + alpha_N * p_last

        return p_m

    def sound_speed(
        self, alphas: torch.Tensor, p: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture speed of sound.

        For perfect gas: a = sqrt(gamma * R * T)
        For incompressible: a = infinity (use large value)
        Mixture: 1/a_m^2 = sum(alpha_i / a_i^2)

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture speed of sound (m/s).
        """
        alphas = self.validate_alphas(alphas)
        T_safe = T.clamp(min=1.0)
        inv_a2 = torch.zeros_like(T)

        for i in range(self._n_phases):
            if i < self._n_phases - 1:
                a_i = alphas[:, i]
            else:
                a_i = self.compute_last_alpha(alphas)

            if self._eos_type[i] == "perfectGas":
                a_sound = torch.sqrt(self._gamma[i] * self._R[i] * T_safe)
                inv_a2 = inv_a2 + a_i / a_sound.pow(2).clamp(min=1e-30)
            else:
                # Incompressible: infinite sound speed → zero contribution
                pass

        # Avoid division by zero if all phases are incompressible
        return torch.sqrt(1.0 / inv_a2.clamp(min=1e-30))

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------

    def advance_phase(
        self,
        alpha_i: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
        delta_t: float,
    ) -> torch.Tensor:
        """Advance a single phase volume fraction (upwind + compression).

        Parameters
        ----------
        alpha_i : torch.Tensor
            ``(n_cells,)`` volume fraction.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        mesh : Any
            Finite volume mesh.
        delta_t : float
            Time step (s).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` updated volume fraction.
        """
        device = alpha_i.device
        dtype = alpha_i.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        flux = phi[:n_internal]
        is_positive = flux >= 0.0
        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)
        alpha_face = torch.where(is_positive, alpha_P, alpha_N)

        phi_max = flux.abs().max().clamp(min=1e-30)
        compression_flux = self._C_alpha * phi_max * (alpha_P - alpha_N)
        alpha_flux = flux * alpha_face + compression_flux

        div_alpha = torch.zeros(n_cells, dtype=dtype, device=device)
        div_alpha = div_alpha + scatter_add(alpha_flux, int_owner, n_cells)
        div_alpha = div_alpha + scatter_add(-alpha_flux, int_neigh, n_cells)

        if mesh.n_faces > n_internal:
            bnd_flux = phi[n_internal:] * gather(alpha_i, owner[n_internal:])
            div_alpha = div_alpha + scatter_add(bnd_flux, owner[n_internal:], n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        alpha_new = alpha_i - delta_t * div_alpha / V
        return alpha_new.clamp(0.0, 1.0)

    def advance(
        self,
        alphas: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
        delta_t: float,
    ) -> torch.Tensor:
        """Advance all independent volume fractions by one time step."""
        alphas = self.validate_alphas(alphas)
        updated = []
        for i in range(self._n_phases - 1):
            updated.append(self.advance_phase(alphas[:, i], phi, mesh, delta_t))
        result = torch.stack(updated, dim=-1)
        return self.validate_alphas(result)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoF(n_phases={self._n_phases}, "
            f"phases=[{phases}], C_alpha={self._C_alpha})"
        )
