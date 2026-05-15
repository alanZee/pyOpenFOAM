"""
rhoCentralFoam — density-based compressible solver using Kurganov-Tadmor scheme.

Implements the Kurganov-Tadmor (KT) semi-discrete central-upwind scheme for
compressible Euler/Navier-Stokes equations.  Unlike SIMPLE/PIMPLE-based solvers,
rhoCentralFoam uses **explicit time marching** with conservative variables.

Algorithm Overview:
1. Store conservative variables: ρ, ρU (momentum), ρE (total energy)
2. Compute primitive variables: U = ρU/ρ, p, T from EOS
3. TVD reconstruction of left/right states at faces using flux limiters
4. Central-upwind flux: F = 0.5*(F_L + F_R) - 0.5*a*(q_R - q_L)
   where a = max wave speed (|u_n| + c) on both sides
5. Update conservative variables: d(ρq)/dt = -∇·F + sources
6. CFL-based adaptive time stepping

References:
- Kurganov & Tadmor (2000), "New high-resolution central schemes for nonlinear
  conservation laws and convection-diffusion equations", J. Comp. Phys. 160, 241-282.
- OpenFOAM rhoCentralFoam solver (applications/solvers/compressible/rhoCentralFoam)

Usage::

    from pyfoam.applications.rho_central_foam import RhoCentralFoam

    solver = RhoCentralFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo, create_air_thermo

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoCentralFoam"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flux limiters for TVD reconstruction
# ---------------------------------------------------------------------------


def _minmod_limiter(r: torch.Tensor) -> torch.Tensor:
    """Minmod limiter: ψ(r) = max(0, min(1, r))."""
    return torch.clamp(r, 0.0, 1.0)


def _van_leer_limiter(r: torch.Tensor) -> torch.Tensor:
    """Van Leer limiter: ψ(r) = (r + |r|) / (1 + |r|)."""
    abs_r = r.abs()
    return (r + abs_r) / (1.0 + abs_r)


def _superbee_limiter(r: torch.Tensor) -> torch.Tensor:
    """Superbee limiter: ψ(r) = max(0, min(2r, 1), min(r, 2))."""
    return torch.max(
        torch.min(2.0 * r, torch.ones_like(r)),
        torch.min(r, 2.0 * torch.ones_like(r)),
    ).clamp(min=0.0)


_LIMITERS = {
    "minmod": _minmod_limiter,
    "vanLeer": _van_leer_limiter,
    "superbee": _superbee_limiter,
}


# ---------------------------------------------------------------------------
# Convergence data for density-based solver
# ---------------------------------------------------------------------------


@dataclass
class CentralFoamConvergenceData:
    """Convergence data for rhoCentralFoam.

    Attributes
    ----------
    rho_residual : float
        L2 residual of density equation.
    rhoU_residual : float
        L2 residual of momentum equation.
    rhoE_residual : float
        L2 residual of energy equation.
    max_speed : float
        Maximum wave speed (for CFL).
    delta_t : float
        Current time step.
    n_steps : int
        Number of time steps completed.
    converged : bool
        Whether the solution converged.
    """

    rho_residual: float = 0.0
    rhoU_residual: float = 0.0
    rhoE_residual: float = 0.0
    max_speed: float = 0.0
    delta_t: float = 0.0
    n_steps: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# rhoCentralFoam solver
# ---------------------------------------------------------------------------


class RhoCentralFoam(SolverBase):
    """Density-based compressible solver using Kurganov-Tadmor central scheme.

    Solves the compressible Euler/Navier-Stokes equations using explicit
    time marching with the KT central-upwind flux scheme and TVD
    reconstruction for shock capturing.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.
    limiter : str
        TVD flux limiter: ``"minmod"``, ``"vanLeer"``, or ``"superbee"``.
        Default is ``"vanLeer"``.
    CFL : float
        Target CFL number for adaptive time stepping. Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        limiter: str = "vanLeer",
        CFL: float = 0.5,
    ) -> None:
        super().__init__(case_path)

        # Thermophysical model
        self.thermo = thermo or create_air_thermo()

        # TVD limiter
        if limiter not in _LIMITERS:
            raise ValueError(
                f"Unknown limiter '{limiter}'. "
                f"Available: {list(_LIMITERS.keys())}"
            )
        self._limiter_name = limiter
        self._limiter_fn = _LIMITERS[limiter]

        # CFL number
        self.CFL = CFL

        # Read settings
        self._read_fv_solution_settings()

        # Initialise conservative and primitive fields
        (
            self.rho, self.rhoU, self.rhoE,
            self.U, self.p, self.T, self.phi,
        ) = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        # Old fields for time derivative residual
        self.rho_old = self.rho.clone()
        self.rhoU_old = self.rhoU.clone()
        self.rhoE_old = self.rhoE.clone()

        logger.info(
            "RhoCentralFoam ready: limiter=%s, CFL=%.2f, thermo=%s",
            self._limiter_name, self.CFL, self.thermo,
        )

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read central scheme settings from fvSolution."""
        fv = self.case.fvSolution

        # CFL and time stepping
        self.CFL = float(fv.get_path("centralCoeffs/CFL", self.CFL))
        self.max_delta_t = float(fv.get_path("centralCoeffs/maxDeltaT", 1e-4))
        self.min_delta_t = float(fv.get_path("centralCoeffs/minDeltaT", 1e-12))

        # Number of correctors (for NS viscous terms)
        self.n_non_orth_correctors = int(
            fv.get_path("centralCoeffs/nNonOrthogonalCorrectors", 0)
        )

        # Convergence
        self.convergence_tolerance = float(
            fv.get_path("centralCoeffs/convergenceTolerance", 1e-4)
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise conservative and primitive fields."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        T_tensor, _ = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # Compute density from EOS
        rho = self.thermo.rho(p, T)

        # Conservative variables
        rhoU = rho.unsqueeze(-1) * U  # (n_cells, 3)
        Cv = self.thermo.Cv()
        gamma = self.thermo.gamma()
        # Total energy per unit volume: ρE = ρ*Cv*T + 0.5*ρ|U|²
        rhoE = rho * Cv * T + 0.5 * rho * (U * U).sum(dim=1)

        return rho, rhoU, rhoE, U, p, T, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        T_data = self.case.read_field("T", 0)
        return U_data, p_data, T_data

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> CentralFoamConvergenceData:
        """Run the rhoCentralFoam solver.

        Returns:
            Final :class:`CentralFoamConvergenceData`.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting rhoCentralFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence = CentralFoamConvergenceData()

        for t, step in time_loop:
            # Adaptive time step based on CFL
            dt = self._compute_adaptive_delta_t()
            dt = min(dt, self.delta_t)

            # Store old fields
            self.rho_old = self.rho.clone()
            self.rhoU_old = self.rhoU.clone()
            self.rhoE_old = self.rhoE.clone()

            # Time integration (forward Euler)
            self.rho, self.rhoU, self.rhoE = self._time_step(
                self.rho, self.rhoU, self.rhoE, dt
            )

            # Update primitive variables
            self._update_primitives()

            # Compute face flux
            self._compute_face_flux()

            # Convergence check
            conv = self._compute_convergence(dt)
            last_convergence = conv

            residuals = {
                "rho": conv.rho_residual,
                "rhoU": conv.rhoU_residual,
                "rhoE": conv.rhoE_residual,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        return last_convergence

    # ------------------------------------------------------------------
    # Adaptive time stepping
    # ------------------------------------------------------------------

    def _compute_adaptive_delta_t(self) -> float:
        """Compute adaptive time step based on CFL condition.

        dt = CFL * min(dx / (|u| + c))

        where dx is cell characteristic length, u is velocity magnitude,
        c is speed of sound.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self.mesh

        # Speed of sound: c = sqrt(gamma * R * T) = sqrt(gamma * p / rho)
        gamma = self.thermo.gamma()
        c = torch.sqrt(
            gamma * self.p.abs().clamp(min=1e-30) / self.rho.abs().clamp(min=1e-30)
        )

        # Velocity magnitude
        U_mag = self.U.norm(dim=1)

        # Wave speed = |U| + c
        wave_speed = U_mag + c

        # Characteristic cell length: V^(1/3) for 3D
        dx = mesh.cell_volumes.pow(1.0 / 3.0).clamp(min=1e-30)

        # Local CFL time step
        local_dt = self.CFL * dx / wave_speed.clamp(min=1e-30)

        # Global minimum
        dt = float(local_dt.min().item())
        dt = max(dt, self.min_delta_t)
        dt = min(dt, self.max_delta_t)

        return dt

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def _time_step(
        self,
        rho: torch.Tensor,
        rhoU: torch.Tensor,
        rhoE: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward Euler time step for conservative variables.

        q^{n+1} = q^n + dt * (source - ∇·F)
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells

        # Compute face fluxes using KT scheme
        flux_rho, flux_rhoU, flux_rhoE = self._compute_kt_fluxes(
            rho, rhoU, rhoE
        )

        # Divergence of fluxes (scatter-add to cells)
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Density flux divergence
        div_rho = torch.zeros(n_cells, dtype=dtype, device=device)
        div_rho = div_rho + scatter_add(flux_rho, int_owner, n_cells)
        div_rho = div_rho + scatter_add(-flux_rho, int_neigh, n_cells)

        # Momentum flux divergence (vector)
        div_rhoU = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        div_rhoU.index_add_(0, int_owner, flux_rhoU)
        div_rhoU.index_add_(0, int_neigh, -flux_rhoU)

        # Energy flux divergence
        div_rhoE = torch.zeros(n_cells, dtype=dtype, device=device)
        div_rhoE = div_rhoE + scatter_add(flux_rhoE, int_owner, n_cells)
        div_rhoE = div_rhoE + scatter_add(-flux_rhoE, int_neigh, n_cells)

        # Viscous sources (for NS equations)
        visc_rhoU, visc_rhoE = self._compute_viscous_sources(rho, rhoU, rhoE)

        # Update conservative variables
        V = mesh.cell_volumes.clamp(min=1e-30)
        rho_new = rho + dt * (-div_rho / V)
        rhoU_new = rhoU + dt * (-div_rhoU / V.unsqueeze(-1) + visc_rhoU)
        rhoE_new = rhoE + dt * (-div_rhoE / V + visc_rhoE)

        # Enforce positivity
        rho_new = rho_new.abs().clamp(min=1e-10)
        # Ensure total energy is positive
        kinetic = 0.5 * (rhoU_new * rhoU_new).sum(dim=1) / rho_new.abs().clamp(min=1e-30)
        rhoE_new = rhoE_new.clamp(min=kinetic + 1e-10)

        return rho_new, rhoU_new, rhoE_new

    # ------------------------------------------------------------------
    # KT central-upwind flux
    # ------------------------------------------------------------------

    def _compute_kt_fluxes(
        self,
        rho: torch.Tensor,
        rhoU: torch.Tensor,
        rhoE: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Kurganov-Tadmor central-upwind face fluxes.

        For each internal face with owner P and neighbour N:
        1. Reconstruct left (P) and right (N) states using TVD limiter
        2. Compute maximum wave speed: a = max(|u_n| + c) on both sides
        3. Central-upwind flux: F = 0.5*(F_L + F_R) - 0.5*a*(q_R - q_L)

        Returns:
            Tuple of (flux_rho, flux_rhoU, flux_rhoE) for internal faces.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_internal = mesh.n_internal_faces

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face normal vectors (unit normal * area)
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1, keepdim=True).clamp(min=1e-30)
        face_normal = face_areas / S_mag  # unit normal
        S_mag_flat = S_mag.squeeze(-1)

        # ---- TVD Reconstruction ----
        # Reconstruct left and right states at faces
        rho_L, rho_R = self._tvd_reconstruct_scalar(rho, int_owner, int_neigh, face_normal)
        rhoU_L, rhoU_R = self._tvd_reconstruct_vector(rhoU, int_owner, int_neigh, face_normal)
        rhoE_L, rhoE_R = self._tvd_reconstruct_scalar(rhoE, int_owner, int_neigh, face_normal)

        # ---- Primitive variables at faces ----
        # Left state
        rho_L_safe = rho_L.abs().clamp(min=1e-30)
        U_L = rhoU_L / rho_L_safe.unsqueeze(-1)
        gamma = self.thermo.gamma()
        Cv = self.thermo.Cv()
        p_L = (gamma - 1.0) * (rhoE_L - 0.5 * rho_L * (U_L * U_L).sum(dim=1))
        p_L = p_L.abs().clamp(min=1e-10)
        c_L = torch.sqrt(gamma * p_L / rho_L_safe)

        # Right state
        rho_R_safe = rho_R.abs().clamp(min=1e-30)
        U_R = rhoU_R / rho_R_safe.unsqueeze(-1)
        p_R = (gamma - 1.0) * (rhoE_R - 0.5 * rho_R * (U_R * U_R).sum(dim=1))
        p_R = p_R.abs().clamp(min=1e-10)
        c_R = torch.sqrt(gamma * p_R / rho_R_safe)

        # ---- Wave speed ----
        # Normal velocity at faces
        u_n_L = (U_L * face_normal).sum(dim=1)
        u_n_R = (U_R * face_normal).sum(dim=1)

        # Maximum wave speed (central-upwind)
        a_L = u_n_L.abs() + c_L
        a_R = u_n_R.abs() + c_R
        a = torch.max(a_L, a_R)

        # ---- Fluxes ----
        # Physical flux: F = [ρu_n, ρu_n*U + p*n, (ρE + p)*u_n]
        # Left flux
        F_rho_L = rho_L * u_n_L
        F_rhoU_L = rho_L * u_n_L.unsqueeze(-1) * U_L + p_L.unsqueeze(-1) * face_normal
        F_rhoE_L = (rhoE_L + p_L) * u_n_L

        # Right flux
        F_rho_R = rho_R * u_n_R
        F_rhoU_R = rho_R * u_n_R.unsqueeze(-1) * U_R + p_R.unsqueeze(-1) * face_normal
        F_rhoE_R = (rhoE_R + p_R) * u_n_R

        # Central-upwind flux (multiplied by face area)
        flux_rho = (0.5 * (F_rho_L + F_rho_R) - 0.5 * a * (rho_R - rho_L)) * S_mag_flat
        flux_rhoU = (0.5 * (F_rhoU_L + F_rhoU_R) - 0.5 * a.unsqueeze(-1) * (rhoU_R - rhoU_L)) * S_mag_flat.unsqueeze(-1)
        flux_rhoE = (0.5 * (F_rhoE_L + F_rhoE_R) - 0.5 * a * (rhoE_R - rhoE_L)) * S_mag_flat

        return flux_rho, flux_rhoU, flux_rhoE

    # ------------------------------------------------------------------
    # TVD reconstruction
    # ------------------------------------------------------------------

    def _tvd_reconstruct_scalar(
        self,
        q: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        face_normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TVD reconstruction of scalar field at faces.

        Uses cell-centre values with gradient-based reconstruction
        and flux limiter to prevent overshoots.

        Returns:
            (q_L, q_R) — left and right reconstructed values at faces.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        q_P = gather(q, owner)
        q_N = gather(q, neighbour)

        # If mesh has too few cells, use piecewise constant
        if mesh.n_cells <= 2:
            return q_P, q_N

        # Compute cell gradients using Gauss theorem
        grad_q = self._compute_scalar_gradient(q)

        # Gradient at face owner/neighbour
        grad_P = gather(grad_q, owner)  # (n_internal, 3)
        grad_N = gather(grad_q, neighbour)

        # Distance vectors from cell centre to face centre
        cc_P = gather(mesh.cell_centres, owner)
        cc_N = gather(mesh.cell_centres, neighbour)
        fc = mesh.face_centres[:mesh.n_internal_faces]

        d_P = fc - cc_P  # (n_internal, 3)
        d_N = fc - cc_N

        # Extrapolated values
        q_P_ext = q_P + (grad_P * d_P).sum(dim=1)
        q_N_ext = q_N + (grad_N * d_N).sum(dim=1)

        # Apply limiter
        # r = (q_upwind - q_2upwind) / (q_downwind - q_upwind)
        # Simplified: use ratio of extrapolated to central difference
        denom = q_N - q_P
        safe_denom = torch.where(
            denom.abs() > 1e-30, denom, torch.sign(denom + 1e-30) * 1e-30
        )

        # Limit extrapolated values to prevent overshoots
        # Use TVD criterion: q_min <= q_face <= q_max
        q_min = torch.min(q_P, q_N)
        q_max = torch.max(q_P, q_N)

        q_P_limited = torch.clamp(q_P_ext, q_min, q_max)
        q_N_limited = torch.clamp(q_N_ext, q_min, q_max)

        return q_P_limited, q_N_limited

    def _tvd_reconstruct_vector(
        self,
        q: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        face_normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TVD reconstruction of vector field at faces.

        Returns:
            (q_L, q_R) — left and right reconstructed vector values.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        q_P = gather(q, owner)  # (n_internal, 3)
        q_N = gather(q, neighbour)

        if mesh.n_cells <= 2:
            return q_P, q_N

        # Compute gradient for each component
        grad_q = self._compute_vector_gradient(q)  # (n_cells, 3, 3)

        grad_P = gather(grad_q, owner)  # (n_internal, 3, 3)
        grad_N = gather(grad_q, neighbour)

        cc_P = gather(mesh.cell_centres, owner)
        cc_N = gather(mesh.cell_centres, neighbour)
        fc = mesh.face_centres[:mesh.n_internal_faces]

        d_P = fc - cc_P
        d_N = fc - cc_N

        # Extrapolated values: q + grad_q · d
        # q_P_ext[i] = q_P[i] + sum_j grad_P[i,j] * d_P[j]
        q_P_ext = q_P + torch.einsum('ijk,ik->ij', grad_P, d_P)
        q_N_ext = q_N + torch.einsum('ijk,ik->ij', grad_N, d_N)

        # Limit each component
        q_min = torch.min(q_P, q_N)
        q_max = torch.max(q_P, q_N)

        q_P_limited = torch.clamp(q_P_ext, q_min, q_max)
        q_N_limited = torch.clamp(q_N_ext, q_min, q_max)

        return q_P_limited, q_N_limited

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _compute_scalar_gradient(self, q: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradient of scalar using Gauss theorem."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face values (linear interpolation)
        q_P = gather(q, int_owner)
        q_N = gather(q, int_neigh)
        q_face = 0.5 * (q_P + q_N)

        face_contrib = q_face.unsqueeze(-1) * mesh.face_areas[:n_internal]

        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        # Boundary contributions
        if mesh.n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            q_bnd = gather(q, bnd_owner)
            bnd_contrib = q_bnd.unsqueeze(-1) * mesh.face_areas[n_internal:]
            grad.index_add_(0, bnd_owner, bnd_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_vector_gradient(self, q: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradient of vector field using Gauss theorem."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face values (linear interpolation)
        q_P = q[int_owner]
        q_N = q[int_neigh]
        q_face = 0.5 * (q_P + q_N)

        # grad[i,j,k] = dq_j/dx_k
        grad = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        face_areas = mesh.face_areas[:n_internal]

        for j in range(3):
            face_contrib = q_face[:, j].unsqueeze(-1) * face_areas
            grad[:, :, j].index_add_(0, int_owner, face_contrib)
            grad[:, :, j].index_add_(0, int_neigh, -face_contrib)

        # Boundary contributions
        if mesh.n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            q_bnd = q[bnd_owner]
            for j in range(3):
                bnd_contrib = q_bnd[:, j].unsqueeze(-1) * mesh.face_areas[n_internal:]
                grad[:, :, j].index_add_(0, bnd_owner, bnd_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    # ------------------------------------------------------------------
    # Viscous sources (Navier-Stokes)
    # ------------------------------------------------------------------

    def _compute_viscous_sources(
        self,
        rho: torch.Tensor,
        rhoU: torch.Tensor,
        rhoE: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute viscous contributions to momentum and energy equations.

        Momentum: ∇·τ where τ = μ(∇U + ∇U^T) - (2/3)μ(∇·U)I
        Energy:  ∇·(τ·U) + ∇·(κ∇T)

        Returns:
            (visc_rhoU, visc_rhoE) — viscous source terms per unit volume.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        rho_safe = rho.abs().clamp(min=1e-30)
        U = rhoU / rho_safe.unsqueeze(-1)

        # Viscosity
        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, mesh.owner[:n_internal]) + gather(mu, mesh.neighbour))

        # Velocity gradient
        grad_U = self._compute_vector_gradient(U)

        # Viscous stress: τ = μ(∇U + ∇U^T) - (2/3)μ(∇·U)I
        div_U = grad_U[:, 0, 0] + grad_U[:, 1, 1] + grad_U[:, 2, 2]
        tau = mu.unsqueeze(-1).unsqueeze(-1) * (grad_U + grad_U.transpose(-1, -2))
        tau[:, 0, 0] -= (2.0 / 3.0) * mu * div_U
        tau[:, 1, 1] -= (2.0 / 3.0) * mu * div_U
        tau[:, 2, 2] -= (2.0 / 3.0) * mu * div_U

        # Viscous momentum source: ∇·τ (via Gauss theorem)
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        face_areas = mesh.face_areas[:n_internal]

        # Face stress (linear interpolation)
        tau_P = gather(tau, int_owner)  # (n_internal, 3, 3)
        tau_N = gather(tau, int_neigh)
        tau_face = 0.5 * (tau_P + tau_N)

        # τ · S (stress dotted with face area vector)
        tau_dot_S = torch.einsum('ijk,ik->ij', tau_face, face_areas)  # (n_internal, 3)

        visc_rhoU = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        visc_rhoU.index_add_(0, int_owner, tau_dot_S)
        visc_rhoU.index_add_(0, int_neigh, -tau_dot_S)

        # Boundary contributions
        if mesh.n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            tau_bnd = gather(tau, bnd_owner)
            tau_bnd_dot_S = torch.einsum('ijk,ik->ij', tau_bnd, mesh.face_areas[n_internal:])
            visc_rhoU.index_add_(0, bnd_owner, tau_bnd_dot_S)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        visc_rhoU = visc_rhoU / V

        # Viscous energy source: ∇·(κ∇T) + ∇·(τ·U)
        # Thermal diffusion
        kappa = self.thermo.kappa(self.T)
        grad_T = self._compute_scalar_gradient(self.T)
        kappa_face = 0.5 * (
            gather(kappa, int_owner) + gather(kappa, int_neigh)
        )
        grad_T_face = 0.5 * (gather(grad_T, int_owner) + gather(grad_T, int_neigh))
        kappa_grad_T = kappa_face.unsqueeze(-1) * grad_T_face

        # κ∇T · S
        kappa_dot_S = (kappa_grad_T * face_areas).sum(dim=1)

        visc_rhoE = torch.zeros(n_cells, dtype=dtype, device=device)
        visc_rhoE = visc_rhoE + scatter_add(kappa_dot_S, int_owner, n_cells)
        visc_rhoE = visc_rhoE + scatter_add(-kappa_dot_S, int_neigh, n_cells)

        # Boundary contributions for thermal
        if mesh.n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            kappa_bnd = gather(kappa, bnd_owner)
            grad_T_bnd = gather(grad_T, bnd_owner)
            kappa_grad_T_bnd = kappa_bnd.unsqueeze(-1) * grad_T_bnd
            kappa_dot_S_bnd = (kappa_grad_T_bnd * mesh.face_areas[n_internal:]).sum(dim=1)
            visc_rhoE = visc_rhoE + scatter_add(kappa_dot_S_bnd, bnd_owner, n_cells)

        # τ·U contribution: (τ·U)·S at faces
        U_face = 0.5 * (U[int_owner] + U[int_neigh])
        tau_U = torch.einsum('ijk,ij->ik', tau_face, U_face)  # (n_internal, 3)
        tau_U_dot_S = (tau_U * face_areas).sum(dim=1)

        visc_rhoE = visc_rhoE + scatter_add(tau_U_dot_S, int_owner, n_cells)
        visc_rhoE = visc_rhoE + scatter_add(-tau_U_dot_S, int_neigh, n_cells)

        visc_rhoE = visc_rhoE / mesh.cell_volumes.clamp(min=1e-30)

        return visc_rhoU, visc_rhoE

    # ------------------------------------------------------------------
    # Update primitives
    # ------------------------------------------------------------------

    def _update_primitives(self) -> None:
        """Update primitive variables from conservative variables."""
        rho_safe = self.rho.abs().clamp(min=1e-30)

        # U = ρU / ρ
        self.U = self.rhoU / rho_safe.unsqueeze(-1)

        # p = (γ-1) * (ρE - 0.5*ρ|U|²)
        gamma = self.thermo.gamma()
        kinetic = 0.5 * rho_safe * (self.U * self.U).sum(dim=1)
        self.p = ((gamma - 1.0) * (self.rhoE - kinetic)).abs().clamp(min=1e-10)

        # T from EOS: T = p / (ρ * R)
        R = self.thermo.R()
        self.T = self.p / (rho_safe * R)
        self.T = self.T.abs().clamp(min=1.0)  # Ensure positive temperature

    def _compute_face_flux(self) -> None:
        """Compute face flux for output/monitoring."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_internal = mesh.n_internal_faces

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face velocity (linear interpolation)
        U_face = 0.5 * (self.U[int_owner] + self.U[int_neigh])

        # Face flux: φ = U · S
        self.phi = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        self.phi[:n_internal] = (U_face * mesh.face_areas[:n_internal]).sum(dim=1)

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def _compute_convergence(self, dt: float) -> CentralFoamConvergenceData:
        """Compute convergence metrics."""
        rho_res = float((self.rho - self.rho_old).abs().mean().item())
        rhoU_res = float((self.rhoU - self.rhoU_old).abs().mean().item())
        rhoE_res = float((self.rhoE - self.rhoE_old).abs().mean().item())

        # Maximum wave speed
        gamma = self.thermo.gamma()
        c = torch.sqrt(
            gamma * self.p.abs().clamp(min=1e-30) / self.rho.abs().clamp(min=1e-30)
        )
        U_mag = self.U.norm(dim=1)
        max_speed = float((U_mag + c).max().item())

        return CentralFoamConvergenceData(
            rho_residual=rho_res,
            rhoU_residual=rhoU_res,
            rhoE_residual=rhoE_res,
            max_speed=max_speed,
            delta_t=dt,
            n_steps=0,
            converged=False,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
