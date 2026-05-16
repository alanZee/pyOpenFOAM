"""
multiphaseInterFoam — N-phase VOF incompressible solver.

Extends interFoam to handle N immiscible incompressible fluids using
the algebraic VOF method. Volume fractions sum to 1:

    sum(alpha_i) = 1

Each phase has its own density and viscosity. Mixture properties are
computed as weighted averages:

    rho = sum(alpha_i * rho_i)
    mu  = sum(alpha_i * mu_i)

Only N-1 volume fractions are transported; the last is computed
from the constraint.

Based on OpenFOAM's multiphaseInterFoam solver.

Usage::

    from pyfoam.applications.multiphase_inter_foam import MultiphaseInterFoam

    phases = [
        {"name": "water", "rho": 1000.0, "mu": 1e-3},
        {"name": "oil",   "rho": 800.0,  "mu": 0.01},
        {"name": "air",   "rho": 1.225,  "mu": 1.8e-5},
    ]
    solver = MultiphaseInterFoam("path/to/case", phases=phases)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseInterFoam"]

logger = logging.getLogger(__name__)


class MultiphaseInterFoam(SolverBase):
    """N-phase VOF incompressible solver.

    Solves the incompressible Navier-Stokes equations for N immiscible
    fluids using the Volume of Fluid method.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        List of phase definitions, each with keys:
        - name: str (e.g., "water", "air", "oil")
        - rho: float (density, kg/m³)
        - mu: float (dynamic viscosity, Pa·s)
    sigma_pairs : dict[tuple[str, str], float], optional
        Surface tension coefficients between phase pairs.
        Keys are (name1, name2) tuples.  Default: no surface tension.
    C_alpha : float
        VOF compression coefficient.  Default 1.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
        sigma_pairs: Dict[tuple, float] | None = None,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.phases = phases
        self.n_phases = len(phases)
        self.sigma_pairs = sigma_pairs or {}
        self.C_alpha = C_alpha

        # Phase properties
        self.phase_names = [p["name"] for p in phases]
        self.rho_phases = torch.tensor(
            [p["rho"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )
        self.mu_phases = torch.tensor(
            [p["mu"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )

        # Read settings
        self._read_fv_solution_settings()

        # Initialise fields
        self.U, self.p, self.alphas, self.phi = self._init_fields()
        self._U_data, self._p_data, self._alpha_datas = self._init_field_data()

        # VOF advection for each transported alpha
        self.vofs = []
        for i in range(self.n_phases - 1):
            vof = VOFAdvection(
                self.mesh, self.alphas[i], self.phi, self.U,
                C_alpha=C_alpha,
            )
            self.vofs.append(vof)

        # Surface tension models (one per pair)
        self.surface_tensions = {}
        for (n1, n2), sigma in self.sigma_pairs.items():
            i1 = self.phase_names.index(n1)
            i2 = self.phase_names.index(n2)
            st = SurfaceTensionModel(sigma=sigma, mesh=self.mesh, n_smooth=1)
            self.surface_tensions[(i1, i2)] = st

        logger.info(
            "MultiphaseInterFoam ready: %d phases, %d sigma pairs",
            self.n_phases, len(self.sigma_pairs),
        )

    def _read_fv_solution_settings(self) -> None:
        """Read PIMPLE settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )

        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    def _init_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Initialise U, p, alpha_i, phi from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        alphas = []
        for i, name in enumerate(self.phase_names):
            field_name = f"alpha.{name}"
            if i < self.n_phases - 1:
                alpha_tensor, _ = self.read_field_tensor(field_name, 0)
                alphas.append(alpha_tensor.to(device=device, dtype=dtype))
            else:
                # Last phase: computed from constraint
                alpha_last = 1.0 - sum(alphas)
                alphas.append(alpha_last.clamp(0.0, 1.0))

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, alphas, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        alpha_datas = []
        for name in self.phase_names:
            try:
                ad = self.case.read_field(f"alpha.{name}", 0)
                alpha_datas.append(ad)
            except Exception:
                alpha_datas.append(None)
        return U_data, p_data, alpha_datas

    def _compute_mixture_rho(self, alphas: list[torch.Tensor]) -> torch.Tensor:
        """Compute mixture density: rho = sum(alpha_i * rho_i)."""
        rho = torch.zeros_like(alphas[0])
        for i in range(self.n_phases):
            rho = rho + alphas[i] * self.rho_phases[i]
        return rho

    def _compute_mixture_mu(self, alphas: list[torch.Tensor]) -> torch.Tensor:
        """Compute mixture viscosity: mu = sum(alpha_i * mu_i)."""
        mu = torch.zeros_like(alphas[0])
        for i in range(self.n_phases):
            mu = mu + alphas[i] * self.mu_phases[i]
        return mu

    def run(self) -> ConvergenceData:
        """Run the multiphaseInterFoam solver."""
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

        logger.info("Starting multiphaseInterFoam run")
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U, self.p, self.alphas, self.phi, conv = (
                self._pimple_vof_iteration()
            )
            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()

    def _pimple_vof_iteration(self):
        """Run one PIMPLE time step with multi-phase VOF advection."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # --- VOF advection for each transported phase ---
            for i in range(self.n_phases - 1):
                self.vofs[i].alpha = alphas[i]
                self.vofs[i].phi = phi
                self.vofs[i].U = U
                alphas[i] = self.vofs[i].advance(self.delta_t)

            # Enforce constraint: last alpha = 1 - sum(others)
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # Renormalise to ensure sum = 1
            total = sum(alphas)
            total = total.clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # Mixture properties
            rho = self._compute_mixture_rho(alphas)
            mu_mix = self._compute_mixture_mu(alphas)

            # --- Momentum predictor ---
            U, A_p, H = self._momentum_predictor(U, p, phi, rho, mu_mix)

            # --- PISO correction loop ---
            for corr in range(self.n_correctors):
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                n_internal = mesh.n_internal_faces
                int_owner = mesh.owner[:n_internal]
                int_neigh = mesh.neighbour
                w = mesh.face_weights[:n_internal]

                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                p = self._solve_pressure_equation(
                    p, phiHbyA, A_p, rho, mesh
                )

                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi = phiHbyA - (p_N - p_P) * A_p_inv_face

                if corr < self.n_correctors - 1:
                    H = self._recompute_H(U, phi, rho, mu_mix)

            # Under-relaxation
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # Convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, alphas, phi, convergence

    def _momentum_predictor(self, U, p, phi, rho, mu_mix):
        """Solve momentum equation with surface tension."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        mu_face = 0.5 * (gather(mu_mix, int_owner) + gather(mu_mix, int_neigh))
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )
        A_p = A_p + rho_V_dt

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        H = H + rho_V_dt.unsqueeze(-1) * self.U

        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Surface tension
        F_sigma = self._compute_surface_tension()
        source = H - grad_p + F_sigma

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    def _compute_surface_tension(self) -> torch.Tensor:
        """Compute surface tension force from all phase pairs."""
        n_cells = self.mesh.n_cells
        device = get_device()
        dtype = get_default_dtype()
        F = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        for (i1, i2), st in self.surface_tensions.items():
            F = F + st.compute_force(self.alphas[i1])

        return F

    def _recompute_H(self, U, phi, rho, mu_mix):
        """Recompute H(U) from corrected velocity."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        mu_face = 0.5 * (gather(mu_mix, int_owner) + gather(mu_mix, int_neigh))
        S_mag = mesh.face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(mesh.cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(mesh.cell_volumes.clamp(min=1e-30), int_neigh)

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        return H

    def _compute_residual(self, field, field_old):
        """Compute the L2 norm of the field change, normalised."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _solve_pressure_equation(self, p, phiHbyA, A_p, rho, mesh):
        """Solve pressure equation for multi-phase flow."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = 0.5 * (rho_P + rho_N)

        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = (
            w * gather(A_p_inv, int_owner)
            + (1.0 - w) * gather(A_p_inv, int_neigh)
        )

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = rho_face * A_p_inv_face * S_mag * delta_f

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.p_max_iter):
            off_diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe
            if (p_new - p).abs().max() < self.p_tolerance:
                break
            p = p_new

        return p

    def _compute_grad(self, phi, mesh):
        """Compute gradient of scalar field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(n_cells, 3, dtype=phi.dtype, device=phi.device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_continuity_error(self, phi, rho):
        """Compute continuity error."""
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        rho_face = 0.5 * (
            gather(rho, owner[:n_internal]) + gather(rho, neighbour)
        )
        mass_flux = phi[:n_internal] * rho_face

        div_rho_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_rho_phi = div_rho_phi + scatter_add(
            mass_flux, owner[:n_internal], n_cells
        )
        div_rho_phi = div_rho_phi + scatter_add(
            -mass_flux, neighbour, n_cells
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        div_rho_phi = div_rho_phi / V

        return float(div_rho_phi.abs().mean().item())

    def _write_fields(self, time):
        """Write U, p, alpha_i to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        for i, name in enumerate(self.phase_names):
            if self._alpha_datas[i] is not None:
                self.write_field(
                    f"alpha.{name}", self.alphas[i],
                    time_str, self._alpha_datas[i],
                )
