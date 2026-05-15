"""
sonicFoam — transient compressible solver for transonic/supersonic flows.

Implements the PISO algorithm for transient compressible Navier-Stokes
equations with shock-capturing TVD schemes.

Based on OpenFOAM's sonicFoam solver:
- Compressible continuity: ∂ρ/∂t + ∇·(ρU) = 0
- Momentum: ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·τ + ρg
- Energy: ∂(ρe)/∂t + ∇·(ρeU) = -p∇·U + ∇·(α∇e) + Su

Key features:
- PISO pressure-velocity coupling (no outer iteration)
- TVD shock-capturing schemes (minmod, vanLeer, superbee)
- Perfect gas EOS: p = ρRT
- Variable viscosity from Sutherland transport model
- Euler time discretisation

Usage::

    from pyfoam.applications.sonic_foam import SonicFoam

    solver = SonicFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
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

__all__ = ["SonicFoam"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TVD flux limiters for shock capturing
# ---------------------------------------------------------------------------

def _van_leer_limiter(r: torch.Tensor) -> torch.Tensor:
    """Van Leer flux limiter: ψ(r) = (r + |r|) / (1 + |r|)."""
    abs_r = r.abs()
    return (r + abs_r) / (1.0 + abs_r)


def _minmod_limiter(r: torch.Tensor) -> torch.Tensor:
    """Minmod flux limiter: ψ(r) = max(0, min(1, r))."""
    return torch.clamp(r, 0.0, 1.0)


def _superbee_limiter(r: torch.Tensor) -> torch.Tensor:
    """Superbee flux limiter: ψ(r) = max(0, min(2r, 1), min(r, 2))."""
    return torch.max(
        torch.zeros_like(r),
        torch.max(
            torch.min(2.0 * r, torch.ones_like(r)),
            torch.min(r, 2.0 * torch.ones_like(r)),
        ),
    )


def _osher_limiter(r: torch.Tensor, beta: float = 1.5) -> torch.Tensor:
    """Osher flux limiter: ψ(r) = max(0, min(r, beta))."""
    return torch.clamp(r, 0.0, beta)


def _sweby_limiter(r: torch.Tensor, beta: float = 1.5) -> torch.Tensor:
    """Sweby flux limiter: ψ(r) = max(0, min(beta*r, 1), min(r, beta))."""
    return torch.max(
        torch.zeros_like(r),
        torch.max(
            torch.min(beta * r, torch.ones_like(r)),
            torch.min(r, beta * torch.ones_like(r)),
        ),
    )


_TVD_LIMITERS = {
    "vanLeer": _van_leer_limiter,
    "minmod": _minmod_limiter,
    "superbee": _superbee_limiter,
    "osher": _osher_limiter,
    "sweby": _sweby_limiter,
}


def get_tvd_limiter(name: str):
    """Get TVD limiter function by name.

    Parameters
    ----------
    name : str
        Limiter name. One of: vanLeer, minmod, superbee, osher, sweby.

    Returns
    -------
    callable
        Limiter function ψ(r).

    Raises
    ------
    ValueError
        If limiter name is not recognised.
    """
    if name not in _TVD_LIMITERS:
        raise ValueError(
            f"Unknown TVD limiter '{name}'. "
            f"Available: {list(_TVD_LIMITERS.keys())}"
        )
    return _TVD_LIMITERS[name]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class SonicFoam(SolverBase):
    """Transient compressible PISO solver (sonicFoam).

    Solves the transient compressible Navier-Stokes equations using
    the PISO algorithm with TVD shock-capturing schemes.

    Designed for transonic and supersonic flows where shocks may form.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
    ) -> None:
        super().__init__(case_path)

        # Thermophysical model
        self.thermo = thermo or create_air_thermo()

        # Read settings
        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.T, self.phi, self.rho = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        # Store old fields for time derivative
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()
        self.T_old = self.T.clone()
        self.rho_old = self.rho.clone()

        # Compressibility: ψ = 1/(R*T) for perfect gas
        self.psi = self._compute_psi(self.T)

        logger.info("SonicFoam ready: %s", self.thermo)
        logger.info("  TVD limiter: %s", self.tvd_limiter_name)

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read PISO settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

        self.n_correctors = int(fv.get_path("PISO/nCorrectors", 2))
        self.n_non_orth_correctors = int(
            fv.get_path("PISO/nNonOrthogonalCorrectors", 0)
        )

        self.convergence_tolerance = float(
            fv.get_path("PISO/convergenceTolerance", 1e-4)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings including TVD limiter selection."""
        fs = self.case.fvSchemes

        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path(
            "laplacianSchemes/default", "Gauss linear corrected"
        ))

        # Read TVD limiter for shock capturing
        # Look for div(rhoPhi,U) or div(phi,U) scheme
        div_u_scheme = str(fs.get_path("divSchemes/div(rhoPhi,U)", ""))
        if not div_u_scheme:
            div_u_scheme = str(fs.get_path("divSchemes/div(phi,U)", ""))

        # Parse limiter from scheme string (e.g. "Gauss limitedLinear vanLeer")
        self.tvd_limiter_name = "vanLeer"  # default
        if "limitedLinear" in div_u_scheme or "LUST" in div_u_scheme:
            for name in _TVD_LIMITERS:
                if name in div_u_scheme:
                    self.tvd_limiter_name = name
                    break
        elif "TVD" in div_u_scheme:
            for name in _TVD_LIMITERS:
                if name in div_u_scheme:
                    self.tvd_limiter_name = name
                    break

        self._tvd_limiter_fn = get_tvd_limiter(self.tvd_limiter_name)

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, T, phi, rho from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        T_tensor, _ = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # Compute initial density from EOS
        rho = self.thermo.rho(p, T)

        return U, p, T, phi, rho

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        T_data = self.case.read_field("T", 0)
        return U_data, p_data, T_data

    # ------------------------------------------------------------------
    # Compressibility
    # ------------------------------------------------------------------

    def _compute_psi(self, T: torch.Tensor) -> torch.Tensor:
        """Compute compressibility ψ = 1/(RT) for perfect gas.

        For a perfect gas: p = ρRT → ρ = p/(RT) = ψ·p
        where ψ = 1/(RT).
        """
        R = self.thermo.R()
        T_safe = T.clamp(min=1e-10)
        return 1.0 / (R * T_safe)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the sonicFoam solver.

        Executes the PISO algorithm in a transient time-stepping loop
        until ``endTime`` is reached.

        Returns:
            Final :class:`ConvergenceData`.
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

        logger.info("Starting sonicFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nCorrectors=%d", self.n_correctors)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            # Run one PISO time step
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._piso_step()
            )
            last_convergence = conv

            # Update compressibility
            self.psi = self._compute_psi(self.T)

            # Check convergence
            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # PISO step for compressible flow
    # ------------------------------------------------------------------

    def _piso_step(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PISO time step for compressible flow.

        Algorithm:
        1. Solve momentum equation (with old pressure)
        2. PISO correction loop:
           a. Compute HbyA and face flux
           b. Solve pressure equation
           c. Correct velocity
           d. Correct flux
        3. Update density from EOS
        4. Solve energy equation

        Returns:
            Tuple of (U, p, T, phi, rho, convergence).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        phi = self.phi.clone()
        rho = self.rho.clone()

        convergence = ConvergenceData()

        # ============================================
        # Step 1: Momentum predictor
        # ============================================
        U, A_p, H = self._momentum_predictor(U, p, phi, rho)

        # ============================================
        # Step 2: PISO correction loop
        # ============================================
        for corr in range(self.n_correctors):
            # HbyA
            HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

            # Face flux
            n_internal = mesh.n_internal_faces
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour
            w = mesh.face_weights[:n_internal]

            HbyA_face = (
                w.unsqueeze(-1) * HbyA[int_owner]
                + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
            )
            phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

            # Solve compressible pressure equation
            p, rho = self._solve_pressure_equation(
                p, phiHbyA, A_p, rho, T, mesh
            )

            # Correct velocity
            grad_p = self._compute_grad(p, mesh)
            U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

            # Correct flux
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
            A_p_inv_face = (
                w * gather(A_p_inv, int_owner)
                + (1.0 - w) * gather(A_p_inv, int_neigh)
            )

            # Compressible flux correction includes density
            rho_face = 0.5 * (gather(rho, int_owner) + gather(rho, int_neigh))
            psi_face = 0.5 * (
                gather(self.psi, int_owner) + gather(self.psi, int_neigh)
            )
            phi = phiHbyA - A_p_inv_face * (p_N - p_P) * rho_face / psi_face.clamp(min=1e-30)

            # Recompute H for subsequent corrections
            if corr < self.n_correctors - 1:
                H = self._recompute_H(U, phi, rho)

        # ============================================
        # Step 3: Update density from EOS
        # ============================================
        rho = self.thermo.rho(p, T)

        # ============================================
        # Step 4: Solve energy equation
        # ============================================
        T = self._solve_energy_equation(T, U, phi, rho, p)

        # Final density update
        rho = self.thermo.rho(p, T)

        # ============================================
        # Compute convergence metrics
        # ============================================
        U_residual = self._compute_residual(U, self.U)
        p_residual = self._compute_residual(p, self.p)
        continuity_error = self._compute_continuity_error(phi, rho)

        convergence.p_residual = p_residual
        convergence.U_residual = U_residual
        convergence.continuity_error = continuity_error
        convergence.outer_iterations = 1  # PISO has no outer loop

        return U, p, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # Momentum equation
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with time derivative and TVD convection.

        ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ∇U)

        The convective term uses TVD-limited interpolation for shock
        capturing.
        """
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

        # Viscosity from transport model
        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # TVD-limited convection for density
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)

        # Compute TVD-limited face density
        rho_face_tvd = self._tvd_interpolate(
            rho, flux, int_owner, int_neigh, mesh
        )

        # Split flux for boundedness
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Time derivative contribution: ρ * V / Δt
        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        # Matrix coefficients with TVD face density
        lower = (-diff_coeff + flux_neg * rho_face_tvd) / V_P
        upper = (-diff_coeff - flux_pos * rho_face_tvd) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face_tvd) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face_tvd) / V_N, int_neigh, n_cells
        )

        # Add time derivative to diagonal
        A_p = A_p + rho_V_dt

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Add time derivative source: ρ * V * U_old / dt
        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        source = H - grad_p

        # Solve
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        return U_solved, A_p, H

    def _recompute_H(
        self, U: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """Recompute H(U) from corrected velocity with TVD convection."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        S_mag = mesh.face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]

        # TVD-limited face density
        rho_face_tvd = self._tvd_interpolate(
            rho, flux, int_owner, int_neigh, mesh
        )

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(mesh.cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(mesh.cell_volumes.clamp(min=1e-30), int_neigh)

        lower = (-diff_coeff + flux_neg * rho_face_tvd) / V_P
        upper = (-diff_coeff - flux_pos * rho_face_tvd) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        return H

    # ------------------------------------------------------------------
    # TVD interpolation
    # ------------------------------------------------------------------

    def _tvd_interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor,
        int_owner: torch.Tensor,
        int_neigh: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """TVD-limited interpolation of cell values to internal faces.

        Implements: φ_f = φ_upwind + ψ(r) * (φ_central - φ_upwind)

        where r = (φ_upwind - φ_2upwind) / (φ_downwind - φ_upwind)
        is the ratio of consecutive gradients.

        Parameters
        ----------
        phi : torch.Tensor
            Cell-centre values (n_cells,).
        face_flux : torch.Tensor
            Face flux for upwind direction (n_internal,).
        int_owner : torch.Tensor
            Owner cell indices for internal faces.
        int_neigh : torch.Tensor
            Neighbour cell indices for internal faces.
        mesh : FvMesh
            The finite volume mesh.

        Returns
        -------
        torch.Tensor
            TVD-limited face values (n_internal,).
        """
        device = phi.device
        dtype = phi.dtype

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)

        # Upwind values
        is_positive = face_flux >= 0.0
        phi_upwind = torch.where(is_positive, phi_P, phi_N)
        phi_downwind = torch.where(is_positive, phi_N, phi_P)

        # Central (linear) interpolation
        w = mesh.face_weights[:mesh.n_internal_faces]
        phi_central = w * phi_P + (1.0 - w) * phi_N

        # For 2-cell mesh or when difference is tiny, use central
        denom = phi_downwind - phi_upwind
        safe_denom = torch.where(
            denom.abs() > 1e-30, denom, torch.ones_like(denom) * 1e-30
        )

        # Compute r using gradient-based estimate
        # r = (φ_upwind - φ_2upwind) / (φ_downwind - φ_upwind)
        # Approximate φ_2upwind using cell gradient
        grad_phi = self._compute_grad_scalar(phi, mesh)

        # Upwind cell centres and face centres
        cc_P = gather(mesh.cell_centres, int_owner)
        cc_N = gather(mesh.cell_centres, int_neigh)
        fc = mesh.face_centres[:mesh.n_internal_faces]

        # Distance from upwind cell centre to face
        d_upwind = torch.where(
            is_positive.unsqueeze(-1), fc - cc_P, fc - cc_N,
        )
        grad_upwind = torch.where(
            is_positive.unsqueeze(-1), gather(grad_phi, int_owner),
            gather(grad_phi, int_neigh),
        )

        # Extrapolated value at face from upwind cell gradient
        phi_face_grad = phi_upwind + (grad_upwind * d_upwind).sum(dim=1)

        # r = (extrapolated - upwind) / (central - upwind)
        r = (phi_face_grad - phi_upwind) / safe_denom

        # Apply limiter
        psi = self._tvd_limiter_fn(r)

        # TVD-limited face value
        phi_face = phi_upwind + psi * (phi_central - phi_upwind)

        return phi_face

    def _compute_grad_scalar(
        self, phi: torch.Tensor, mesh: Any
    ) -> torch.Tensor:
        """Compute gradient of scalar field using Gauss theorem."""
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

    # ------------------------------------------------------------------
    # Pressure equation (compressible)
    # ------------------------------------------------------------------

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        mesh: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve compressible pressure equation.

        For compressible flow: ∇·(ρ HbyA) - ∇·(ρ/ψ ∇p) = 0

        This includes the compressibility ψ = 1/(RT).
        """
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        # Face density and compressibility
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = 0.5 * (rho_P + rho_N)

        psi_P = gather(self.psi, int_owner)
        psi_N = gather(self.psi, int_neigh)
        psi_face = 0.5 * (psi_P + psi_N)

        # 1/A_p on faces
        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = (
            w * gather(A_p_inv, int_owner)
            + (1.0 - w) * gather(A_p_inv, int_neigh)
        )

        # Laplacian coefficient: ρ/(ψ * A_p) * |S|/|d|
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = rho_face * A_p_inv_face * S_mag * delta_f / psi_face.clamp(min=1e-30)

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Source: divergence of ρ*HbyA flux
        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

        # Solve using Jacobi iteration
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

        # Update density from EOS after pressure update
        rho = self.thermo.rho(p, T)

        return p, rho

    # ------------------------------------------------------------------
    # Energy equation
    # ------------------------------------------------------------------

    def _solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Solve energy equation with TVD convection.

        ∂(ρe)/∂t + ∇·(ρeU) = -p∇·U + ∇·(α∇e) + Su

        where e = Cv*T is the specific internal energy.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Thermal conductivity
        kappa = self.thermo.kappa(T)
        kappa_face = 0.5 * (gather(kappa, int_owner) + gather(kappa, int_neigh))

        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = kappa_face * S_mag * delta_f

        # TVD-limited convection for temperature
        flux = phi[:n_internal]
        T_face_tvd = self._tvd_interpolate(
            T, flux, int_owner, int_neigh, mesh
        )

        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cp = self.thermo.Cp()
        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Time derivative
        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        # Matrix coefficients
        lower = (-diff_coeff + flux_neg * rho_face * cp) / V_P
        upper = (-diff_coeff - flux_pos * rho_face * cp) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face * cp) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face * cp) / V_N, int_neigh, n_cells
        )

        # Add time derivative
        diag = diag + rho_V_dt * cp

        # Source: viscous dissipation + p·div(U) + time source
        mu = self.thermo.mu(T)
        grad_U = self._compute_grad_vector(U, mesh)
        S_mag_sq = (grad_U * grad_U).sum(dim=(1, 2))
        phi_viscous = mu * S_mag_sq

        div_U = self._compute_div(U, phi, mesh)
        source = phi_viscous + p * div_U + rho_V_dt * cp * self.T_old

        # Solve using Jacobi iteration
        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.T_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            T_P = gather(T, int_owner)
            T_N = gather(T, int_neigh)
            off_diag = off_diag + scatter_add(lower * T_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * T_P, int_neigh, n_cells)

            T_new = (source - off_diag) / diag_safe

            if (T_new - T).abs().max() < self.T_tolerance:
                break
            T = T_new

        return T

    # ------------------------------------------------------------------
    # Gradient and divergence utilities
    # ------------------------------------------------------------------

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
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

    def _compute_grad_vector(self, U: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of vector field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        grad_U = torch.zeros(n_cells, 3, 3, dtype=U.dtype, device=U.device)
        for j in range(3):
            face_contrib = U_face[:, j].unsqueeze(-1) * face_areas
            grad_U[:, :, j].index_add_(0, int_owner, face_contrib)
            grad_U[:, :, j].index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-30)
        return grad_U / V

    def _compute_div(
        self, U: torch.Tensor, phi: torch.Tensor, mesh: Any
    ) -> torch.Tensor:
        """Compute divergence of vector field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = 0.5 * (U_P + U_N)

        flux = (U_face * face_areas).sum(dim=1)

        div = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    # ------------------------------------------------------------------
    # Residual and continuity
    # ------------------------------------------------------------------

    def _compute_residual(
        self, current: torch.Tensor, old: torch.Tensor
    ) -> float:
        """Compute L2 residual between current and old fields."""
        diff = (current - old).abs()
        if diff.dim() > 1:
            diff = diff.norm(dim=1)
        return float(diff.mean().item())

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor
    ) -> float:
        """Compute continuity error for compressible flow."""
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

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
