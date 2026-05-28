"""
adjointTurbulenceFoam — adjoint turbulence optimisation solver.

Extends the continuous adjoint method to include turbulence model
sensitivities for gradient-based design optimisation.  Solves the
adjoint turbulence equations in addition to the adjoint
Navier-Stokes equations, enabling the computation of sensitivities
that account for the effect of turbulence model parameters on the
objective function.

The adjoint turbulence equations are obtained by differentiating
the RANS equations with respect to the turbulent viscosity field:

    Adjoint k equation:
        -(U·∇)ka - ∇·((ν + ν_t/σ_k)∇ka) + β*ω*ka = S_k^a

    Adjoint ω equation:
        -(U·∇)ωa - ∇·((ν + ν_t/σ_ω)∇ωa) + 2σ_ω2*ω*ωa = S_ω^a

where ka, ωa are adjoint turbulent quantities, and S_k^a, S_ω^a
are source terms derived from the functional derivative of the
objective with respect to the turbulence model.

Algorithm (per outer iteration):
1. Solve adjoint momentum (SIMPLE-like) — inherited from AdjointFoam
2. Solve adjoint k equation
3. Solve adjoint omega equation
4. Compute turbulence sensitivity
5. Correct adjoint velocity from adjoint pressure
6. Check convergence

Usage::

    from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

    solver = AdjointTurbulenceFoam("path/to/case")
    result = solver.run()

    # Access turbulence sensitivity
    turb_sensitivity = solver.turbulence_sensitivity
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .adjoint_foam import AdjointFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["AdjointTurbulenceFoam"]

logger = logging.getLogger(__name__)


class AdjointTurbulenceFoam(AdjointFoam):
    """Adjoint turbulence optimisation solver.

    Extends :class:`AdjointFoam` with adjoint turbulence equations
    for k and omega, enabling the computation of design sensitivities
    that account for the turbulence model.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    objective : str, optional
        Objective function type. ``"drag"`` (default) minimises drag.
    turb_model : str, optional
        Turbulence model type. ``"kOmegaSST"`` (default).

    Attributes
    ----------
    ka : torch.Tensor
        ``(n_cells,)`` adjoint turbulent kinetic energy.
    omega_a : torch.Tensor
        ``(n_cells,)`` adjoint specific dissipation rate.
    turbulence_sensitivity : torch.Tensor
        ``(n_cells,)`` turbulence sensitivity field.
    k : torch.Tensor
        ``(n_cells,)`` turbulent kinetic energy from primal.
    omega : torch.Tensor
        ``(n_cells,)`` specific dissipation rate from primal.
    nut : torch.Tensor
        ``(n_cells,)`` turbulent viscosity from primal.
    """

    # kOmegaSST model constants (Menter 2003)
    _SIGMA_K: float = 0.85
    _SIGMA_W: float = 0.5
    _BETA: float = 0.075
    _SIGMA_D: float = 0.5  # cross-diffusion coefficient

    def __init__(
        self,
        case_path: Union[str, Path],
        objective: str = "drag",
        turb_model: str = "kOmegaSST",
    ) -> None:
        # Call SolverBase.__init__ directly (skip AdjointFoam.__init__
        # because we need to add turbulence fields before the adjoint
        # field initialisation).
        super(AdjointFoam, self).__init__(case_path)

        self.nu = self._read_nu()
        self.objective = objective
        self.turb_model = turb_model

        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Primal fields
        self.U, self.p, self.phi_primal = self._init_primal_fields()

        # Turbulence fields from primal
        self.k, self.omega, self.nut = self._init_turbulence_fields()

        # Adjoint fields
        self.Ua, self.pa = self._init_adjoint_fields()

        # Adjoint turbulence fields
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self.ka = torch.zeros(n_cells, dtype=dtype, device=device)
        self.omega_a = torch.zeros(n_cells, dtype=dtype, device=device)

        # Sensitivity fields
        self.sensitivity = torch.zeros(n_cells, dtype=dtype, device=device)
        self.turbulence_sensitivity = torch.zeros(
            n_cells, dtype=dtype, device=device,
        )

        self._U_data, self._p_data, self._Ua_data, self._pa_data = \
            self._init_field_data()

        logger.info(
            "AdjointTurbulenceFoam ready: nu=%.6e, objective=%s, turb=%s",
            self.nu, objective, turb_model,
        )

    # ------------------------------------------------------------------
    # Turbulence field initialisation
    # ------------------------------------------------------------------

    def _init_turbulence_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise turbulence fields k, omega, nut from 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        try:
            k_tensor, _ = self.read_field_tensor("k", 0)
            k = k_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            k = torch.full((n_cells,), 1e-4, dtype=dtype, device=device)

        try:
            omega_tensor, _ = self.read_field_tensor("omega", 0)
            omega = omega_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            omega = torch.full((n_cells,), 1.0, dtype=dtype, device=device)

        # Compute turbulent viscosity: nut = k / omega
        nut = (k / omega.clamp(min=1e-10)).to(device=device, dtype=dtype)

        return k, omega, nut

    # ------------------------------------------------------------------
    # Turbulence model coefficients (read from fvSolution if available)
    # ------------------------------------------------------------------

    def _read_turb_coeffs(self) -> None:
        """Read turbulence model coefficients from fvSolution."""
        fv = self.case.fvSolution
        self._SIGMA_K = float(fv.get_path("turbulence/sigmaK", self._SIGMA_K))
        self._SIGMA_W = float(fv.get_path("turbulence/sigmaW", self._SIGMA_W))
        self._BETA = float(fv.get_path("turbulence/beta", self._BETA))
        self._SIGMA_D = float(fv.get_path("turbulence/sigmaD", self._SIGMA_D))

    # ------------------------------------------------------------------
    # Adjoint turbulence equation assembly
    # ------------------------------------------------------------------

    def _solve_adjoint_k(self) -> torch.Tensor:
        """Solve the adjoint turbulent kinetic energy equation.

        -(U·∇)ka - ∇·((nu + nut/sigma_k)∇ka) + beta*omega*ka = S_k^a

        The source term S_k^a is derived from the objective function
        derivative with respect to nut (through k).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` updated adjoint k.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        face_areas = mesh.face_areas

        # Effective diffusivity: nu + nut/sigma_k
        nu_eff = self.nu + self.nut / self._SIGMA_K

        # Face-interpolated diffusivity (harmonic mean for internal faces)
        nu_P = gather(nu_eff, int_owner)
        nu_N = gather(nu_eff, int_neigh)
        nu_face = 2.0 * nu_P * nu_N / (nu_P + nu_N).clamp(min=1e-30)

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = nu_face * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        # Diffusion coefficients
        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        # Diagonal from diffusion
        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Damping term: beta * omega
        diag = diag + self._BETA * self.omega * cell_volumes

        # Convection (upwind)
        n_faces = mesh.n_faces
        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N
        if n_faces > n_internal:
            U_face[n_internal:] = self.U[mesh.owner[n_internal:]]
        phi = (U_face * face_areas).sum(dim=1)

        flux = phi[:n_internal]
        is_pos = flux >= 0.0
        flux_pos = torch.where(is_pos, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_pos, flux, torch.zeros_like(flux))

        conv_lower = flux_neg / V_P
        conv_upper = flux_pos / V_N

        diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_conv = diag_conv + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag_conv = diag_conv + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        diag = diag + diag_conv

        # Source: from turbulence model sensitivity
        source = self._turbulence_k_source()

        # Jacobi iteration
        diag_safe = diag.abs().clamp(min=1e-30)
        ka = self.ka.clone()

        for _ in range(self.Ua_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            ka_P = gather(ka, int_owner)
            ka_N = gather(ka, int_neigh)
            combined_lower = lower + conv_lower
            combined_upper = upper + conv_upper
            off_diag = off_diag + scatter_add(
                combined_lower * ka_N, int_owner, n_cells,
            )
            off_diag = off_diag + scatter_add(
                combined_upper * ka_P, int_neigh, n_cells,
            )

            ka_new = (source - off_diag) / diag_safe

            if (ka_new - ka).abs().max() < self.Ua_tolerance:
                ka = ka_new
                break
            ka = ka_new

        return ka

    def _solve_adjoint_omega(self) -> torch.Tensor:
        """Solve the adjoint specific dissipation rate equation.

        -(U·∇)ωa - ∇·((nu + nut/sigma_w)∇ωa) + 2*sigma_d/omega*ωa = S_ω^a

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` updated adjoint omega.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        face_areas = mesh.face_areas

        # Effective diffusivity: nu + nut/sigma_w
        nu_eff = self.nu + self.nut / self._SIGMA_W

        nu_P = gather(nu_eff, int_owner)
        nu_N = gather(nu_eff, int_neigh)
        nu_face = 2.0 * nu_P * nu_N / (nu_P + nu_N).clamp(min=1e-30)

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = nu_face * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Damping: 2 * sigma_d / omega (cross-diffusion stabilization)
        diag = diag + 2.0 * self._SIGMA_D * cell_volumes

        # Convection (upwind)
        n_faces = mesh.n_faces
        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N
        if n_faces > n_internal:
            U_face[n_internal:] = self.U[mesh.owner[n_internal:]]
        phi = (U_face * face_areas).sum(dim=1)

        flux = phi[:n_internal]
        is_pos = flux >= 0.0
        flux_pos = torch.where(is_pos, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_pos, flux, torch.zeros_like(flux))

        conv_lower = flux_neg / V_P
        conv_upper = flux_pos / V_N

        diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_conv = diag_conv + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag_conv = diag_conv + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        diag = diag + diag_conv

        # Source
        source = self._turbulence_omega_source()

        diag_safe = diag.abs().clamp(min=1e-30)
        omega_a = self.omega_a.clone()

        for _ in range(self.Ua_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            oa_P = gather(omega_a, int_owner)
            oa_N = gather(omega_a, int_neigh)
            combined_lower = lower + conv_lower
            combined_upper = upper + conv_upper
            off_diag = off_diag + scatter_add(
                combined_lower * oa_N, int_owner, n_cells,
            )
            off_diag = off_diag + scatter_add(
                combined_upper * oa_P, int_neigh, n_cells,
            )

            omega_a_new = (source - off_diag) / diag_safe

            if (omega_a_new - omega_a).abs().max() < self.Ua_tolerance:
                omega_a = omega_a_new
                break
            omega_a = omega_a_new

        return omega_a

    # ------------------------------------------------------------------
    # Turbulence source terms
    # ------------------------------------------------------------------

    def _turbulence_k_source(self) -> torch.Tensor:
        """Compute source term for adjoint k equation.

        S_k^a = dJ/dk ≈ nut/k * P_k (production sensitivity)

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` source for adjoint k equation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Simplified: source proportional to production rate
        # P_k = nut * |S|^2 where S is the strain rate magnitude
        # dJ/dk = -dJ/d(nut) * d(nut)/dk = -dJ/d(nut) / omega
        # For drag minimisation, source on boundary-adjacent cells
        source = self._objective_source()
        source_mag = source.norm(dim=1)

        # Scale by nut/k (sensitivity of nut to k)
        k_safe = self.k.clamp(min=1e-10)
        turb_source = source_mag * self.nut / (k_safe * self.omega.clamp(min=1e-10))

        return turb_source

    def _turbulence_omega_source(self) -> torch.Tensor:
        """Compute source term for adjoint omega equation.

        S_ω^a ≈ nut/omega * production sensitivity

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` source for adjoint omega equation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        source = self._objective_source()
        source_mag = source.norm(dim=1)

        omega_safe = self.omega.clamp(min=1e-10)
        turb_source = -source_mag * self.nut / (self.k.clamp(min=1e-10) * omega_safe)

        return turb_source

    # ------------------------------------------------------------------
    # Turbulence sensitivity computation
    # ------------------------------------------------------------------

    def _compute_turbulence_sensitivity(self) -> torch.Tensor:
        """Compute turbulence model sensitivity field.

        Combines adjoint k and omega to determine how sensitive
        the objective is to changes in the turbulence field.

        turbulence_sensitivity_i = ka_i * k_i + ωa_i * ω_i

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulence sensitivity.
        """
        return self.ka * self.k + self.omega_a * self.omega

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the adjoint turbulence solver.

        Solves the coupled adjoint Navier-Stokes and adjoint
        turbulence equations.  After convergence, computes both
        shape and turbulence sensitivity fields.

        Returns
        -------
        ConvergenceData
            Final convergence information.
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

        logger.info("Starting adjointTurbulenceFoam run")

        Ua_bc = self._build_adjoint_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        for t, step in time_loop:
            # Solve adjoint momentum (inherited from AdjointFoam)
            self.Ua = self._solve_adjoint_momentum(Ua_bc)
            Ua_old = self.Ua.clone()

            # Solve adjoint pressure
            self.pa = self._solve_adjoint_pressure(Ua_bc)
            pa_old = self.pa.clone()
            self.pa = self.alpha_pa * self.pa + (1 - self.alpha_pa) * pa_old

            # Solve adjoint turbulence equations
            self.ka = self._solve_adjoint_k()
            self.omega_a = self._solve_adjoint_omega()

            # Correct adjoint velocity
            self._correct_adjoint_velocity()

            # Compute residuals
            Ua_residual = float(
                (self.Ua - Ua_old).norm() / (self.Ua.norm().clamp(min=1e-30))
            )
            pa_residual = float(
                (self.pa - pa_old).norm() / (self.pa.norm().clamp(min=1e-30))
            )

            conv = ConvergenceData()
            conv.U_residual = Ua_residual
            conv.p_residual = pa_residual

            residuals = {"Ua": Ua_residual, "pa": pa_residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info(
                    "Adjoint turbulence converged at step %d (t=%.6g)",
                    step + 1, t,
                )
                break

        # Compute sensitivities
        self.sensitivity = self._compute_sensitivity()
        self.turbulence_sensitivity = self._compute_turbulence_sensitivity()

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("adjointTurbulenceFoam completed")
        logger.info(
            "  shape sensitivity range: [%.6e, %.6e]",
            self.sensitivity.min().item(),
            self.sensitivity.max().item(),
        )
        logger.info(
            "  turbulence sensitivity range: [%.6e, %.6e]",
            self.turbulence_sensitivity.min().item(),
            self.turbulence_sensitivity.max().item(),
        )

        conv.converged = converged if converged else False
        return conv
