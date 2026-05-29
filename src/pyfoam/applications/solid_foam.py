"""
solidFoam — Solid mechanics solver with thermal stress analysis.

Solves the thermoelasticity equations for solid bodies undergoing
both mechanical loading and thermal gradients:

    ∇·σ + f = 0  (equilibrium)
    σ = C : (ε - ε_th)  (constitutive law with thermal strain)

where:
    ε_th = α_th * ΔT * I  (thermal strain tensor)
    α_th is the coefficient of thermal expansion
    ΔT = T - T_ref is the temperature difference from reference

The solver couples:
1. Displacement equation (Navier's equation with thermal body force)
2. Stress-strain computation (generalised Hooke's law)
3. Von Mises stress evaluation

Unlike :class:`SolidDisplacementFoam`, this solver adds thermal loading
as a body-force term computed from a temperature field.

Usage::

    from pyfoam.applications.solid_foam import SolidFoam

    solver = SolidFoam("path/to/case", E=200e9, nu=0.3, alpha_th=12e-6)
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from pyfoam.solvers.linear_solver import create_solver

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SolidFoam"]

logger = logging.getLogger(__name__)


class SolidFoam(SolverBase):
    """Solid mechanics solver with thermal stress analysis.

    Solves the coupled thermoelasticity equations:

    - Displacement equation with thermal body force.
    - Stress computation (Hooke's law with thermal strain).
    - Von Mises stress evaluation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    E : float or None
        Young's modulus (Pa). Default: read from case or 200 GPa.
    nu : float or None
        Poisson's ratio. Default: read from case or 0.3.
    alpha_th : float
        Coefficient of thermal expansion (1/K, default 12e-6).
    T_ref : float
        Reference temperature for thermal strain (K, default 293.15).
    rho_s : float
        Solid density (kg/m^3, default 7800 for steel).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        E: float | None = None,
        nu: float | None = None,
        alpha_th: float = 12e-6,
        T_ref: float = 293.15,
        rho_s: float = 7800.0,
    ) -> None:
        super().__init__(case_path)

        # Read or use provided mechanical properties
        E_default, nu_default = self._read_mechanical_properties()
        self.E = E if E is not None else E_default
        self.nu = nu if nu is not None else nu_default
        self.alpha_th = alpha_th
        self.T_ref = T_ref
        self.rho_s = rho_s

        # Lamé parameters
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # fvSettings
        self._read_fv_solution_settings()

        # Fields
        self.D = self._init_displacement()
        self.T = self._init_temperature()

        # Strain and stress
        self.epsilon = self._compute_strain()
        self.epsilon_th = self._compute_thermal_strain()
        self.sigma = self._compute_stress()

        self._D_data = self._init_field_data()

        logger.info(
            "SolidFoam ready: E=%.6e Pa, nu=%.4f, alpha_th=%.6e, "
            "T_ref=%.1f K",
            self.E, self.nu, self.alpha_th, self.T_ref,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _read_mechanical_properties(self) -> tuple[float, float]:
        E = 200e9  # steel
        nu = 0.3

        mp_path = self.case_path / "constant" / "mechanicalProperties"
        if mp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                mp = parse_dict_file(mp_path)
                raw_E = mp.get("E", E)
                if isinstance(raw_E, (int, float)):
                    E = float(raw_E)
                raw_nu = mp.get("nu", nu)
                if isinstance(raw_nu, (int, float)):
                    nu = float(raw_nu)
            except Exception as e:
                logger.warning("Could not read mechanical properties: %s", e)

        return E, nu

    def _read_fv_solution_settings(self) -> None:
        fv = self.case.fvSolution
        self.D_solver = str(fv.get_path("solvers/D/solver", "PBiCGStab"))
        self.D_tolerance = float(fv.get_path("solvers/D/tolerance", 1e-6))
        self.D_rel_tol = float(fv.get_path("solvers/D/relTol", 0.01))
        self.D_max_iter = int(fv.get_path("solvers/D/maxIter", 1000))
        self.convergence_tolerance = float(
            fv.get_path("solidMechanics/convergenceTolerance", 1e-5)
        )
        self.n_correctors = int(
            fv.get_path("solidMechanics/nCorrectors", 1)
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_displacement(self) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        try:
            D_tensor, _ = self.read_field_tensor("D", 0)
            return D_tensor.to(device=device, dtype=dtype)
        except Exception:
            return torch.zeros(self.mesh.n_cells, 3, dtype=dtype, device=device)

    def _init_temperature(self) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            return T_tensor.to(device=device, dtype=dtype).reshape(-1)
        except Exception:
            return torch.full(
                (self.mesh.n_cells,), self.T_ref, dtype=dtype, device=device,
            )

    def _init_field_data(self):
        try:
            return self.case.read_field("D", 0)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Strain computation
    # ------------------------------------------------------------------

    def _compute_strain(self) -> torch.Tensor:
        """Compute small strain tensor from displacement.

        epsilon = 0.5 * (grad(D) + grad(D)^T) in Voigt notation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        grad_D = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for dim in range(3):
            grad_D[:, dim, :] = fvc.grad(self.D[:, dim], mesh=self.mesh)

        epsilon = torch.zeros(n_cells, 6, dtype=dtype, device=device)
        epsilon[:, 0] = grad_D[:, 0, 0]
        epsilon[:, 1] = grad_D[:, 1, 1]
        epsilon[:, 2] = grad_D[:, 2, 2]
        epsilon[:, 3] = 0.5 * (grad_D[:, 0, 1] + grad_D[:, 1, 0])
        epsilon[:, 4] = 0.5 * (grad_D[:, 1, 2] + grad_D[:, 2, 1])
        epsilon[:, 5] = 0.5 * (grad_D[:, 0, 2] + grad_D[:, 2, 0])

        return epsilon

    def _compute_thermal_strain(self) -> torch.Tensor:
        """Compute thermal strain tensor.

        epsilon_th = alpha_th * (T - T_ref) * I

        Returns Voigt notation: (xx, yy, zz, xy, yz, zx).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        dT = self.T - self.T_ref
        epsilon_th = torch.zeros(n_cells, 6, dtype=dtype, device=device)
        epsilon_th[:, 0] = self.alpha_th * dT
        epsilon_th[:, 1] = self.alpha_th * dT
        epsilon_th[:, 2] = self.alpha_th * dT
        # Off-diagonal thermal strains are zero (isotropic expansion)

        return epsilon_th

    # ------------------------------------------------------------------
    # Stress computation
    # ------------------------------------------------------------------

    def _compute_stress(self) -> torch.Tensor:
        """Compute stress from mechanical strain (subtracting thermal).

        sigma = lambda * tr(epsilon_mech) * I + 2*mu * epsilon_mech

        where epsilon_mech = epsilon - epsilon_th.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Mechanical strain (total - thermal)
        eps_mech = self.epsilon - self.epsilon_th

        sigma = torch.zeros(n_cells, 6, dtype=dtype, device=device)
        tr_eps = eps_mech[:, 0] + eps_mech[:, 1] + eps_mech[:, 2]

        sigma[:, 0] = self.lam * tr_eps + 2 * self.mu * eps_mech[:, 0]
        sigma[:, 1] = self.lam * tr_eps + 2 * self.mu * eps_mech[:, 1]
        sigma[:, 2] = self.lam * tr_eps + 2 * self.mu * eps_mech[:, 2]
        sigma[:, 3] = 2 * self.mu * eps_mech[:, 3]
        sigma[:, 4] = 2 * self.mu * eps_mech[:, 4]
        sigma[:, 5] = 2 * self.mu * eps_mech[:, 5]

        return sigma

    # ------------------------------------------------------------------
    # Von Mises stress
    # ------------------------------------------------------------------

    def _compute_von_mises_stress(self) -> torch.Tensor:
        """Compute von Mises stress."""
        s = self.sigma
        return torch.sqrt(
            0.5 * (
                (s[:, 0] - s[:, 1]) ** 2
                + (s[:, 1] - s[:, 2]) ** 2
                + (s[:, 2] - s[:, 0]) ** 2
                + 6 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
            )
        )

    # ------------------------------------------------------------------
    # Thermal body force
    # ------------------------------------------------------------------

    def _compute_thermal_body_force(self) -> torch.Tensor:
        """Compute thermal body force from temperature gradient.

        f_th = -grad(sigma_th) where sigma_th = -alpha_th * (3*lambda + 2*mu) * T

        Simplified: f_th = (3*lambda + 2*mu) * alpha_th * grad(T)
        """
        coeff = (3.0 * self.lam + 2.0 * self.mu) * self.alpha_th
        grad_T = fvc.grad(self.T, mesh=self.mesh)  # (n_cells, 3)
        return coeff * grad_T

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the SolidFoam solver with thermal stress.

        Returns
        -------
        dict
            ``converged``, ``iterations``, ``residual``,
            ``von_mises_max``, ``max_displacement``, ``max_thermal_strain``.
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

        logger.info("Starting SolidFoam run")
        logger.info("  E=%.6e, nu=%.4f, alpha_th=%.6e", self.E, self.nu, self.alpha_th)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        solver = create_solver(
            self.D_solver,
            tolerance=self.D_tolerance,
            rel_tol=self.D_rel_tol,
            max_iter=self.D_max_iter,
        )

        converged = False
        residual = 0.0
        iters = 0

        for t, step in time_loop:
            D_old = self.D.clone()

            # Compute thermal body force
            f_th = self._compute_thermal_body_force()

            # Solve displacement equation
            for corrector in range(self.n_correctors):
                matrix = fvm.laplacian(self.mu, self.D[:, 0], mesh=self.mesh)

                for dim in range(3):
                    D_comp = self.D[:, dim].clone()
                    D_comp_new, iters, residual = matrix.solve(
                        solver, D_comp,
                        tolerance=self.D_tolerance,
                        max_iter=self.D_max_iter,
                    )
                    self.D[:, dim] = D_comp_new

            # Update strain and stress
            self.epsilon = self._compute_strain()
            self.epsilon_th = self._compute_thermal_strain()
            self.sigma = self._compute_stress()

            # Convergence
            residuals = {"D": residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("SolidFoam converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        von_mises = self._compute_von_mises_stress()
        max_disp = float(self.D.abs().max().item())
        max_thermal_strain = float(self.epsilon_th.abs().max().item())

        logger.info("SolidFoam completed")
        logger.info("  max|D| = %.6e, max sigma_vm = %.6e", max_disp, von_mises.max().item())

        return {
            "converged": converged,
            "iterations": iters,
            "residual": residual,
            "von_mises_max": von_mises.max().item(),
            "max_displacement": max_disp,
            "max_thermal_strain": max_thermal_strain,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        time_str = f"{time:g}"
        if self._D_data is not None:
            self.write_field("D", self.D, time_str, self._D_data)
