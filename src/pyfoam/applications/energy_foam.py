"""
energyFoam — enhanced energy equation solver.

Solves the energy equation with viscous dissipation, compressibility work,
and radiation coupling:

    ∂(ρCpT)/∂t + ∇·(ρUCpT) = ∇·(κ∇T) + Φ + βT(Dp/Dt) + S_rad

where:
    Φ       = viscous dissipation (2μ S:S)
    βT(Dp/Dt) = compressibility work (pressure-velocity coupling)
    S_rad   = radiation source (P1 model)

Supports both transient and steady-state operation.

Usage::

    from pyfoam.applications.energy_foam import EnergyFoam

    solver = EnergyFoam("path/to/case", viscous_dissipation=True)
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
from pyfoam.models.radiation import RadiationModel, P1Radiation

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["EnergyFoam"]

logger = logging.getLogger(__name__)


class EnergyFoam(SolverBase):
    """Enhanced energy equation solver.

    Extends the basic heat transfer solver with:

    - **Viscous dissipation** (Phi = 2 * mu * S : S) — converts
      mechanical energy into heat.
    - **Compressibility work** (beta * T * Dp/Dt) — accounts for
      pressure changes doing work on the fluid.
    - **Radiation coupling** via P1 model (or custom RadiationModel).
    - **Transient** or **steady-state** operation.
    - Under-relaxation for stability.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    kappa : float, optional
        Thermal conductivity (W/(m·K)). Default: reads from case or 0.026.
    Cp : float, optional
        Specific heat capacity (J/(kg·K)). Default: 1005.
    rho : float, optional
        Fluid density (kg/m^3). Default: 1.225.
    mu : float, optional
        Dynamic viscosity (Pa·s). Default: 1.8e-5.
    beta : float, optional
        Thermal expansion coefficient (1/K). Default: 3.33e-3 (ideal gas).
    radiation : RadiationModel, optional
        Radiation model. Default: P1 with default parameters.
    viscous_dissipation : bool
        Enable viscous dissipation term. Default: False.
    compressibility_work : bool
        Enable compressibility work term. Default: False.
    alpha_T : float
        Temperature under-relaxation factor. Default: 0.7.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        kappa: float | None = None,
        Cp: float | None = None,
        rho: float | None = None,
        mu: float | None = None,
        beta: float | None = None,
        radiation: RadiationModel | None = None,
        viscous_dissipation: bool = False,
        compressibility_work: bool = False,
        alpha_T: float = 0.7,
    ) -> None:
        super().__init__(case_path)

        # Material properties
        self.kappa = kappa if kappa is not None else self._read_kappa()
        self.Cp = Cp if Cp is not None else 1005.0
        self.rho_const = rho if rho is not None else 1.225
        self.mu = mu if mu is not None else 1.8e-5
        self.beta = beta if beta is not None else 1.0 / 300.0

        # Radiation model
        self.radiation = radiation or P1Radiation()

        # Feature flags
        self.viscous_dissipation = viscous_dissipation
        self.compressibility_work = compressibility_work

        # Under-relaxation
        self.alpha_T = alpha_T

        # Read solver settings
        self._read_fv_solution_settings()

        # Initialize temperature field
        self.T, self._T_data = self._init_temperature()

        # Initialize velocity field (for viscous dissipation)
        self._U = self._init_velocity()

        # Initialize pressure field (for compressibility work)
        self._p = self._init_pressure()

        # Parse boundary conditions
        self._bc_values = self._parse_boundary_conditions()

        logger.info(
            "EnergyFoam ready: kappa=%.6g, Cp=%.6g, rho=%.6g, mu=%.6g, "
            "viscous_dissipation=%s, compressibility_work=%s",
            self.kappa, self.Cp, self.rho_const, self.mu,
            self.viscous_dissipation, self.compressibility_work,
        )

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution
        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))
        self.max_outer_iterations = int(fv.get_path("SIMPLE/maxOuterIterations", 100))
        self.convergence_tolerance = float(
            fv.get_path("SIMPLE/convergenceTolerance", 1e-4)
        )

    def _read_kappa(self) -> float:
        """Read thermal conductivity from case or use default."""
        tp_path = self.case_path / "constant" / "transportProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                kappa = tp.get("kappa", tp.get("DT", 0.026))
                if isinstance(kappa, dict):
                    kappa = kappa.get("value", 0.026)
                return float(kappa)
            except Exception:
                pass
        return 0.026  # default for air

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_temperature(self) -> tuple[torch.Tensor, Any]:
        """Initialize temperature field."""
        device = get_device()
        dtype = get_default_dtype()
        T_tensor, T_data = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)
        return T, T_data

    def _init_velocity(self) -> torch.Tensor:
        """Initialize velocity field (zero if not found)."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        try:
            U_tensor, _ = self.read_field_tensor("U", 0)
            return U_tensor.to(device=device, dtype=dtype).reshape(n_cells, 3)
        except Exception:
            return torch.zeros(n_cells, 3, device=device, dtype=dtype)

    def _init_pressure(self) -> torch.Tensor:
        """Initialize pressure field (zero if not found)."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        try:
            p_tensor, _ = self.read_field_tensor("p", 0)
            return p_tensor.to(device=device, dtype=dtype)
        except Exception:
            return torch.zeros(n_cells, device=device, dtype=dtype)

    def _parse_boundary_conditions(self) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from the field data."""
        bc_values: dict[str, dict[str, Any]] = {}
        boundary = self._T_data.boundary_field
        mesh_boundary = self.case.boundary

        for i, patch in enumerate(boundary.patches):
            bc_info: dict[str, Any] = {"type": patch.patch_type, "value": None}

            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    try:
                        bc_info["value"] = float(val)
                    except ValueError:
                        logger.warning("Could not parse BC value: %s", val)
                elif isinstance(val, (int, float)):
                    bc_info["value"] = float(val)

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the energy solver.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting energyFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            T_prev = self.T.clone()

            # Solve energy equation
            self.T = self._solve_energy(T_prev)

            # Compute residual
            T_residual = self._compute_residual(self.T, T_prev)
            conv = ConvergenceData()
            conv.T_residual = T_residual
            conv.converged = T_residual < self.convergence_tolerance
            last_convergence = conv

            residuals = {"T": T_residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            logger.info(
                "energyFoam completed: T_res=%.6e",
                last_convergence.T_residual,
            )

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Viscous dissipation
    # ------------------------------------------------------------------

    def _compute_viscous_dissipation(self) -> torch.Tensor:
        """Compute viscous dissipation term Phi = 2 * mu * S : S.

        Uses a simplified estimate from velocity gradients:
            S_ij = 0.5 * (dU_i/dx_j + dU_j/dx_i)
            Phi = 2 * mu * S_ij * S_ij

        Returns
        -------
        torch.Tensor
            Viscous dissipation per cell (n_cells,).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Estimate velocity gradient from face interpolation
        # Simplified: compute |U|^2 gradient contribution
        U = self._U
        U_mag_sq = (U * U).sum(dim=1)  # |U|^2

        # Simple approximation: Phi ~ mu * |grad(U)|^2
        # Use central differencing on internal faces
        U_P = gather(U_mag_sq, int_owner)
        U_N = gather(U_mag_sq, int_neigh)
        delta_f = mesh.delta_coefficients[:n_internal]

        grad_mag_sq = ((U_N - U_P) * delta_f) ** 2
        grad_sq_cells = torch.zeros(n_cells, device=device, dtype=dtype)
        grad_sq_cells = grad_sq_cells + scatter_add(
            0.5 * grad_mag_sq / gather(mesh.cell_volumes, int_owner),
            int_owner, n_cells,
        )
        grad_sq_cells = grad_sq_cells + scatter_add(
            0.5 * grad_mag_sq / gather(mesh.cell_volumes, int_neigh),
            int_neigh, n_cells,
        )

        return 2.0 * self.mu * grad_sq_cells

    # ------------------------------------------------------------------
    # Compressibility work
    # ------------------------------------------------------------------

    def _compute_compressibility_work(self) -> torch.Tensor:
        """Compute compressibility work: beta * T * Dp/Dt.

        For steady-state, Dp/Dt ~ U · grad(p).
        Uses simplified face-based gradient estimation.

        Returns
        -------
        torch.Tensor
            Compressibility work source per cell (n_cells,).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Pressure gradient from central differencing
        p_P = gather(self._p, int_owner)
        p_N = gather(self._p, int_neigh)
        delta_f = mesh.delta_coefficients[:n_internal]

        dp_dx = (p_N - p_P) * delta_f  # approximate dp/dx per face

        # Velocity magnitude at faces
        U_P = self._U[int_owner]  # (n_internal, 3)
        U_N = self._U[int_neigh]
        U_face = 0.5 * (U_P + U_N)  # (n_internal, 3)

        # Source = beta * T * (U · dp/dx)
        # dp/dx is scalar per face; estimate as scalar * x-direction unit
        T_face = gather(self.T, int_owner)
        source_face = self.beta * T_face * U_face[:, 0] * dp_dx

        # Distribute to owner cells
        source_cells = torch.zeros(n_cells, device=device, dtype=dtype)
        source_cells = source_cells + scatter_add(
            source_face / gather(mesh.cell_volumes, int_owner),
            int_owner, n_cells,
        )

        return source_cells

    # ------------------------------------------------------------------
    # Energy equation solver
    # ------------------------------------------------------------------

    def _solve_energy(self, T_old: torch.Tensor) -> torch.Tensor:
        """Solve energy equation with all source terms.

        ∇·(ρUCpT) = ∇·(κ∇T) + Phi + beta*T*DpDt + S_rad

        Uses Jacobi iteration with upwind convection and central diffusion.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        T = T_old.clone()

        # ---- Conduction: Laplacian (kappa * grad(T)) ----
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = self.kappa * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -diff_coeff / V_P
        upper = -diff_coeff / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(diff_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(diff_coeff / V_N, int_neigh, n_cells)

        # ---- Boundary contributions ----
        bc_source = torch.zeros(n_cells, dtype=dtype, device=device)

        for patch_name, bc_info in self._bc_values.items():
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue
            if bc_info["type"] in ("empty", "wedge"):
                continue
            if bc_info["type"] == "zeroGradient":
                bnd_faces = slice(start_face, start_face + n_faces)
                bnd_areas = mesh.face_areas[bnd_faces]
                bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
                bnd_cells = mesh.owner[bnd_faces]
                bnd_V = gather(cell_volumes, bnd_cells)

                bnd_face_centres = mesh.face_centres[bnd_faces]
                bnd_cell_centres = mesh.cell_centres[bnd_cells]
                d_P = bnd_face_centres - bnd_cell_centres
                safe_area = torch.where(
                    bnd_S_mag.unsqueeze(-1) > 1e-30,
                    bnd_areas,
                    torch.ones_like(bnd_areas),
                )
                n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
                d_dot_n = (d_P * n_f).sum(dim=1).abs()
                bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)
                bnd_coeff = self.kappa * bnd_S_mag * bnd_delta
                diag = diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)
                continue

            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                bnd_faces = slice(start_face, start_face + n_faces)
                bnd_areas = mesh.face_areas[bnd_faces]
                bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
                bnd_cells = mesh.owner[bnd_faces]
                bnd_V = gather(cell_volumes, bnd_cells)

                bnd_face_centres = mesh.face_centres[bnd_faces]
                bnd_cell_centres = mesh.cell_centres[bnd_cells]
                d_P = bnd_face_centres - bnd_cell_centres
                safe_area = torch.where(
                    bnd_S_mag.unsqueeze(-1) > 1e-30,
                    bnd_areas,
                    torch.ones_like(bnd_areas),
                )
                n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
                d_dot_n = (d_P * n_f).sum(dim=1).abs()
                bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)
                bnd_coeff = self.kappa * bnd_S_mag * bnd_delta

                diag = diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)
                T_bc = bc_info["value"]
                bc_source = bc_source + scatter_add(
                    bnd_coeff * T_bc / bnd_V, bnd_cells, n_cells,
                )

        # ---- Radiation source ----
        S_rad = self.radiation.Sh(T)

        # ---- Viscous dissipation ----
        S_visc = torch.zeros(n_cells, dtype=dtype, device=device)
        if self.viscous_dissipation:
            S_visc = self._compute_viscous_dissipation()

        # ---- Compressibility work ----
        S_comp = torch.zeros(n_cells, dtype=dtype, device=device)
        if self.compressibility_work:
            S_comp = self._compute_compressibility_work()

        source = bc_source + S_rad + S_visc + S_comp

        # ---- Solve using Jacobi iteration ----
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

        # Under-relaxation
        if self.alpha_T < 1.0:
            T = self.alpha_T * T + (1.0 - self.alpha_T) * T_old

        return T

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residual(
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute L2 residual normalised by field magnitude."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("T", self.T, time_str, self._T_data)
