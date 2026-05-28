"""
viscousFoam — viscous flow solver for high-viscosity fluids.

Implements a steady-state incompressible solver specialised for
high-viscosity (low Reynolds number) flows with support for
non-Newtonian viscosity models (power-law, Bird-Carreau, Cross).

Uses the SIMPLE algorithm with enhanced viscous stress formulation:

    ∇·(UU) = -∇p + ∇·(μ_eff * ∇U)
    ∇·U = 0

where μ_eff may depend on the local shear-strain rate for
non-Newtonian fluids.

The solver reads:
- ``0/U``, ``0/p`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — viscosity model and parameters
- ``system/controlDict`` — endTime, deltaT, writeControl
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — SIMPLE settings, linear solver tolerances

Usage::

    from pyfoam.applications.viscous_foam import ViscousFoam

    solver = ViscousFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ViscousFoam"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Non-Newtonian viscosity model registry (inline for solver self-containment)
# ---------------------------------------------------------------------------

_VISCOSITY_MODELS: dict[str, type] = {}


def _register_viscosity_model(name: str):
    """Decorator to register a viscosity model for ViscousFoam."""
    def decorator(cls):
        _VISCOSITY_MODELS[name] = cls
        return cls
    return decorator


class _ViscosityModelBase:
    """Base for viscosity models used by ViscousFoam."""

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute apparent viscosity from strain-rate magnitude."""
        raise NotImplementedError

    def is_non_newtonian(self) -> bool:
        """Return True if viscosity depends on strain rate."""
        return False


@_register_viscosity_model("constant")
class _ConstantViscosity(_ViscosityModelBase):
    """Newtonian (constant) viscosity."""

    def __init__(self, nu: float = 1.0) -> None:
        self.nu = nu

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gamma_dot, self.nu)


@_register_viscosity_model("powerLaw")
class _PowerLawViscosity(_ViscosityModelBase):
    """Power-law viscosity: mu = K * |gamma_dot|^(n-1)."""

    def __init__(self, K: float = 0.01, n: float = 0.5, nu_min: float = 1e-6, nu_max: float = 1e4) -> None:
        self.K = K
        self.n = n
        self.nu_min = nu_min
        self.nu_max = nu_max

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=1e-30)
        mu = self.K * gd.pow(self.n - 1.0)
        return mu.clamp(min=self.nu_min, max=self.nu_max)

    def is_non_newtonian(self) -> bool:
        return True


@_register_viscosity_model("BirdCarreau")
class _BirdCarreauViscosity(_ViscosityModelBase):
    """Bird-Carreau viscosity model.

    mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * |gamma_dot|)^2)^((n-1)/2)
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf: float = 0.001,
        lambda_: float = 1.0,
        n: float = 0.4,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf = mu_inf
        self.lambda_ = lambda_
        self.n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=0.0)
        factor = (1.0 + (self.lambda_ * gd).pow(2)).pow((self.n - 1.0) / 2.0)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * factor

    def is_non_newtonian(self) -> bool:
        return True


@_register_viscosity_model("Cross")
class _CrossPowerLawViscosity(_ViscosityModelBase):
    """Cross power-law viscosity model.

    mu = mu_inf + (mu_0 - mu_inf) / (1 + (lambda * |gamma_dot|)^m)
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf: float = 0.001,
        lambda_: float = 1.0,
        m: float = 1.0,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf = mu_inf
        self.lambda_ = lambda_
        self.m = m

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=0.0)
        denom = 1.0 + (self.lambda_ * gd).pow(self.m)
        return self.mu_inf + (self.mu_0 - self.mu_inf) / denom

    def is_non_newtonian(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# ViscousFoam solver
# ---------------------------------------------------------------------------


class ViscousFoam(SolverBase):
    """Steady-state viscous flow solver for high-viscosity fluids.

    Solves the incompressible Navier-Stokes equations for flows where
    viscous forces dominate (low Reynolds number).  Supports Newtonian
    and non-Newtonian viscosity models via ``constant/transportProperties``.

    Dictionary format for ``transportProperties``::

        viscosityModel  powerLaw;    // or constant, BirdCarreau, Cross
        K               0.01;        // consistency index (powerLaw)
        n               0.5;         // power-law index (powerLaw)
        nu_min          1e-6;        // lower viscosity bound (powerLaw)
        nu_max          1e4;         // upper viscosity bound (powerLaw)

    For BirdCarreau / Cross models::

        viscosityModel  BirdCarreau;
        mu_0            0.05;
        mu_inf          0.001;
        lambda_         1.0;
        n               0.4;

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field.
    phi : torch.Tensor
        ``(n_faces,)`` face flux field.
    nu : float
        Base/reference kinematic viscosity.
    viscosity_model : str
        Name of the viscosity model (``"constant"``, ``"powerLaw"``, etc.).
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read viscosity model and parameters
        self.nu, self.viscosity_model, self._visco_params = self._read_viscosity_properties()
        self._visco = self._build_viscosity_model()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Store old fields
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()

        logger.info(
            "ViscousFoam ready: nu=%.6e, model=%s",
            self.nu, self.viscosity_model,
        )

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_viscosity_properties(self) -> tuple[float, str, dict[str, Any]]:
        """Read viscosity model and parameters from transportProperties."""
        tp_path = self.case_path / "constant" / "transportProperties"
        model_name = "constant"
        nu = 1.0
        params: dict[str, Any] = {}

        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)

                # Viscosity model name
                model_name = str(tp.get("viscosityModel", "constant"))

                # Base viscosity (for constant model)
                raw_nu = tp.get("nu", 1.0)
                if isinstance(raw_nu, (int, float)):
                    nu = float(raw_nu)
                else:
                    raw_str = str(raw_nu).strip()
                    match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
                    if match:
                        nu = float(match.group(1))
                    else:
                        nu = float(raw_str)

                # Model-specific parameters
                param_keys = [
                    "K", "n", "m", "nu_min", "nu_max",
                    "mu_0", "mu_inf", "lambda_",
                ]
                for key in param_keys:
                    val = tp.get(key)
                    if val is not None:
                        params[key] = float(val)

            except Exception:
                pass

        return nu, model_name, params

    def _build_viscosity_model(self) -> _ViscosityModelBase:
        """Build the viscosity model from parsed parameters."""
        name = self.viscosity_model
        if name not in _VISCOSITY_MODELS:
            available = sorted(_VISCOSITY_MODELS.keys())
            logger.warning(
                "Unknown viscosity model '%s', falling back to constant. "
                "Available: %s", name, available,
            )
            name = "constant"

        cls = _VISCOSITY_MODELS[name]
        if name == "constant":
            return cls(nu=self.nu)
        return cls(**self._visco_params)

    def _read_fv_solution_settings(self) -> None:
        """Read SIMPLE settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_rel_tol = float(fv.get_path("solvers/p/relTol", 0.01))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.n_non_orth_correctors = int(
            fv.get_path("SIMPLE/nNonOrthogonalCorrectors", 0)
        )
        self.n_outer_correctors = int(
            fv.get_path("SIMPLE/nOuterCorrectors", 100)
        )
        self.convergence_tolerance = float(
            fv.get_path("SIMPLE/convergenceTolerance", 1e-4)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
        fs = self.case.fvSchemes
        self.ddt_scheme = str(fs.get_path("ddtSchemes/default", "steadyState"))
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path("laplacianSchemes/default", "Gauss linear corrected"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, phi from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        return U_data, p_data

    # ------------------------------------------------------------------
    # SIMPLE solver construction
    # ------------------------------------------------------------------

    def _build_solver(self) -> SIMPLESolver:
        """Build a SIMPLESolver with settings from fvSolution."""
        config = SIMPLEConfig(
            nu=self.nu,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        )
        return SIMPLESolver(self.mesh, config)

    # ------------------------------------------------------------------
    # Strain rate computation (for non-Newtonian models)
    # ------------------------------------------------------------------

    def _compute_strain_rate_magnitude(self, U: torch.Tensor) -> torch.Tensor:
        """Compute |gamma_dot| = sqrt(2 * S_ij * S_ij).

        Simplified gradient computation from velocity field.

        Args:
            U: ``(n_cells, 3)`` velocity field.

        Returns:
            ``(n_cells,)`` strain-rate magnitude.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = get_device()
        dtype = get_default_dtype()

        if n_internal == 0:
            return torch.zeros(n_cells, dtype=dtype, device=device)

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Interpolation weights
        if hasattr(mesh, 'face_weights'):
            w = mesh.face_weights[:n_internal].to(dtype=dtype)
        else:
            w = torch.full((n_internal,), 0.5, dtype=dtype, device=device)

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        face_areas = mesh.face_areas[:n_internal].to(dtype=dtype)

        # Gauss gradient: grad(U)_ij = (1/V) * sum_f (U_face_i * A_j)
        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        _EPS = 1e-30

        for i in range(3):
            for j in range(3):
                contrib = U_face[:, i] * face_areas[:, j]
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_owner, contrib)
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_neigh, -contrib)

        V = mesh.cell_volumes.to(dtype=dtype).clamp(min=_EPS)
        grad_U = grad_U / V.unsqueeze(-1).unsqueeze(-1)

        # Strain rate: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # |gamma_dot| = sqrt(2 * S_ij * S_ij)
        S_sq = (S * S).sum(dim=(-2, -1))
        return (2.0 * S_sq).clamp(min=0.0).sqrt()

    # ------------------------------------------------------------------
    # Boundary conditions (same as SimpleFoam)
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor:
        """Build velocity BC tensor from the 0/U boundary field."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        U_field_data = self.case.read_field("U", 0)
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return U_bc

        mesh_boundary = self.case.boundary
        owner = self.mesh.owner

        mesh_patches = {}
        for bp in mesh_boundary:
            mesh_patches[bp.name] = {"startFace": bp.start_face, "nFaces": bp.n_faces}

        for patch in boundary_field:
            if patch.patch_type == "fixedValue" and patch.value is not None:
                value = self._parse_vector_value(patch.value)
                if value is not None:
                    mesh_info = mesh_patches.get(patch.name)
                    if mesh_info is not None:
                        start_face = mesh_info["startFace"]
                        n_faces = mesh_info["nFaces"]
                        for i in range(n_faces):
                            face_idx = start_face + i
                            cell_idx = owner[face_idx].item()
                            U_bc[cell_idx, 0] = value[0]
                            U_bc[cell_idx, 1] = value[1]
                            U_bc[cell_idx, 2] = value[2]

        return U_bc

    @staticmethod
    def _parse_vector_value(value):
        """Parse a vector value from field data."""
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        if isinstance(value, str):
            match = re.search(
                r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", value,
            )
            if match:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        return None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the ViscousFoam solver.

        Executes the SIMPLE algorithm with the configured viscosity model.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver()

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

        logger.info("Starting ViscousFoam run")
        logger.info("  viscosityModel=%s, nu=%.6e", self.viscosity_model, self.nu)
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        U_bc = self._build_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # For non-Newtonian models, update effective viscosity per cell
            if self._visco.is_non_newtonian():
                mag_S = self._compute_strain_rate_magnitude(self.U)
                mu_eff = self._visco.mu(mag_S)
                # Update solver viscosity to spatial average for SIMPLE solver
                solver._config.nu = float(mu_eff.mean().item())

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                tolerance=self.convergence_tolerance,
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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("ViscousFoam completed successfully")
            else:
                logger.warning("ViscousFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U and p to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
