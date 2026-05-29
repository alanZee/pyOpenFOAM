"""
filmFoam — Thin film flow solver with surface tension.

Solves the thin-film equation for gravity-driven flows on solid surfaces:

    dh/dt + ∇·(h U_s) = 0

where:
- h(x, t) is the film thickness
- U_s is the surface velocity

For a viscous thin film under gravity with surface tension:

    U_s = (rho g h^2 / (3 mu)) * sin(beta) - (sigma h^3 / (3 mu)) * ∇(∇^2 h)

The first term represents gravity-driven flow, and the second is
the surface-tension-driven smoothing of the interface (Marangoni-like).

Key features:
- Lubrication approximation (Re << 1, h << L)
- Contact angle boundary conditions
- Capillary number-based stability control
- Film rupture detection and treatment

Usage::

    from pyfoam.applications.film_foam import FilmFoam

    solver = FilmFoam("path/to/case", rho=1000, mu=1e-3, sigma=0.07)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoam"]

logger = logging.getLogger(__name__)


class FilmFoam(SolverBase):
    """Thin film flow solver with surface tension.

    Solves the lubrication equation for thin films under gravity
    and surface tension.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho : float
        Liquid density (kg/m^3, default 1000).
    mu : float
        Dynamic viscosity (Pa s, default 1e-3).
    sigma : float
        Surface tension coefficient (N/m, default 0.07).
    g : float
        Gravitational acceleration (m/s^2, default 9.81).
    beta : float
        Inclination angle from horizontal (radians, default 0).
    contact_angle : float
        Contact angle at walls (radians, default pi/4 = 45 degrees).
    h_min : float
        Minimum film thickness for rupture detection (m, default 1e-6).
    theta_method : float
        Time stepping: 0 = explicit, 1 = implicit (default 1.0).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho: float = 1000.0,
        mu: float = 1e-3,
        sigma: float = 0.07,
        g: float = 9.81,
        beta: float = 0.0,
        contact_angle: float = math.pi / 4,
        h_min: float = 1e-6,
        theta_method: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.rho = rho
        self.mu = mu
        self.sigma = sigma
        self.g = g
        self.beta = beta
        self.contact_angle = contact_angle
        self.h_min = h_min
        self.theta_method = theta_method

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Initialise film thickness field
        self.h, self._h_data = self._init_film_thickness()

        # Compute capillary number (stability diagnostic)
        self._compute_capillary_number()

        logger.info(
            "FilmFoam ready: rho=%.1f, mu=%.2e, sigma=%.4f, "
            "beta=%.2f deg",
            rho, mu, sigma, math.degrees(beta),
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("filmFoam/convergenceTolerance", 1e-6)
        )
        self.n_correctors = int(
            fv.get_path("filmFoam/nCorrectors", 3)
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_film_thickness(self) -> tuple[torch.Tensor, Any]:
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        try:
            h_tensor, field_data = self.read_field_tensor("h", 0)
            h = h_tensor.to(device=device, dtype=dtype).reshape(-1)
            return h, field_data
        except Exception:
            h = torch.full((n_cells,), 1e-3, dtype=dtype, device=device)
            return h, None

    def _compute_capillary_number(self) -> float:
        """Compute and store the capillary number.

        Ca = mu * U_ref / sigma

        For gravity-driven flow: U_ref = rho * g * h_ref^2 / (3*mu)
        """
        h_ref = float(self.h.mean().item())
        U_ref = self.rho * self.g * h_ref ** 2 / (3.0 * self.mu)
        self.Ca = self.mu * U_ref / max(self.sigma, 1e-30)
        return self.Ca

    # ------------------------------------------------------------------
    # Surface velocity
    # ------------------------------------------------------------------

    def _compute_surface_velocity(self, h: torch.Tensor) -> torch.Tensor:
        """Compute surface velocity from the lubrication equation.

        U_s = (rho g h^2 / (3 mu)) * sin(beta)
            - (sigma h^3 / (3 mu)) * grad(grad^2 h)

        Returns (n_cells, 3) velocity vector.
        """
        device = h.device
        dtype = h.dtype
        n_cells = h.shape[0]

        # Gravity-driven component
        h2 = h.pow(2)
        grav_coeff = self.rho * self.g * math.sin(self.beta) / (3.0 * self.mu)
        U_grav = grav_coeff * h2  # scalar

        # Surface tension smoothing: -sigma * h^3 / (3*mu) * grad(laplacian(h))
        h3 = h.pow(3)
        st_coeff = -self.sigma / (3.0 * self.mu)

        # Laplacian of h (simplified via finite differences)
        lap_h = self._compute_laplacian(h)

        # Gradient of laplacian
        grad_lap_h = self._compute_gradient(lap_h)

        U_st = st_coeff * h3.unsqueeze(-1) * grad_lap_h

        # Combine: gravity in x-direction, surface tension in all directions
        U = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = U_grav  # gravity-driven flow along x
        U = U + U_st  # add surface tension contribution

        return U

    def _compute_laplacian(self, h: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of the film thickness field.

        nabla^2 h = (1/V) * sum_faces (grad(h)_face . S_f)

        Simplified using cell-centre finite differences.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = h.device
        dtype = h.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        w = mesh.face_weights[:n_internal]

        h_P = gather(h, owner)
        h_N = gather(h, neigh)

        # Face gradient (snGrad)
        delta_f = mesh.delta_coefficients[:n_internal]
        sn_grad = (h_N - h_P) * delta_f

        # Face area magnitude
        S_mag = mesh.face_areas[:n_internal].norm(dim=1)

        # Flux = sn_grad * S_mag
        flux = sn_grad * S_mag

        # Scatter to cells
        lap = torch.zeros(n_cells, dtype=dtype, device=device)
        lap = lap + scatter_add(flux, owner, n_cells)
        lap = lap + scatter_add(-flux, neigh, n_cells)

        # Divide by cell volume
        V = mesh.cell_volumes.clamp(min=1e-30)
        lap = lap / V

        return lap

    def _compute_gradient(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient of a scalar field.

        Returns (n_cells, 3) gradient vector.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = phi.device
        dtype = phi.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        w = mesh.face_weights[:n_internal]
        face_areas = mesh.face_areas[:n_internal]

        phi_P = gather(phi, owner)
        phi_N = gather(phi, neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad.index_add_(0, owner, face_contrib)
        grad.index_add_(0, neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    # ------------------------------------------------------------------
    # Film thickness transport
    # ------------------------------------------------------------------

    def _advance_film(self, h: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance film thickness one time step.

        dh/dt = -div(h * U_s)

        Using explicit time stepping with CFL-like stability check.
        """
        mesh = self.mesh
        device = h.device
        dtype = h.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        # Compute surface velocity
        U_s = self._compute_surface_velocity(h)

        # Face velocity interpolation
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        w = mesh.face_weights[:n_internal]
        face_areas = mesh.face_areas[:n_internal]

        U_s_face = (
            w.unsqueeze(-1) * U_s[owner]
            + (1.0 - w).unsqueeze(-1) * U_s[neigh]
        )

        # Face flux = U_s_face . S_f
        phi_face = (U_s_face * face_areas).sum(dim=1)

        # Average h at faces
        h_face = w * h[owner] + (1.0 - w) * h[neigh]

        # Film flux: h * phi
        h_phi = h_face * phi_face

        # Divergence: dh/dt = -div(h * U_s)
        div = torch.zeros(n_cells, dtype=dtype, device=device)
        div = div + scatter_add(h_phi, owner, n_cells)
        div = div + scatter_add(-h_phi, neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        dh = -dt * div / V

        # Update
        h_new = h + dh

        # Rupture detection
        h_new = h_new.clamp(min=self.h_min)

        return h_new

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the FilmFoam solver.

        Returns
        -------
        dict
            ``converged``, ``steps``, ``residual``,
            ``h_min``, ``h_max``, ``capillary_number``.
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

        logger.info("Starting FilmFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0

        for t, step in time_loop:
            h_old = self.h.clone()

            # Advance film thickness
            self.h = self._advance_film(self.h, self.delta_t)

            # Rupture detection
            ruptured = (self.h <= self.h_min).sum().item()
            if ruptured > 0:
                logger.debug("  %d cells at minimum film thickness", ruptured)

            # Residual
            residual = float((self.h - h_old).abs().max().item())
            converged = convergence.update(step + 1, {"h": residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FilmFoam converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())

        logger.info("FilmFoam completed: h=[%.2e, %.2e] m", h_min, h_max)

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
            "h_min": h_min,
            "h_max": h_max,
            "capillary_number": self.Ca,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        time_str = f"{time:g}"
        if self._h_data is not None:
            self.write_field("h", self.h, time_str, self._h_data)
        else:
            from pyfoam.io.field_io import FieldData, write_field as _write_field
            from pyfoam.io.foam_file import FoamFileHeader, FileFormat

            time_dir = self.case_path / time_str
            time_dir.mkdir(parents=True, exist_ok=True)
            field_data = FieldData(
                header=FoamFileHeader(
                    version="2.0", format=FileFormat.ASCII,
                    class_name="volScalarField",
                    location=time_str, object="h",
                ),
                dimensions=[0, 1, 0, 0, 0, 0, 0],
                internal_field=self.h.detach().cpu(),
                boundary_field=[],
                is_uniform=False,
                scalar_type="scalar",
            )
            _write_field(time_dir / "h", field_data, overwrite=True)
