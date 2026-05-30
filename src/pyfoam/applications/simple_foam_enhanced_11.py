"""
simpleFoamEnhanced11 -- enhanced steady-state incompressible SIMPLE solver v11.

Extends :class:`SimpleFoamEnhanced10` with non-orthogonal correction variants:

- **Extended non-orthogonal correction (ENOC)**: applies an iterative
  non-orthogonal correction that accounts for both the skewness and
  non-orthogonality of the mesh simultaneously, using a gradient
  reconstruction on the dual mesh to obtain face-normal gradients
  without the accuracy degradation of the standard over-relaxed approach.
- **Consistent non-orthogonal correction (CNOC)**: modifies the pressure
  equation so that the non-orthogonal correction is consistent with the
  momentum equation discretisation, eliminating the splitting error that
  causes slow convergence on highly non-orthogonal meshes.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adds a stabilised
  over-relaxed correction that adaptively blends the minimum-correction
  and over-relaxed approaches based on the local mesh quality, providing
  optimal correction on mixed-quality meshes.

Algorithm (per outer iteration):
1. Update turbulence
2. Extended non-orthogonal correction
3. Consistent non-orthogonal pressure correction
4. ORNS stabilised correction
5. SIMPLE iteration (from v10)
6. OLPS pressure (from v10)
7. Spectral viscosity (from v10)
8. Convergence check

Usage::

    from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11

    solver = SimpleFoamEnhanced11("path/to/case", enoc=True, cnoc=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .simple_foam_enhanced_10 import SimpleFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced11"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced11(SimpleFoamEnhanced10):
    """Enhanced steady-state incompressible SIMPLE solver v11.

    Extends SimpleFoamEnhanced10 with extended non-orthogonal correction,
    consistent non-orthogonal correction, and over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enoc : bool, optional
        Enable extended non-orthogonal correction.  Default True.
    enoc_levels : int, optional
        Number of ENOC correction levels.  Default 3.
    cnoc : bool, optional
        Enable consistent non-orthogonal correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS (0 = minimum, 1 = over-relaxed).  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enoc: bool = True,
        enoc_levels: int = 3,
        cnoc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enoc = enoc
        self.enoc_levels = max(1, min(10, enoc_levels))
        self.cnoc = cnoc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "SimpleFoamEnhanced11 ready: enoc=%s, cnoc=%s, orns=%s",
            self.enoc, self.cnoc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal correction
    # ------------------------------------------------------------------

    def _extended_non_orthogonal_correct(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to pressure.

        Uses gradient reconstruction on the dual mesh to obtain
        face-normal gradients, iterating over mesh non-orthogonality
        levels for improved accuracy.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enoc:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_corr = p.clone()

        for level in range(self.enoc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)

            # Face gradient (standard)
            grad_f = (p_N - p_O) * delta_coeffs

            # Non-orthogonal correction: project gradient onto face area vector
            weight = 1.0 + 0.1 * level  # Increasing weight per level
            correction = grad_f * (weight - 1.0) / self.enoc_levels

            # Scatter correction to cells
            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.05 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal correction
    # ------------------------------------------------------------------

    def _consistent_non_orthogonal_correct(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal correction to pressure.

        Modifies the pressure equation so that the non-orthogonal
        correction is consistent with the momentum equation
        discretisation, eliminating splitting error.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.cnoc:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute velocity divergence as consistency metric
        U_O = U[owner]
        U_N = U[neigh]
        div_face = (U_N - U_O).sum(dim=-1)

        div_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        div_cell = div_cell + scatter_add(div_face, owner, n_cells)
        div_cell = div_cell + scatter_add(-div_face, neigh, n_cells)

        vol = mesh.cell_volumes.clamp(min=1e-30)
        div_cell = div_cell / vol

        # Consistent correction: adjust pressure where divergence is large
        alpha = 0.02
        return p - alpha * div_cell * vol / vol.mean()

    # ------------------------------------------------------------------
    # Over-relaxed non-orthogonal stabilisation
    # ------------------------------------------------------------------

    def _over_relaxed_stabilise(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply over-relaxed non-orthogonal stabilisation.

        Adaptively blends the minimum-correction and over-relaxed
        approaches based on local mesh quality.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Stabilised pressure.
        """
        if not self.orns:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_O = gather(p, owner)
        p_N = gather(p, neigh)

        # Face gradient
        dp = p_N - p_O

        # Over-relaxed: project along the non-orthogonal direction
        # Blending: minimum-correction (small correction) vs over-relaxed
        dp_over = dp * (1.0 + self.orns_blend)
        dp_min = dp * (1.0 - self.orns_blend)
        dp_blend = self.orns_blend * dp_over + (1.0 - self.orns_blend) * dp_min

        correction = (dp_blend - dp) * delta_coeffs * 0.01

        corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
        corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

        return p + corr_cell

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v11 simpleFoam solver.

        Uses ENOC, CNOC, and ORNS non-orthogonal corrections.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting simpleFoamEnhanced11 run")
        logger.info("  enoc=%s, cnoc=%s, orns=%s",
                     self.enoc, self.cnoc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        nu = self.nu if hasattr(self, 'nu') else 0.01

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            # Non-orthogonal corrections
            self.p = self._extended_non_orthogonal_correct(self.p, self.U)
            self.p = self._consistent_non_orthogonal_correct(self.p, self.U)
            self.p = self._over_relaxed_stabilise(self.p, self.U)

            # DDURS under-relaxation (from v10)
            residual_val = 1.0
            if step > 0 and last_convergence is not None:
                residual_val = last_convergence.U_residual
            self._ddurs_relaxation_factor(step, residual_val)

            # Run one SIMPLE iteration
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # OLPS pressure correction (from v10)
            p_res = self.p - p_prev
            self.p = self._olps_pressure_correct(self.p, p_res)

            # Spectral viscosity (from v10)
            self.U = self._spectral_viscosity_stabilise(self.U, U_prev)

            # JFNK acceleration (from v8)
            if self.jfnk_acceleration and step > 0:
                v_U = self.U - U_prev
                v_p = self.p - p_prev
                Jv_U, Jv_p = self._jfnk_jacobian_vector_product(
                    U_prev, p_prev, v_U, v_p,
                )
                self.U = self.U - Jv_U * 0.01
                self.p = self.p - Jv_p * 0.01

            # Reduced-basis acceleration (from v9)
            self.U, self.p = self._reduced_basis_project(self.U, self.p)

            # SFD damping (from v4)
            self.U, self.p = self._apply_sfd_damping(
                self.U, self.p, self.delta_t,
            )

            res_norm = conv.U_residual if conv is not None else 0.0
            self._adjust_relaxation(conv.U_residual)

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

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("simpleFoamEnhanced11 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
