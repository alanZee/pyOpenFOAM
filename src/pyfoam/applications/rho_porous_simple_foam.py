"""
rhoPorousSimpleFoam — steady-state compressible solver with porous media zones.

Implements the SIMPLE algorithm for steady-state compressible
Navier-Stokes equations with Darcy-Forchheimer porous media resistance,
energy equation coupling, and optional Multiple Reference Frame (MRF) zones.

Combines :class:`RhoSimpleFoam` (compressible SIMPLE) with
:class:`PorousSimpleFoam` (porous resistance):

- **Compressible**: density from EOS, energy equation, variable viscosity
- **Darcy-Forchheimer**: ``S = -(mu/D * U + rho*|U|*C/2 * U)``
- **MRF zones**: Coriolis and centrifugal forces in rotating regions

Algorithm (per outer iteration):

1. Momentum predictor with porous resistance and MRF forces
2. Compressible pressure equation
3. Density update from EOS
4. Energy equation
5. Turbulence update (if active)
6. Convergence check

The porous media model reads from ``constant/porosityProperties``::

    porosity1
    {
        type            DarcyForchheimer;
        cellZone        porosity;
        d   (5e7 -1000 -1000);     // Darcy coefficients [1/m^2]
        f   (0 0 0);               // Forchheimer coefficients [1/m]
    }

The MRF zone reads from ``constant/MRFProperties``::

    MRFProperties
    {
        cellZone        rotor;
        origin          (0 0 0);
        axis            (0 0 1);
        omega           100;
    }

For compressible flow the resistance uses **dynamic viscosity** mu
(not kinematic viscosity nu) and includes **density** in the
Forchheimer term, matching OpenFOAM's compressible porous treatment.

**Darcy-Forchheimer Model** (compressible form):

The resistance per cell direction is:

    Cd_i = mu * d_i + rho * |U| * f_i / 2

Diagonal contribution: ``+V * tr(Cd)``
Anisotropic source: ``-V * (Cd - I*tr(Cd)) . U``

This matches OpenFOAM's ``DarcyForchheimer::apply()`` for compressible
solvers.

Usage::

    from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

    solver = RhoPorousSimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.thermo import BasicThermo

from .rho_simple_foam import RhoSimpleFoam
from .porous_simple_foam import (
    PorousZoneProperties,
    MRFZoneProperties,
)
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor
from pyfoam.solvers.coupled_solver import ConvergenceData

__all__ = ["RhoPorousSimpleFoam"]

logger = logging.getLogger(__name__)


class RhoPorousSimpleFoam(RhoSimpleFoam):
    """Steady-state compressible SIMPLE solver with porous media zones.

    Extends :class:`RhoSimpleFoam` with Darcy-Forchheimer porous resistance
    and optional MRF (Multiple Reference Frame) zones for compressible flows
    through porous media (e.g. packed-bed reactors, filters in high-speed flow).

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.
    porous_zones : list[PorousZoneProperties], optional
        Porous zone definitions. If None, reads from ``constant/porosityProperties``.
    mrf_zones : list[MRFZoneProperties], optional
        MRF zone definitions. If None, reads from ``constant/MRFProperties``.

    Attributes
    ----------
    porous_zones : list[PorousZoneProperties]
        The porous zone definitions.
    mrf_zones : list[MRFZoneProperties]
        The MRF zone definitions.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        porous_zones: list[PorousZoneProperties] | None = None,
        mrf_zones: list[MRFZoneProperties] | None = None,
    ) -> None:
        super().__init__(case_path, thermo=thermo)

        # Read or use provided porous properties
        self.porous_zones = (
            porous_zones if porous_zones is not None
            else self._read_porous_properties()
        )

        # Read or use provided MRF properties
        self.mrf_zones = (
            mrf_zones if mrf_zones is not None
            else self._read_mrf_properties()
        )

        # Precompute MRF tensors
        device = get_device()
        dtype = get_default_dtype()
        self._mrf_data: list[dict[str, Any]] = []

        for mrf in self.mrf_zones:
            omega_vec = torch.tensor(mrf.omega_vec, dtype=dtype, device=device)
            origin = torch.tensor(mrf.origin, dtype=dtype, device=device)
            omega_mag = omega_vec.norm().item()

            cell_mask = self._build_cell_mask(mrf.cell_zone)

            # r = centre - origin for MRF cells
            r = self.mesh.cell_centres[cell_mask] - origin.unsqueeze(0)

            # Centrifugal force
            omega_sq = omega_vec.dot(omega_vec)
            omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
            centrifugal = (
                r * omega_sq
                - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)
            )

            self._mrf_data.append({
                "name": mrf.name,
                "cell_mask": cell_mask,
                "omega_vec": omega_vec,
                "omega_mag": omega_mag,
                "centrifugal": centrifugal,
                "origin": origin,
            })

        logger.info(
            "RhoPorousSimpleFoam ready: %d porous zone(s), %d MRF zone(s)",
            len(self.porous_zones), len(self.mrf_zones),
        )
        for pz in self.porous_zones:
            logger.info("  Porous zone: %s", pz)
        for mrf in self.mrf_zones:
            logger.info("  MRF zone: %s", mrf)

    # ------------------------------------------------------------------
    # Property reading (reuse from PorousSimpleFoam pattern)
    # ------------------------------------------------------------------

    def _read_porous_properties(self) -> list[PorousZoneProperties]:
        """Read porous properties from ``constant/porosityProperties``."""
        pp_path = self.case_path / "constant" / "porosityProperties"
        if not pp_path.exists():
            logger.info("constant/porosityProperties not found; no porous zones")
            return []

        try:
            from pyfoam.io.dictionary import parse_dict_file
            pp = parse_dict_file(pp_path)

            zones = []
            for key, value in pp.items():
                if not isinstance(value, dict):
                    continue
                zone_type = str(value.get("type", "")).strip()
                if zone_type != "DarcyForchheimer":
                    logger.warning("Skipping non-DarcyForchheimer zone: %s", key)
                    continue

                cell_zone = str(value.get("cellZone", key)).strip()
                d = self._parse_resistance_vector(value.get("d", (0, 0, 0)))
                f = self._parse_resistance_vector(value.get("f", (0, 0, 0)))

                zones.append(PorousZoneProperties(
                    name=key,
                    cell_zone=cell_zone,
                    d=d,
                    f=f,
                ))

            return zones

        except Exception as e:
            logger.warning("Could not read porosityProperties: %s", e)
            return []

    def _read_mrf_properties(self) -> list[MRFZoneProperties]:
        """Read MRF properties from ``constant/MRFProperties``."""
        mrf_path = self.case_path / "constant" / "MRFProperties"
        if not mrf_path.exists():
            logger.info("constant/MRFProperties not found; no MRF zones")
            return []

        try:
            from pyfoam.io.dictionary import parse_dict_file
            mrf = parse_dict_file(mrf_path)

            zones = []
            for key, value in mrf.items():
                if not isinstance(value, dict):
                    continue

                cell_zone = str(value.get("cellZone", key)).strip()
                origin = self._parse_vector(value.get("origin", (0, 0, 0)))
                axis = self._parse_vector(value.get("axis", (0, 0, 1)))
                omega = float(value.get("omega", 0.0))

                zones.append(MRFZoneProperties(
                    name=key,
                    cell_zone=cell_zone,
                    origin=origin,
                    axis=axis,
                    omega=omega,
                ))

            return zones

        except Exception as e:
            logger.warning("Could not read MRFProperties: %s", e)
            return []

    def _build_cell_mask(self, zone_name: str) -> torch.Tensor:
        """Build a boolean mask for cells in a named zone.

        If the zone name is ``"all"``, returns a mask selecting all cells.
        Otherwise, reads the cell zone from the mesh.
        """
        n_cells = self.mesh.n_cells

        if zone_name == "all":
            return torch.ones(n_cells, dtype=torch.bool)

        try:
            zone_path = self.case_path / "constant" / "polyMesh" / "cellZones"
            if zone_path.exists():
                from pyfoam.io.dictionary import parse_dict_file
                zones = parse_dict_file(zone_path)
                if zone_name in zones:
                    zone_data = zones[zone_name]
                    if isinstance(zone_data, (list, tuple)):
                        indices = [int(i) for i in zone_data]
                        mask = torch.zeros(n_cells, dtype=torch.bool)
                        for idx in indices:
                            if 0 <= idx < n_cells:
                                mask[idx] = True
                        return mask
        except Exception as e:
            logger.debug("Could not read cellZones: %s", e)

        logger.warning(
            "Cell zone '%s' not found; applying to ALL cells", zone_name,
        )
        return torch.ones(n_cells, dtype=torch.bool)

    @staticmethod
    def _parse_vector(value: Any) -> tuple[float, float, float]:
        """Parse a vector from dictionary value."""
        import re
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        if isinstance(value, str):
            match = re.search(
                r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                value,
            )
            if match:
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                )

        return (0.0, 0.0, 0.0)

    @staticmethod
    def _parse_resistance_vector(value: Any) -> tuple[float, float, float]:
        """Parse a resistance vector (d or f coefficients).

        Negative values are treated as multipliers of the maximum positive
        component (OpenFOAM convention).
        """
        vec = RhoPorousSimpleFoam._parse_vector(value)

        max_cmpt = max(vec)
        if max_cmpt < 0:
            return (0.0, 0.0, 0.0)

        result = list(vec)
        for i in range(3):
            if result[i] < 0:
                result[i] = result[i] * (-max_cmpt)

        return tuple(result)

    # ------------------------------------------------------------------
    # Momentum predictor (override with porous + MRF)
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with variable density, porous resistance, and MRF.

        Extends the base momentum predictor with:
        - Darcy-Forchheimer resistance: ``S = -(mu/D U + rho |U| C/2 U)``
        - MRF Coriolis and centrifugal forces

        For compressible flow, the resistance uses **dynamic viscosity** mu
        and includes **density** in the Forchheimer term:

            Cd_i = mu * d_i + rho * |U| * f_i / 2

        This matches OpenFOAM's compressible DarcyForchheimer model.
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

        # Viscosity: molecular or effective
        if mu_eff is not None:
            mu = mu_eff
        else:
            mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # Convection (upwind) with variable density
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Matrix coefficients
        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

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

        # ============================================
        # Porous resistance: Darcy-Forchheimer (compressible form)
        # Cd_i = mu * d_i + rho * |U| * f_i / 2
        # ============================================
        for zone in self._build_porous_data():
            mask = zone["cell_mask"]
            if not mask.any():
                continue

            d = zone["d"]  # (3,) Darcy coefficients
            f = zone["f"]  # (3,) Forchheimer coefficients

            U_zone = U[mask]
            V_zone = cell_volumes_safe[mask]
            rho_zone = rho[mask]

            # Dynamic viscosity for this zone
            if mu_eff is not None:
                mu_zone = mu_eff[mask]
            else:
                mu_zone = self.thermo.mu(T=self.T)[mask]

            U_mag = U_zone.norm(dim=1)

            # Resistance tensor: Cd_i = mu * d_i + rho * |U| * f_i / 2
            Cd = (
                mu_zone.unsqueeze(-1) * d.unsqueeze(0)
                + rho_zone.unsqueeze(-1) * U_mag.unsqueeze(-1)
                * f.unsqueeze(0) * 0.5
            )  # (n_zone, 3)

            # Isotropic part: tr(Cd) = sum of diagonal
            isoCd = Cd.sum(dim=1)  # (n_zone,)

            # Add to diagonal: += V * tr(Cd)  (implicit resistance)
            diag[mask] = diag[mask] + V_zone * isoCd

            # Anisotropic source: -= V * (Cd - I*isoCd) . U
            aniso = Cd - isoCd.unsqueeze(-1) * torch.ones_like(Cd)
            aniso_source = V_zone.unsqueeze(-1) * aniso * U_zone
            source[mask] = source[mask] - aniso_source

        # ============================================
        # MRF: Coriolis and centrifugal forces
        # ============================================
        for mrf in self._mrf_data:
            mask = mrf["cell_mask"]
            if not mask.any():
                continue

            omega_vec = mrf["omega_vec"]
            omega_mag = mrf["omega_mag"]
            centrifugal = mrf["centrifugal"]

            # Centrifugal force (explicit)
            source[mask] = source[mask] + centrifugal

            # Coriolis force (semi-implicit)
            U_zone = U[mask]
            omega_cross_U = torch.zeros_like(U_zone)
            omega_cross_U[:, 0] = omega_vec[1] * U_zone[:, 2] - omega_vec[2] * U_zone[:, 1]
            omega_cross_U[:, 1] = omega_vec[2] * U_zone[:, 0] - omega_vec[0] * U_zone[:, 2]
            omega_cross_U[:, 2] = omega_vec[0] * U_zone[:, 1] - omega_vec[1] * U_zone[:, 0]
            source[mask] = source[mask] - 2.0 * omega_cross_U

            # Implicit part: +2|omega| on diagonal
            diag[mask] = diag[mask] + 2.0 * omega_mag

        # ============================================
        # Boundary conditions
        # ============================================
        if n_faces := mesh.n_faces > n_internal:
            U_bc = self._build_boundary_conditions()
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    bnd_owner = owner[n_internal:]
                    bnd_areas = mesh.face_areas[n_internal:]
                    bnd_face_centres = mesh.face_centres[n_internal:]

                    owner_centres = mesh.cell_centres[bnd_owner]
                    d_P = bnd_face_centres - owner_centres
                    d_full = 2.0 * d_P
                    bnd_S_mag = bnd_areas.norm(dim=1)
                    safe_S_mag = torch.where(
                        bnd_S_mag > 1e-30, bnd_S_mag, torch.ones_like(bnd_S_mag)
                    )
                    n_hat = bnd_areas / safe_S_mag.unsqueeze(-1)
                    d_dot_n = (d_full * n_hat).sum(dim=1).abs()
                    bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

                    if mu_eff is not None:
                        bnd_mu = gather(mu_eff, bnd_owner)
                    else:
                        bnd_mu = gather(mu, bnd_owner)
                    bnd_face_coeff = bnd_mu * bnd_S_mag * bnd_delta

                    bnd_bc_mask = bc_mask[bnd_owner]
                    bnd_face_coeff_masked = bnd_face_coeff * bnd_bc_mask.float()

                    bnd_V = gather(cell_volumes_safe, bnd_owner)
                    bnd_face_coeff_pv = bnd_face_coeff_masked / bnd_V

                    diag = diag + scatter_add(bnd_face_coeff_pv, bnd_owner, n_cells)

                    for comp in range(3):
                        u_bc_comp = U_bc[bnd_owner, comp].nan_to_num(0.0)
                        source_contrib = bnd_face_coeff_pv * u_bc_comp
                        source[:, comp] = (
                            source[:, comp]
                            + scatter_add(source_contrib, bnd_owner, n_cells)
                        )

        # ============================================
        # Solve: A_p * U = source
        # ============================================
        A_p = diag.clone()
        diag_safe = diag.abs().clamp(min=1e-30)
        U_solved = source / diag_safe.unsqueeze(-1)

        # Under-relaxation
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        # Re-apply boundary conditions at fixed-value cells
        U_bc = self._build_boundary_conditions()
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any():
                U_new[bc_mask] = U_bc[bc_mask]

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Build porous data tensors (lazy, cached per __init__ state)
    # ------------------------------------------------------------------

    def _build_porous_data(self) -> list[dict[str, Any]]:
        """Build porous zone data tensors.

        Returns:
            List of dicts with keys ``name``, ``cell_mask``, ``d``, ``f``.
        """
        device = get_device()
        dtype = get_default_dtype()

        data = []
        for pz in self.porous_zones:
            cell_mask = self._build_cell_mask(pz.cell_zone)
            d = torch.tensor(pz.d, dtype=dtype, device=device)
            f = torch.tensor(pz.f, dtype=dtype, device=device)
            data.append({
                "name": pz.name,
                "cell_mask": cell_mask,
                "d": d,
                "f": f,
            })
        return data

    # ------------------------------------------------------------------
    # Main run loop (override for porous logging)
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the rhoPorousSimpleFoam solver.

        Executes the compressible SIMPLE algorithm with porous media
        resistance in a time-stepping loop until convergence or
        ``endTime`` is reached.

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

        logger.info("Starting rhoPorousSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  porous zones: %d", len(self.porous_zones))
        logger.info("  MRF zones: %d", len(self.mrf_zones))
        logger.info(
            "  relaxation: alpha_U=%.2f, alpha_p=%.2f, alpha_T=%.2f",
            self.alpha_U, self.alpha_p, self.alpha_T,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence model (if active)
            mu_eff = self._update_turbulence()

            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._simple_iteration(mu_eff=mu_eff)
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

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info(
                    "rhoPorousSimpleFoam completed successfully (converged)"
                )
            else:
                logger.warning(
                    "rhoPorousSimpleFoam completed without full convergence"
                )

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Boundary condition builder (for momentum predictor)
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor | None:
        """Build boundary condition tensor for velocity.

        Returns:
            ``(n_cells, 3)`` tensor with prescribed values at boundary
            cells and NaN at interior cells, or ``None`` if no BCs.
        """
        # Use the first time step data
        U_data = self.case.read_field("U", 0)
        if U_data is None:
            return None

        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        U_bc = torch.full((n_cells, 3), float("nan"), dtype=dtype, device=device)

        # Mark boundary cells from U boundary field
        try:
            for patch_name, patch_data in U_data.boundary.items():
                bc_type = patch_data.get("type", "")
                if bc_type == "fixedValue":
                    value = patch_data.get("value")
                    if value is not None:
                        # Get face range for this patch
                        patch_info = self.mesh.boundary.get(patch_name)
                        if patch_info is not None:
                            start = patch_info.get("startFace", 0)
                            n_faces_patch = patch_info.get("nFaces", 0)
                            # Map boundary faces to owner cells
                            for fi in range(start, start + n_faces_patch):
                                if fi < len(self.mesh.owner):
                                    cell = self.mesh.owner[fi].item()
                                    if isinstance(value, (list, tuple)) and len(value) >= 3:
                                        U_bc[cell, 0] = float(value[0])
                                        U_bc[cell, 1] = float(value[1])
                                        U_bc[cell, 2] = float(value[2])
                                    elif isinstance(value, torch.Tensor) and value.numel() >= 3:
                                        U_bc[cell] = value[:3].to(dtype=dtype, device=device)
        except Exception:
            # Fallback: no boundary conditions
            return None

        # If no BCs were set, return None
        if torch.isnan(U_bc).all():
            return None

        return U_bc
