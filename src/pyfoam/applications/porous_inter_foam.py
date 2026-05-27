"""
PorousInterFoam — porous media two-phase VOF solver.

Implements a two-phase incompressible solver combining interFoam with
Darcy-Forchheimer porous media resistance.  This is the pyOpenFOAM
equivalent of OpenFOAM's ``porousInterFoam`` solver.

Extends :class:`InterFoam` with:

- **Darcy-Forchheimer resistance**: ``S = -(μ d + ρ|U| f/2) U`` in porous zones
- **VOF advection** with compression (MULES-like)
- **Surface tension** at the interface (CSF model)

The porous media model reads from ``constant/porosityProperties``::

    porosity1
    {
        type            DarcyForchheimer;
        cellZone        porosity;
        d   (5e7 -1000 -1000);     // Darcy coefficients [1/m²]
        f   (0 0 0);               // Forchheimer coefficients [1/m]
    }

Algorithm modifications (per PIMPLE outer iteration):

1. Advance VOF volume fraction α
2. Compute mixture properties (ρ_mix, μ_mix)
3. Add porous resistance to momentum predictor
4. Standard PISO pressure-velocity coupling

Usage::

    from pyfoam.applications.porous_inter_foam import PorousInterFoam

    solver = PorousInterFoam("path/to/case")
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

from .inter_foam import InterFoam
from .porous_simple_foam import PorousZoneProperties

__all__ = ["PorousInterFoam"]

logger = logging.getLogger(__name__)


class PorousInterFoam(InterFoam):
    """Porous media two-phase VOF incompressible solver.

    Combines the interFoam VOF two-phase solver with Darcy-Forchheimer
    porous media resistance.  Suitable for modelling two-phase flow
    through filters, packed beds, and other porous structures.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    porous_zones : list[PorousZoneProperties], optional
        Porous zone definitions. If None, reads from
        ``constant/porosityProperties``.
    rho1, rho2, mu1, mu2, sigma, C_alpha
        Fluid properties (see :class:`InterFoam`).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        porous_zones: list[PorousZoneProperties] | None = None,
        rho1: float = 1000.0,
        rho2: float = 1.225,
        mu1: float = 1e-3,
        mu2: float = 1.8e-5,
        sigma: float = 0.07,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(
            case_path,
            rho1=rho1, rho2=rho2,
            mu1=mu1, mu2=mu2,
            sigma=sigma, C_alpha=C_alpha,
        )

        # Read or use provided porous properties
        self.porous_zones = (
            porous_zones if porous_zones is not None
            else self._read_porous_properties()
        )

        # Pre-build porous zone data (cell masks + resistance tensors)
        device = get_device()
        dtype = get_default_dtype()
        self._porous_data: list[dict[str, Any]] = []

        for pz in self.porous_zones:
            cell_mask = self._build_cell_mask(pz.cell_zone)
            d_tensor = torch.tensor(pz.d, dtype=dtype, device=device)
            f_tensor = torch.tensor(pz.f, dtype=dtype, device=device)
            self._porous_data.append({
                "name": pz.name,
                "cell_mask": cell_mask,
                "d": d_tensor,
                "f": f_tensor,
            })

        logger.info(
            "PorousInterFoam ready: %d porous zone(s), "
            "rho1=%.1f, rho2=%.3f, sigma=%.3f",
            len(self.porous_zones), rho1, rho2, sigma,
        )

    # ------------------------------------------------------------------
    # Porous property reading (reuse from PorousSimpleFoam pattern)
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
                    continue
                cell_zone = str(value.get("cellZone", key)).strip()
                d = self._parse_resistance_vector(value.get("d", (0, 0, 0)))
                f = self._parse_resistance_vector(value.get("f", (0, 0, 0)))
                zones.append(PorousZoneProperties(
                    name=key, cell_zone=cell_zone, d=d, f=f,
                ))
            return zones
        except Exception as e:
            logger.warning("Could not read porosityProperties: %s", e)
            return []

    @staticmethod
    def _parse_resistance_vector(value: Any) -> tuple[float, float, float]:
        """Parse a resistance vector, handling negative multipliers."""
        from .porous_simple_foam import PorousSimpleFoam
        return PorousSimpleFoam._parse_resistance_vector(value)

    def _build_cell_mask(self, zone_name: str) -> torch.Tensor:
        """Build a boolean mask for cells in a named zone."""
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

        logger.warning("Cell zone '%s' not found; applying to ALL cells", zone_name)
        return torch.ones(n_cells, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Porous momentum modification
    # ------------------------------------------------------------------

    def _apply_porous_resistance(
        self,
        A_p: torch.Tensor,
        source: torch.Tensor,
        U: torch.Tensor,
        rho: torch.Tensor,
        mu_mix: torch.Tensor,
        cell_volumes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Darcy-Forchheimer porous resistance to momentum equation.

        For each porous zone:
        - ``Cd_i = μ_mix * d_i + 0.5 * ρ * |U| * f_i``
        - Diagonal: ``+= V * tr(Cd)``
        - Source: ``-= V * (Cd - I*tr(Cd)) · U``

        Parameters
        ----------
        A_p : Tensor
            Momentum diagonal, shape ``(n_cells,)``.
        source : Tensor
            Momentum source, shape ``(n_cells, 3)``.
        U : Tensor
            Current velocity, shape ``(n_cells, 3)``.
        rho : Tensor
            Mixture density, shape ``(n_cells,)``.
        mu_mix : Tensor
            Mixture viscosity, shape ``(n_cells,)``.
        cell_volumes : Tensor
            Cell volumes, shape ``(n_cells,)``.

        Returns
        -------
        A_p, source : tuple[Tensor, Tensor]
            Modified diagonal and source.
        """
        device = get_device()
        dtype = get_default_dtype()

        for zone in self._porous_data:
            mask = zone["cell_mask"]
            if not mask.any():
                continue

            d = zone["d"]
            f = zone["f"]

            U_zone = U[mask]
            V_zone = cell_volumes[mask].clamp(min=1e-30)
            mu_zone = mu_mix[mask]
            rho_zone = rho[mask]

            U_mag = U_zone.norm(dim=1)  # (n_zone,)

            # Resistance per component: Cd_i = μ * d_i + 0.5 * ρ * |U| * f_i
            Cd = (
                mu_zone.unsqueeze(-1) * d.unsqueeze(0)
                + 0.5 * rho_zone.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0)
            )  # (n_zone, 3)

            isoCd = Cd.sum(dim=1)  # (n_zone,)

            # Diagonal: += V * tr(Cd)
            A_p[mask] = A_p[mask] + V_zone * isoCd

            # Anisotropic source: -= V * (Cd - I*isoCd) · U
            aniso = Cd - isoCd.unsqueeze(-1) * torch.ones_like(Cd)
            source[mask] = source[mask] - V_zone.unsqueeze(-1) * aniso * U_zone

        return A_p, source

    # ------------------------------------------------------------------
    # Override momentum predictor to include porous resistance
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with surface tension and porous resistance.

        Extends the base interFoam momentum predictor by adding
        Darcy-Forchheimer resistance in porous cells after the standard
        assembly.
        """
        # Call base interFoam momentum predictor
        U_new, A_p, H = super()._momentum_predictor(
            U, p, phi, rho, mu_mix,
        )

        if not self._porous_data:
            return U_new, A_p, H

        # Add porous resistance
        cell_volumes = self.mesh.cell_volumes

        # Reconstruct source from H (source = H + A_p * U)
        source = H + A_p.unsqueeze(-1) * U_new

        A_p, source = self._apply_porous_resistance(
            A_p, source, U_new, rho, mu_mix, cell_volumes,
        )

        # Re-solve: U = source / A_p
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_new = source / A_p_safe.unsqueeze(-1)

        # Re-apply under-relaxation
        if self.alpha_U < 1.0:
            U_new = self.alpha_U * U_new + (1.0 - self.alpha_U) * U

        # Update H
        H = source - A_p.unsqueeze(-1) * U_new

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Override run to log porous info
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the PorousInterFoam solver.

        Returns:
            Final :class:`ConvergenceData`.
        """
        logger.info("Starting PorousInterFoam run")
        logger.info("  porous zones: %d", len(self.porous_zones))
        return super().run()
