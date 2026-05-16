"""
YPlus — y+ calculation function object for wall-bounded flows.

Computes the dimensionless wall distance y+ at wall patches, which is
essential for determining the appropriate turbulence model and wall
treatment.

Physics
-------
y+ is defined as:

    y+ = y · u_τ / ν

where:
    - y is the distance from the wall to the first cell centre
    - u_τ = √(τ_w / ρ) is the friction velocity
    - τ_w is the wall shear stress
    - ν = μ/ρ is the kinematic viscosity

Wall treatment guidelines:
    - y+ < 1:   Resolved (low-Re) wall treatment
    - 30 < y+ < 300: Wall function treatment
    - y+ > 300: Too coarse for wall-bounded flows

References
----------
- OpenFOAM ``yPlus`` function object source
- Spalding, D.B. "A single formula for the law of the wall" (1961)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["YPlus"]

logger = logging.getLogger(__name__)


class YPlus(FunctionObject):
    """Compute y+ (dimensionless wall distance) at wall patches.

    Configuration keys:

    - ``patches``: list of wall patch names (default: all wall patches)
    - ``rho``: reference density (default: 1.0)
    - ``mu``: dynamic viscosity (default: 1.0)

    Example controlDict entry::

        yPlus1
        {
            type            yPlus;
            libs            ("libfieldFunctionObjects.so");
            patches         (movingWall fixedWalls);
            rho             1.0;
            mu              1e-3;
        }
    """

    def __init__(self, name: str = "yPlus", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._patches: List[str] = self.config.get("patches", [])
        self._rho: float = float(self.config.get("rho", 1.0))
        self._mu: float = float(self.config.get("mu", 1.0))

        # Results storage
        self._y_plus_history: List[Dict[str, torch.Tensor]] = []
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and validate patches."""
        self._mesh = mesh
        self._fields = fields

        # Auto-detect wall patches if none specified
        if not self._patches and hasattr(mesh, "boundary"):
            self._patches = [
                p["name"] for p in mesh.boundary
                if p.get("type", "").lower() in ("wall",)
            ]

        logger.info("YPlus '%s' initialised: patches=%s", self.name, self._patches)

    def execute(self, time: float) -> None:
        """Compute y+ at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required for YPlus. Skipping.")
            return

        y_plus = self._compute_y_plus(U)
        self._y_plus_history.append({k: v.detach().cpu() for k, v in y_plus.items()})
        self._times.append(time)

        # Log y+ statistics per patch
        for patch_name, yp in y_plus.items():
            self._log.info(
                "t=%g  patch='%s'  y+_avg=%.2f  y+_min=%.2f  y+_max=%.2f",
                time, patch_name,
                yp.mean().item(), yp.min().item(), yp.max().item()
            )

    def _compute_y_plus(self, U_field) -> Dict[str, torch.Tensor]:
        """Compute y+ on each wall patch.

        Returns:
            Dictionary mapping patch names to ``(n_faces,)`` tensors of y+ values.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh

        # Get velocity data
        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        face_centres = mesh.face_centres.to(device=device, dtype=dtype)
        cell_centres = mesh.cell_centres.to(device=device, dtype=dtype)

        nu = self._mu / self._rho  # kinematic viscosity

        y_plus_patches = {}

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long
            )

            # Face normals
            S = face_areas[face_indices]  # (n_faces, 3)
            S_mag = S.norm(dim=1, keepdim=True)  # (n_faces, 1)
            n = S / S_mag.clamp(min=1e-30)  # unit normals

            # Owner cells
            owner = mesh.owner[face_indices]

            # Distance from cell centre to face (wall-normal distance)
            x_P = cell_centres[owner]  # (n_faces, 3)
            x_f = face_centres[face_indices]  # (n_faces, 3)
            d = torch.abs(torch.sum((x_f - x_P) * n, dim=1))  # (n_faces,)

            # Velocity at owner cell
            U_P = U_data[owner]  # (n_faces, 3)

            # Tangential velocity
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)  # (n_faces, 1)
            U_t = U_P - U_n * n  # (n_faces, 3)
            U_t_mag = U_t.norm(dim=1)  # (n_faces,)

            # Wall shear stress: τ_w = μ * U_t / d
            tau_w = self._mu * U_t_mag / d.clamp(min=1e-30)  # (n_faces,)

            # Friction velocity: u_τ = √(τ_w / ρ)
            u_tau = torch.sqrt(tau_w / self._rho)  # (n_faces,)

            # y+ = y * u_τ / ν
            y_p = d * u_tau / nu  # (n_faces,)

            y_plus_patches[patch_name] = y_p

        return y_plus_patches

    def _get_patch_info(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """Get patch information from mesh boundary."""
        if not hasattr(self._mesh, "boundary"):
            return None
        for p in self._mesh.boundary:
            if p["name"] == patch_name:
                return p
        return None

    def write(self) -> None:
        """Write y+ data to output files."""
        if self._output_path is None or not self._times:
            return

        yplus_file = self._output_path / "yPlus.dat"
        with open(yplus_file, "w") as f:
            f.write("# Time  patch  y+_avg  y+_min  y+_max\n")
            for i, t in enumerate(self._times):
                for patch_name, yp in self._y_plus_history[i].items():
                    f.write(
                        f"{t:.6e}  {patch_name}  "
                        f"{yp.mean():.4f} {yp.min():.4f} {yp.max():.4f}\n"
                    )
        logger.info("Wrote y+ to %s", yplus_file)

    @property
    def y_plus_history(self) -> List[Dict[str, torch.Tensor]]:
        """y+ history (per patch per time step)."""
        return self._y_plus_history

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# Register
FunctionObjectRegistry.register("yPlus", YPlus)
