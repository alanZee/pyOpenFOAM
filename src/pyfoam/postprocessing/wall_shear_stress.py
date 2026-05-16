"""
WallShearStress — wall shear stress computation function object.

Computes the wall shear stress τ_w on wall patches using the velocity
gradient at the wall.

Physics
-------
Wall shear stress is defined as:

    τ_w = μ (∂U/∂n)|_wall

where n is the wall-normal direction and μ is the dynamic viscosity.

For a Newtonian fluid:

    τ_w = μ * (∇U + (∇U)^T) · n |_wall

The friction velocity is:

    u_τ = √(τ_w / ρ)

References
----------
- OpenFOAM ``wallShearStress`` function object source
- Schlichting & Gersten, "Boundary Layer Theory", 8th ed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["WallShearStress"]

logger = logging.getLogger(__name__)


class WallShearStress(FunctionObject):
    """Compute wall shear stress on wall patches.

    Configuration keys:

    - ``patches``: list of wall patch names (default: all wall patches)
    - ``rho``: reference density (default: 1.0)
    - ``writeFields``: if True, write τ_w as a boundary field (default: False)

    Example controlDict entry::

        wallShearStress1
        {
            type            wallShearStress;
            libs            ("libfieldFunctionObjects.so");
            patches         (movingWall fixedWalls);
            rho             1.0;
            writeFields     true;
        }
    """

    def __init__(self, name: str = "wallShearStress", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._patches: List[str] = self.config.get("patches", [])
        self._rho: float = float(self.config.get("rho", 1.0))
        self._write_fields: bool = self.config.get("writeFields", False)

        # Results storage
        self._tau_w_history: List[Dict[str, torch.Tensor]] = []
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

        logger.info("WallShearStress '%s' initialised: patches=%s", self.name, self._patches)

    def execute(self, time: float) -> None:
        """Compute wall shear stress at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required for WallShearStress. Skipping.")
            return

        tau_w = self._compute_wall_shear_stress(U)
        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        # Log average wall shear stress per patch
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            self._log.info(
                "t=%g  patch='%s'  tau_w_avg=%.6e  tau_w_max=%.6e",
                time, patch_name, tau_mag.mean().item(), tau_mag.max().item()
            )

    def _compute_wall_shear_stress(self, U_field) -> Dict[str, torch.Tensor]:
        """Compute wall shear stress on each patch.

        Returns:
            Dictionary mapping patch names to ``(n_faces, 3)`` tensors
            of wall shear stress vectors.
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

        # Get viscosity
        mu = self._get_viscosity()

        tau_w_patches = {}

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long
            )

            # Face normals (outward from domain)
            S = face_areas[face_indices]  # (n_faces, 3)
            S_mag = S.norm(dim=1, keepdim=True)  # (n_faces, 1)
            n = S / S_mag.clamp(min=1e-30)  # unit normals

            # Owner cells
            owner = mesh.owner[face_indices]

            # Cell centres of owner cells
            x_P = cell_centres[owner]  # (n_faces, 3)

            # Face centres
            x_f = face_centres[face_indices]  # (n_faces, 3)

            # Distance from cell centre to face
            d_Pf = x_f - x_P  # (n_faces, 3)
            d_Pf_mag = d_Pf.norm(dim=1, keepdim=True)  # (n_faces, 1)

            # Velocity at owner cell
            U_P = U_data[owner]  # (n_faces, 3)

            # Wall-normal component of velocity
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)  # (n_faces, 1)

            # Tangential velocity (remove wall-normal component)
            U_t = U_P - U_n * n  # (n_faces, 3)

            # Wall shear stress: τ_w = μ * U_t / d
            # where d is the distance from cell centre to wall face
            tau_w = mu * U_t / d_Pf_mag.clamp(min=1e-30)  # (n_faces, 3)

            tau_w_patches[patch_name] = tau_w

        return tau_w_patches

    def _get_viscosity(self) -> float:
        """Get dynamic viscosity."""
        return float(self.config.get("mu", 1.0))

    def _get_patch_info(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """Get patch information from mesh boundary."""
        if not hasattr(self._mesh, "boundary"):
            return None
        for p in self._mesh.boundary:
            if p["name"] == patch_name:
                return p
        return None

    def write(self) -> None:
        """Write wall shear stress data to output files."""
        if self._output_path is None or not self._times:
            return

        # Write wallShearStress.dat
        wss_file = self._output_path / "wallShearStress.dat"
        with open(wss_file, "w") as f:
            f.write("# Time  patch  tau_w_x  tau_w_y  tau_w_z  tau_w_mag_avg  tau_w_mag_max\n")
            for i, t in enumerate(self._times):
                for patch_name, tau in self._tau_w_history[i].items():
                    tau_mag = tau.norm(dim=1)
                    f.write(
                        f"{t:.6e}  {patch_name}  "
                        f"{tau[:, 0].mean():.6e} {tau[:, 1].mean():.6e} {tau[:, 2].mean():.6e}  "
                        f"{tau_mag.mean():.6e} {tau_mag.max():.6e}\n"
                    )
        logger.info("Wrote wall shear stress to %s", wss_file)

    @property
    def tau_w_history(self) -> List[Dict[str, torch.Tensor]]:
        """Wall shear stress history (per patch per time step)."""
        return self._tau_w_history

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# Register
FunctionObjectRegistry.register("wallShearStress", WallShearStress)
