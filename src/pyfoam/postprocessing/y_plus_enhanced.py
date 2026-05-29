"""
YPlusEnhanced — enhanced y+ computation with wall function recommendations.

Extends the basic :class:`~pyfoam.postprocessing.y_plus.YPlus` function
object with:

- Wall function regime classification (low-Re, buffer, log-law)
- Recommended wall treatment per patch
- Spatial y+ distribution statistics (percentiles, histograms)
- Time-averaged y+ tracking
- y+ convergence monitoring

Physics
-------
y+ = y * u_tau / nu

Regime classification:
    - y+ < 5:    Viscous sublayer (low-Re resolved)
    - 5 < y+ < 30: Buffer layer (transition)
    - 30 < y+ < 300: Log-law region (wall functions)
    - y+ > 300: Too coarse for wall-bounded flows

Configuration keys:

- ``patches``: wall patch names (default: auto-detect)
- ``rho``: density (default: 1.0)
- ``mu``: dynamic viscosity (default: 1.0)
- ``percentiles``: list of percentiles to compute (default: [5, 25, 50, 75, 95])
- ``histogramBins``: number of histogram bins (default: 50)

Usage::

    ype = YPlusEnhanced("yPlusEnhanced1", {"rho": 1.0, "mu": 1e-5})
    ype.initialise(mesh, fields)
    ype.execute(1.0)
    recommendation = ype.get_wall_treatment_recommendation("movingWall")

References
----------
- OpenFOAM ``yPlus`` function object
- Spalding, D.B. "A single formula for the law of the wall" (1961)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["YPlusEnhanced", "WallTreatment", "YPatchStats"]

logger = logging.getLogger(__name__)


# Regime thresholds
_VISCOUS_SUBLAYER_MAX = 5.0
_BUFFER_LAYER_MAX = 30.0
_LOG_LAW_MAX = 300.0


class WallTreatment:
    """Wall treatment recommendation."""

    LOW_RE = "low-Re resolved"
    BUFFER = "buffer layer (refine mesh or use blended)"
    WALL_FUNCTION = "wall function"
    TOO_COARSE = "too coarse (refine mesh)"

    @staticmethod
    def classify(y_plus_mean: float) -> str:
        """Classify the wall treatment regime based on mean y+.

        Args:
            y_plus_mean: Mean y+ value.

        Returns:
            Wall treatment recommendation string.
        """
        if y_plus_mean < _VISCOUS_SUBLAYER_MAX:
            return WallTreatment.LOW_RE
        elif y_plus_mean < _BUFFER_LAYER_MAX:
            return WallTreatment.BUFFER
        elif y_plus_mean < _LOG_LAW_MAX:
            return WallTreatment.WALL_FUNCTION
        else:
            return WallTreatment.TOO_COARSE


@dataclass
class YPatchStats:
    """Statistics for y+ on a single wall patch.

    Attributes:
        patch_name: Patch name.
        mean: Mean y+.
        min: Minimum y+.
        max: Maximum y+.
        std: Standard deviation of y+.
        percentiles: Dict mapping percentile (0-100) to y+ value.
        regime: Wall treatment regime string.
        n_faces: Number of wall faces.
    """

    patch_name: str = ""
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    regime: str = ""
    n_faces: int = 0


class YPlusEnhanced(FunctionObject):
    """Enhanced y+ computation with wall function recommendations.

    Configuration keys:

    - ``patches``: wall patch names (default: auto-detect)
    - ``rho``: density (default: 1.0)
    - ``mu``: dynamic viscosity (default: 1.0)
    - ``percentiles``: list of percentiles (default: [5, 25, 50, 75, 95])
    - ``histogramBins``: number of histogram bins (default: 50)
    """

    def __init__(self, name: str = "yPlusEnhanced", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._patches: List[str] = self.config.get("patches", [])
        self._rho: float = float(self.config.get("rho", 1.0))
        self._mu: float = float(self.config.get("mu", 1.0))
        self._percentile_levels: List[int] = self.config.get("percentiles", [5, 25, 50, 75, 95])
        self._n_bins: int = int(self.config.get("histogramBins", 50))

        # Results storage
        self._patch_history: List[Dict[str, YPatchStats]] = []
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and validate patches."""
        self._mesh = mesh
        self._fields = fields

        if not self._patches and hasattr(mesh, "boundary"):
            self._patches = [
                p["name"] for p in mesh.boundary
                if p.get("type", "").lower() in ("wall",)
            ]

        logger.info("YPlusEnhanced '%s' initialised: patches=%s", self.name, self._patches)

    def execute(self, time: float) -> None:
        """Compute enhanced y+ at the current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required for YPlusEnhanced. Skipping.")
            return

        y_plus_per_patch = self._compute_y_plus(U)

        # Compute statistics for each patch
        stats: Dict[str, YPatchStats] = {}
        for patch_name, yp in y_plus_per_patch.items():
            stats[patch_name] = self._compute_patch_stats(patch_name, yp)

        self._patch_history.append(stats)
        self._times.append(time)

        for patch_name, ps in stats.items():
            self._log.info(
                "t=%g  patch='%s'  y+_mean=%.2f  y+_min=%.2f  y+_max=%.2f  regime='%s'",
                time, patch_name, ps.mean, ps.min, ps.max, ps.regime,
            )

    def _compute_y_plus(self, U_field) -> Dict[str, torch.Tensor]:
        """Compute y+ on each wall patch."""
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        face_centres = mesh.face_centres.to(device=device, dtype=dtype)
        cell_centres = mesh.cell_centres.to(device=device, dtype=dtype)

        nu = self._mu / self._rho
        y_plus_patches: Dict[str, torch.Tensor] = {}

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            S = face_areas[face_indices]
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=1e-30)

            owner = mesh.owner[face_indices]
            x_P = cell_centres[owner]
            x_f = face_centres[face_indices]
            d = torch.abs(torch.sum((x_f - x_P) * n, dim=1))

            U_P = U_data[owner]
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)
            U_t = U_P - U_n * n
            U_t_mag = U_t.norm(dim=1)

            tau_w = self._mu * U_t_mag / d.clamp(min=1e-30)
            u_tau = torch.sqrt(tau_w / self._rho)
            y_p = d * u_tau / nu

            y_plus_patches[patch_name] = y_p

        return y_plus_patches

    def _compute_patch_stats(self, patch_name: str, yp: torch.Tensor) -> YPatchStats:
        """Compute statistics for a patch's y+ values."""
        yp_np = yp.detach().cpu()

        # Percentiles
        percentiles: Dict[int, float] = {}
        for p_level in self._percentile_levels:
            percentiles[p_level] = float(torch.quantile(yp_np, p_level / 100.0).item())

        mean_val = float(yp_np.mean().item())
        regime = WallTreatment.classify(mean_val)

        return YPatchStats(
            patch_name=patch_name,
            mean=mean_val,
            min=float(yp_np.min().item()),
            max=float(yp_np.max().item()),
            std=float(yp_np.std().item()),
            percentiles=percentiles,
            regime=regime,
            n_faces=yp_np.shape[0],
        )

    def _get_patch_info(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """Get patch information from mesh boundary."""
        if not hasattr(self._mesh, "boundary"):
            return None
        for p in self._mesh.boundary:
            if p["name"] == patch_name:
                return p
        return None

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_wall_treatment_recommendation(self, patch_name: str) -> str:
        """Get the wall treatment recommendation for a patch.

        Uses the most recent time step's y+ statistics.

        Args:
            patch_name: Name of the wall patch.

        Returns:
            Recommendation string.

        Raises:
            RuntimeError: If no data is available.
        """
        if not self._patch_history:
            raise RuntimeError("No data available. Call execute() first.")

        latest = self._patch_history[-1]
        if patch_name not in latest:
            raise KeyError(f"Patch '{patch_name}' not found. Available: {list(latest.keys())}")

        return latest[patch_name].regime

    def get_latest_stats(self, patch_name: str) -> YPatchStats:
        """Get the latest y+ statistics for a patch.

        Args:
            patch_name: Name of the wall patch.

        Returns:
            :class:`YPatchStats` from the most recent time step.
        """
        if not self._patch_history:
            raise RuntimeError("No data available. Call execute() first.")

        latest = self._patch_history[-1]
        if patch_name not in latest:
            raise KeyError(f"Patch '{patch_name}' not found.")
        return latest[patch_name]

    @property
    def patch_history(self) -> List[Dict[str, YPatchStats]]:
        """Full history of patch statistics."""
        return self._patch_history

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def write(self) -> None:
        """Write enhanced y+ data to output files."""
        if self._output_path is None or not self._times:
            return

        # Main y+ statistics file
        stats_file = self._output_path / "yPlusEnhanced.dat"
        with open(stats_file, "w") as f:
            header = "# Time  patch  y+_avg  y+_min  y+_max  y+_std"
            for p in self._percentile_levels:
                header += f"  y+_{p}pct"
            header += "  regime"
            f.write(header + "\n")

            for i, t in enumerate(self._times):
                for patch_name, ps in self._patch_history[i].items():
                    line = (
                        f"{t:.6e}  {patch_name}  "
                        f"{ps.mean:.4f} {ps.min:.4f} {ps.max:.4f} {ps.std:.4f}"
                    )
                    for p in self._percentile_levels:
                        line += f"  {ps.percentiles.get(p, 0.0):.4f}"
                    line += f"  {ps.regime}"
                    f.write(line + "\n")

        # Recommendation summary
        rec_file = self._output_path / "wall_treatment_recommendation.txt"
        with open(rec_file, "w") as f:
            f.write("# Wall Treatment Recommendations\n")
            f.write(f"# Based on {len(self._times)} time steps\n\n")
            if self._patch_history:
                latest = self._patch_history[-1]
                for patch_name, ps in latest.items():
                    f.write(f"  {patch_name:20s}  y+_mean={ps.mean:8.2f}  -> {ps.regime}\n")

        logger.info("Wrote YPlusEnhanced to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced", YPlusEnhanced)
