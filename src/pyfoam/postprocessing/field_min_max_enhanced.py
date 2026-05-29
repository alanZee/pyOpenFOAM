"""
FieldMinMaxEnhanced — Enhanced field min/max with cell location and spatial info.

Extends :class:`~pyfoam.postprocessing.field_min_max.FieldMinMax` with:

- **Cell centre coordinates** for min/max locations
- **Per-patch** min/max analysis
- **Time history** tracking with convergence monitoring
- **Statistical measures** (mean, std, range) in addition to min/max
- **Gradient information** at extrema locations

Configuration keys (beyond FieldMinMax):

- ``computeGradient``: compute gradient at extrema (default: False)
- ``perPatch``: compute per-patch min/max (default: False)
- ``trackHistory``: keep full time history (default: True)
- ``patches``: list of patch names for per-patch analysis (default: all)

References
----------
- OpenFOAM ``fieldMinMax`` function object source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max import FieldMinMax, MinMaxResult

__all__ = ["FieldMinMaxEnhanced", "EnhancedMinMaxResult"]

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMinMaxResult:
    """Extended min/max result with spatial coordinates and statistics.

    Attributes:
        time: Simulation time.
        field_name: Field name.
        min_value: Global minimum value.
        max_value: Global maximum value.
        min_location: Cell index of minimum.
        max_location: Cell index of maximum.
        min_coords: ``(x, y, z)`` coordinates of minimum cell centre.
        max_coords: ``(x, y, z)`` coordinates of maximum cell centre.
        mean: Field mean value.
        std: Field standard deviation.
        range: max - min.
        gradient_at_min: Gradient vector at minimum (if computed).
        gradient_at_max: Gradient vector at maximum (if computed).
    """

    time: float = 0.0
    field_name: str = ""
    min_value: float = 0.0
    max_value: float = 0.0
    min_location: int = -1
    max_location: int = -1
    min_coords: tuple = (0.0, 0.0, 0.0)
    max_coords: tuple = (0.0, 0.0, 0.0)
    mean: float = 0.0
    std: float = 0.0
    range: float = 0.0
    gradient_at_min: Optional[tuple] = None
    gradient_at_max: Optional[tuple] = None


class FieldMinMaxEnhanced(FieldMinMax):
    """Enhanced field min/max with cell location and spatial statistics.

    Extends :class:`FieldMinMax` with spatial information and statistics.

    Additional configuration keys (beyond FieldMinMax):

    - ``computeGradient``: compute gradient at extrema (default: False)
    - ``perPatch``: compute per-patch min/max (default: False)
    - ``trackHistory``: keep full time history (default: True)
    - ``patches``: list of patch names for per-patch analysis

    Example controlDict entry::

        fieldMinMaxEnhanced1
        {
            type            fieldMinMaxEnhanced;
            libs            ("libfieldFunctionObjects.so");
            field           p;
            computeGradient true;
            perPatch        true;
        }
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_gradient: bool = self.config.get("computeGradient", False)
        self._per_patch: bool = self.config.get("perPatch", False)
        self._track_history: bool = self.config.get("trackHistory", True)
        self._patch_names: List[str] = self.config.get("patches", [])

        # Enhanced results
        self._enhanced_results: List[EnhancedMinMaxResult] = []
        self._patch_results: Dict[str, List[MinMaxResult]] = {}

    def execute(self, time: float) -> None:
        """Compute enhanced min/max at the current time step."""
        if not self._enabled:
            return

        field = self._fields.get(self._field_name)
        if field is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        # Compute base result
        base_result = self._compute_min_max(field, time)

        # Enhance with spatial information
        enhanced = self._enhance_result(field, base_result, time)

        if self._track_history:
            self._enhanced_results.append(enhanced)

        # Also store in base class results
        self._results.append(base_result)

        # Per-patch analysis
        if self._per_patch:
            self._compute_per_patch(field, time)

        if self._do_log:
            self._log.info(
                "t=%g  field='%s'  min=%.6g @ cell %d (%.3f, %.3f, %.3f)  "
                "max=%.6g @ cell %d (%.3f, %.3f, %.3f)  mean=%.6g  std=%.6g",
                time, self._field_name,
                enhanced.min_value, enhanced.min_location,
                *enhanced.min_coords,
                enhanced.max_value, enhanced.max_location,
                *enhanced.max_coords,
                enhanced.mean, enhanced.std,
            )

    def _enhance_result(
        self, field, base_result: MinMaxResult, time: float,
    ) -> EnhancedMinMaxResult:
        """Add spatial coordinates and statistics to a base result."""
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        elif hasattr(field, "data"):
            data = field.data.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        # Compute statistics
        if data.dim() == 1:
            mean_val = float(data.mean().item())
            std_val = float(data.std().item())
        elif data.dim() == 2 and data.shape[1] == 3:
            mag = data.norm(dim=1)
            mean_val = float(mag.mean().item())
            std_val = float(mag.std().item())
        else:
            mean_val = float(data.mean().item())
            std_val = float(data.std().item())

        range_val = base_result.max_value - base_result.min_value

        # Get cell centre coordinates
        min_coords = (0.0, 0.0, 0.0)
        max_coords = (0.0, 0.0, 0.0)
        if self._mesh is not None and hasattr(self._mesh, "cell_centres"):
            cc = self._mesh.cell_centres.to(device=device, dtype=dtype)
            n_cells = cc.shape[0]

            if 0 <= base_result.min_location < n_cells:
                c = cc[base_result.min_location]
                min_coords = (c[0].item(), c[1].item(), c[2].item())

            if 0 <= base_result.max_location < n_cells:
                c = cc[base_result.max_location]
                max_coords = (c[0].item(), c[1].item(), c[2].item())

        # Gradient computation (simple finite-difference approximation)
        grad_at_min = None
        grad_at_max = None
        if self._compute_gradient and self._mesh is not None:
            grad_at_min, grad_at_max = self._compute_gradients_at_extrema(
                data, base_result, device, dtype,
            )

        return EnhancedMinMaxResult(
            time=time,
            field_name=self._field_name,
            min_value=base_result.min_value,
            max_value=base_result.max_value,
            min_location=base_result.min_location,
            max_location=base_result.max_location,
            min_coords=min_coords,
            max_coords=max_coords,
            mean=mean_val,
            std=std_val,
            range=range_val,
            gradient_at_min=grad_at_min,
            gradient_at_max=grad_at_max,
        )

    def _compute_gradients_at_extrema(
        self,
        data: torch.Tensor,
        result: MinMaxResult,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """Compute gradient at min and max locations using cell neighbours.

        Uses simple central-difference approximation with neighbour cells.
        """
        mesh = self._mesh
        if not hasattr(mesh, "owner") or not hasattr(mesh, "neighbour"):
            return None, None

        owner = mesh.owner
        neighbour = mesh.neighbour
        n_internal = mesh.n_internal_faces

        # Build adjacency for min and max cells
        grad_min = self._estimate_cell_gradient(
            data, result.min_location, owner, neighbour, n_internal, device, dtype,
        )
        grad_max = self._estimate_cell_gradient(
            data, result.max_location, owner, neighbour, n_internal, device, dtype,
        )

        return grad_min, grad_max

    def _estimate_cell_gradient(
        self,
        data: torch.Tensor,
        cell_idx: int,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_internal: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple]:
        """Estimate gradient at a cell using neighbour values (scalar only)."""
        if data.dim() != 1:
            return None

        if cell_idx < 0:
            return None

        # Find neighbours
        neighbours = []
        for fi in range(n_internal):
            o = int(owner[fi].item())
            n = int(neighbour[fi].item())
            if o == cell_idx:
                neighbours.append(n)
            elif n == cell_idx:
                neighbours.append(o)

        if not neighbours:
            return None

        # Average gradient magnitude (simplified)
        cell_val = data[cell_idx].item()
        grad_mags = []
        for nb in neighbours:
            if nb < data.shape[0]:
                grad_mags.append(abs(data[nb].item() - cell_val))

        avg_grad = sum(grad_mags) / len(grad_mags) if grad_mags else 0.0
        return (avg_grad, 0.0, 0.0)  # Simplified: scalar gradient magnitude

    def _compute_per_patch(self, field, time: float) -> None:
        """Compute min/max per boundary patch."""
        mesh = self._mesh
        if mesh is None or not hasattr(mesh, "boundary"):
            return

        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        owner = mesh.owner

        # Auto-detect patches if not specified
        patches = self._patch_names
        if not patches:
            patches = [p["name"] for p in mesh.boundary]

        for patch_name in patches:
            patch_info = None
            for p in mesh.boundary:
                if p["name"] == patch_name:
                    patch_info = p
                    break

            if patch_info is None:
                continue

            start = patch_info["startFace"]
            n_faces = patch_info["nFaces"]

            # Get owner cells of patch faces
            face_indices = torch.arange(start, start + n_faces, device=device, dtype=torch.long)
            cell_indices = owner[face_indices]

            # Filter valid indices
            valid = cell_indices[cell_indices < data.shape[0]]
            if valid.numel() == 0:
                continue

            if data.dim() == 1:
                patch_data = data[valid]
                min_val = patch_data.min()
                max_val = patch_data.max()
                min_loc = int(valid[patch_data.argmin()].item())
                max_loc = int(valid[patch_data.argmax()].item())
            else:
                # Vector: use magnitude
                patch_data = data[valid]
                mag = patch_data.norm(dim=1)
                min_val = mag.min()
                max_val = mag.max()
                min_loc = int(valid[mag.argmin()].item())
                max_loc = int(valid[mag.argmax()].item())

            result = MinMaxResult(
                time=time,
                field_name=self._field_name,
                min_value=min_val.item(),
                max_value=max_val.item(),
                min_location=min_loc,
                max_location=max_loc,
            )

            if patch_name not in self._patch_results:
                self._patch_results[patch_name] = []
            self._patch_results[patch_name].append(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enhanced_results(self) -> List[EnhancedMinMaxResult]:
        """All enhanced min/max results."""
        return self._enhanced_results

    @property
    def patch_results(self) -> Dict[str, List[MinMaxResult]]:
        """Per-patch min/max results.  ``{patch_name: [results]}``."""
        return self._patch_results

    def get_latest_enhanced(self) -> Optional[EnhancedMinMaxResult]:
        """Get the most recent enhanced result."""
        return self._enhanced_results[-1] if self._enhanced_results else None

    def get_patch_latest(self, patch_name: str) -> Optional[MinMaxResult]:
        """Get the latest result for a specific patch.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Latest MinMaxResult for the patch, or None.
        """
        results = self._patch_results.get(patch_name, [])
        return results[-1] if results else None

    def write(self) -> None:
        """Write enhanced min/max data to output files."""
        # Write base class output
        super().write()

        if self._output_path is None:
            return

        # Write enhanced results
        if self._enhanced_results:
            enhanced_file = self._output_path / "fieldMinMaxEnhanced.dat"
            with open(enhanced_file, "w") as f:
                header = (
                    "# Time  field  min  minCell  min_x  min_y  min_z  "
                    "max  maxCell  max_x  max_y  max_z  mean  std  range"
                )
                if self._compute_gradient:
                    header += "  grad_min  grad_max"
                f.write(header + "\n")

                for r in self._enhanced_results:
                    line = (
                        f"{r.time:.6e}  {r.field_name}  "
                        f"{r.min_value:.10g}  {r.min_location}  "
                        f"{r.min_coords[0]:.6e}  {r.min_coords[1]:.6e}  {r.min_coords[2]:.6e}  "
                        f"{r.max_value:.10g}  {r.max_location}  "
                        f"{r.max_coords[0]:.6e}  {r.max_coords[1]:.6e}  {r.max_coords[2]:.6e}  "
                        f"{r.mean:.10g}  {r.std:.10g}  {r.range:.10g}"
                    )
                    if self._compute_gradient and r.gradient_at_min is not None:
                        line += f"  {r.gradient_at_min[0]:.6e}  {r.gradient_at_max[0]:.6e}"
                    f.write(line + "\n")

            logger.info("Wrote enhanced FieldMinMax to %s", enhanced_file)

        # Write per-patch results
        if self._patch_results:
            for patch_name, results in self._patch_results.items():
                patch_file = self._output_path / f"fieldMinMax_{patch_name}.dat"
                with open(patch_file, "w") as f:
                    f.write("# Time  min  minCell  max  maxCell\n")
                    for r in results:
                        f.write(
                            f"{r.time:.6e}  {r.min_value:.10g}  {r.min_location}  "
                            f"{r.max_value:.10g}  {r.max_location}\n"
                        )
                logger.info("Wrote patch FieldMinMax to %s", patch_file)


# Register
from pyfoam.postprocessing.function_object import FunctionObjectRegistry
FunctionObjectRegistry.register("fieldMinMaxEnhanced", FieldMinMaxEnhanced)
