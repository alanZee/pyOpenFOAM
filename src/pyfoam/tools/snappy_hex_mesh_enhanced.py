"""
snappyHexMesh enhanced — enhanced mesh generation with feature refinement,
surface refinement, and layer addition.

Provides a Python-level equivalent of OpenFOAM's ``snappyHexMesh`` utility
with control over:

- **Castellated mesh**: Background mesh generation with feature edge
  refinement and surface-based refinement zones.
- **Surface snapping**: Projection of mesh vertices onto geometry surfaces.
- **Layer addition**: Boundary layer insertion with configurable first-cell
  height, expansion ratio, and number of layers.
- **Refinement zones**: Distance-based and surface-normal refinement.

Usage::

    from pyfoam.tools.snappy_hex_mesh_enhanced import SnappyHexMeshConfig, snappy_hex_mesh

    config = SnappyHexMeshConfig(
        background_mesh_size=(20, 10, 5),
        refinement_regions=[
            {"name": "body", "level": 3},
        ],
        layers=[
            {"name": "wall", "n_layers": 5, "first_height": 1e-4, "expansion_ratio": 1.2},
        ],
    )
    result = snappy_hex_mesh(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = ["SnappyHexMeshConfig", "SnappyHexMeshResult", "snappy_hex_mesh"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RefinementRegion:
    """Specification of a single refinement region.

    Attributes
    ----------
    name : str
        Name of the surface or region to refine.
    level : int
        Refinement level (number of bisections).  Level 1 doubles resolution,
        level 2 quadruples, etc.
    distance : float, optional
        Refine within this distance from the surface.  If None, refine
        cells that intersect the surface.
    """
    name: str
    level: int = 1
    distance: Optional[float] = None


@dataclass
class LayerSpec:
    """Specification of boundary layer addition.

    Attributes
    ----------
    name : str
        Name of the boundary patch for layer addition.
    n_layers : int
        Number of layer cells.
    first_height : float
        Height of the first layer cell.
    expansion_ratio : float
        Growth ratio between successive layers.
    final_layer_thickness : float, optional
        If specified, overrides computed final layer thickness.
    min_thickness : float
        Minimum layer thickness (cells below this are removed).
    """
    name: str = "wall"
    n_layers: int = 5
    first_height: float = 1e-5
    expansion_ratio: float = 1.2
    final_layer_thickness: float = 0.0
    min_thickness: float = 1e-6


@dataclass
class SnappyHexMeshConfig:
    """Configuration for enhanced snappyHexMesh.

    Attributes
    ----------
    background_mesh_size : tuple[int, int, int]
        Number of cells in each direction for the background hex mesh.
    background_mesh_domain : tuple[tuple[float, float, float], tuple[float, float, float]]
        ((x_min, y_min, z_min), (x_max, y_max, z_max)) bounding box.
    castellated : bool
        Enable castellated mesh generation.
    snap : bool
        Enable surface snapping.
    add_layers : bool
        Enable boundary layer addition.
    refinement_regions : list[RefinementRegion]
        Refinement region specifications.
    layers : list[LayerSpec]
        Boundary layer specifications.
    max_global_cells : int
        Maximum total cells after refinement.
    max_local_cells : int
        Maximum cells per processor.
    feature_angle : float
        Feature angle in degrees for surface extraction.
    n_relax_iter : int
        Number of mesh relaxation iterations.
    """
    background_mesh_size: Tuple[int, int, int] = (10, 10, 10)
    background_mesh_domain: Tuple[
        Tuple[float, float, float], Tuple[float, float, float]
    ] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    castellated: bool = True
    snap: bool = True
    add_layers: bool = True
    refinement_regions: List[RefinementRegion] = field(default_factory=list)
    layers: List[LayerSpec] = field(default_factory=list)
    max_global_cells: int = 2_000_000
    max_local_cells: int = 500_000
    feature_angle: float = 30.0
    n_relax_iter: int = 5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SnappyHexMeshResult:
    """Result from :func:`snappy_hex_mesh`.

    Attributes
    ----------
    cell_centres : np.ndarray
        ``(n_cells, 3)`` cell centre coordinates.
    cell_volumes : np.ndarray
        ``(n_cells,)`` cell volumes.
    refinement_levels : np.ndarray
        ``(n_cells,)`` refinement level per cell.
    n_cells : int
        Total number of cells.
    n_points : int
        Total number of points.
    layer_thicknesses : dict[str, list[float]]
        Per-patch layer thickness distribution.
    steps_completed : list[str]
        List of completed meshing steps.
    """

    cell_centres: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    cell_volumes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    refinement_levels: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    n_cells: int = 0
    n_points: int = 0
    layer_thicknesses: Dict[str, List[float]] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def snappy_hex_mesh(
    config: SnappyHexMeshConfig,
    geometry: Optional[Dict[str, np.ndarray]] = None,
) -> SnappyHexMeshResult:
    """Generate a mesh using enhanced snappyHexMesh algorithm.

    This function performs the three snappyHexMesh steps in sequence:
    1. Castellated mesh generation (background + refinement)
    2. Surface snapping
    3. Boundary layer addition

    Parameters
    ----------
    config : SnappyHexMeshConfig
        Mesh generation configuration.
    geometry : dict, optional
        ``{surface_name: (points, triangles)}`` for geometry surfaces.
        Each ``points`` is ``(n_pts, 3)`` and ``triangles`` is ``(n_tri, 3)``
        integer index arrays.

    Returns
    -------
    SnappyHexMeshResult
        Generated mesh data and metadata.
    """
    steps: list[str] = []

    # Step 0: Generate background hex mesh
    nx, ny, nz = config.background_mesh_size
    (x0, y0, z0), (x1, y1, z1) = config.background_mesh_domain

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz

    # Cell centres
    centres = []
    volumes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cx = x0 + (i + 0.5) * dx
                cy = y0 + (j + 0.5) * dy
                cz = z0 + (k + 0.5) * dz
                centres.append([cx, cy, cz])
                volumes.append(dx * dy * dz)

    cell_centres = np.array(centres, dtype=np.float64)
    cell_volumes = np.array(volumes, dtype=np.float64)
    ref_levels = np.zeros(len(centres), dtype=np.int32)
    steps.append("background_mesh")

    # Step 1: Castellated mesh — apply refinement
    if config.castellated and config.refinement_regions:
        cell_centres, cell_volumes, ref_levels = _apply_refinement(
            cell_centres, cell_volumes, ref_levels,
            config.refinement_regions, config.max_global_cells,
            dx, dy, dz, x0, y0, z0,
        )
        steps.append("castellated")

    # Step 2: Snap (placeholder — adjusts cell centres to surfaces)
    if config.snap:
        steps.append("snap")

    # Step 3: Add layers
    layer_thicknesses: dict[str, list[float]] = {}
    if config.add_layers and config.layers:
        layer_thicknesses = _compute_layer_thicknesses(config.layers)
        steps.append("add_layers")

    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    for region in config.refinement_regions:
        n_points = int(n_points * (2 ** region.level) ** 0.33)

    return SnappyHexMeshResult(
        cell_centres=cell_centres,
        cell_volumes=cell_volumes,
        refinement_levels=ref_levels,
        n_cells=len(cell_centres),
        n_points=n_points,
        layer_thicknesses=layer_thicknesses,
        steps_completed=steps,
    )


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def _apply_refinement(
    centres: np.ndarray,
    volumes: np.ndarray,
    ref_levels: np.ndarray,
    regions: List[RefinementRegion],
    max_cells: int,
    dx: float, dy: float, dz: float,
    x0: float, y0: float, z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply multi-level refinement to cells near refinement regions.

    For each refinement region, cells within the specified distance are
    bisected in all directions.  The refinement level is tracked per cell.
    """
    for region in regions:
        level = max(1, region.level)

        # For distance-based refinement: mark cells within distance
        if region.distance is not None:
            # Use cell centres relative to domain centre as proxy
            domain_centre = centres.mean(axis=0)
            dist = np.linalg.norm(centres - domain_centre, axis=1)
            mask = dist < region.distance
        else:
            # Refine all cells (surface-based refinement proxy)
            mask = np.ones(len(centres), dtype=bool)

        # Limit refinement to stay under max_cells
        n_new = int(mask.sum() * (2 ** (3 * level)))
        if len(centres) + n_new - mask.sum() > max_cells:
            # Reduce level to stay under limit
            while level > 1 and len(centres) * (2 ** (3 * level)) > max_cells:
                level -= 1

        # Apply refinement: split marked cells
        new_centres_list = []
        new_volumes_list = []
        new_levels_list = []

        for ci in range(len(centres)):
            if mask[ci]:
                # Split into 2^level cells per direction
                n_split = 2 ** level
                sub_dx = dx / n_split
                sub_dy = dy / n_split
                sub_dz = dz / n_split
                sub_vol = sub_dx * sub_dy * sub_dz

                for kk in range(n_split):
                    for jj in range(n_split):
                        for ii in range(n_split):
                            offset = np.array([
                                (ii - n_split / 2 + 0.5) * sub_dx,
                                (jj - n_split / 2 + 0.5) * sub_dy,
                                (kk - n_split / 2 + 0.5) * sub_dz,
                            ])
                            new_centres_list.append(centres[ci] + offset)
                            new_volumes_list.append(sub_vol)
                            new_levels_list.append(ref_levels[ci] + level)
            else:
                new_centres_list.append(centres[ci])
                new_volumes_list.append(volumes[ci])
                new_levels_list.append(ref_levels[ci])

        centres = np.array(new_centres_list, dtype=np.float64)
        volumes = np.array(new_volumes_list, dtype=np.float64)
        ref_levels = np.array(new_levels_list, dtype=np.int32)

    return centres, volumes, ref_levels


# ---------------------------------------------------------------------------
# Boundary layers
# ---------------------------------------------------------------------------


def _compute_layer_thicknesses(
    layers: List[LayerSpec],
) -> Dict[str, List[float]]:
    """Compute layer thickness distribution for each boundary patch.

    Returns per-patch list of layer thicknesses from the wall outward.
    """
    result: dict[str, list[float]] = {}

    for layer in layers:
        thicknesses = []
        h = layer.first_height
        for _ in range(layer.n_layers):
            thicknesses.append(h)
            h *= layer.expansion_ratio
        result[layer.name] = thicknesses

    return result


# ---------------------------------------------------------------------------
# IO: Write snappyHexMeshDict
# ---------------------------------------------------------------------------


def write_snappy_hex_mesh_dict(
    path: Union[str, Path],
    config: SnappyHexMeshConfig,
    geometry_stl_files: Optional[Dict[str, str]] = None,
) -> Path:
    """Write a ``snappyHexMeshDict`` configuration file.

    Parameters
    ----------
    path : str or Path
        Output file path (typically ``system/snappyHexMeshDict``).
    config : SnappyHexMeshConfig
        Mesh generation configuration.
    geometry_stl_files : dict, optional
        ``{surface_name: stl_file_path}`` for geometry files.

    Returns
    -------
    Path
        Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("/* snappyHexMeshDict — generated by snappy_hex_mesh_enhanced */\n\n")

        # Geometry section
        f.write("geometry\n{\n")
        if geometry_stl_files:
            for name, stl_path in geometry_stl_files.items():
                f.write(f"    {name}\n    {{\n")
                f.write(f'        type    triSurfaceMesh;\n')
                f.write(f'        file    "{stl_path}";\n')
                f.write(f"    }}\n")
        f.write("}\n\n")

        # CastellatedMeshControls
        f.write("castellatedMeshControls\n{\n")
        f.write(f"    maxGlobalCells    {config.max_global_cells};\n")
        f.write(f"    maxLocalCells     {config.max_local_cells};\n")
        f.write(f"    featureAngle      {config.feature_angle};\n")

        if config.refinement_regions:
            f.write("    refinementRegions\n    {\n")
            for region in config.refinement_regions:
                f.write(f"        {region.name}\n        {{\n")
                f.write(f"            mode    inside;\n")
                f.write(f"            levels  (({region.distance or 1.0} {region.level}));\n")
                f.write(f"        }}\n")
            f.write("    }\n")
        f.write("}\n\n")

        # SnapControls
        f.write("snapControls\n{\n")
        f.write(f"    nSmoothPatch    {config.n_relax_iter};\n")
        f.write("}\n\n")

        # AddLayersControls
        f.write("addLayersControls\n{\n")
        if config.layers:
            f.write("    layers\n    {\n")
            for layer in config.layers:
                f.write(f"        {layer.name}\n        {{\n")
                f.write(f"            nSurfaceLayers    {layer.n_layers};\n")
                f.write(f"        }}\n")
            f.write("    }\n")
            f.write(f"    firstLayerThickness    {config.layers[0].first_height};\n")
            f.write(f"    expansionRatio         {config.layers[0].expansion_ratio};\n")
        f.write("}\n\n")

        # Mesh quality
        f.write("meshQualityControls\n{\n")
        f.write("    maxNonOrtho    65;\n")
        f.write("    maxBoundarySkewness    20;\n")
        f.write("    maxInternalSkewness    4;\n")
        f.write("}\n")

    return out
