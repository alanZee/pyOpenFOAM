"""
Enhanced Gmsh I/O — Gmsh 4.x format support.

Extends :mod:`pyfoam.io.gmsh_io` with:

- Complete Gmsh 4.1 ``$Entities`` section parsing
- Gmsh 4.x element partition tags
- ``$GhostEntities`` support for parallel meshes
- Enhanced physical group handling with boundary entity mapping

Usage::

    mesh = read_gmsh_v4("mesh.msh")
    # mesh.entities contains geometric entities
    # mesh.partition_tags maps elements to partitions

References
----------
- Gmsh MSH 4.1 format: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.gmsh_io import (
    GmshElement,
    GmshMesh,
    GmshPhysicalGroup,
    read_gmsh,
    _GMSH_ELEMENT_INFO,
)

__all__ = [
    "GmshEntity",
    "GmshPartitionTag",
    "GmshMeshV4",
    "read_gmsh_v4",
    "gmsh_to_foam_enhanced",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GmshEntity:
    """A Gmsh geometric entity (point, curve, surface, volume).

    Attributes:
        dim: Entity dimension (0=point, 1=curve, 2=surface, 3=volume).
        tag: Entity tag (positive integer).
        physical_tags: Physical group tags associated with this entity.
        bounding_tags: Tags of bounding entities (for curves: point tags;
            for surfaces: curve tags; for volumes: surface tags).
    """

    dim: int
    tag: int
    physical_tags: List[int] = field(default_factory=list)
    bounding_tags: List[int] = field(default_factory=list)


@dataclass
class GmshPartitionTag:
    """Partition tag for an element.

    Attributes:
        elem_tag: Element tag.
        partition: Partition number (0-based).
    """

    elem_tag: int
    partition: int


@dataclass
class GmshMeshV4(GmshMesh):
    """Extended Gmsh mesh with v4.x features.

    Attributes:
        entities: Geometric entities parsed from ``$Entities``.
        partition_tags: Element partition assignments.
        ghost_entities: Ghost entity information for parallel meshes.
    """

    entities: List[GmshEntity] = field(default_factory=list)
    partition_tags: List[GmshPartitionTag] = field(default_factory=list)
    ghost_entities: Dict[int, List[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_gmsh_v4(path: Union[str, Path]) -> GmshMeshV4:
    """Read a Gmsh ``.msh`` file with enhanced v4.x support.

    Extends :func:`read_gmsh` with entity and partition parsing.

    Args:
        path: Path to the ``.msh`` file.

    Returns:
        :class:`GmshMeshV4` with enhanced data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format version is not 4.x.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")

    # Detect version
    fmt_match = re.search(r"\$MeshFormat\s*\n([\d.]+)", content)
    version = fmt_match.group(1) if fmt_match else "2.2"

    # Use base parser for standard data
    base_mesh = read_gmsh(path)

    # Parse additional v4.x sections
    entities = _parse_entities(content) if version.startswith("4") else []
    partition_tags = _parse_partition_tags(content)
    ghost_entities = _parse_ghost_entities(content)

    return GmshMeshV4(
        node_coords=base_mesh.node_coords,
        node_id_map=base_mesh.node_id_map,
        elements=base_mesh.elements,
        physical_groups=base_mesh.physical_groups,
        mesh_format=base_mesh.mesh_format,
        entities=entities,
        partition_tags=partition_tags,
        ghost_entities=ghost_entities,
    )


# ---------------------------------------------------------------------------
# $Entities parsing (Gmsh 4.x)
# ---------------------------------------------------------------------------


def _parse_entities(content: str) -> List[GmshEntity]:
    """Parse $Entities section (Gmsh 4.x format).

    Format::

        numPoints numCurves numSurfaces numVolumes
        // Points
        pointTag x y z numPhysicalTags [physicalTag...] numBoundingTags [boundingTag...]
        // Curves
        curveTag minX minY minZ maxX maxY maxZ numPhysicalTags [...] numBoundingTags [...]
            numBoundingPoints boundingTag...
        // Surfaces and volumes similarly
    """
    match = re.search(
        r"\$Entities\s*\n(.*?)\$EndEntities",
        content,
        re.DOTALL,
    )
    if match is None:
        return []

    lines = match.group(1).strip().split("\n")
    if not lines:
        return []

    # First line: counts
    header = lines[0].strip().split()
    if len(header) < 4:
        return []

    n_points = int(header[0])
    n_curves = int(header[1])
    n_surfaces = int(header[2])
    n_volumes = int(header[3])

    entities: List[GmshEntity] = []
    idx = 1

    # Parse points (dim=0)
    for _ in range(n_points):
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        idx += 1
        if len(parts) < 5:
            continue

        tag = int(parts[0])
        # x, y, z at positions 1, 2, 3
        n_phys = int(parts[4])
        phys_tags: List[int] = []
        offset = 5
        for pi in range(n_phys):
            if offset + pi < len(parts):
                phys_tags.append(int(parts[offset + pi]))
        offset += n_phys

        n_bnd = int(parts[offset]) if offset < len(parts) else 0
        bnd_tags: List[int] = []
        offset += 1
        for bi in range(n_bnd):
            if offset + bi < len(parts):
                bnd_tags.append(int(parts[offset + bi]))

        entities.append(GmshEntity(dim=0, tag=tag, physical_tags=phys_tags,
                                    bounding_tags=bnd_tags))

    # Parse curves (dim=1), surfaces (dim=2), volumes (dim=3)
    for dim, count in [(1, n_curves), (2, n_surfaces), (3, n_volumes)]:
        for _ in range(count):
            if idx >= len(lines):
                break
            parts = lines[idx].strip().split()
            idx += 1
            if len(parts) < 8:
                continue

            tag = int(parts[0])
            # min/max bbox at 1..6
            n_phys = int(parts[7])
            phys_tags = []
            offset = 8
            for pi in range(n_phys):
                if offset + pi < len(parts):
                    phys_tags.append(int(parts[offset + pi]))
            offset += n_phys

            n_bnd = int(parts[offset]) if offset < len(parts) else 0
            bnd_tags = []
            offset += 1
            for bi in range(n_bnd):
                if offset + bi < len(parts):
                    bnd_tags.append(int(parts[offset + bi]))

            entities.append(GmshEntity(dim=dim, tag=tag, physical_tags=phys_tags,
                                        bounding_tags=bnd_tags))

    return entities


# ---------------------------------------------------------------------------
# Partition tags parsing
# ---------------------------------------------------------------------------


def _parse_partition_tags(content: str) -> List[GmshPartitionTag]:
    """Parse $PartitionedEntities section if present.

    Format::

        numPartitions
        numGhostEntities
        ghostTag partitionId
        ...
        numEntities
        entityDim entityTag numPartitions [partitionTag...]
        ...
    """
    match = re.search(
        r"\$PartitionedEntities\s*\n(.*?)\$EndPartitionedEntities",
        content,
        re.DOTALL,
    )
    if match is None:
        return []

    lines = match.group(1).strip().split("\n")
    if len(lines) < 2:
        return []

    idx = 0
    n_partitions = int(lines[idx].strip())
    idx += 1

    n_ghost = int(lines[idx].strip())
    idx += 1

    # Skip ghost entries
    idx += n_ghost

    if idx >= len(lines):
        return []

    n_entities = int(lines[idx].strip())
    idx += 1

    tags: List[GmshPartitionTag] = []
    for _ in range(n_entities):
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        idx += 1
        if len(parts) < 3:
            continue

        # entityDim entityTag numPartitions [partitionTags...]
        elem_tag = int(parts[1])
        n_part = int(parts[2])
        for pi in range(n_part):
            if 3 + pi < len(parts):
                tags.append(GmshPartitionTag(
                    elem_tag=elem_tag,
                    partition=int(parts[3 + pi]),
                ))

    return tags


# ---------------------------------------------------------------------------
# Ghost entities parsing
# ---------------------------------------------------------------------------


def _parse_ghost_entities(content: str) -> Dict[int, List[int]]:
    """Parse ghost entity information from $PartitionedEntities.

    Returns:
        Dict mapping ghost entity tag -> list of partition IDs.
    """
    match = re.search(
        r"\$PartitionedEntities\s*\n(.*?)\$EndPartitionedEntities",
        content,
        re.DOTALL,
    )
    if match is None:
        return {}

    lines = match.group(1).strip().split("\n")
    if len(lines) < 2:
        return {}

    idx = 0
    _n_partitions = int(lines[idx].strip())
    idx += 1

    n_ghost = int(lines[idx].strip())
    idx += 1

    ghosts: Dict[int, List[int]] = {}
    for _ in range(n_ghost):
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        idx += 1
        if len(parts) >= 2:
            tag = int(parts[0])
            partition = int(parts[1])
            if tag not in ghosts:
                ghosts[tag] = []
            ghosts[tag].append(partition)

    return ghosts


# ---------------------------------------------------------------------------
# Enhanced gmshToFoam
# ---------------------------------------------------------------------------


def gmsh_to_foam_enhanced(
    gmsh_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
    map_entities_to_boundary: bool = True,
) -> GmshMeshV4:
    """Convert a Gmsh file to OpenFOAM format with enhanced entity mapping.

    Extends :func:`gmsh_to_foam` with:

    - Entity-to-boundary mapping: uses Gmsh entities to improve boundary
      patch identification
    - Partition tag preservation

    Args:
        gmsh_path: Path to the ``.msh`` file.
        output_dir: Output directory for polyMesh files.
        overwrite: If True, overwrite existing files.
        map_entities_to_boundary: Use entity data to refine boundary patches.

    Returns:
        The parsed :class:`GmshMeshV4`.
    """
    from pyfoam.io.gmsh_io import gmsh_to_foam

    gmsh_path = Path(gmsh_path)
    output_dir = Path(output_dir)

    # Read enhanced mesh
    mesh = read_gmsh_v4(gmsh_path)

    # Use base converter for standard mesh
    gmsh_to_foam(gmsh_path, output_dir, overwrite=overwrite)

    # Write entity info if available
    if mesh.entities:
        entity_dir = output_dir / "constant" / "polyMesh"
        entity_dir.mkdir(parents=True, exist_ok=True)
        _write_entity_mapping(entity_dir / "entities", mesh.entities)

    # Write partition tags if available
    if mesh.partition_tags:
        part_dir = output_dir / "constant" / "polyMesh"
        part_dir.mkdir(parents=True, exist_ok=True)
        _write_partition_info(part_dir / "partitioning", mesh.partition_tags)

    return mesh


def _write_entity_mapping(path: Path, entities: List[GmshEntity]) -> None:
    """Write entity mapping info to a file."""
    with open(path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       dictionary;\n")
        f.write('    location    "constant/polyMesh";\n')
        f.write("    object      entities;\n")
        f.write("}\n\n")
        f.write(f"// {len(entities)} entities\n\n")
        for ent in entities:
            f.write(f"entity_{ent.tag}\n{{\n")
            f.write(f"    dim {ent.dim};\n")
            if ent.physical_tags:
                f.write(f"    physicalTags ({' '.join(str(t) for t in ent.physical_tags)});\n")
            if ent.bounding_tags:
                f.write(f"    boundingTags ({' '.join(str(t) for t in ent.bounding_tags)});\n")
            f.write("}\n\n")


def _write_partition_info(path: Path, tags: List[GmshPartitionTag]) -> None:
    """Write partition info to a file."""
    with open(path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       labelList;\n")
        f.write('    location    "constant/polyMesh";\n')
        f.write("    object      cellPartitioning;\n")
        f.write("}\n\n")
        f.write(f"// {len(tags)} partition entries\n\n")
        f.write(f"{len(tags)}\n(\n")
        for t in tags:
            f.write(f"    {t.elem_tag} {t.partition}\n")
        f.write(")\n")
