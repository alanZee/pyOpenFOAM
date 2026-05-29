"""
ReconstructParEnhanced — enhanced parallel reconstruction with zone support.

Extends :class:`~pyfoam.parallel.reconstruct_par.ReconstructPar` with:

- Reconstruction of cell zones (e.g. ``rotor``, ``solid``)
- Reconstruction of face zones (e.g. ``internalInterface``)
- Zone-aware field mapping during reconstruction

This mirrors additional features in OpenFOAM's ``reconstructPar`` that
handle zone data beyond basic mesh and field merging.

Usage::

    recon = ReconstructParEnhanced(case_dir)
    recon.discover()
    recon.reconstruct_mesh()
    recon.reconstruct_zones()
    recon.reconstruct_fields(time=0)

References
----------
- OpenFOAM ``reconstructPar`` utility source (zone handling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par import ReconstructPar, ReconstructResult

__all__ = ["ReconstructParEnhanced", "ZoneInfo", "EnhancedReconstructResult"]

logger = logging.getLogger(__name__)


@dataclass
class ZoneInfo:
    """Information about a reconstructed zone.

    Attributes:
        name: Zone name.
        zone_type: ``"cellZone"`` or ``"faceZone"``.
        n_entries: Number of entries (cells or faces) in the zone.
    """

    name: str
    zone_type: str  # "cellZone" or "faceZone"
    n_entries: int = 0


@dataclass
class EnhancedReconstructResult:
    """Result of an enhanced reconstruction operation.

    Attributes:
        base: Base reconstruction result.
        zones: List of reconstructed zone information.
    """

    base: ReconstructResult
    zones: List[ZoneInfo] = dc_field(default_factory=list)


class ReconstructParEnhanced(ReconstructPar):
    """Enhanced parallel reconstruction with zone support.

    Extends :class:`ReconstructPar` to handle cell and face zones.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        super().__init__(case_dir)
        self._cell_zones: Dict[str, List[int]] = {}  # zone_name -> global cell IDs
        self._face_zones: Dict[str, List[int]] = {}  # zone_name -> global face IDs

    @property
    def cell_zones(self) -> Dict[str, List[int]]:
        """Reconstructed cell zone mappings."""
        return self._cell_zones

    @property
    def face_zones(self) -> Dict[str, List[int]]:
        """Reconstructed face zone mappings."""
        return self._face_zones

    # ------------------------------------------------------------------
    # Zone discovery
    # ------------------------------------------------------------------

    def discover_zones(self) -> Dict[str, List[str]]:
        """Discover zone names across processor directories.

        Returns:
            Dictionary with keys ``"cellZones"`` and ``"faceZones"``,
            each listing zone names found in the processor meshes.
        """
        if not self._processor_dirs:
            self.discover()

        cell_zone_names: Set[str] = set()
        face_zone_names: Set[str] = set()

        for i, proc_dir in enumerate(self._processor_dirs):
            zones_dir = proc_dir / "constant" / "polyMesh" / "zones"
            if not zones_dir.exists():
                continue

            for zone_file in zones_dir.iterdir():
                if zone_file.name == "cellZones":
                    names = self._read_zone_names(zone_file)
                    cell_zone_names.update(names)
                elif zone_file.name == "faceZones":
                    names = self._read_zone_names(zone_file)
                    face_zone_names.update(names)

        result = {
            "cellZones": sorted(cell_zone_names),
            "faceZones": sorted(face_zone_names),
        }
        logger.info(
            "Discovered %d cell zones, %d face zones",
            len(result["cellZones"]),
            len(result["faceZones"]),
        )
        return result

    # ------------------------------------------------------------------
    # Zone reconstruction
    # ------------------------------------------------------------------

    def reconstruct_zones(
        self,
        zone_names: Optional[Dict[str, List[str]]] = None,
    ) -> List[ZoneInfo]:
        """Reconstruct cell and face zones from processor subdirectories.

        Args:
            zone_names: Optional dict with ``"cellZones"`` and ``"faceZones"``
                lists. If None, auto-discovers zone names.

        Returns:
            List of :class:`ZoneInfo` for all reconstructed zones.
        """
        if not self._processor_dirs:
            self.discover()

        if zone_names is None:
            zone_names = self.discover_zones()

        all_zones: List[ZoneInfo] = []

        # Reconstruct cell zones
        for zone_name in zone_names.get("cellZones", []):
            global_cells = self._reconstruct_cell_zone(zone_name)
            self._cell_zones[zone_name] = global_cells
            all_zones.append(ZoneInfo(
                name=zone_name,
                zone_type="cellZone",
                n_entries=len(global_cells),
            ))

        # Reconstruct face zones
        for zone_name in zone_names.get("faceZones", []):
            global_faces = self._reconstruct_face_zone(zone_name)
            self._face_zones[zone_name] = global_faces
            all_zones.append(ZoneInfo(
                name=zone_name,
                zone_type="faceZone",
                n_entries=len(global_faces),
            ))

        logger.info("Reconstructed %d zones total", len(all_zones))
        return all_zones

    def _reconstruct_cell_zone(self, zone_name: str) -> List[int]:
        """Reconstruct a cell zone from all processors.

        Each processor stores local cell indices; we remap them to global
        indices using the accumulated cell offset.

        Args:
            zone_name: Name of the cell zone.

        Returns:
            List of global cell indices in the zone.
        """
        global_cells: List[int] = []
        cell_offset = 0

        for i, proc_dir in enumerate(self._processor_dirs):
            zones_dir = proc_dir / "constant" / "polyMesh" / "zones"
            cell_zones_file = zones_dir / "cellZones"

            if not cell_zones_file.exists():
                # Compute cell offset even if no zones
                owner_file = proc_dir / "constant" / "polyMesh" / "owner"
                if owner_file.exists():
                    owner = self._read_label_list_ascii(owner_file)
                    n_cells = int(owner.max().item()) + 1 if owner.numel() > 0 else 0
                    cell_offset += n_cells
                continue

            local_indices = self._read_zone_entries(cell_zones_file, zone_name)
            global_cells.extend([idx + cell_offset for idx in local_indices])

            # Update cell offset
            owner_file = proc_dir / "constant" / "polyMesh" / "owner"
            if owner_file.exists():
                owner = self._read_label_list_ascii(owner_file)
                n_cells = int(owner.max().item()) + 1 if owner.numel() > 0 else 0
                cell_offset += n_cells

        return sorted(global_cells)

    def _reconstruct_face_zone(self, zone_name: str) -> List[int]:
        """Reconstruct a face zone from all processors.

        Args:
            zone_name: Name of the face zone.

        Returns:
            List of global face indices in the zone.
        """
        global_faces: List[int] = []
        face_offset = 0

        for i, proc_dir in enumerate(self._processor_dirs):
            zones_dir = proc_dir / "constant" / "polyMesh" / "zones"
            face_zones_file = zones_dir / "faceZones"

            if not face_zones_file.exists():
                faces_file = proc_dir / "constant" / "polyMesh" / "faces"
                if faces_file.exists():
                    proc_faces = self._read_faces_ascii(faces_file)
                    face_offset += len(proc_faces)
                continue

            local_indices = self._read_zone_entries(face_zones_file, zone_name)
            global_faces.extend([idx + face_offset for idx in local_indices])

            # Update face offset
            faces_file = proc_dir / "constant" / "polyMesh" / "faces"
            if faces_file.exists():
                proc_faces = self._read_faces_ascii(faces_file)
                face_offset += len(proc_faces)

        return sorted(global_faces)

    # ------------------------------------------------------------------
    # Zone-aware field reconstruction
    # ------------------------------------------------------------------

    def reconstruct_zone_fields(
        self,
        time: str | int | float = "0",
        zone_name: Optional[str] = None,
        field_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct fields restricted to a zone.

        After full field reconstruction, extracts only cells/faces that
        belong to the specified zone.

        Args:
            time: Time step.
            zone_name: Cell zone name. If None, returns full fields.
            field_names: Fields to reconstruct.

        Returns:
            Dict mapping field names to zone-filtered tensors.
        """
        fields = self.reconstruct_fields(time=time, field_names=field_names)

        if zone_name is None:
            return fields

        if zone_name not in self._cell_zones:
            logger.warning("Zone '%s' not found. Available: %s",
                           zone_name, list(self._cell_zones.keys()))
            return fields

        cell_indices = self._cell_zones[zone_name]
        if not cell_indices:
            return {}

        idx_tensor = torch.tensor(cell_indices, dtype=INDEX_DTYPE, device=self._device)

        result: Dict[str, torch.Tensor] = {}
        for fname, data in fields.items():
            if data.shape[0] > idx_tensor.max():
                result[fname] = data[idx_tensor]
            else:
                result[fname] = data

        return result

    # ------------------------------------------------------------------
    # Full enhanced reconstruction
    # ------------------------------------------------------------------

    def reconstruct_case_enhanced(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
    ) -> EnhancedReconstructResult:
        """Reconstruct the full case including zones.

        Args:
            output_dir: Output directory.
            field_names: Fields to reconstruct.

        Returns:
            :class:`EnhancedReconstructResult` with statistics.
        """
        base_result = self.reconstruct_case(
            output_dir=output_dir,
            field_names=field_names,
        )

        zones = self.reconstruct_zones()

        # Write zone data to output
        out = Path(output_dir) if output_dir else self._case_dir
        for zone_info in zones:
            if zone_info.zone_type == "cellZone" and zone_info.name in self._cell_zones:
                zone_dir = out / "constant" / "polyMesh" / "zones"
                zone_dir.mkdir(parents=True, exist_ok=True)
                self._write_zone_file(
                    zone_dir / "cellZones",
                    zone_info.name,
                    self._cell_zones[zone_info.name],
                    label_class="labelListList",
                )
            elif zone_info.zone_type == "faceZone" and zone_info.name in self._face_zones:
                zone_dir = out / "constant" / "polyMesh" / "zones"
                zone_dir.mkdir(parents=True, exist_ok=True)
                self._write_zone_file(
                    zone_dir / "faceZones",
                    zone_info.name,
                    self._face_zones[zone_info.name],
                    label_class="labelListList",
                )

        return EnhancedReconstructResult(
            base=base_result,
            zones=zones,
        )

    # ------------------------------------------------------------------
    # Zone file I/O helpers
    # ------------------------------------------------------------------

    def _read_zone_names(self, filepath: Path) -> List[str]:
        """Read zone names from a zone file."""
        try:
            with open(filepath) as f:
                lines = f.readlines()
            ds = self._find_data_start(lines)
            n = int(lines[ds].strip())
            names = []
            for i in range(ds + 2, ds + 2 + n):
                name = lines[i].strip()
                if name and name not in ("(", ")"):
                    names.append(name)
            return names
        except (ValueError, IndexError, FileNotFoundError):
            return []

    def _read_zone_entries(
        self, filepath: Path, zone_name: str
    ) -> List[int]:
        """Read zone entries (cell/face indices) from a zone file.

        The file format is::

            N
            (
            zone_name
            {
                n
                (i1 i2 ... in)
            }
            ...
            )

        Args:
            filepath: Path to the zone file.
            zone_name: Name of the zone to extract.

        Returns:
            List of local indices in the zone.
        """
        try:
            with open(filepath) as f:
                content = f.read()

            # Find the zone block
            zone_start = content.find(zone_name)
            if zone_start == -1:
                return []

            # Find the opening brace after zone name
            brace_start = content.find("{", zone_start)
            if brace_start == -1:
                return []

            # Find the closing brace
            brace_count = 1
            idx = brace_start + 1
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1
            block = content[brace_start + 1: idx - 1]

            # Parse the entries: n (i1 i2 ...)
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                return []

            n = int(lines[0])
            entries: List[int] = []
            for line in lines[1:]:
                line = line.strip().strip("()")
                if line:
                    entries.extend(int(v) for v in line.split())

            return entries[:n]
        except (ValueError, IndexError):
            return []

    def _write_zone_file(
        self,
        filepath: Path,
        zone_name: str,
        indices: List[int],
        label_class: str = "labelListList",
    ) -> None:
        """Write a zone file with zone name and indices.

        Args:
            filepath: Output file path.
            zone_name: Zone name.
            indices: Global indices.
            label_class: FoamFile class name.
        """
        with open(filepath, "a") as f:
            # Write header if file is empty
            if filepath.stat().st_size == 0:
                self._write_header(f, label_class, filepath.name, "constant/polyMesh")

            n = len(indices)
            f.write(f"{zone_name}\n")
            f.write("{\n")
            f.write(f"    {n}\n")
            f.write("    (\n")
            for idx in indices:
                f.write(f"        {idx}\n")
            f.write("    )\n")
            f.write("}\n")
