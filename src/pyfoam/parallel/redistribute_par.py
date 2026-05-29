"""
RedistributePar — redistribute a parallel case across a different number of processors.

Takes an existing parallel case (``processor0/``, ``processor1/``, ...) and
redistributes the mesh and fields across a new processor count.  This mirrors
OpenFOAM's ``redistributePar`` utility.

Workflow:

1. Read the existing parallel decomposition
2. Compute a new decomposition with the target processor count
3. Remap cells from the old decomposition to the new one
4. Write the redistributed processor directories

Usage::

    redist = RedistributePar(case_dir, target_n_procs=4)
    redist.discover()
    redist.redistribute(output_dir="case_new")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = ["RedistributePar", "RedistributeResult"]

logger = logging.getLogger(__name__)


@dataclass
class RedistributeResult:
    """Result of a redistribution operation.

    Attributes:
        source_n_procs: Number of source processors.
        target_n_procs: Number of target processors.
        n_cells: Total number of redistributed cells.
        n_time_steps: Number of time steps redistributed.
        imbalance_ratio: Cell count imbalance ratio (max/mean).
    """

    source_n_procs: int = 0
    target_n_procs: int = 0
    n_cells: int = 0
    n_time_steps: int = 0
    imbalance_ratio: float = 1.0


class RedistributePar:
    """Redistribute a parallel case to a different processor count.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        if target_n_procs < 1:
            raise ValueError("target_n_procs must be >= 1")

        self._case_dir = Path(case_dir)
        self._target_n_procs = target_n_procs
        self._device = get_device()
        self._dtype = get_default_dtype()

        self._source_n_procs: int = 0
        self._processor_dirs: List[Path] = []
        self._time_steps: List[str] = []
        self._global_cell_map: Optional[torch.Tensor] = None

    @property
    def case_dir(self) -> Path:
        return self._case_dir

    @property
    def target_n_procs(self) -> int:
        return self._target_n_procs

    @property
    def source_n_procs(self) -> int:
        return self._source_n_procs

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> Tuple[int, List[str]]:
        """Discover source processor directories and time steps.

        Returns:
            Tuple of (n_processors, time_step_names).

        Raises:
            FileNotFoundError: If no processor directories exist.
        """
        proc_dirs = sorted(self._case_dir.glob("processor[0-9]*"))
        proc_dirs = [d for d in proc_dirs if d.is_dir()]

        if not proc_dirs:
            raise FileNotFoundError(
                f"No processor directories found in {self._case_dir}"
            )

        self._processor_dirs = proc_dirs
        self._source_n_procs = len(proc_dirs)

        # Find common time steps
        all_times: Optional[set] = None
        for pd in proc_dirs:
            proc_times = set()
            for entry in pd.iterdir():
                if entry.is_dir():
                    try:
                        float(entry.name)
                        proc_times.add(entry.name)
                    except ValueError:
                        continue
            if all_times is None:
                all_times = proc_times
            else:
                all_times &= proc_times

        self._time_steps = sorted(all_times or [], key=lambda x: float(x))
        return self._source_n_procs, self._time_steps

    # ------------------------------------------------------------------
    # Cell mapping
    # ------------------------------------------------------------------

    def compute_cell_mapping(self, n_global_cells: int) -> torch.Tensor:
        """Compute a new round-robin cell-to-processor mapping.

        Args:
            n_global_cells: Total number of cells in the global mesh.

        Returns:
            Tensor of shape ``(n_global_cells,)`` mapping each cell to a
            target processor index.
        """
        mapping = torch.arange(n_global_cells, dtype=INDEX_DTYPE) % self._target_n_procs
        self._global_cell_map = mapping
        return mapping

    def compute_load_balanced_mapping(
        self,
        n_global_cells: int,
        cell_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a load-balanced cell-to-processor mapping.

        Uses a greedy bin-packing approach to distribute cells across
        processors, minimizing the maximum load.

        Args:
            n_global_cells: Total number of cells.
            cell_weights: Optional per-cell weights. If None, all cells
                have equal weight.

        Returns:
            Tensor of shape ``(n_global_cells,)`` mapping cells to processors.
        """
        if cell_weights is None:
            # Round-robin for equal weights
            return self.compute_cell_mapping(n_global_cells)

        # Greedy: assign each cell to the processor with lowest current load
        loads = torch.zeros(self._target_n_procs, dtype=self._dtype)
        mapping = torch.zeros(n_global_cells, dtype=INDEX_DTYPE)

        # Sort cells by weight (descending) for better balance
        sorted_indices = torch.argsort(cell_weights, descending=True)

        for idx in sorted_indices:
            w = cell_weights[idx].item()
            min_proc = int(torch.argmin(loads).item())
            mapping[idx] = min_proc
            loads[min_proc] += w

        self._global_cell_map = mapping
        return mapping

    # ------------------------------------------------------------------
    # Redistribution
    # ------------------------------------------------------------------

    def redistribute(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
    ) -> RedistributeResult:
        """Redistribute the case to the target processor count.

        Args:
            output_dir: Output directory. If None, overwrites in place.
            field_names: Fields to redistribute. If None, auto-detect.

        Returns:
            :class:`RedistributeResult` with redistribution statistics.
        """
        if not self._processor_dirs:
            self.discover()

        out = Path(output_dir) if output_dir else self._case_dir

        # Read global mesh structure from all source processors
        total_cells = 0
        proc_cell_counts: List[int] = []
        for proc_dir in self._processor_dirs:
            mesh_dir = proc_dir / "constant" / "polyMesh"
            owner_file = mesh_dir / "owner"
            if owner_file.exists():
                owner = self._read_label_list(owner_file)
                n_cells = int(owner.max().item()) + 1 if owner.numel() > 0 else 0
            else:
                n_cells = 0
            proc_cell_counts.append(n_cells)
            total_cells += n_cells

        if total_cells == 0:
            logger.warning("No cells found in source case")
            return RedistributeResult()

        # Compute new cell mapping
        mapping = self.compute_cell_mapping(total_cells)

        # Group cells by target processor
        target_cell_lists: List[List[int]] = [[] for _ in range(self._target_n_procs)]
        for global_cell in range(total_cells):
            target_proc = int(mapping[global_cell].item())
            target_cell_lists[target_proc].append(global_cell)

        # Auto-detect fields from first time step
        if field_names is None and self._time_steps:
            proc0_time = self._processor_dirs[0] / self._time_steps[0]
            if proc0_time.exists():
                field_names = [
                    e.name for e in proc0_time.iterdir()
                    if e.is_file() and not e.name.startswith(".")
                ]

        # Build a global-to-source mapping: for each global cell, which
        # source processor owns it and what is its local index
        cell_to_source: List[Tuple[int, int]] = []
        cell_counter = 0
        for proc_idx, n_cells in enumerate(proc_cell_counts):
            for local_idx in range(n_cells):
                cell_to_source.append((proc_idx, local_idx))
                cell_counter += 1

        # Write target processor directories
        for target_proc in range(self._target_n_procs):
            proc_dir = out / f"processor{target_proc}"
            proc_dir.mkdir(parents=True, exist_ok=True)

            # Copy mesh structure (simplified: write cell mapping info)
            mesh_dir = proc_dir / "constant" / "polyMesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            cell_list = target_cell_lists[target_proc]

            # Write a cell mapping file for reference
            self._write_cell_mapping(mesh_dir / "cellProcAddressing", cell_list)

            # Copy boundary from source processor 0
            src_boundary = self._processor_dirs[0] / "constant" / "polyMesh" / "boundary"
            if src_boundary.exists():
                shutil.copy2(src_boundary, mesh_dir / "boundary")

            # Copy and filter fields for each time step
            if field_names:
                for ts in self._time_steps:
                    time_dir = proc_dir / ts
                    time_dir.mkdir(parents=True, exist_ok=True)

                    for fname in field_names:
                        self._redistribute_field(
                            time_dir / fname,
                            fname,
                            ts,
                            cell_list,
                            cell_to_source,
                        )

        # Compute imbalance ratio
        counts = [len(cl) for cl in target_cell_lists]
        mean_cells = sum(counts) / len(counts) if counts else 1.0
        max_cells = max(counts) if counts else 1
        imbalance = max_cells / mean_cells if mean_cells > 0 else 1.0

        return RedistributeResult(
            source_n_procs=self._source_n_procs,
            target_n_procs=self._target_n_procs,
            n_cells=total_cells,
            n_time_steps=len(self._time_steps),
            imbalance_ratio=imbalance,
        )

    def _redistribute_field(
        self,
        output_path: Path,
        field_name: str,
        time_str: str,
        target_cells: List[int],
        cell_to_source: List[Tuple[int, int]],
    ) -> None:
        """Redistribute a field to a target processor."""
        # Read all source fields and build a global array
        global_field: List[float] = []
        for proc_dir in self._processor_dirs:
            filepath = proc_dir / time_str / field_name
            if not filepath.exists():
                continue
            try:
                data = self._read_internal_field(filepath)
                global_field.extend(data.cpu().numpy().tolist())
            except (ValueError, FileNotFoundError):
                continue

        if not global_field:
            return

        # Extract target cells
        target_values = []
        for gc in target_cells:
            if gc < len(global_field):
                target_values.append(global_field[gc])

        if not target_values:
            return

        # Write the field
        n = len(target_values)
        with open(output_path, "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       volScalarField;\n")
            f.write(f'    location    "{time_str}";\n')
            f.write(f"    object      {field_name};\n")
            f.write("}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(f"internalField nonuniform {n}\n(\n")
            for v in target_values:
                f.write(f"{v:.10g}\n")
            f.write(");\n\n")
            f.write("boundaryField\n{\n}\n")

    def _write_cell_mapping(self, filepath: Path, cell_list: List[int]) -> None:
        """Write cell addressing file."""
        n = len(cell_list)
        with open(filepath, "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write('    location    "constant/polyMesh";\n')
            f.write("    object      cellProcAddressing;\n")
            f.write("}\n\n")
            f.write(f"{n}\n(\n")
            for c in cell_list:
                f.write(f"{c}\n")
            f.write(")\n")

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _find_data_start(self, lines: List[str]) -> int:
        data_start = 0
        brace_count = 0
        for i, line in enumerate(lines):
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
                if brace_count == 0:
                    data_start = i + 1
                    break
        while data_start < len(lines) and lines[data_start].strip() == "":
            data_start += 1
        return data_start

    def _read_label_list(self, filepath: Path) -> torch.Tensor:
        with open(filepath) as f:
            lines = f.readlines()
        ds = self._find_data_start(lines)
        n = int(lines[ds].strip())
        values = []
        for i in range(ds + 2, ds + 2 + n):
            values.append(int(lines[i].strip()))
        return torch.tensor(values, dtype=INDEX_DTYPE, device=self._device)

    def _read_internal_field(self, filepath: Path) -> torch.Tensor:
        with open(filepath) as f:
            content = f.read()

        marker = "internalField"
        idx = content.find(marker)
        if idx == -1:
            raise ValueError(f"No internalField in {filepath}")

        rest = content[idx + len(marker):].strip()
        if rest.startswith("uniform"):
            val = float(rest.split()[1].rstrip(";"))
            return torch.tensor([val], dtype=self._dtype, device=self._device)

        n_start = rest.find("nonuniform") + len("nonuniform")
        n_end = rest.find("\n", n_start)
        n = int(rest[n_start:n_end].strip())

        paren_start = rest.find("(") + 1
        paren_end = rest.find(")")
        values_str = rest[paren_start:paren_end]
        values = [float(v) for v in values_str.split() if v]

        return torch.tensor(values, dtype=self._dtype, device=self._device)
