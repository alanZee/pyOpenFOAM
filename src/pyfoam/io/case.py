"""
OpenFOAM case directory representation.

An OpenFOAM case has the structure::

    case/
    ├── 0/                  # Initial/boundary conditions
    │   ├── U
    │   ├── p
    │   └── ...
    ├── constant/
    │   ├── polyMesh/
    │   │   ├── points
    │   │   ├── faces
    │   │   ├── owner
    │   │   ├── neighbour
    │   │   └── boundary
    │   └── transportProperties
    ├── system/
    │   ├── controlDict
    │   ├── fvSchemes
    │   └── fvSolution
    └── [time dirs...]

Usage::

    from pyfoam.io.case import Case

    case = Case("path/to/case")
    print(case.controlDict["application"])
    print(case.time_dirs)
    mesh = case.mesh
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

from pyfoam.io.dictionary import FoamDict, parse_dict, parse_dict_file
from pyfoam.io.field_io import FieldData, read_field
from pyfoam.io.mesh_io import MeshData, BoundaryPatch, read_boundary, read_mesh

__all__ = ["Case"]


class Case:
    """Represents a complete OpenFOAM case directory.

    Provides access to:
    - Configuration files (controlDict, fvSchemes, fvSolution)
    - Time directories and their fields
    - Mesh data (constant/polyMesh)

    Attributes:
        root: Root path of the case directory.
        controlDict: Parsed controlDict.
        fvSchemes: Parsed fvSchemes.
        fvSolution: Parsed fvSolution.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self.root}")

        # Lazy-loaded caches
        self._control_dict: Optional[FoamDict] = None
        self._fv_schemes: Optional[FoamDict] = None
        self._fv_solution: Optional[FoamDict] = None
        self._mesh: Optional[MeshData] = None
        self._boundary: Optional[list[BoundaryPatch]] = None
        self._time_dirs: Optional[list[str]] = None

    def __repr__(self) -> str:
        return f"Case({self.root!r})"

    # -- Properties ---------------------------------------------------------

    @property
    def controlDict(self) -> FoamDict:
        """Parsed ``system/controlDict``."""
        if self._control_dict is None:
            self._control_dict = self._load_system_dict("controlDict")
        return self._control_dict

    @property
    def fvSchemes(self) -> FoamDict:
        """Parsed ``system/fvSchemes``."""
        if self._fv_schemes is None:
            self._fv_schemes = self._load_system_dict("fvSchemes")
        return self._fv_schemes

    @property
    def fvSolution(self) -> FoamDict:
        """Parsed ``system/fvSolution``."""
        if self._fv_solution is None:
            self._fv_solution = self._load_system_dict("fvSolution")
        return self._fv_solution

    @property
    def mesh(self) -> MeshData:
        """Mesh data from ``constant/polyMesh``."""
        if self._mesh is None:
            self._mesh = self._load_mesh()
        return self._mesh

    @property
    def boundary(self) -> list[BoundaryPatch]:
        """Boundary patch definitions from ``constant/polyMesh/boundary``."""
        if self._boundary is None:
            _, self._boundary = read_boundary(self.mesh_dir / "boundary")
        return self._boundary

    @property
    def time_dirs(self) -> list[str]:
        """Sorted list of time directory names (e.g., ``['0', '1', '10']``)."""
        if self._time_dirs is None:
            self._time_dirs = self._scan_time_dirs()
        return self._time_dirs

    @property
    def constant_dir(self) -> Path:
        """Path to the ``constant`` directory."""
        return self.root / "constant"

    @property
    def system_dir(self) -> Path:
        """Path to the ``system`` directory."""
        return self.root / "system"

    @property
    def mesh_dir(self) -> Path:
        """Path to the ``constant/polyMesh`` directory."""
        return self.constant_dir / "polyMesh"

    # -- Time directory access ----------------------------------------------

    def get_time_dir(self, time: Union[str, float, int]) -> Path:
        """Get the path to a specific time directory.

        Args:
            time: Time value (e.g., ``0``, ``0.5``, ``"latestTime"``).

        Returns:
            Path to the time directory.

        Raises:
            KeyError: If the time directory does not exist.
        """
        time_str = str(time) if not isinstance(time, str) else time

        if time_str == "latestTime":
            dirs = self.time_dirs
            if not dirs:
                raise KeyError("No time directories found")
            return self.root / dirs[-1]

        time_path = self.root / time_str
        if not time_path.is_dir():
            raise KeyError(f"Time directory '{time_str}' not found")
        return time_path

    def list_fields(self, time: Union[str, float, int] = 0) -> list[str]:
        """List field files in a time directory.

        Args:
            time: Time value.

        Returns:
            List of field file names.
        """
        time_dir = self.get_time_dir(time)
        fields = []
        for p in time_dir.iterdir():
            if p.is_file() and not p.name.startswith("."):
                # Skip non-field files
                fields.append(p.name)
        return sorted(fields)

    def read_field(self, name: str, time: Union[str, float, int] = 0) -> FieldData:
        """Read a field file from a time directory.

        Args:
            name: Field name (e.g., ``"U"``, ``"p"``).
            time: Time value.

        Returns:
            :class:`FieldData` with parsed field information.
        """
        time_dir = self.get_time_dir(time)
        return read_field(time_dir / name)

    # -- Internal loading ---------------------------------------------------

    def _load_system_dict(self, name: str) -> FoamDict:
        """Load a system dictionary file."""
        path = self.system_dir / name
        if not path.exists():
            return FoamDict()
        return parse_dict_file(path)

    def _load_mesh(self) -> MeshData:
        """Load mesh from constant/polyMesh."""
        mesh_dir = self.mesh_dir
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
        return read_mesh(mesh_dir)

    def _scan_time_dirs(self) -> list[str]:
        """Scan for time directories (numeric names)."""
        time_pattern = re.compile(r"^\d+(\.\d*)?$")
        dirs: list[str] = []
        for p in self.root.iterdir():
            if p.is_dir() and time_pattern.match(p.name):
                dirs.append(p.name)
        # Sort numerically (0, 0.1, 0.5, 1, 10)
        def _sort_key(name: str) -> float:
            try:
                return float(name)
            except ValueError:
                return float("inf")
        return sorted(dirs, key=_sort_key)

    # -- Convenience methods ------------------------------------------------

    def has_field(self, name: str, time: Union[str, float, int] = 0) -> bool:
        """Check if a field file exists in a time directory.

        Args:
            name: Field name.
            time: Time value.

        Returns:
            True if the field file exists.
        """
        try:
            time_dir = self.get_time_dir(time)
            return (time_dir / name).is_file()
        except KeyError:
            return False

    def has_mesh(self) -> bool:
        """Check if the mesh directory exists and has required files."""
        mesh_dir = self.mesh_dir
        required = ["points", "faces", "owner", "neighbour", "boundary"]
        return all((mesh_dir / f).exists() for f in required)

    def get_application(self) -> str:
        """Get the application name from controlDict.

        Returns:
            Application name (e.g., ``"simpleFoam"``).
        """
        return self.controlDict.get("application", "")

    def get_start_time(self) -> float:
        """Get startTime from controlDict."""
        return float(self.controlDict.get("startTime", 0))

    def get_end_time(self) -> float:
        """Get endTime from controlDict."""
        return float(self.controlDict.get("endTime", 0))

    def get_delta_t(self) -> float:
        """Get deltaT from controlDict."""
        return float(self.controlDict.get("deltaT", 1))
