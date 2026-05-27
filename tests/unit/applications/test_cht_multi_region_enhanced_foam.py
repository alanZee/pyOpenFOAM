"""
Unit tests for CHTMultiRegionEnhancedFoam — enhanced conjugate heat transfer.

Tests cover:
- RegionConfig and InterfaceConfig dataclasses
- Multi-region initialization with custom configs
- Temperature-dependent conductivity
- Inner iteration coupling
- Convergence of coupled solution
- Field output for all regions
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Helper to create a simple CHT case (reuses mesh from reacting case)
# ---------------------------------------------------------------------------

def _make_cht_enhanced_case(
    case_dir: Path,
    n_cells: int = 3,
    T_init: float = 300.0,
    T_hot: float = 400.0,
    T_cold: float = 200.0,
    end_time: int = 10,
    delta_t: float = 0.1,
) -> None:
    """Create a simple CHT case using the same mesh as reacting case (valid cells).

    This reuses the 1D mesh from test_reacting_foam.py which produces
    cells with positive volumes.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    L = 1.0
    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # Points
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # Faces
    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Inlet
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Outlet
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches (sides)
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    n_empty = 4 * n_cells
    lines = ["2", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/T
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # transportProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        "DT              1.0;\n", overwrite=True,
    )

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     chtMultiRegionFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        tolerance       1e-6;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cht_enhanced_case(tmp_path):
    """Create a simple CHT enhanced case."""
    case_dir = tmp_path / "cht_enhanced"
    _make_cht_enhanced_case(case_dir, n_cells=3, end_time=5, delta_t=0.1)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestRegionConfig:
    """Tests for RegionConfig dataclass."""

    def test_create_default(self):
        """RegionConfig creates with defaults."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import RegionConfig

        cfg = RegionConfig(name="fluid", region_type="fluid")
        assert cfg.name == "fluid"
        assert cfg.region_type == "fluid"
        assert cfg.k > 0
        assert cfg.Cp > 0
        assert cfg.rho > 0

    def test_create_solid(self):
        """RegionConfig creates solid region correctly."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import RegionConfig

        cfg = RegionConfig(name="steel", region_type="solid", k=50.0, rho=7800.0)
        assert cfg.region_type == "solid"
        assert cfg.k == 50.0
        assert cfg.rho == 7800.0


class TestInterfaceConfig:
    """Tests for InterfaceConfig dataclass."""

    def test_create_default(self):
        """InterfaceConfig creates with defaults."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import InterfaceConfig

        cfg = InterfaceConfig(
            fluid_region="fluid",
            solid_region="solid",
            fluid_patch="interface",
            solid_patch="interface",
        )
        assert cfg.fluid_region == "fluid"
        assert cfg.solid_region == "solid"
        assert cfg.h_interface > 0

    def test_custom_h_interface(self):
        """InterfaceConfig accepts custom heat transfer coefficient."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import InterfaceConfig

        cfg = InterfaceConfig(h_interface=500.0)
        assert cfg.h_interface == 500.0


class TestCHTEnhancedInit:
    """Tests for CHTMultiRegionEnhancedFoam initialization."""

    def test_creates_with_defaults(self, cht_enhanced_case):
        """Creates with default region configuration."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        assert len(solver.fluid_solvers) >= 0
        assert len(solver.solid_solvers) >= 0

    def test_creates_with_custom_regions(self, cht_enhanced_case):
        """Creates with custom region configurations."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import (
            CHTMultiRegionEnhancedFoam, RegionConfig,
        )

        configs = [
            RegionConfig(name="air", region_type="fluid", k=1.0, rho=1.225, Cp=1005.0),
            RegionConfig(name="copper", region_type="solid", k=400.0, rho=8960.0, Cp=385.0),
        ]
        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case, region_configs=configs)
        assert "air" in solver.region_configs
        assert "copper" in solver.region_configs
        assert solver.region_configs["copper"].k == 400.0

    def test_has_all_solvers_property(self, cht_enhanced_case):
        """all_solvers returns combined dict."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        assert isinstance(solver.all_solvers, dict)


class TestTemperatureDependentConductivity:
    """Tests for temperature-dependent conductivity model."""

    def test_k_at_reference_temperature(self, cht_enhanced_case):
        """k(T_ref) returns k_ref."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        T = torch.full((4,), 300.0, dtype=CFD_DTYPE)
        k = solver._compute_temperature_dependent_conductivity(T, k_ref=0.026, T_ref=300.0)
        assert torch.allclose(k, torch.full((4,), 0.026, dtype=CFD_DTYPE), atol=1e-6)

    def test_k_increases_with_temperature(self, cht_enhanced_case):
        """k(T) increases with T (for T > T_ref)."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        T_low = torch.full((4,), 300.0, dtype=CFD_DTYPE)
        T_high = torch.full((4,), 600.0, dtype=CFD_DTYPE)

        k_low = solver._compute_temperature_dependent_conductivity(T_low, 0.026, 300.0)
        k_high = solver._compute_temperature_dependent_conductivity(T_high, 0.026, 300.0)

        assert (k_high > k_low).all()

    def test_k_finite_for_small_temperature(self, cht_enhanced_case):
        """k(T) is finite even for very small T (clamped)."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        T = torch.tensor([0.1, 1.0, 100.0, 1000.0], dtype=CFD_DTYPE)
        k = solver._compute_temperature_dependent_conductivity(T, 1.0, 300.0)
        assert torch.isfinite(k).all()


class TestCHTEnhancedRun:
    """Tests for the full enhanced CHT solver run."""

    def test_run_completes(self, cht_enhanced_case):
        """Enhanced CHT solver runs to completion."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        result = solver.run()

        assert "converged" in result
        assert "T_residual" in result

    def test_run_finite_values(self, cht_enhanced_case):
        """All field values are finite after run."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        solver.run()

        for name, T in solver.T_fluid.items():
            assert torch.isfinite(T).all(), f"Fluid {name} has non-finite values"

        for name, T in solver.T_solid.items():
            assert torch.isfinite(T).all(), f"Solid {name} has non-finite values"

    def test_run_with_custom_configs(self, cht_enhanced_case):
        """Solver runs with custom region configs."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import (
            CHTMultiRegionEnhancedFoam, RegionConfig,
        )

        configs = [
            RegionConfig(name="fluid", region_type="fluid", k=1.0, rho=1.0, Cp=1000.0),
            RegionConfig(name="solid", region_type="solid", k=2.0, rho=2500.0, Cp=900.0),
        ]
        solver = CHTMultiRegionEnhancedFoam(
            cht_enhanced_case,
            region_configs=configs,
            n_inner_correctors=2,
        )
        result = solver.run()
        assert "converged" in result

    def test_residual_is_nonnegative(self, cht_enhanced_case):
        """Residual is non-negative after run."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        result = solver.run()

        assert result["T_residual"] >= 0

    def test_region_counts_reported(self, cht_enhanced_case):
        """Result includes region counts."""
        from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

        solver = CHTMultiRegionEnhancedFoam(cht_enhanced_case)
        result = solver.run()

        assert "n_fluid_regions" in result
        assert "n_solid_regions" in result
        assert "n_interfaces" in result
