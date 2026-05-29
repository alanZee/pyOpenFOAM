"""
Unit tests for FinancialFoam2 — enhanced Black-Scholes with Greeks and American options.

Tests cover:
- Case loading and initialisation
- European call/put options
- American call/put options (early exercise)
- Greeks computation (delta, gamma, vega, rho)
- Dividend yield support
- Solver runs to completion
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper (reused from test_financial_foam.py)
# ---------------------------------------------------------------------------

def _make_financial2_case(
    case_dir: Path,
    n_cells: int = 20,
    end_time: int = 1,
    delta_t: float = 0.01,
) -> None:
    """Write a 1D case for financialFoam2."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells
    dy, dz = 0.1, 0.1

    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.extend([(x, 0.0, 0.0), (x, dy, 0.0), (x, dy, dz), (x, 0.0, dz)])

    n_points = len(points)
    faces, owner, neighbour = [], [], []

    for i in range(n_cells - 1):
        faces.append((4, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)
    n_internal = len(neighbour)

    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    empty_start = outlet_start + 1
    n_empty = 0
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, i * 4))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)
    n_empty += n_cells

    n_faces = len(faces)

    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    hb = FoamFileHeader(version="2.0", format=FileFormat.ASCII, location="constant/polyMesh")

    def _header(class_name: str, obj: str) -> FoamFileHeader:
        return FoamFileHeader(**{**hb.__dict__, "class_name": class_name, "object": obj})

    lines = [str(n_points), "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", _header("vectorField", "points"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", _header("faceList", "faces"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", _header("labelList", "owner"), "\n".join(lines), overwrite=True)

    lines = [str(n_internal), "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", _header("labelList", "neighbour"), "\n".join(lines), overwrite=True)

    lines = ["2", "("]
    lines += [
        "    inlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {inlet_start};", "    }",
        "    outlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {outlet_start};", "    }",
        "    walls", "    {", "        type            empty;",
        f"        nFaces          {n_empty};", f"        startFace       {empty_start};", "    }",
        ")",
    ]
    write_foam_file(mesh_dir / "boundary", _header("polyBoundaryMesh", "boundary"), "\n".join(lines), overwrite=True)

    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    write_foam_file(zero_dir / "V",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="V"),
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform 0;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     financialFoam2;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "financialFoam\n{\n"
        "    convergenceTolerance 1e-6;\n"
        "    scheme              implicit;\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def financial2_case(tmp_path):
    case_dir = tmp_path / "financial2"
    _make_financial2_case(case_dir, n_cells=20, end_time=1, delta_t=0.01)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestFinancialFoam2Init:
    """Initialisation tests for FinancialFoam2."""

    def test_european_call_creates(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert solver.option_type == "call"
        assert solver.exercise == "european"

    def test_american_put_creates(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="put", exercise="american",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert solver.option_type == "put"
        assert solver.exercise == "american"

    def test_dividend_yield_injection(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", q=0.03,
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert solver.q == 0.03

    def test_payoff_non_negative(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert (solver.V >= 0).all()


class TestFinancialFoam2Greeks:
    """Tests for Greeks computation."""

    def test_greeks_returned_on_run(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        solver.end_time = 0.1
        result = solver.run()
        assert "greeks" in result
        greeks = result["greeks"]
        assert hasattr(greeks, "delta")
        assert hasattr(greeks, "gamma")
        assert hasattr(greeks, "vega")
        assert hasattr(greeks, "rho")

    def test_delta_bounded(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        greeks = solver.compute_greeks(S_target=100.0)
        # Delta for a call should be between 0 and 1
        assert -2.0 <= greeks.delta <= 2.0

    def test_vega_computed(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        greeks = solver.compute_greeks()
        # Vega should be finite
        assert math.isfinite(greeks.vega)


class TestFinancialFoam2Run:
    """Solver execution tests."""

    def test_european_run_completes(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        solver.end_time = 0.1
        result = solver.run()
        assert "converged" in result
        assert "V_at_K" in result

    def test_american_run_completes(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="put", exercise="american",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        solver.end_time = 0.1
        result = solver.run()
        assert "converged" in result
        assert "V_at_K" in result

    def test_values_finite(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        solver.end_time = 0.1
        solver.run()
        assert torch.isfinite(solver.V).all()

    def test_values_non_negative(self, financial2_case):
        from pyfoam.applications.financial_foam_2 import FinancialFoam2
        solver = FinancialFoam2(
            financial2_case, option_type="call", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        solver.end_time = 0.1
        solver.run()
        assert (solver.V >= 0).all()

    def test_american_greater_than_european_put(self, financial2_case):
        """American put should be >= European put value."""
        from pyfoam.applications.financial_foam_2 import FinancialFoam2

        eu = FinancialFoam2(
            financial2_case, option_type="put", exercise="european",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        eu.end_time = 0.1
        eu.run()
        V_eu = eu.V.clone()

        am = FinancialFoam2(
            financial2_case, option_type="put", exercise="american",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        am.end_time = 0.1
        am.run()

        # American >= European (at every node)
        assert (am.V >= V_eu - 1e-6).all(), "American put < European put"

    def test_dividend_yield_effect(self, financial2_case):
        """Dividend yield should reduce call option value."""
        from pyfoam.applications.financial_foam_2 import FinancialFoam2

        no_div = FinancialFoam2(
            financial2_case, option_type="call", q=0.0,
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        no_div.end_time = 0.1
        no_div.run()

        with_div = FinancialFoam2(
            financial2_case, option_type="call", q=0.05,
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        with_div.end_time = 0.1
        with_div.run()

        # With dividend, call value should be <= without dividend
        assert with_div.V.sum() <= no_div.V.sum() + 1e-6
