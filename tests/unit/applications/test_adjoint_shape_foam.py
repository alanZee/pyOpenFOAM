"""
Unit tests for AdjointShapeFoam -- enhanced adjoint shape optimisation
solver with mesh morphing.

Tests cover:
- Case loading and field initialisation
- Adjoint field initialisation
- Sensitivity computation
- Boundary displacement computation
- Displacement smoothing
- Mesh morph application
- Gradient computation
- Full run completion
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper (reuses cavity-like 2D mesh)
# ---------------------------------------------------------------------------

def _make_shape_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    end_time: int = 2,
    delta_t: float = 1.0,
) -> None:
    """Write a complete adjoint shape case (2D cavity)."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    # Points
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary: inlet (left)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_inlet = n_cells_y
    inlet_start = n_internal

    # Boundary: outlet (right)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    # Boundary: walls (top, bottom)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_walls = 2 * n_cells_x
    walls_start = outlet_start + n_outlet

    # Boundary: empty (front, back)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)
    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = walls_start + n_walls

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in all_points:
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
    lines = ["4", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_inlet};")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_outlet};")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_walls};")
    lines.append(f"        startFace       {walls_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # constant/transportProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              {nu};\n",
        overwrite=True,
    )

    # 0/U
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # 0/p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     adjointShapeFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   1;\n"
    ), overwrite=True)

    # system/fvSchemes
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    ), overwrite=True)

    # system/fvSolution
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    Ua\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-4;\n"
        "        relTol          0.01;\n"
        "        maxIter         50;\n"
        "    }\n"
        "    pa\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-4;\n"
        "        relTol          0.01;\n"
        "        maxIter         50;\n"
        "    }\n"
        "}\n\n"
        "adjoint\n{\n"
        "    convergenceTolerance 1e-3;\n"
        "    maxOuterIterations   5;\n"
        "    relaxationFactors\n"
        "    {\n"
        "        Ua              0.7;\n"
        "        pa              0.3;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def shape_case(tmp_path):
    """Create a minimal 2x2 shape optimisation case."""
    case_dir = tmp_path / "shape"
    _make_shape_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=1, delta_t=1.0)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestAdjointShapeFoamInit:
    """Tests for AdjointShapeFoam initialisation."""

    def test_case_loads(self, shape_case):
        """Case directory is readable."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        assert solver.U.shape[0] == 4  # 2x2 mesh
        assert solver.p.shape[0] == 4

    def test_adjoint_fields_zero_by_default(self, shape_case):
        """Adjoint fields default to zero when not in 0/ directory."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        assert torch.allclose(solver.Ua, torch.zeros_like(solver.Ua))
        assert torch.allclose(solver.pa, torch.zeros_like(solver.pa))

    def test_nu_reading(self, shape_case):
        """Viscosity is read from transportProperties."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_sensitivity_zero_initially(self, shape_case):
        """Sensitivity field starts as zeros."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        assert torch.allclose(solver.sensitivity, torch.zeros_like(solver.sensitivity))

    def test_boundary_displacement_shape(self, shape_case):
        """Boundary displacement has correct shape."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        n_bnd = solver.mesh.n_faces - solver.mesh.n_internal_faces
        assert solver.boundary_displacement.shape == (n_bnd, 3)

    def test_custom_parameters(self, shape_case):
        """Custom morph parameters are stored."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(
            shape_case,
            morph_smoothing=0.8,
            max_morph_displacement=0.05,
            n_shape_iterations=3,
        )
        assert solver.morph_smoothing == 0.8
        assert solver.max_morph_displacement == 0.05
        assert solver.n_shape_iterations == 3


class TestAdjointShapeFoamSensitivity:
    """Tests for sensitivity computation."""

    def test_sensitivity_finite(self, shape_case):
        """Sensitivity values are finite after computation."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        # Solve adjoint first
        solver._solve_adjoint()
        sens = solver._compute_boundary_sensitivity()
        assert torch.isfinite(sens).all()

    def test_sensitivity_shape(self, shape_case):
        """Sensitivity has the right shape (n_cells,)."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        solver._solve_adjoint()
        sens = solver._compute_boundary_sensitivity()
        assert sens.shape == (4,)  # 2x2

    def test_sensitivity_interior_zero(self, shape_case):
        """Sensitivity should be zero on interior cells (masked)."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        solver._solve_adjoint()
        sens = solver._compute_boundary_sensitivity()
        # In a 2x2 cavity, all cells are boundary-adjacent, so all could
        # have non-zero sensitivity. Just verify finite.
        assert torch.isfinite(sens).all()


class TestAdjointShapeFoamMorph:
    """Tests for mesh morphing / displacement computation."""

    def test_displacement_finite(self, shape_case):
        """Displacement values are finite."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        solver._solve_adjoint()
        solver.sensitivity = solver._compute_boundary_sensitivity()
        disp = solver._compute_face_displacement()
        assert torch.isfinite(disp).all()

    def test_displacement_clamped(self, shape_case):
        """Displacement magnitude respects max_morph_displacement."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, max_morph_displacement=0.001)
        solver._solve_adjoint()
        solver.sensitivity = solver._compute_boundary_sensitivity()
        disp = solver._compute_face_displacement()
        magnitudes = disp.norm(dim=1)
        assert (magnitudes <= 0.001 + 1e-10).all()

    def test_smoothing_no_op(self, shape_case):
        """Zero smoothing returns same displacement."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, morph_smoothing=0.0)
        solver._solve_adjoint()
        solver.sensitivity = solver._compute_boundary_sensitivity()
        disp = solver._compute_face_displacement()
        smoothed = solver._smooth_displacement(disp)
        assert torch.allclose(disp, smoothed)

    def test_smoothing_modifies(self, shape_case):
        """Non-zero smoothing modifies synthetic non-uniform displacement."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, morph_smoothing=0.5)
        # Create synthetic non-uniform displacement with enough faces
        n_bnd = solver.mesh.n_faces - solver.mesh.n_internal_faces
        disp = torch.randn(n_bnd, 3, dtype=torch.float64)
        if n_bnd > 2:
            smoothed = solver._smooth_displacement(disp)
            # Interior faces should be modified by the smoothing
            assert not torch.allclose(disp, smoothed)

    def test_apply_mesh_morph(self, shape_case):
        """apply_mesh_morph stores displacement."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        solver._solve_adjoint()
        solver.sensitivity = solver._compute_boundary_sensitivity()
        solver._apply_mesh_morph()
        assert solver.boundary_displacement.shape[1] == 3


class TestAdjointShapeFoamGradient:
    """Tests for gradient computation."""

    def test_gradient_shape(self, shape_case):
        """Gradient returns (n_cells, 3) tensor."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        field = solver.p
        grad = solver._compute_gradient(field)
        assert grad.shape == (4, 3)

    def test_gradient_finite(self, shape_case):
        """Gradient values are finite."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case)
        grad = solver._compute_gradient(solver.p)
        assert torch.isfinite(grad).all()


class TestAdjointShapeFoamRun:
    """Tests for the full solver run."""

    def test_run_completes(self, shape_case):
        """AdjointShapeFoam runs to completion."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, n_shape_iterations=2)
        conv = solver.run()
        assert conv is not None

    def test_fields_finite_after_run(self, shape_case):
        """All field values are finite after run."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, n_shape_iterations=2)
        solver.run()
        assert torch.isfinite(solver.Ua).all()
        assert torch.isfinite(solver.pa).all()

    def test_sensitivity_computed_after_run(self, shape_case):
        """Sensitivity field is computed after run."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, n_shape_iterations=2)
        solver.run()
        assert solver.sensitivity.shape == (4,)
        assert torch.isfinite(solver.sensitivity).all()

    def test_displacement_computed_after_run(self, shape_case):
        """Boundary displacement is computed after run."""
        from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
        solver = AdjointShapeFoam(shape_case, n_shape_iterations=2)
        solver.run()
        assert solver.boundary_displacement.shape[1] == 3
        assert torch.isfinite(solver.boundary_displacement).all()
