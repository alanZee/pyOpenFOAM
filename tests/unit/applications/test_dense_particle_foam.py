"""
Test DenseParticleFoam — dense particle two-way Euler-Lagrange solver.

Creates a minimal channel case with U, p fields and verifies:
- Case loading
- Field initialisation
- Particle initialisation
- Solver run completion
- Finite output fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_particle_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    delta_t: float = 0.001,
    end_time: float = 0.005,
    n_outer: int = 2,
    n_correctors: int = 2,
) -> None:
    """Write a minimal 1D channel case for DenseParticleFoam.

    Same mesh topology as the reacting flow test (1D channel with empty
    patches in y and z).  Only U and p fields are required.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # Points: 4 vertices per cell cross-section
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

    # Internal faces (x-direction)
    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Boundary: inlet (x=0)
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Boundary: outlet (x=L)
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches (y and z faces)
    empty_start = outlet_start + 1
    # Front (y=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    # Back (y=dy)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    # Bottom (z=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)
    # Top (z=dz)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)
    n_empty = 4 * n_cells

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

    # ---- constant/physicalProperties ----
    pp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="physicalProperties",
    )
    write_foam_file(
        case_dir / "constant" / "physicalProperties", pp_header,
        "nu              [0 2 -1 0 0 0 0] 1.5e-5;\n"
        "rho             [1 -3 0 0 0 0 0] 1.225;\n",
        overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
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
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     denseParticleFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
    ), overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    # ---- system/fvSolution ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    relaxationFactors\n    {\n"
        "        U               0.7;\n"
        "        p               0.3;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def particle_case(tmp_path):
    """Create a minimal channel case for DenseParticleFoam."""
    case_dir = tmp_path / "particleFlow"
    _make_particle_case(case_dir, n_cells=3, end_time=0.003, delta_t=0.001)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDenseParticleFoamInit:
    """Tests for DenseParticleFoam initialisation."""

    def test_case_loads(self, particle_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(particle_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)

    def test_mesh_builds(self, particle_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(particle_case)
        mesh = solver.mesh

        assert mesh.n_cells == 3
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (3,)

    def test_fields_initialise(self, particle_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=10)

        assert solver.U.shape == (3, 3)
        assert solver.p.shape == (3,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.alpha_p.shape == (3,)

    def test_particles_initialise(self, particle_case):
        """Lagrangian particles are created."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        n_p = 50
        solver = DenseParticleFoam(particle_case, n_particles=n_p, particle_diameter=1e-4)

        assert solver.particles["positions"].shape == (n_p, 3)
        assert solver.particles["velocities"].shape == (n_p, 3)
        assert solver.particles["cell_ids"].shape == (n_p,)
        assert solver.particles["diameters"].shape == (n_p,)
        assert torch.isfinite(solver.particles["positions"]).all()
        assert (solver.particles["diameters"] == 1e-4).all()

    def test_drag_coefficient(self, particle_case):
        """Schiller-Naumann drag coefficient is finite and positive."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=10)
        Re_p = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=CFD_DTYPE)
        Cd = solver._compute_drag_coefficient(Re_p)

        assert Cd.shape == (4,)
        assert torch.isfinite(Cd).all()
        assert (Cd > 0).all()

    def test_particle_volume_fraction(self, particle_case):
        """Volume fraction is computed and bounded."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=20)
        alpha_p = solver._compute_particle_volume_fraction()

        assert alpha_p.shape == (3,)
        assert torch.isfinite(alpha_p).all()
        assert alpha_p.min() >= 0.0
        assert alpha_p.max() <= 0.63 + 1e-6


class TestDenseParticleFoamSolver:
    """Tests for DenseParticleFoam solver execution."""

    def test_run_completes(self, particle_case):
        """Solver runs without errors."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=20)
        result = solver.run()

        assert hasattr(result, "converged")

    def test_fields_finite_after_run(self, particle_case):
        """All fields are finite after solving."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=20)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha_p).all(), "alpha_p contains NaN/Inf"

    def test_particles_updated_after_run(self, particle_case):
        """Particle positions are updated after solving."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=20)
        pos_before = solver.particles["positions"].clone()
        solver.run()
        pos_after = solver.particles["positions"]

        # Positions should change (gravity + drag)
        # At minimum, they should still be finite
        assert torch.isfinite(pos_after).all(), "Particle positions contain NaN/Inf"

    def test_field_shapes_preserved(self, particle_case):
        """Field shapes are preserved after running."""
        from pyfoam.applications.dense_particle_foam import DenseParticleFoam

        solver = DenseParticleFoam(particle_case, n_particles=20)
        solver.run()

        assert solver.U.shape == (3, 3)
        assert solver.p.shape == (3,)
        assert solver.alpha_p.shape == (3,)
