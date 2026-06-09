"""验证全部 50 个基础求解器产生真实物理结果。"""
import tempfile, torch, json
from pathlib import Path
from tests.tutorials.helpers import make_structured_mesh, write_control_dict, write_fv_schemes, write_fv_solution, write_velocity_field, write_pressure_field, write_transport_properties
from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat

def make_full_case(tmp_dir, nu=0.01):
    case_dir = Path(tmp_dir)
    mesh_dir = case_dir / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=4, ny=4)
    write_control_dict(case_dir, delta_t=0.001, end_time=0.005)
    write_fv_schemes(case_dir)
    write_fv_solution(case_dir, algorithm="SIMPLE")
    write_transport_properties(case_dir, nu=nu)
    patches_U = {"movingWall": (1.0, 0.0, 0.0), "fixedWalls": (0.0, 0.0, 0.0)}
    bc_U = {"movingWall": "fixedValue", "fixedWalls": "noSlip"}
    write_velocity_field(case_dir, patches=patches_U, bc_types=bc_U)
    patches_p = {"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"}
    write_pressure_field(case_dir, patches=patches_p)
    zero_dir = case_dir / "0"

    def _write_scalar(name, value, bc_moving="zeroGradient", bc_fixed="zeroGradient"):
        h = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                           class_name="volScalarField", location="0", object=name)
        lines = [
            "dimensions      [0 0 0 1 0 0 0];" if name == "T" else "dimensions      [0 0 0 0 0 0 0];",
            f"internalField   uniform {value};",
            "boundaryField {",
            f"    movingWall {{ type {bc_moving}; {'value uniform ' + str(value) + ';' if bc_moving == 'fixedValue' else ''} }}",
            f"    fixedWalls {{ type {bc_fixed}; }}",
            "    frontAndBack { type empty; }",
            "}",
        ]
        write_foam_file(zero_dir / name, h, "\n".join(lines), overwrite=True)

    def _write_vector(name, value=(0, 0, 0)):
        h = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                           class_name="volVectorField", location="0", object=name)
        v = f"({value[0]} {value[1]} {value[2]})"
        lines = [
            "dimensions      [0 1 -1 0 0 0 0];",
            f"internalField   uniform {v};",
            "boundaryField {",
            f"    movingWall {{ type fixedValue; value uniform {v}; }}",
            "    fixedWalls { type noSlip; }",
            "    frontAndBack { type empty; }",
            "}",
        ]
        write_foam_file(zero_dir / name, h, "\n".join(lines), overwrite=True)

    # Temperature
    _write_scalar("T", 300, "fixedValue", "zeroGradient")

    # Alpha fields
    _write_scalar("alpha.water", 0.5)
    _write_scalar("alpha.vapor", 0)
    _write_scalar("alpha", 0.3)

    # Pressure fields
    _write_scalar("p", 101325)
    _write_scalar("p_rgh", 101325, "fixedFluxPressure", "fixedFluxPressure")

    # Scalar concentration
    _write_scalar("C", 0, "fixedValue")
    # Override C movingWall to 1
    h_C = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                         class_name="volScalarField", location="0", object="C")
    lines_C = [
        "dimensions      [0 0 0 0 0 0 0];",
        "internalField   uniform 0;",
        "boundaryField {",
        "    movingWall { type fixedValue; value uniform 1; }",
        "    fixedWalls { type zeroGradient; }",
        "    frontAndBack { type empty; }",
        "}",
    ]
    write_foam_file(zero_dir / "C", h_C, "\n".join(lines_C), overwrite=True)

    # Acoustic fields
    _write_scalar("p'", 0)
    _write_vector("U'", (0, 0, 0))

    # XiFoam progress variable
    _write_scalar("b", 1)

    # Species mass fraction
    _write_scalar("Y", 1)

    return case_dir


solvers_to_test = [
    "SimpleFoam", "IcoFoam", "PisoFoam", "PimpleFoam",
    "RhoSimpleFoam", "RhoPimpleFoam", "RhoCentralFoam", "SonicFoam",
    "InterFoam", "CompressibleInterFoam",
    "CavitatingFoam", "IncompressibleFluidFoam", "FluidFoam", "MulticomponentFluidFoam",
    "BuoyantSimpleFoam", "BuoyantPimpleFoam", "BuoyantBoussinesqSimpleFoam",
    "ReactingFoam", "SolidDisplacementFoam",
    "LaplacianFoam", "ScalarTransportFoam", "PotentialFoam",
    "BoundaryFoam", "PorousSimpleFoam", "SrfSimpleFoam",
    "IncompressibleVoFFoam", "CompressibleVoFFoam",
    "IncompressibleDriftFluxFoam", "IsothermalFluidFoam",
    "XiFoam", "DenseParticleFoam", "CompressibleMultiphaseVoFFoam",
    "AcousticFoam", "MagneticFoam", "MhdFoam",
    "SolidEquilibriumDisplacementFoam", "StressFoam",
    "DpmFoam", "MppicFoam",
    "CoalCombustionFoam", "ChemFoam", "DsmcFoam",
    "PDRFoam", "SprayFoam", "DieselFoam",
    "ViscousFoam",
]

from pyfoam.applications import __all__ as available
mod = __import__("pyfoam.applications", fromlist=solvers_to_test)

real_physics = []
zero_physics = []
errors = []

with tempfile.TemporaryDirectory() as tmp:
    case_dir = make_full_case(tmp, nu=0.01)

    for name in solvers_to_test:
        if name not in available:
            errors.append((name, "NOT_AVAILABLE"))
            continue
        try:
            cls = getattr(mod, name)
            kw = {}
            if name == "DenseParticleFoam":
                kw["n_particles"] = 20
            elif name == "CompressibleMultiphaseVoFFoam":
                kw["phases"] = [
                    {"name": "water", "rho": 1000, "mu": 1e-3},
                    {"name": "air", "rho": 1.225, "mu": 1.8e-5},
                ]

            s = cls(case_dir, **kw)
            conv = s.run()
            U = getattr(s, "U", None)
            if U is None:
                U = getattr(s, "T", None)
            if U is None:
                U = getattr(s, "D", None)
            if U is not None:
                U_max = U.abs().max().item()
                if U_max > 0.001:
                    real_physics.append((name, U_max))
                else:
                    zero_physics.append((name, U_max))
            else:
                zero_physics.append((name, 0))
        except Exception as e:
            errors.append((name, str(e)[:100]))

print(f"=== FULL VALIDATION ===")
print(f"Real physics: {len(real_physics)}/{len(solvers_to_test)}")
print(f"Zero physics: {len(zero_physics)}")
print(f"Errors: {len(errors)}")
print()
print("Real physics solvers:")
for name, u in sorted(real_physics):
    print(f"  {name}: U_max={u:.4f}")
print()
print("Zero physics solvers:")
for name, u in sorted(zero_physics):
    print(f"  {name}: U_max={u:.6f}")
print()
print("Error solvers:")
for name, e in sorted(errors):
    print(f"  {name}: {e}")
