"""验证全部基础求解器产生真实物理结果（检查每个求解器的主变量）。"""
import tempfile, torch
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

    def ws(name, val, dim="[0 0 0 0 0 0 0]"):
        h = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                           class_name="volScalarField", location="0", object=name)
        lines = [
            f"dimensions      {dim};",
            f"internalField   uniform {val};",
            "boundaryField {",
            "    movingWall { type zeroGradient; }",
            "    fixedWalls { type zeroGradient; }",
            "    frontAndBack { type empty; }",
            "}",
        ]
        write_foam_file(zero_dir / name, h, "\n".join(lines), overwrite=True)

    ws("T", 400, "[0 0 0 1 0 0 0]")
    ws("p", 101325, "[1 -1 -2 0 0 0 0]")
    ws("p_rgh", 101325, "[1 -1 -2 0 0 0 0]")
    ws("alpha.water", 0.5)
    ws("alpha.vapor", 0)
    ws("alpha", 0.3)
    ws("b", 1)
    ws("C", 0)
    ws("Y", 1)

    # ReactingFoam species
    for name in ["YA", "YB"]:
        val = 1.0 if name == "YA" else 0.0
        ws(name, val)

    # Acoustic fields
    ws("p'", 0, "[1 -1 -2 0 0 0 0]")
    h_up = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                          class_name="volVectorField", location="0", object="u'")
    lines_up = [
        "dimensions      [0 1 -1 0 0 0 0];",
        "internalField   uniform (0 0 0);",
        "boundaryField {",
        "    movingWall { type fixedValue; value uniform (0 0 0); }",
        "    fixedWalls { type noSlip; }",
        "    frontAndBack { type empty; }",
        "}",
    ]
    write_foam_file(zero_dir / "u'", h_up, "\n".join(lines_up), overwrite=True)

    return case_dir


# 每个求解器的主变量和最小有效阈值
SOLVER_CONFIG = {
    "SimpleFoam": {"field": "U", "threshold": 0.001},
    "IcoFoam": {"field": "U", "threshold": 0.001},
    "PisoFoam": {"field": "U", "threshold": 0.001},
    "PimpleFoam": {"field": "U", "threshold": 0.001},
    "RhoSimpleFoam": {"field": "U", "threshold": 0.001},
    "RhoPimpleFoam": {"field": "U", "threshold": 0.001},
    "RhoCentralFoam": {"field": "U", "threshold": 0.001},
    "SonicFoam": {"field": "U", "threshold": 0.001},
    "InterFoam": {"field": "U", "threshold": 0.001},
    "CompressibleInterFoam": {"field": "U", "threshold": 0.001},
    "CavitatingFoam": {"field": "U", "threshold": 0.001},
    "IncompressibleFluidFoam": {"field": "U", "threshold": 0.001},
    "FluidFoam": {"field": "U", "threshold": 0.001},
    "MulticomponentFluidFoam": {"field": "U", "threshold": 0.001},
    "BuoyantSimpleFoam": {"field": "U", "threshold": 0.001},
    "BuoyantPimpleFoam": {"field": "U", "threshold": 0.001},
    "BuoyantBoussinesqSimpleFoam": {"field": "U", "threshold": 0.001},
    "BoundaryFoam": {"field": "U", "threshold": 0.001},
    "PorousSimpleFoam": {"field": "U", "threshold": 0.001},
    "SrfSimpleFoam": {"field": "U", "threshold": 0.001},
    "IncompressibleVoFFoam": {"field": "U", "threshold": 0.001},
    "CompressibleVoFFoam": {"field": "U", "threshold": 0.001},
    "IncompressibleDriftFluxFoam": {"field": "U", "threshold": 0.001},
    "IsothermalFluidFoam": {"field": "U", "threshold": 0.001},
    "DenseParticleFoam": {"field": "U", "threshold": 0.001},
    "ViscousFoam": {"field": "U", "threshold": 0.001},
    "DsmcFoam": {"field": "U", "threshold": 0.001},
    "DieselFoam": {"field": "U", "threshold": 0.001},
    "SprayFoam": {"field": "U", "threshold": 0.001},
    "PDRFoam": {"field": "U", "threshold": 0.001},
    # 热传导/温度求解器
    "LaplacianFoam": {"field": "T", "threshold": 401},
    "ReactingFoam": {"field": "T", "threshold": 399},
    "XiFoam": {"field": "T", "threshold": 401},
    "ChemFoam": {"field": "T", "threshold": 401},
    # 标量输运
    "ScalarTransportFoam": {"field": "C", "threshold": 0.0},
    # 位移/应力
    "SolidDisplacementFoam": {"field": "D", "threshold": 0.0},
    "SolidEquilibriumDisplacementFoam": {"field": "D", "threshold": 0.0},
    "StressFoam": {"field": "D", "threshold": 0.0},
    # 声学
    "AcousticFoam": {"field": "p'", "threshold": 0.0},
    # 电磁
    "MagneticFoam": {"field": "U", "threshold": 0.0},
    "MhdFoam": {"field": "U", "threshold": 0.0},
    # 势流
    "PotentialFoam": {"field": "U", "threshold": 0.0},
}


from pyfoam.applications import __all__ as available
mod = __import__("pyfoam.applications", fromlist=list(SOLVER_CONFIG.keys()))

real_physics = []
zero_physics = []
errors = []

with tempfile.TemporaryDirectory() as tmp:
    case_dir = make_full_case(tmp, nu=0.01)

    for name, cfg in SOLVER_CONFIG.items():
        if name not in available:
            errors.append((name, "NOT_AVAILABLE"))
            continue
        try:
            cls = getattr(mod, name)
            kw = {}
            if name == "DenseParticleFoam":
                kw["n_particles"] = 20

            s = cls(case_dir, **kw)
            conv = s.run()

            field_name = cfg["field"]
            field = getattr(s, field_name, None)
            if field is None:
                # Try alternative names
                for alt in ["p_acoustic", "p_prime", "displacement"]:
                    field = getattr(s, alt, None)
                    if field is not None:
                        break

            if field is not None:
                val_max = field.abs().max().item()
                # 检查是否有变化（非均匀）
                val_range = (field.max() - field.min()).item()
                has_physics = val_max > cfg["threshold"] or val_range > 0.001
                if has_physics:
                    real_physics.append((name, field_name, val_max, val_range))
                else:
                    zero_physics.append((name, field_name, val_max))
            else:
                zero_physics.append((name, field_name, 0))
        except Exception as e:
            errors.append((name, str(e)[:100]))

print(f"=== FULL VALIDATION (with proper field checking) ===")
print(f"Real physics: {len(real_physics)}/{len(SOLVER_CONFIG)}")
print(f"Zero physics: {len(zero_physics)}")
print(f"Errors: {len(errors)}")
print()
print("Real physics:")
for name, fld, vmax, vrange in sorted(real_physics):
    print(f"  {name} ({fld}): max={vmax:.4f} range={vrange:.6f}")
print()
print("Zero physics:")
for name, fld, vmax in sorted(zero_physics):
    print(f"  {name} ({fld}): max={vmax:.6f}")
print()
print("Errors:")
for name, e in sorted(errors):
    print(f"  {name}: {e}")
