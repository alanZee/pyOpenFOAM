"""Run remaining unvalidated cases."""
import json, os, sys, time, shutil, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pyfoam.applications as apps

REF_BASE = Path("validation/reference/openfoam")
TUT_BASE = Path(".reference/OpenFOAM-13/tutorials")
OUTPUT = Path("validation/per_case_data")
WORK = Path("validation/_work")

# All remaining cases with their tutorial paths and solvers
remaining = [
    ("incompressibleVoF_damBreak", "incompressibleVoF/damBreak", "IncompressibleVoFFoam"),
    ("incompressibleVoF_damBreakFine", "incompressibleVoF/damBreak", "IncompressibleVoFFoam"),
    ("incompressibleVoF_damBreakLaminarFine", "incompressibleVoF/damBreakLaminar", "IncompressibleVoFFoam"),
    ("incompressibleVoF_damBreakTracer", "incompressibleVoF/damBreakTracer", "IncompressibleVoFFoam"),
    ("incompressibleVoF_damBreakPorousBaffle", "incompressibleVoF/damBreakPorousBaffle", "IncompressibleVoFFoam"),
    ("incompressibleVoF_damBreakInjection", "incompressibleVoF/damBreakInjection", "IncompressibleVoFFoam"),
    ("compressibleVoF_damBreakInjection", "compressibleVoF/damBreakInjection", "CompressibleVoFFoam"),
    ("legacy_incompressible_icoFoam_cavity", "legacy/incompressible/icoFoam/cavity/cavity", "IcoFoam"),
    ("incompressibleFluid_pitzDailySteadyMappedToPart", "incompressibleFluid/pitzDailySteadyMappedToPart", "IncompressibleFluidFoam"),
    ("incompressibleFluid_pitzDailySteadyMappedToRefined", "incompressibleFluid/pitzDailySteadyMappedToRefined", "IncompressibleFluidFoam"),
    ("incompressibleFluid_drivaerFastback", "incompressibleFluid/drivaerFastback", "IncompressibleFluidFoam"),
    ("multicomponentFluid_SandiaD_LTS", "multicomponentFluid/SandiaD_LTS", "MulticomponentFluidFoam"),
    ("multiRegion_CHT", "multiRegion/CHT/circuitBoardCooling", "ChtMultiRegionFoam"),
    ("multiRegion_film", "multiRegion/film/cylinder", "FilmFoam"),
    ("mesh_blockMesh_pipe", "mesh/blockMesh/pipe", "PotentialFoam"),
    ("mesh_blockMesh_sphere", "mesh/blockMesh/sphere", "PotentialFoam"),
    ("mesh_blockMesh_sphere7", "mesh/blockMesh/sphere7", "PotentialFoam"),
    ("mesh_refineMesh_refineFieldDirs", "mesh/refineMesh/refineFieldDirs", "PotentialFoam"),
    ("mesh_snappyHexMesh", "mesh/snappyHexMesh/flange", "PotentialFoam"),
    ("mesh_snappyHexMesh_pipe", "mesh/snappyHexMesh/pipe", "PotentialFoam"),
    ("mesh_spiralPipe", "mesh/spiralPipe", "PotentialFoam"),
    ("fluid_roomHeating", "fluid/roomHeating", "FluidFoam"),
    ("film_rivuletPanel", "multiRegion/film/rivuletPanel", "FilmFoam"),
    ("XiFluid_moriyoshiHomogeneous", "XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous", "XiFoam"),
]

results = []
for case, tut_rel, solver_name in remaining:
    tut_path = TUT_BASE / tut_rel
    ref_dir = REF_BASE / case
    work_dir = WORK / f"rem_{case}"

    print(f"\n--- {case} ---")

    if not tut_path.exists():
        print(f"  SKIP: Tutorial path {tut_rel} not found")
        results.append({"case": case, "status": "SKIP", "reason": "no_tutorial"})
        continue

    # Check prerequisites
    has_0 = (tut_path / "0").exists() or (tut_path / "0.orig").exists()
    has_mesh = (ref_dir / "constant" / "polyMesh").exists() or (tut_path / "constant" / "polyMesh").exists()

    if not has_0:
        print(f"  SKIP: No initial conditions")
        results.append({"case": case, "status": "SKIP", "reason": "no_0"})
        continue

    if not has_mesh:
        print(f"  SKIP: No mesh")
        results.append({"case": case, "status": "SKIP", "reason": "no_mesh"})
        continue

    # Setup
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    if (tut_path / "0").exists():
        shutil.copytree(tut_path / "0", work_dir / "0")
    else:
        shutil.copytree(tut_path / "0.orig", work_dir / "0")

    const_dir = work_dir / "constant"
    const_dir.mkdir()
    if (ref_dir / "constant" / "polyMesh").exists():
        shutil.copytree(ref_dir / "constant" / "polyMesh", const_dir / "polyMesh")
    elif (tut_path / "constant" / "polyMesh").exists():
        shutil.copytree(tut_path / "constant" / "polyMesh", const_dir / "polyMesh")
    for f in os.listdir(tut_path / "constant"):
        fp = tut_path / "constant" / f
        if f != "polyMesh" and os.path.isfile(fp):
            shutil.copy2(fp, const_dir / f)

    if (ref_dir / "system").exists():
        shutil.copytree(ref_dir / "system", work_dir / "system")
    else:
        shutil.copytree(tut_path / "system", work_dir / "system")

    # Limit iterations
    cd_path = work_dir / "system" / "controlDict"
    if cd_path.exists():
        try:
            content = cd_path.read_text(encoding="utf-8", errors="replace")
            content = re.sub(r"endTime\s+\S+;", "endTime         5;", content)
            cd_path.write_text(content)
        except:
            pass

    # Run
    try:
        solver_cls = getattr(apps, solver_name, None)
        if solver_cls is None:
            print(f"  SKIP: Solver {solver_name} not found")
            results.append({"case": case, "status": "SKIP", "reason": "no_solver"})
            continue

        print(f"  Running {solver_name}...")
        start = time.time()
        solver = solver_cls(str(work_dir))
        solver.run()
        elapsed = time.time() - start
        print(f"  OK: {elapsed:.1f}s")
        results.append({"case": case, "status": "OK", "solver": solver_name, "elapsed": round(elapsed, 1)})
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}")
        results.append({"case": case, "status": "ERROR", "error": str(e)[:200]})

    # Cleanup
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

# Save
with open(OUTPUT / "remaining_runs.json", "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for r in results if r["status"] == "OK")
err = sum(1 for r in results if r["status"] == "ERROR")
skip = sum(1 for r in results if r["status"] == "SKIP")
print(f"\n=== Summary: {ok} OK, {err} ERROR, {skip} SKIP ===")
