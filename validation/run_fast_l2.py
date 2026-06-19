"""Run fast L2 comparisons for key benchmark cases."""
import json, os, sys, time, shutil, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REF_BASE = Path("validation/reference/openfoam")
TUT_BASE = Path(".reference/OpenFOAM-13/tutorials")
OUTPUT = Path("validation/per_case_data")
WORK = Path("validation/_work")

import pyfoam.applications as apps
from pyfoam.io.field_io import read_field
import torch


def compute_l2(ref_t, py_t):
    if ref_t.shape != py_t.shape:
        return {"l2": None, "error": "shape_mismatch", "ref_shape": list(ref_t.shape), "py_shape": list(py_t.shape)}
    ref_norm = torch.norm(ref_t).item()
    if ref_norm < 1e-15:
        return {"l2": None, "error": "zero_norm"}
    diff_norm = torch.norm(ref_t - py_t).item()
    max_abs = torch.max(torch.abs(ref_t - py_t)).item()
    return {"l2": round(diff_norm / ref_norm, 8), "max_abs": round(max_abs, 8), "ref_norm": round(ref_norm, 4)}


def setup_and_run(case, tut_rel, solver_name, max_end_time=5):
    tut_path = TUT_BASE / tut_rel
    ref_dir = REF_BASE / case
    work_dir = WORK / f"fast_{case}"

    if not tut_path.exists():
        return {"case": case, "status": "SKIP", "reason": "no_tutorial"}

    has_0 = (tut_path / "0").exists() or (tut_path / "0.orig").exists()
    has_mesh = (ref_dir / "constant" / "polyMesh").exists() or (tut_path / "constant" / "polyMesh").exists()
    if not has_0 or not has_mesh:
        return {"case": case, "status": "SKIP", "reason": "no_0_or_mesh"}

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
    mesh_src = ref_dir / "constant" / "polyMesh" if (ref_dir / "constant" / "polyMesh").exists() else tut_path / "constant" / "polyMesh"
    shutil.copytree(mesh_src, const_dir / "polyMesh")
    for f in os.listdir(tut_path / "constant"):
        fp = tut_path / "constant" / f
        if f != "polyMesh" and os.path.isfile(fp):
            shutil.copy2(fp, const_dir / f)

    sys_src = ref_dir / "system" if (ref_dir / "system").exists() else tut_path / "system"
    shutil.copytree(sys_src, work_dir / "system")

    cd_path = work_dir / "system" / "controlDict"
    if cd_path.exists():
        content = cd_path.read_text(encoding="utf-8", errors="replace")
        content = re.sub(r"endTime\s+\S+;", f"endTime         {max_end_time};", content)
        cd_path.write_text(content)

    # Run
    try:
        solver_cls = getattr(apps, solver_name, None)
        if solver_cls is None:
            return {"case": case, "status": "SKIP", "reason": f"no_solver_{solver_name}"}

        start = time.time()
        solver = solver_cls(str(work_dir))
        solver.run()
        elapsed = time.time() - start

        # Find output time dirs
        py_times = []
        for d in os.listdir(work_dir):
            dp = work_dir / d
            if os.path.isdir(dp):
                try:
                    float(d)
                    py_times.append(d)
                except ValueError:
                    pass
        ref_times = []
        for d in os.listdir(ref_dir):
            dp = ref_dir / d
            if os.path.isdir(dp):
                try:
                    float(d)
                    ref_times.append(d)
                except ValueError:
                    pass

        if not py_times:
            return {"case": case, "status": "OK_NO_OUTPUT", "elapsed": round(elapsed, 1)}

        py_final = max(py_times, key=float)
        ref_final = max(ref_times, key=float) if ref_times else None

        # Compare fields
        field_errors = {}
        if ref_final:
            py_dir = work_dir / py_final
            ref_d = ref_dir / ref_final
            try:
                ref_fields = {f for f in os.listdir(ref_d) if os.path.isfile(ref_d / f) and f != "uniform"}
                py_fields = {f for f in os.listdir(py_dir) if os.path.isfile(py_dir / f) and f != "uniform"}
            except OSError:
                ref_fields = py_fields = set()

            for fname in sorted(ref_fields & py_fields):
                try:
                    py_fd = read_field(str(py_dir / fname))
                    ref_fd = read_field(str(ref_d / fname))
                    if (py_fd.internal_field is not None and ref_fd.internal_field is not None
                            and isinstance(py_fd.internal_field, torch.Tensor)
                            and isinstance(ref_fd.internal_field, torch.Tensor)
                            and py_fd.internal_field.numel() > 0 and ref_fd.internal_field.numel() > 0):
                        err = compute_l2(ref_fd.internal_field, py_fd.internal_field)
                        field_errors[fname] = err
                except Exception as e:
                    field_errors[fname] = {"error": str(e)[:80]}

        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "case": case, "status": "OK", "solver": solver_name,
            "elapsed": round(elapsed, 1), "py_time": py_final,
            "ref_time": ref_final, "field_errors": field_errors,
        }

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"case": case, "status": "ERROR", "error": str(e)[:200]}


# Key benchmark cases (fast to run)
fast_cases = [
    ("incompressibleFluid_planarCouette", "incompressibleFluid/planarCouette", "IncompressibleFluidFoam", 2),
    ("incompressibleFluid_planarPoiseuille", "incompressibleFluid/planarPoiseuille", "IncompressibleFluidFoam", 2),
    ("incompressibleFluid_cavity", "incompressibleFluid/cavity", "IncompressibleFluidFoam", 2),
    ("fluid_shockTube", "fluid/shockTube", "FluidFoam", 0.01),
    ("fluid_cavity", "fluid/cavity", "FluidFoam", 2),
    ("incompressibleFluid_pitzDaily", "incompressibleFluid/pitzDaily", "IncompressibleFluidFoam", 2),
    ("incompressibleFluid_cylinder", "incompressibleFluid/cylinder", "IncompressibleFluidFoam", 2),
    ("potentialFoam_pitzDaily", "potentialFoam/pitzDaily", "PotentialFoam", 1),
    ("fluid_angledDuct", "fluid/angledDuct", "FluidFoam", 2),
    ("incompressibleFluid_TJunction", "incompressibleFluid/TJunction", "IncompressibleFluidFoam", 2),
    ("shockFluid_shockTube", "shockFluid/shockTube", "RhoCentralFoam", 0.01),
    ("fluid_blockedChannel", "fluid/blockedChannel", "FluidFoam", 0.01),
    ("incompressibleFluid_blockedChannel", "incompressibleFluid/blockedChannel", "IncompressibleFluidFoam", 2),
    ("incompressibleFluid_elipsekkLOmega", "incompressibleFluid/elipsekkLOmega", "IncompressibleFluidFoam", 2),
    ("solidDisplacement_beamEndLoad", "solidDisplacement/beamEndLoad", "SolidDisplacementFoam", 1),
    ("legacy_incompressible_icoFoam_cavity", "legacy/incompressible/icoFoam/cavity/cavity", "IcoFoam", 0.5),
    ("compressibleVoF_damBreak", "compressibleVoF/damBreak", "CompressibleVoFFoam", 2),
    ("incompressibleVoF_damBreakLaminar", "incompressibleVoF/damBreakLaminar", "IncompressibleVoFFoam", 2),
    ("XiFluid_1D", "XiFluid/1D", "XiFoam", 0.01),
    ("multiphaseEuler_bubbleColumn", "multiphaseEuler/bubbleColumn", "MultiphaseEulerFoam", 5),
]

print(f"Running {len(fast_cases)} fast L2 comparisons...")
all_results = []
for i, (case, tut_rel, solver, end_time) in enumerate(fast_cases):
    print(f"[{i+1}/{len(fast_cases)}] {case}...", end=" ", flush=True)
    result = setup_and_run(case, tut_rel, solver, end_time)
    all_results.append(result)

    if result["status"] == "OK":
        n_fields = len(result.get("field_errors", {}))
        l2_vals = [v["l2"] for v in result.get("field_errors", {}).values() if isinstance(v, dict) and v.get("l2") is not None]
        avg_l2 = sum(l2_vals) / len(l2_vals) if l2_vals else 0
        print(f"OK ({result['elapsed']}s, {n_fields} fields, avg L2={avg_l2:.4f})")
    else:
        print(f"{result['status']}: {result.get('reason', result.get('error', ''))[:60]}")

# Save
with open(OUTPUT / "fast_l2_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

ok = sum(1 for r in all_results if r["status"] == "OK")
with_l2 = sum(1 for r in all_results if r.get("field_errors"))
print(f"\n=== {ok} OK, {with_l2} with L2 data ===")

# Print L2 summary table
print("\nL2 Error Summary:")
print(f"{'Case':<50s} {'Solver':<30s} {'U':>10s} {'p':>10s} {'Other':>10s}")
print("-" * 110)
for r in all_results:
    if r.get("field_errors"):
        errs = r["field_errors"]
        u_l2 = f"{errs['U']['l2']:.4f}" if "U" in errs and isinstance(errs["U"], dict) and errs["U"].get("l2") else "N/A"
        p_l2 = f"{errs['p']['l2']:.4f}" if "p" in errs and isinstance(errs["p"], dict) and errs["p"].get("l2") else "N/A"
        other = sum(1 for v in errs.values() if isinstance(v, dict) and v.get("l2") is not None and v.get("l2", 0) > 0)
        print(f"{r['case']:<50s} {r.get('solver',''):<30s} {u_l2:>10s} {p_l2:>10s} {other:>10d}")
