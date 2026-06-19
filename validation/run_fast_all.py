"""Fast batch L2 using SimpleFoam for all feasible cases."""
import json, os, sys, time, shutil, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REF_BASE = Path("validation/reference/openfoam")
TUT_BASE = Path(".reference/OpenFOAM-13/tutorials")
OUTPUT = Path("validation/per_case_data")
WORK = Path("validation/_work")

from pyfoam.io.field_io import read_field
import torch
import pyfoam.applications as apps


def compute_l2(ref_t, py_t):
    if ref_t.shape != py_t.shape:
        return None
    ref_norm = torch.norm(ref_t).item()
    if ref_norm < 1e-15:
        return None
    return torch.norm(ref_t - py_t).item() / ref_norm


def run_fast(case_name, tut_path, ref_dir):
    """Run case with SimpleFoam (fast, stable) for 1 time unit."""
    work_dir = WORK / f"fast2_{case_name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # 0/
    if (tut_path / "0").exists():
        shutil.copytree(tut_path / "0", work_dir / "0")
    elif (tut_path / "0.orig").exists():
        shutil.copytree(tut_path / "0.orig", work_dir / "0")
    else:
        return {"status": "SKIP", "reason": "no_0"}

    # constant/
    const_dir = work_dir / "constant"
    const_dir.mkdir()
    mesh_src = ref_dir / "constant" / "polyMesh" if (ref_dir / "constant" / "polyMesh").exists() else tut_path / "constant" / "polyMesh"
    if not mesh_src.exists():
        return {"status": "SKIP", "reason": "no_mesh"}
    shutil.copytree(mesh_src, const_dir / "polyMesh")
    for f in os.listdir(tut_path / "constant"):
        fp = tut_path / "constant" / f
        if f != "polyMesh" and os.path.isfile(fp):
            shutil.copy2(fp, const_dir / f)

    # system/
    sys_src = ref_dir / "system" if (ref_dir / "system").exists() else tut_path / "system"
    shutil.copytree(sys_src, work_dir / "system")

    # Force SimpleFoam with single timestep
    cd_path = work_dir / "system" / "controlDict"
    if cd_path.exists():
        content = cd_path.read_text(encoding="utf-8", errors="replace")
        content = re.sub(r"endTime\s+\S+;", "endTime         0.006;", content)
        content = re.sub(r"deltaT\s+\S+;", "deltaT          0.005;", content)
        content = re.sub(r"application\s+\S+;", "application     foamRun;", content)
        content = re.sub(r"solver\s+\S+;", "solver          incompressibleFluid;", content)
        content = re.sub(r"nOuterIterations\s+\d+;", "nOuterIterations 1;", content)
        cd_path.write_text(content)

    try:
        start = time.time()
        solver = apps.SimpleFoam(str(work_dir))
        solver.run()
        elapsed = time.time() - start
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:80]}"}

    # Find output
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
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "OK_NO_OUTPUT", "elapsed": round(elapsed, 1)}

    py_final = max(py_times, key=float)
    ref_final = max(ref_times, key=float) if ref_times else None

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
                    l2 = compute_l2(ref_fd.internal_field, py_fd.internal_field)
                    if l2 is not None:
                        field_errors[fname] = round(l2, 6)
            except Exception:
                pass

    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "status": "OK", "solver": "SimpleFoam", "elapsed": round(elapsed, 1),
        "py_time": py_final, "ref_time": ref_final, "field_errors": field_errors,
    }


def main():
    # Build tutorial map
    tut_cases = {}
    for root, dirs, files in os.walk(TUT_BASE):
        if "system" in os.listdir(root) and "constant" in os.listdir(root):
            rel = os.path.relpath(root, TUT_BASE)
            flat = rel.replace(os.sep, "_").replace("/", "_")
            tut_cases[flat] = Path(root)

    ref_cases = sorted([d for d in os.listdir(REF_BASE) if os.path.isdir(REF_BASE / d)])
    print(f"Running {len(ref_cases)} cases with SimpleFoam (fast)...")

    results = []
    ok = skip = err = 0

    for i, case in enumerate(ref_cases):
        ref_dir = REF_BASE / case

        # Find tutorial
        tut_path = None
        if case in tut_cases:
            tut_path = tut_cases[case]
        else:
            for tut in tut_cases:
                if tut.startswith(case + "_"):
                    tut_path = tut_cases[tut]
                    break

        if tut_path is None:
            results.append({"case": case, "status": "SKIP", "reason": "no_tutorial"})
            skip += 1
            print(f"[{i+1}] {case}: SKIP (no tutorial)")
            continue

        print(f"[{i+1}/{len(ref_cases)}] {case}...", end=" ", flush=True)
        result = run_fast(case, tut_path, ref_dir)
        result["case"] = case
        results.append(result)

        if result["status"] == "OK":
            ok += 1
            n = len(result.get("field_errors", {}))
            print(f"OK ({result['elapsed']}s, {n} fields)")
        elif result["status"] == "ERROR":
            err += 1
            print(f"ERR: {result.get('error', '')[:50]}")
        else:
            skip += 1
            print(f"{result['status']}")

    # Save
    with open(OUTPUT / "fast_l2_all.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== {ok} OK, {err} ERROR, {skip} SKIP ===")

    # Summary
    with_l2 = [r for r in results if r.get("field_errors")]
    print(f"Cases with L2 data: {len(with_l2)}")
    print(f"\n{'Case':<55s} {'Fields':>6s} {'Avg L2':>10s} {'Max L2':>10s}")
    print("-" * 85)
    for r in with_l2:
        vals = [v for v in r["field_errors"].values() if v is not None]
        if vals:
            print(f"{r['case']:<55s} {len(vals):>6d} {sum(vals)/len(vals):>10.4f} {max(vals):>10.4f}")


if __name__ == "__main__":
    main()
