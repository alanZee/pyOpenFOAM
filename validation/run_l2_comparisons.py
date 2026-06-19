"""Run pyOpenFOAM on cases and compute L2 errors against OpenFOAM reference."""
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
        return None
    ref_norm = torch.norm(ref_t).item()
    if ref_norm < 1e-15:
        return None
    diff = torch.norm(ref_t - py_t).item()
    return diff / ref_norm


def setup_case(case, tut_path, ref_dir, work_dir):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # 0/
    if (tut_path / "0").exists():
        shutil.copytree(tut_path / "0", work_dir / "0")
    elif (tut_path / "0.orig").exists():
        shutil.copytree(tut_path / "0.orig", work_dir / "0")
    else:
        return False

    # constant/
    const_dir = work_dir / "constant"
    const_dir.mkdir()
    if (ref_dir / "constant" / "polyMesh").exists():
        shutil.copytree(ref_dir / "constant" / "polyMesh", const_dir / "polyMesh")
    elif (tut_path / "constant" / "polyMesh").exists():
        shutil.copytree(tut_path / "constant" / "polyMesh", const_dir / "polyMesh")
    else:
        return False
    for f in os.listdir(tut_path / "constant"):
        fp = tut_path / "constant" / f
        if f != "polyMesh" and os.path.isfile(fp):
            shutil.copy2(fp, const_dir / f)

    # system/
    if (ref_dir / "system").exists():
        shutil.copytree(ref_dir / "system", work_dir / "system")
    else:
        shutil.copytree(tut_path / "system", work_dir / "system")

    # Limit iterations
    cd_path = work_dir / "system" / "controlDict"
    if cd_path.exists():
        content = cd_path.read_text(encoding="utf-8", errors="replace")
        content = re.sub(r"endTime\s+\S+;", "endTime         10;", content)
        cd_path.write_text(content)
    return True


def get_final_time(case_dir):
    times = []
    try:
        for d in os.listdir(case_dir):
            dp = case_dir / d
            if os.path.isdir(dp):
                try:
                    float(d)
                    times.append(d)
                except ValueError:
                    pass
    except OSError:
        pass
    return max(times, key=float) if times else None


def compare_fields(py_dir, ref_dir):
    errors = {}
    try:
        ref_fields = {f for f in os.listdir(ref_dir) if os.path.isfile(ref_dir / f) and f != "uniform"}
        py_fields = {f for f in os.listdir(py_dir) if os.path.isfile(py_dir / f) and f != "uniform"}
    except OSError:
        return errors

    for fname in sorted(ref_fields & py_fields):
        try:
            py_fd = read_field(str(py_dir / fname))
            ref_fd = read_field(str(ref_dir / fname))
            if (py_fd.internal_field is not None and ref_fd.internal_field is not None
                    and isinstance(py_fd.internal_field, torch.Tensor)
                    and isinstance(ref_fd.internal_field, torch.Tensor)
                    and py_fd.internal_field.numel() > 0 and ref_fd.internal_field.numel() > 0):
                l2 = compute_l2(ref_fd.internal_field, py_fd.internal_field)
                if l2 is not None:
                    errors[fname] = round(l2, 6)
        except Exception:
            pass
    return errors


def main():
    # Build tutorial map
    tut_cases = {}
    for root, dirs, files in os.walk(TUT_BASE):
        if "system" in os.listdir(root) and "constant" in os.listdir(root):
            rel = os.path.relpath(root, TUT_BASE)
            flat = rel.replace(os.sep, "_").replace("/", "_")
            tut_cases[flat] = Path(root)

    # Load existing results
    with open(OUTPUT / "analysis_results.json") as f:
        results = json.load(f)

    with open("validation/results/all_tutorials_validation.json") as f:
        tut_data = json.load(f)
    tut_lookup = {t["path"].replace("/", "_"): t for t in tut_data.get("tutorials", [])}

    # Find cases to run
    to_run = []
    for r in results:
        case = r["case_name"]
        if r.get("l2_computed"):
            continue

        tut_path = None
        if case in tut_cases:
            tut_path = tut_cases[case]
        else:
            for tut in tut_cases:
                if tut.startswith(case + "_"):
                    tut_path = tut_cases[tut]
                    break
        if tut_path is None:
            continue

        has_0 = (tut_path / "0").exists() or (tut_path / "0.orig").exists()
        ref_dir = REF_BASE / case
        has_mesh = (ref_dir / "constant" / "polyMesh").exists() or (tut_path / "constant" / "polyMesh").exists()

        if has_0 and has_mesh:
            solver_name = r.get("solver_pyfoam", "")
            if not solver_name and case in tut_lookup:
                solver_name = tut_lookup[case].get("mapped_to", "")
            if solver_name:
                to_run.append((case, tut_path, solver_name))

    print(f"Cases to run: {len(to_run)}")

    batch = []
    for i, (case, tut_path, solver_name) in enumerate(to_run[:25]):
        ref_dir = REF_BASE / case
        work_dir = WORK / f"l2_{case}"

        print(f"[{i+1}/{min(25, len(to_run))}] {case} ({solver_name})...", end=" ", flush=True)

        if not setup_case(case, tut_path, ref_dir, work_dir):
            print("SKIP (setup failed)")
            batch.append({"case": case, "status": "SKIP", "reason": "setup"})
            continue

        try:
            solver_cls = getattr(apps, solver_name, None)
            if solver_cls is None:
                # Try base solver name
                base = solver_name.replace("Foam", "").replace("Enhanced", "")
                for attr in dir(apps):
                    if attr.lower().startswith(base.lower()) and "Enhanced" not in attr:
                        solver_cls = getattr(apps, attr)
                        solver_name = attr
                        break
            if solver_cls is None:
                print("SKIP (no solver)")
                batch.append({"case": case, "status": "SKIP", "reason": "no_solver"})
                continue

            start = time.time()
            solver = solver_cls(str(work_dir))
            solver.run()
            elapsed = time.time() - start

            py_final = get_final_time(work_dir)
            ref_final = get_final_time(ref_dir)

            field_errors = {}
            if py_final and ref_final:
                field_errors = compare_fields(work_dir / py_final, ref_dir / ref_final)

            print(f"OK ({elapsed:.1f}s, {len(field_errors)} fields, py_t={py_final})")
            batch.append({
                "case": case, "status": "OK", "solver": solver_name,
                "elapsed": round(elapsed, 1), "py_time": py_final, "ref_time": ref_final,
                "field_errors": field_errors, "n_fields": len(field_errors),
            })

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {str(e)[:80]}")
            batch.append({"case": case, "status": "ERROR", "error": str(e)[:200]})

        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    # Save
    with open(OUTPUT / "l2_comparison_results.json", "w") as f:
        json.dump(batch, f, indent=2)

    ok = sum(1 for r in batch if r["status"] == "OK")
    with_l2 = sum(1 for r in batch if r.get("field_errors"))
    print(f"\n=== {ok} OK, {with_l2} with L2 comparison data ===")

    # Print L2 summary
    for r in batch:
        if r.get("field_errors"):
            print(f"\n{r['case']} ({r['solver']}):")
            for fname, l2 in r["field_errors"].items():
                print(f"  {fname}: L2={l2:.6f} ({l2*100:.3f}%)")


if __name__ == "__main__":
    main()
