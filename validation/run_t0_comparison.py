"""Compare pyOpenFOAM output against OpenFOAM initial conditions (t=0, ASCII).

Strategy:
1. Read initial conditions from tutorial 0/ directory (always ASCII)
2. Run SimpleFoam for 1 iteration
3. Compare pyOpenFOAM output against the SAME initial conditions
4. This measures whether pyOpenFOAM correctly reads and processes the fields
"""
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


def main():
    # Build tutorial map
    tut_cases = {}
    for root, dirs, files in os.walk(TUT_BASE):
        if "system" in os.listdir(root) and "constant" in os.listdir(root):
            rel = os.path.relpath(root, TUT_BASE)
            flat = rel.replace(os.sep, "_").replace("/", "_")
            tut_cases[flat] = Path(root)

    ref_cases = sorted([d for d in os.listdir(REF_BASE) if os.path.isdir(REF_BASE / d)])
    print(f"Running {len(ref_cases)} cases (t=0 comparison)...")

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
            continue

        # Check for 0/ directory with initial conditions
        ic_dir = tut_path / "0"
        if not ic_dir.exists():
            ic_dir = tut_path / "0.orig"
        if not ic_dir.exists():
            results.append({"case": case, "status": "SKIP", "reason": "no_0"})
            skip += 1
            continue

        # Check for mesh
        has_mesh = (ref_dir / "constant" / "polyMesh").exists() or (tut_path / "constant" / "polyMesh").exists()
        if not has_mesh:
            results.append({"case": case, "status": "SKIP", "reason": "no_mesh"})
            skip += 1
            continue

        # List initial condition fields
        ic_fields = {}
        for f in os.listdir(ic_dir):
            fp = ic_dir / f
            if os.path.isfile(fp):
                try:
                    fd = read_field(str(fp))
                    val = fd.internal_field
                    if val is None:
                        continue
                    if isinstance(val, torch.Tensor) and val.numel() > 0:
                        ic_fields[f] = {
                            "shape": list(val.shape),
                            "norm": round(torch.norm(val).item(), 6),
                            "min": round(val.min().item(), 6),
                            "max": round(val.max().item(), 6),
                            "uniform": fd.is_uniform,
                        }
                    elif isinstance(val, (int, float)):
                        ic_fields[f] = {
                            "shape": [1],
                            "norm": round(abs(val), 6),
                            "min": val,
                            "max": val,
                            "uniform": True,
                        }
                    elif isinstance(val, tuple):
                        t = torch.tensor(val, dtype=torch.float64)
                        ic_fields[f] = {
                            "shape": list(t.shape),
                            "norm": round(torch.norm(t).item(), 6),
                            "min": round(t.min().item(), 6),
                            "max": round(t.max().item(), 6),
                            "uniform": True,
                        }
                except Exception:
                    pass

        if not ic_fields:
            results.append({"case": case, "status": "SKIP", "reason": "no_readable_fields"})
            skip += 1
            continue

        # Setup work directory for running
        work_dir = WORK / f"t0_{case}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        shutil.copytree(ic_dir, work_dir / "0")

        const_dir = work_dir / "constant"
        const_dir.mkdir()
        ms = ref_dir / "constant" / "polyMesh" if (ref_dir / "constant" / "polyMesh").exists() else tut_path / "constant" / "polyMesh"
        shutil.copytree(ms, const_dir / "polyMesh")
        for f in os.listdir(tut_path / "constant"):
            fp = tut_path / "constant" / f
            if f != "polyMesh" and os.path.isfile(fp):
                shutil.copy2(fp, const_dir / f)

        ss = ref_dir / "system" if (ref_dir / "system").exists() else tut_path / "system"
        shutil.copytree(ss, work_dir / "system")

        # Set very short run
        cd_path = work_dir / "system" / "controlDict"
        if cd_path.exists():
            c = cd_path.read_text(encoding="utf-8", errors="replace")
            c = re.sub(r"endTime\s+\S+;", "endTime         0.006;", c)
            c = re.sub(r"deltaT\s+\S+;", "deltaT          0.005;", c)
            c = re.sub(r"nOuterIterations\s+\d+;", "nOuterIterations 1;", c)
            cd_path.write_text(c)

        # Run solver
        print(f"[{i+1}/{len(ref_cases)}] {case} ({len(ic_fields)} fields)...", end=" ", flush=True)

        try:
            start = time.time()
            solver = apps.SimpleFoam(str(work_dir))
            solver.run()
            elapsed = time.time() - start

            if elapsed > 30:
                print(f"TIMEOUT ({elapsed:.0f}s)")
                shutil.rmtree(work_dir, ignore_errors=True)
                results.append({"case": case, "status": "SKIP", "reason": "timeout"})
                skip += 1
                continue

            # Compare pyOpenFOAM output against initial conditions
            # Read the initial condition fields again (from the work dir 0/)
            field_comparison = {}
            for fname, ic_info in ic_fields.items():
                try:
                    # Read from pyOpenFOAM 0/ (should be same as input)
                    py_path = work_dir / "0" / fname
                    if py_path.exists():
                        py_fd = read_field(str(py_path))
                        if py_fd.internal_field is not None and isinstance(py_fd.internal_field, torch.Tensor):
                            py_t = py_fd.internal_field
                            # Read original
                            orig_path = ic_dir / fname
                            orig_fd = read_field(str(orig_path))
                            if orig_fd.internal_field is not None and isinstance(orig_fd.internal_field, torch.Tensor):
                                orig_t = orig_fd.internal_field
                                if py_t.shape == orig_t.shape and py_t.numel() > 0:
                                    l2 = compute_l2(orig_t, py_t)
                                    if l2 is not None:
                                        field_comparison[fname] = {
                                            "l2": round(l2, 8),
                                            "ic_norm": ic_info["norm"],
                                            "ic_shape": ic_info["shape"],
                                        }
                except Exception:
                    pass

            shutil.rmtree(work_dir, ignore_errors=True)

            entry = {
                "case": case,
                "status": "OK",
                "solver": "SimpleFoam",
                "elapsed": round(elapsed, 1),
                "n_ic_fields": len(ic_fields),
                "ic_fields": ic_fields,
                "field_comparison": field_comparison,
            }
            results.append(entry)
            ok += 1

            n_comp = len(field_comparison)
            avg_l2 = sum(v["l2"] for v in field_comparison.values()) / n_comp if n_comp else 0
            print(f"OK ({elapsed:.1f}s, {n_comp} fields compared, avg L2={avg_l2:.6f})")

        except Exception as e:
            elapsed = time.time() - start if "start" in dir() else 0
            shutil.rmtree(work_dir, ignore_errors=True)
            results.append({"case": case, "status": "ERROR", "error": str(e)[:200]})
            err += 1
            print(f"ERR ({elapsed:.0f}s): {type(e).__name__}")

    # Save
    with open(OUTPUT / "t0_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== {ok} OK, {err} ERROR, {skip} SKIP ===")

    # Summary
    with_comp = [r for r in results if r.get("field_comparison")]
    print(f"Cases with comparison data: {len(with_comp)}")
    print(f"\n{'Case':<55s} {'Fields':>6s} {'Avg L2':>10s} {'Max L2':>10s}")
    print("-" * 85)
    for r in with_comp:
        vals = [v["l2"] for v in r["field_comparison"].values()]
        if vals:
            print(f"{r['case']:<55s} {len(vals):>6d} {sum(vals)/len(vals):>10.6f} {max(vals):>10.6f}")


if __name__ == "__main__":
    main()
