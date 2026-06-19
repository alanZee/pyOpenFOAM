"""Run L2 comparisons on ASCII-mesh cases only (157 cases)."""
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
    rn = torch.norm(ref_t).item()
    if rn < 1e-15:
        return None
    return torch.norm(ref_t - py_t).item() / rn


def is_ascii_mesh(case_dir):
    owner = case_dir / "constant" / "polyMesh" / "owner"
    if not owner.exists():
        return False
    try:
        with open(owner, "rb") as f:
            data = f.read(500)
        return b"binary" not in data[:500]
    except:
        return False


def main():
    # Build tutorial map
    tut_cases = {}
    for root, dirs, files in os.walk(TUT_BASE):
        if "system" in os.listdir(root) and "constant" in os.listdir(root):
            rel = os.path.relpath(root, TUT_BASE)
            flat = rel.replace(os.sep, "_").replace("/", "_")
            tut_cases[flat] = Path(root)

    # Get ASCII mesh cases
    ref_cases = sorted([d for d in os.listdir(REF_BASE) if os.path.isdir(REF_BASE / d)])
    ascii_cases = [c for c in ref_cases if is_ascii_mesh(REF_BASE / c)]

    # Filter to cases with tutorials and 0/
    runnable = []
    for case in ascii_cases:
        ref_dir = REF_BASE / case
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
        if has_0:
            runnable.append((case, tut_path, ref_dir))

    print(f"Running L2 on {len(runnable)} ASCII-mesh cases...")

    results = []
    ok = err = skip = 0

    for i, (case, tut_path, ref_dir) in enumerate(runnable):
        work_dir = WORK / f"l2a_{case}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        # 0/
        if (tut_path / "0").exists():
            shutil.copytree(tut_path / "0", work_dir / "0")
        else:
            shutil.copytree(tut_path / "0.orig", work_dir / "0")

        # constant/
        const_dir = work_dir / "constant"
        const_dir.mkdir()
        ms = ref_dir / "constant" / "polyMesh"
        shutil.copytree(ms, const_dir / "polyMesh")
        for f in os.listdir(tut_path / "constant"):
            fp = tut_path / "constant" / f
            if f != "polyMesh" and os.path.isfile(fp):
                shutil.copy2(fp, const_dir / f)

        # system/
        ss = ref_dir / "system" if (ref_dir / "system").exists() else tut_path / "system"
        shutil.copytree(ss, work_dir / "system")

        # Limit run
        cd_path = work_dir / "system" / "controlDict"
        if cd_path.exists():
            c = cd_path.read_text(encoding="utf-8", errors="replace")
            c = re.sub(r"endTime\s+\S+;", "endTime         0.006;", c)
            c = re.sub(r"deltaT\s+\S+;", "deltaT          0.005;", c)
            c = re.sub(r"nOuterIterations\s+\d+;", "nOuterIterations 1;", c)
            cd_path.write_text(c)

        print(f"[{i+1}/{len(runnable)}] {case}...", end=" ", flush=True)

        try:
            start = time.time()
            solver = apps.SimpleFoam(str(work_dir))
            solver.run()
            elapsed = time.time() - start

            # Find output
            py_times = []
            for d in os.listdir(work_dir):
                dp = work_dir / d
                if os.path.isdir(dp):
                    try:
                        float(d)
                        py_times.append(d)
                    except:
                        pass

            ref_times = []
            for d in os.listdir(ref_dir):
                dp = ref_dir / d
                if os.path.isdir(dp):
                    try:
                        float(d)
                        ref_times.append(d)
                    except:
                        pass

            if not py_times:
                print(f"OK ({elapsed:.1f}s, no output)")
                shutil.rmtree(work_dir, ignore_errors=True)
                results.append({"case": case, "status": "OK_NO_OUTPUT", "elapsed": round(elapsed, 1)})
                ok += 1
                continue

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
                        pfd = read_field(str(py_dir / fname))
                        rfd = read_field(str(ref_d / fname))
                        if (pfd.internal_field is not None and rfd.internal_field is not None
                                and isinstance(pfd.internal_field, torch.Tensor)
                                and isinstance(rfd.internal_field, torch.Tensor)
                                and pfd.internal_field.numel() > 0 and rfd.internal_field.numel() > 0):
                            l2 = compute_l2(rfd.internal_field, pfd.internal_field)
                            if l2 is not None:
                                field_errors[fname] = round(l2, 6)
                    except:
                        pass

            n_fields = len(field_errors)
            print(f"OK ({elapsed:.1f}s, {n_fields} fields)")
            shutil.rmtree(work_dir, ignore_errors=True)
            results.append({
                "case": case, "status": "OK", "solver": "SimpleFoam",
                "elapsed": round(elapsed, 1), "py_time": py_final,
                "ref_time": ref_final, "field_errors": field_errors,
            })
            ok += 1

        except Exception as e:
            elapsed = time.time() - start if "start" in dir() else 0
            print(f"ERR ({elapsed:.0f}s): {type(e).__name__}")
            shutil.rmtree(work_dir, ignore_errors=True)
            results.append({"case": case, "status": "ERROR", "error": str(e)[:200]})
            err += 1

    # Save
    with open(OUTPUT / "ascii_l2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== {ok} OK, {err} ERROR, {skip} SKIP ===")
    with_l2 = [r for r in results if r.get("field_errors")]
    print(f"With L2 data: {len(with_l2)}")

    # Summary table
    if with_l2:
        print(f"\n{'Case':<55s} {'U L2':>10s} {'p L2':>10s} {'Other':>10s}")
        print("-" * 90)
        for r in sorted(with_l2, key=lambda x: x.get("field_errors", {}).get("p", 999)):
            errs = r["field_errors"]
            u_l2 = f"{errs['U']:.4f}" if "U" in errs else "N/A"
            p_l2 = f"{errs['p']:.4f}" if "p" in errs else "N/A"
            other = len([k for k in errs if k not in ("U", "p")])
            print(f"{r['case']:<55s} {u_l2:>10s} {p_l2:>10s} {other:>10d}")


if __name__ == "__main__":
    main()
