"""
Batch-run OpenFOAM v11 reference cases via Docker and save results.

Usage:
    CUDA_VISIBLE_DEVICES="" python validation/reference/batch_reference.py [--cases cavity couette channel]
"""

import subprocess
import os
import json
import re
import sys

DOCKER_IMAGE = "openfoam/openfoam11-paraview510"
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "openfoam")

# Maps pyOpenFOAM case names to OpenFOAM v11 tutorial paths and solvers
REFERENCE_CASES = {
    # Legacy icoFoam cavity
    "cavity_icoFoam": {
        "tutorial": "/opt/openfoam11/tutorials/legacy/incompressible/icoFoam/cavity/cavity",
        "solver": "icoFoam",
        "description": "Lid-driven cavity (icoFoam, Re=100, 20x20)",
    },
    # SIMPLE cavity (steady-state)
    "cavity_simpleFoam": {
        "tutorial": "/opt/openfoam11/tutorials/incompressibleFluid/cavity",
        "solver": "simpleFoam",
        "description": "Lid-driven cavity (simpleFoam, steady-state)",
    },
    # PotentialFoam
    "potentialFoam_cylinder": {
        "tutorial": "/opt/openfoam11/tutorials/potentialFoam/cylinder",
        "solver": "potentialFoam",
        "description": "Potential flow around cylinder",
    },
    # Dam break
    "damBreak": {
        "tutorial": "/opt/openfoam11/tutorials/multiphase/interFoam/laminar/damBreak/damBreak",
        "solver": "interFoam",
        "description": "Dam break (interFoam, VOF)",
    },
    # Shock tube
    "shockTube": {
        "tutorial": "/opt/openfoam11/tutorials/basic/shockTube",
        "solver": "rhoCentralFoam",
        "description": "Sod shock tube",
    },
}


def run_in_docker(cmd: str, volume_map: dict | None = None) -> str:
    """Run a command inside the OpenFOAM Docker container."""
    docker_cmd = ["docker", "run", "--rm"]
    if volume_map:
        for host_path, container_path in volume_map.items():
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
    docker_cmd.extend([
        "-w", "/work",
        "--entrypoint", "bash",
        DOCKER_IMAGE,
        "-c", cmd,
    ])
    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr


def extract_centerline(U_file: str, nx: int, ny: int, direction: str = "vertical") -> list:
    """Extract centerline velocity profile from OpenFOAM U field file."""
    with open(U_file) as f:
        content = f.read()

    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)", content)
    if not m:
        return []
    n = int(m.group(1))
    data_start = m.end()

    lines = content[data_start:].split("\n")
    ux_values = []
    uy_values = []
    for line in lines:
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            parts = line[1:-1].split()
            if len(parts) >= 3:
                ux_values.append(float(parts[0]))
                uy_values.append(float(parts[1]))
                if len(ux_values) >= n:
                    break

    profile = []
    if direction == "vertical":
        # x=const centerline, vary y
        i_mid = nx // 2
        for j in range(ny):
            cell = j * nx + i_mid
            y = (j + 0.5) / ny
            if cell < len(ux_values):
                profile.append({"y": y, "Ux": ux_values[cell], "Uy": uy_values[cell]})
    elif direction == "horizontal":
        # y=const centerline, vary x
        j_mid = ny // 2
        for i in range(nx):
            cell = j_mid * nx + i
            x = (i + 0.5) / nx
            if cell < len(ux_values):
                profile.append({"x": x, "Ux": ux_values[cell], "Uy": uy_values[cell]})

    return profile


def run_reference_case(case_name: str, case_config: dict) -> dict:
    """Run a single OpenFOAM reference case in Docker."""
    print(f"  Running {case_name}...")

    tutorial = case_config["tutorial"]
    solver = case_config["solver"]

    # Create output dir
    out_dir = os.path.join(REFERENCE_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)

    # Run blockMesh + solver in Docker
    cmd = f"""
        source /opt/openfoam11/etc/bashrc 2>/dev/null
        cp -r {tutorial}/* /work/ 2>/dev/null || true
        if [ -f system/blockMeshDict ]; then
            blockMesh 2>&1 | tail -3
        fi
        timeout 120 {solver} 2>&1 | tail -10
        FINAL=$(ls -d [0-9]* 2>/dev/null | sort -n | tail -1)
        if [ -n "$FINAL" ]; then
            cp -r $FINAL /work/result 2>/dev/null || true
        fi
    """

    result = run_in_docker(cmd, volume_map={out_dir: "/work"})

    # Save log
    with open(os.path.join(out_dir, "run.log"), "w") as f:
        f.write(result)

    # Check for results
    result_dir = os.path.join(out_dir, "result")
    if os.path.exists(result_dir):
        print(f"    Results saved to {out_dir}/result/")
        return {"status": "OK", "output_dir": out_dir}
    else:
        print(f"    FAILED: no results produced")
        return {"status": "FAILED", "log": result[:500]}


def main():
    cases_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(REFERENCE_CASES.keys())

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    results = {}

    print(f"Running {len(cases_to_run)} OpenFOAM v11 reference cases...")

    for case_name in cases_to_run:
        if case_name not in REFERENCE_CASES:
            print(f"  Unknown case: {case_name}")
            continue
        results[case_name] = run_reference_case(case_name, REFERENCE_CASES[case_name])

    # Save summary
    summary_path = os.path.join(REFERENCE_DIR, "reference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
