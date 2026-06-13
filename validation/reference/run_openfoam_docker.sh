#!/bin/bash
# Run an OpenFOAM tutorial case inside Docker and extract results
# Usage: bash run_openfoam_docker.sh <tutorial_path> <solver_name>
# Example: bash run_openfoam_docker.sh incompressibleFluid/cavity incompressibleFluid

set -e

TUTORIAL_PATH="${1:?Usage: $0 <tutorial_path> <solver_name>}"
SOLVER="${2:?Usage: $0 <tutorial_path> <solver_name>}"
DOCKER_IMAGE="openfoam/openfoam11-paraview510"
RESULT_DIR="validation/reference/openfoam/$(echo $TUTORIAL_PATH | tr '/' '_')"
mkdir -p "$RESULT_DIR"

echo "=== Running OpenFOAM reference: $TUTORIAL_PATH (solver: $SOLVER) ==="

# Find the tutorial in the reference directory
TUTORIAL_BASE=".reference/OpenFOAM-13/tutorials"
TUTORIAL_DIR="$TUTORIAL_BASE/$TUTORIAL_PATH"

if [ ! -d "$TUTORIAL_DIR" ]; then
    echo "ERROR: Tutorial not found: $TUTORIAL_DIR"
    exit 1
fi

# Copy tutorial to a temp directory for Docker mounting
TMPDIR=$(mktemp -d)
cp -r "$TUTORIAL_DIR"/* "$TMPDIR/" 2>/dev/null || true
# Also copy the Allrun script if it exists
if [ -f "$TUTORIAL_DIR/Allrun" ]; then
    cp "$TUTORIAL_DIR/Allrun" "$TMPDIR/"
fi

# Run inside Docker
docker run --rm \
    -v "$(cd $TMPDIR && pwd):/work" \
    -w /work \
    --entrypoint bash \
    "$DOCKER_IMAGE" -c "
        source /opt/openfoam11/etc/bashrc 2>/dev/null
        cd /work

        # Run blockMesh if system/blockMeshDict exists
        if [ -f system/blockMeshDict ]; then
            blockMesh 2>&1 | tail -5
        fi

        # Run the solver for a short time
        # Modify controlDict to limit runtime
        if [ -f system/controlDict ]; then
            # Run solver
            timeout 120 $SOLVER 2>&1 | tail -10
        fi

        # List result directories
        echo '=== Result directories ==='
        ls -d [0-9]* 2>/dev/null | sort -n | tail -5

        # Get final time directory
        FINAL=\$(ls -d [0-9]* 2>/dev/null | sort -t/ -k1 -n | tail -1)
        if [ -n \"\$FINAL\" ]; then
            echo \"=== Final time: \$FINAL ===\"
            # Copy results
            cp -r \$FINAL /work/result_final 2>/dev/null || true
        fi
    " 2>&1 | tee "$RESULT_DIR/run.log"

# Copy results back
if [ -d "$TMPDIR/result_final" ]; then
    cp -r "$TMPDIR/result_final"/* "$RESULT_DIR/" 2>/dev/null || true
    echo "Results saved to $RESULT_DIR/"
fi

# Cleanup
rm -rf "$TMPDIR"

echo "=== Done: $TUTORIAL_PATH ==="
