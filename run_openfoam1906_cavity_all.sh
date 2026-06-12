#!/bin/bash
# Run multiple OpenFOAM v1906 reference cases
export LD_LIBRARY_PATH=/tmp/openfoam1906/usr/lib/x86_64-linux-gnu:/tmp/openfoam1906/usr/lib:$LD_LIBRARY_PATH
export FOAM_ETC=/tmp/openfoam1906/usr/share/openfoam/etc
export WM_PROJECT_DIR=/tmp/openfoam1906/usr/share/openfoam

# Cavity cases with different mesh sizes
for NX in 8 16 32; do
    CASE_DIR=/tmp/cavity_${NX}x${NX}
    rm -rf $CASE_DIR
    mkdir -p $CASE_DIR/0 $CASE_DIR/constant/polyMesh $CASE_DIR/system

    # blockMeshDict
    cat > $CASE_DIR/system/blockMeshDict << EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale   1;
vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ($NX $NX 1) simpleGrading (1 1 1)
);
edges
();
boundary
(
    movingWall
    {
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (1 5 4 0)
            (2 6 5 1)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
EOF

    # controlDict
    cat > $CASE_DIR/system/controlDict << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1.0;
deltaT          0.005;
writeControl    timeStep;
writeInterval   200;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
EOF

    # fvSchemes
    cat > $CASE_DIR/system/fvSchemes << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes
{
    default         Euler;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear uncorrected;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         uncorrected;
}
EOF

    # fvSolution
    cat > $CASE_DIR/system/fvSolution << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.01;
    }
    pFinal
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.01;
    }
    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0.1;
    }
    UFinal
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0.1;
    }
}
PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
EOF

    # U field
    cat > $CASE_DIR/0/U << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    fixedWalls
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
EOF

    # p field
    cat > $CASE_DIR/0/p << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
EOF

    # transportProperties
    cat > $CASE_DIR/constant/transportProperties << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
nu              nu [ 0 2 -1 0 0 0 0 ] 0.01;
EOF

    echo "=== Running ${NX}x${NX} ==="
    cd $CASE_DIR && /tmp/openfoam1906/usr/bin/blockMesh 2>&1 | tail -2
    cd $CASE_DIR && /tmp/openfoam1906/usr/bin/icoFoam 2>&1 | tail -3

    # Extract results
    FINAL=$(ls -d $CASE_DIR/[0-9]* 2>/dev/null | sort -t/ -k4 -n | tail -1 | xargs basename)
    if [ -n "$FINAL" ] && [ -f "$CASE_DIR/$FINAL/U" ]; then
        python3 -c "
import re
with open('$CASE_DIR/$FINAL/U') as f:
    content = f.read()
match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', content)
if match:
    n_cells = int(match.group(1))
    values = re.findall(r'\(([^)]+)\)', content[match.end():])
    ux_values = []
    uy_values = []
    for v in values[:n_cells]:
        parts = v.split()
        if len(parts) >= 2:
            try:
                ux = float(parts[0])
                uy = float(parts[1])
                ux_values.append(ux)
                uy_values.append(uy)
            except:
                pass
    if ux_values:
        print(f'  ${NX}x${NX}: Ux_max={max(ux_values):.6f} Ux_min={min(ux_values):.6f} Uy_max={max(uy_values):.6f} Uy_min={min(uy_values):.6f}')
"
    fi
done
