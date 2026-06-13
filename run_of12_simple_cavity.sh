#!/bin/bash
# Run lid-driven cavity with SIMPLEFoam (steady-state) using OpenFOAM v12
export LD_LIBRARY_PATH=/tmp/openfoam12/opt/openfoam12/platforms/linux64GccDPInt32Opt/lib:/tmp/openfoam12/opt/openfoam12/platforms/linux64GccDPInt32Opt/lib/dummy:$LD_LIBRARY_PATH
export FOAM_ETC=/tmp/openfoam12/opt/openfoam12/etc
export WM_PROJECT_DIR=/tmp/openfoam12/opt/openfoam12
export PATH=$WM_PROJECT_DIR/platforms/linux64GccDPInt32Opt/bin:$WM_PROJECT_DIR/bin:$PATH
BLOCKMESH=$WM_PROJECT_DIR/platforms/linux64GccDPInt32Opt/bin/blockMesh
SIMPLEFOAM=$WM_PROJECT_DIR/bin/simpleFoam

echo "=== SIMPLEFoam Cavity Reference (OpenFOAM v12) ==="
for NX in 8 16 32; do
    CASE_DIR=/tmp/simple_cavity_${NX}x${NX}
    rm -rf $CASE_DIR
    mkdir -p $CASE_DIR/0 $CASE_DIR/constant/polyMesh $CASE_DIR/system

    cat > $CASE_DIR/system/blockMeshDict << EOF
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 0.1)(1 0 0.1)(1 1 0.1)(0 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) ($NX $NX 1) simpleGrading (1 1 1));
edges ();
boundary (movingWall { type wall; faces ((3 7 6 2)); } fixedWalls { type wall; faces ((0 4 7 3)(1 5 4 0)(2 6 5 1)); } frontAndBack { type empty; faces ((0 3 2 1)(4 5 6 7)); });
EOF

    cat > $CASE_DIR/system/controlDict << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 100;
deltaT 1; writeControl timeStep; writeInterval 100; purgeWrite 0;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
EOF

    cat > $CASE_DIR/system/fvSchemes << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }
EOF

    cat > $CASE_DIR/system/fvSolution << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.01; }
pFinal { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.01; }
U { solver PBiCGStab; preconditioner DILU; tolerance 1e-05; relTol 0.1; }
UFinal { solver PBiCGStab; preconditioner DILU; tolerance 1e-05; relTol 0.1; } }
SIMPLE { nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0;
relaxationFactors { fields { p 0.3; } equations { U 0.7; } } }
EOF

    cat > $CASE_DIR/0/U << 'EOF'
FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField { movingWall { type fixedValue; value uniform (1 0 0); }
fixedWalls { type noSlip; } frontAndBack { type empty; } }
EOF

    cat > $CASE_DIR/0/p << 'EOF'
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField { movingWall { type zeroGradient; }
fixedWalls { type zeroGradient; } frontAndBack { type empty; } }
EOF

    cat > $CASE_DIR/constant/physicalProperties << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object physicalProperties; }
transportModel Newtonian; nu nu [ 0 2 -1 0 0 0 0 ] 0.01;
EOF

    cat > $CASE_DIR/constant/momentumTransport << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }
simulationType laminar;
EOF

    echo "--- Running ${NX}x${NX} ---"
    cd $CASE_DIR && $BLOCKMESH 2>&1 | tail -2
    cd $CASE_DIR && $SIMPLEFOAM 2>&1 | tail -3
    FINAL=$(ls -d $CASE_DIR/[0-9]* 2>/dev/null | sort -t/ -k4 -n | tail -1)
    python3 -c "
import re
with open('$FINAL/U') as f:
    c = f.read()
m = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', c)
if m:
    n = int(m.group(1))
    vals = re.findall(r'\(([^)]+)\)', c[m.end():])
    ux = []
    for v in vals[:n]:
        parts = v.strip().split()
        if len(parts) >= 1:
            try: ux.append(float(parts[0]))
            except: pass
    if ux:
        print(f'  ${NX}x${NX}: Ux_max={max(ux):.6f} Ux_min={min(ux):.6f}')
"
done
