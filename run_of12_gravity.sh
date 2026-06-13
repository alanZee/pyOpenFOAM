#!/bin/bash
# Run damBreak-like case with OpenFOAM v12 (gravity-driven flow)
export LD_LIBRARY_PATH=/tmp/openfoam12/opt/openfoam12/platforms/linux64GccDPInt32Opt/lib:/tmp/openfoam12/opt/openfoam12/platforms/linux64GccDPInt32Opt/lib/dummy:$LD_LIBRARY_PATH
export FOAM_ETC=/tmp/openfoam12/opt/openfoam12/etc
export WM_PROJECT_DIR=/tmp/openfoam12/opt/openfoam12
export PATH=$WM_PROJECT_DIR/platforms/linux64GccDPInt32Opt/bin:$WM_PROJECT_DIR/bin:$PATH
BLOCKMESH=$WM_PROJECT_DIR/platforms/linux64GccDPInt32Opt/bin/blockMesh
ICOFOAM=$WM_PROJECT_DIR/platforms/linux64GccDPInt32Opt/bin/icoFoam

# Simple gravity-driven flow: tall column of fluid
echo "=== Gravity Column Reference (OpenFOAM v12) ==="
CASE_DIR=/tmp/gravity_v12
rm -rf $CASE_DIR
mkdir -p $CASE_DIR/0 $CASE_DIR/constant/polyMesh $CASE_DIR/system

# Tall column: 1x4 mesh, fluid initially in bottom half
cat > $CASE_DIR/system/blockMeshDict << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((0 0 0)(1 0 0)(1 4 0)(0 4 0)(0 0 0.1)(1 0 0.1)(1 4 0.1)(0 4 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (4 16 1) simpleGrading (1 1 1));
edges ();
boundary (
    top { type patch; faces ((3 7 6 2)); }
    bottom { type wall; faces ((0 4 7 3)(1 5 4 0)(2 6 5 1)); }
    walls { type wall; faces ((0 4 5 1)(3 2 6 7)); }
    frontAndBack { type empty; faces ((0 3 2 1)(4 5 6 7)); }
);
EOF

cat > $CASE_DIR/system/controlDict << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 0.5;
deltaT 0.001; writeControl timeStep; writeInterval 500; purgeWrite 0;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
EOF

cat > $CASE_DIR/system/fvSchemes << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linear; }
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
PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
EOF

cat > $CASE_DIR/0/U << 'EOF'
FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField {
    top { type pressureInletOutletVelocity; value uniform (0 0 0); }
    bottom { type noSlip; }
    walls { type noSlip; }
    frontAndBack { type empty; }
}
EOF

cat > $CASE_DIR/0/p << 'EOF'
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField {
    top { type totalPressure; p0 uniform 0; }
    bottom { type zeroGradient; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }
}
EOF

cat > $CASE_DIR/constant/physicalProperties << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object physicalProperties; }
transportModel Newtonian; nu nu [ 0 2 -1 0 0 0 0 ] 1e-05;
EOF

cat > $CASE_DIR/constant/momentumTransport << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }
simulationType laminar;
EOF

# Create gravity file
cat > $CASE_DIR/constant/g << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object g; }
dimensions [0 1 -2 0 0 0 0]; value (0 -9.81 0);
EOF

cd $CASE_DIR && $BLOCKMESH 2>&1 | tail -2
cd $CASE_DIR && $ICOFOAM 2>&1 | tail -5

FINAL=$(ls -d $CASE_DIR/[0-9]* 2>/dev/null | sort -t/ -k4 -n | tail -1)
python3 -c "
import re
with open('$FINAL/U') as f:
    c = f.read()
m = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', c)
if m:
    n = int(m.group(1))
    vals = re.findall(r'\(([^)]+)\)', c[m.end():])
    uy = []
    for v in vals[:n]:
        parts = v.strip().split()
        if len(parts) >= 2:
            try: uy.append(float(parts[1]))
            except: pass
    if uy:
        print(f'  Gravity: Uy_max={max(uy):.6f} Uy_min={min(uy):.6f}')
"
