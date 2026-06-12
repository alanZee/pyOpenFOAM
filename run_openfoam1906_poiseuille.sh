#!/bin/bash
# Run Poiseuille flow reference case with OpenFOAM v1906
export LD_LIBRARY_PATH=/tmp/openfoam1906/usr/lib/x86_64-linux-gnu:/tmp/openfoam1906/usr/lib:$LD_LIBRARY_PATH
export FOAM_ETC=/tmp/openfoam1906/usr/share/openfoam/etc
export WM_PROJECT_DIR=/tmp/openfoam1906/usr/share/openfoam

CASE_DIR=/tmp/poiseuille_ref
rm -rf $CASE_DIR
mkdir -p $CASE_DIR/0 $CASE_DIR/constant/polyMesh $CASE_DIR/system

# blockMeshDict - 1D channel
cat > $CASE_DIR/system/blockMeshDict << 'EOF'
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
    (2 0 0)
    (2 1 0)
    (0 1 0)
    (0 0 0.1)
    (2 0 0.1)
    (2 1 0.1)
    (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)
);
edges
();
boundary
(
    inlet
    {
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }
    walls
    {
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
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
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
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
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
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
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
    relaxationFactors
    {
        fields
        {
            p               0.3;
        }
        equations
        {
            U               0.7;
        }
    }
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
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
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
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
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
transportModel  Newtonian;
nu              nu [ 0 2 -1 0 0 0 0 ] 0.01;
EOF

# turbulenceProperties
cat > $CASE_DIR/constant/turbulenceProperties << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType  laminar;
EOF

echo "=== Running Poiseuille flow ==="
cd $CASE_DIR && /tmp/openfoam1906/usr/bin/blockMesh 2>&1 | tail -2
cd $CASE_DIR && /tmp/openfoam1906/usr/bin/simpleFoam 2>&1 | tail -5

# Extract results
python3 -c "
import re
with open('$CASE_DIR/100/U') as f:
    content = f.read()
match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', content)
if match:
    n_cells = int(match.group(1))
    values = re.findall(r'\(([^)]+)\)', content[match.end():])
    ux_values = []
    for v in values[:n_cells]:
        parts = v.split()
        if len(parts) >= 1:
            try:
                ux = float(parts[0])
                ux_values.append(ux)
            except:
                pass
    if ux_values:
        print(f'  Ux_max={max(ux_values):.6f} Ux_min={min(ux_values):.6f}')
        # Analytical: U_max = 1.5 * U_avg for Poiseuille flow
        print(f'  Analytical U_max = 1.5 * 1.0 = 1.5')
"
