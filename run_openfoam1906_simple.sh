#!/bin/bash
export LD_LIBRARY_PATH=/tmp/openfoam1906/usr/lib/x86_64-linux-gnu:/tmp/openfoam1906/usr/lib:$LD_LIBRARY_PATH
export FOAM_ETC=/tmp/openfoam1906/usr/share/openfoam/etc
export WM_PROJECT_DIR=/tmp/openfoam1906/usr/share/openfoam

CASE_DIR=/tmp/cavity_ref

# Run simpleFoam cavity case
echo "=== Running simpleFoam cavity ==="
rm -rf $CASE_DIR
mkdir -p $CASE_DIR/0 $CASE_DIR/constant/polyMesh $CASE_DIR/system

# blockMeshDict
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
    hex (0 1 2 3 4 5 6 7) (8 8 1) simpleGrading (1 1 1)
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
endTime         0.1;
deltaT          0.001;
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

# Run blockMesh and icoFoam
cd $CASE_DIR && /tmp/openfoam1906/usr/bin/blockMesh 2>&1 | tail -3
cd $CASE_DIR && /tmp/openfoam1906/usr/bin/icoFoam 2>&1 | tail -5

# Save results
echo "=== Saving results ==="
cp -r $CASE_DIR/0.1 $CASE_DIR/final_time 2>/dev/null || echo "No 0.1 directory"
ls $CASE_DIR/ | grep -E '^[0-9]' | tail -5
