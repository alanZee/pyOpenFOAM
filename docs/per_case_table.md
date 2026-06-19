#### Combustion Xi (5 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| XiFluid_1D | XiFoam | Y | - | - |
| XiFluid_engine2Valve2D | XiFoam | Y | - | - |
| XiFluid_kivaTest | XiFoam | Y | 17 | - |
| XiFluid_moriyoshiHomogeneous |  | N | 14 | - |
| XiFluid_stratified | XiFoam | Y | - | - |

#### Compressible Multiphase VoF (1 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| compressibleMultiphaseVoF_damBreak4phaseLaminar | CompressibleMultiphase... | Y | 15 | - |

#### Compressible Shock (8 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| shockFluid_LadenburgJet60psi | RhoCentralFoam | Y | 5 | - |
| shockFluid_biconic25-55Run35 | RhoCentralFoam | Y | 4 | - |
| shockFluid_diffuserIntake | RhoCentralFoam | Y | 9 | - |
| shockFluid_forwardStep | RhoCentralFoam | Y | 5 | - |
| shockFluid_movingCone | RhoCentralFoam | Y | 7 | - |
| shockFluid_obliqueShock | RhoCentralFoam | Y | 5 | - |
| shockFluid_shockTube | RhoCentralFoam | Y | 6 | - |
| shockFluid_wedge15Ma5 | RhoCentralFoam | Y | 5 | - |

#### Compressible VoF (8 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| compressibleVoF_ballValve | CompressibleVoFFoam | Y | - | - |
| compressibleVoF_climbingRod | CompressibleVoFFoam | Y | 11 | - |
| compressibleVoF_damBreak | CompressibleVoFFoam | Y | 13 | - |
| compressibleVoF_damBreakInjection |  | N | 13 | - |
| compressibleVoF_depthCharge2D | CompressibleVoFFoam | Y | 8 | - |
| compressibleVoF_depthCharge3D | CompressibleVoFFoam | Y | 4 | - |
| compressibleVoF_sloshingTank2D | CompressibleVoFFoam | Y | 6 | - |
| compressibleVoF_throttle | CompressibleVoFFoam | Y | 10 | - |

#### Dense Particle (5 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| incompressibleDenseParticleFluid_Goldschmidt | DenseParticleFoam | Y | 4 | - |
| incompressibleDenseParticleFluid_GoldschmidtMPPIC | DenseParticleFoam | Y | 3 | - |
| incompressibleDenseParticleFluid_column | DenseParticleFoam | Y | 4 | - |
| incompressibleDenseParticleFluid_cyclone | DenseParticleFoam | Y | - | - |
| incompressibleDenseParticleFluid_injectionChannel | DenseParticleFoam | Y | 4 | - |

#### Drift Flux (3 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| incompressibleDriftFlux_dahl | IncompressibleDriftFlu... | Y | 7 | - |
| incompressibleDriftFlux_mixerVessel2DMRF | IncompressibleDriftFlu... | Y | 11 | - |
| incompressibleDriftFlux_tank3D | IncompressibleDriftFlu... | Y | 9 | - |

#### Film (1 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| film_rivuletPanel | FilmFoam | Y | 6 | - |

#### General Fluid (31 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| fluid_BernardCells | FluidFoam | Y | 3 | - |
| fluid_aerofoilNACA0012 | FluidFoam | Y | 10 | - |
| fluid_aerofoilNACA0012Steady | FluidFoam | Y | 10 | - |
| fluid_angledDuct | FluidFoam | Y | 9 | 3 |
| fluid_angledDuctExplicitFixedCoeff | FluidFoam | Y | 9 | 3 |
| fluid_angledDuctLTS | FluidFoam | Y | 10 | 3 |
| fluid_annularThermalMixer | FluidFoam | Y | 8 | - |
| fluid_blockedChannel | FluidFoam | Y | 6 | - |
| fluid_buoyantCavity | FluidFoam | Y | 10 | 3 |
| fluid_cavity | FluidFoam | Y | 5 | - |
| fluid_decompressionTank | FluidFoam | Y | - | - |
| fluid_externalCoupledCavity | FluidFoam | Y | 10 | 3 |
| fluid_forwardStep | FluidFoam | Y | 6 | 3 |
| fluid_helmholtzResonance | FluidFoam | Y | - | - |
| fluid_hotRadiationRoom | FluidFoam | Y | 11 | 3 |
| fluid_hotRadiationRoomFvDOM | FluidFoam | Y | 64 | - |
| fluid_hotRoom | FluidFoam | Y | 10 | - |
| fluid_hotRoomBoussinesq | FluidFoam | Y | 10 | - |
| fluid_hotRoomBoussinesqSteady | FluidFoam | Y | 10 | - |
| fluid_hotRoomComfort | FluidFoam | Y | 11 | - |
| fluid_iglooWithFridges | FluidFoam | Y | 10 | - |
| fluid_mixerVessel2DMRF | FluidFoam | Y | 10 | - |
| fluid_nacaAirfoil | FluidFoam | Y | - | - |
| fluid_pitzDaily | FluidFoam | Y | 17 | - |
| fluid_prism | FluidFoam | Y | 9 | - |
| fluid_roomHeating |  | N | - | - |
| fluid_shockTube | FluidFoam | Y | 6 | - |
| fluid_squareBend | FluidFoam | Y | 9 | - |
| fluid_squareBendLiq | FluidFoam | Y | 9 | - |
| fluid_squareBendLiqSteady | FluidFoam | Y | 9 | - |
| fluid_stackPlume | FluidFoam | Y | - | - |

#### Incompressible Steady-State (55 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| incompressibleFluid_T3A | IncompressibleFluidFoam | Y | 8 | - |
| incompressibleFluid_TJunction | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_TJunctionFan | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_airFoil2D | IncompressibleFluidFoam | Y | 5 | - |
| incompressibleFluid_ballValve | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_blockedChannel | IncompressibleFluidFoam | Y | 4 | - |
| incompressibleFluid_boxTurb16 | IncompressibleFluidFoam | Y | 4 | - |
| incompressibleFluid_cavity | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_cavityCoupledU | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_channel395 | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_cylinder | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_drivaerFastback |  | N | - | - |
| incompressibleFluid_ductSecondaryFlow | IncompressibleFluidFoam | Y | 4 | - |
| incompressibleFluid_elipsekkLOmega | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_flowWithOpenBoundary | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_hopperParticles |  | N | - | - |
| incompressibleFluid_impeller | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_mixerSRF | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_mixerVessel2D | IncompressibleFluidFoam | Y | 8 | - |
| incompressibleFluid_mixerVessel2DMRF | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_mixerVesselHorizontal2DParticles |  | N | - | - |
| incompressibleFluid_moodyChart |  | N | 3 | - |
| incompressibleFluid_motorBike | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_motorBikeSteady | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_motorBike_motorBike | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_movingCone | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_offsetCylinder | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_oscillatingInlet | IncompressibleFluidFoam | Y | 8 | - |
| incompressibleFluid_pipeCyclic | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_pitzDaily | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_pitzDailyLES | IncompressibleFluidFoam | Y | 14 | - |
| incompressibleFluid_pitzDailyLESDevelopedInlet | IncompressibleFluidFoam | Y | 12 | - |
| incompressibleFluid_pitzDailyLTS | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_pitzDailyPulse | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_pitzDailyScalarTransport |  | N | 1 | - |
| incompressibleFluid_pitzDailySteady | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_pitzDailySteadyExperimentalInlet | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_pitzDailySteadyMappedToPart |  | N | 6 | - |
| incompressibleFluid_pitzDailySteadyMappedToRefined |  | N | - | - |
| incompressibleFluid_planarContraction | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_planarCouette | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_planarPoiseuille | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_porousBlockage | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_propeller | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_roomResidenceTime | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_rotor2D | IncompressibleFluidFoam | Y | 8 | - |
| incompressibleFluid_rotor2DSRF | IncompressibleFluidFoam | Y | 7 | - |
| incompressibleFluid_rotorDisk | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_simpleRushtonMRF | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_simpleRushtonNCC | IncompressibleFluidFoam | Y | - | - |
| incompressibleFluid_turbineSiting | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_venturiTube | IncompressibleFluidFoam | Y | 3 | - |
| incompressibleFluid_waveSubSurface | IncompressibleFluidFoam | Y | 1 | - |
| incompressibleFluid_windAroundBuildings | IncompressibleFluidFoam | Y | 6 | - |
| incompressibleFluid_wingMotion | IncompressibleFluidFoam | Y | - | - |

#### Incompressible VoF (39 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| incompressibleVoF_DTCHull | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_DTCHullMoving | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_DTCHullWave | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_angledDuct | IncompressibleVoFFoam | Y | 10 | - |
| incompressibleVoF_capillaryRise | IncompressibleVoFFoam | Y | 7 | - |
| incompressibleVoF_cavitatingBullet | IncompressibleVoFFoam | Y | 2 | - |
| incompressibleVoF_climbingRod | IncompressibleVoFFoam | Y | 8 | - |
| incompressibleVoF_containerDischarge2D | IncompressibleVoFFoam | Y | 7 | - |
| incompressibleVoF_damBreak |  | N | 10 | - |
| incompressibleVoF_damBreak3D | IncompressibleVoFFoam | Y | 5 | - |
| incompressibleVoF_damBreakFine |  | N | 10 | - |
| incompressibleVoF_damBreakInjection |  | N | 10 | - |
| incompressibleVoF_damBreakLaminar | IncompressibleVoFFoam | Y | 4 | - |
| incompressibleVoF_damBreakLaminarFine |  | N | 7 | - |
| incompressibleVoF_damBreakPorousBaffle |  | N | 10 | - |
| incompressibleVoF_damBreakTracer |  | N | - | - |
| incompressibleVoF_floatingObject | IncompressibleVoFFoam | Y | 12 | - |
| incompressibleVoF_floatingObjectWaves | IncompressibleVoFFoam | Y | 15 | - |
| incompressibleVoF_forcedUpstreamWave | IncompressibleVoFFoam | Y | 1 | - |
| incompressibleVoF_mixerVessel | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_mixerVessel2DMRF | IncompressibleVoFFoam | Y | 5 | - |
| incompressibleVoF_mixerVesselHorizontal2D | IncompressibleVoFFoam | Y | 10 | - |
| incompressibleVoF_nozzleFlow2D | IncompressibleVoFFoam | Y | 4 | - |
| incompressibleVoF_parshallFlume | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_planingHullW3 | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_propeller | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_rotatingCube | IncompressibleVoFFoam | Y | - | - |
| incompressibleVoF_sloshingCylinder | IncompressibleVoFFoam | Y | 9 | - |
| incompressibleVoF_sloshingTank2D | IncompressibleVoFFoam | Y | 5 | - |
| incompressibleVoF_sloshingTank2D3DoF | IncompressibleVoFFoam | Y | 5 | - |
| incompressibleVoF_sloshingTank3D | IncompressibleVoFFoam | Y | 6 | - |
| incompressibleVoF_sloshingTank3D3DoF | IncompressibleVoFFoam | Y | 10 | - |
| incompressibleVoF_sloshingTank3D6DoF | IncompressibleVoFFoam | Y | 7 | - |
| incompressibleVoF_testTubeMixer | IncompressibleVoFFoam | Y | 6 | - |
| incompressibleVoF_trayedPipe | IncompressibleVoFFoam | Y | 5 | - |
| incompressibleVoF_waterChannel | IncompressibleVoFFoam | Y | 8 | - |
| incompressibleVoF_wave | IncompressibleVoFFoam | Y | 2 | - |
| incompressibleVoF_wave3D | IncompressibleVoFFoam | Y | 2 | - |
| incompressibleVoF_weirOverflow | IncompressibleVoFFoam | Y | 4 | - |

#### Isothermal Film (1 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| isothermalFilm_rivuletPanel | FilmFoam | Y | 5 | - |

#### Isothermal Fluid (2 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| isothermalFluid_potentialFreeSurfaceMovingOscillatingBox | IsothermalFluidFoam | Y | 5 | - |
| isothermalFluid_potentialFreeSurfaceOscillatingBox | IsothermalFluidFoam | Y | 4 | - |

#### Legacy (15 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| legacy_basic_laplacianFoam_flange | LaplacianFoam | Y | 4 | - |
| legacy_compressible_rhoPorousSimpleFoam_angledDuctImplicit | RhoPorousSimpleFoam | Y | 9 | - |
| legacy_electromagnetics_mhdFoam_hartmann | MhdFoam | Y | 8 | - |
| legacy_incompressible_adjointShapeOptimisationFoam_pitzDaily | AdjointFoam | Y | 10 | - |
| legacy_incompressible_icoFoam_cavity | IcoFoam | Y | - | - |
| legacy_incompressible_icoFoam_elbow | IcoFoam | Y | 3 | - |
| legacy_incompressible_porousSimpleFoam_angledDuctImplicit | PorousSimpleFoam | Y | 5 | - |
| legacy_incompressible_shallowWaterFoam_squareBump | ShallowWaterFoam | Y | 12 | - |
| legacy_lagrangian_dsmcFoam_freeSpacePeriodic | DsmcFoam | Y | 20 | - |
| legacy_lagrangian_dsmcFoam_freeSpaceStream | DsmcFoam | Y | 20 | - |
| legacy_lagrangian_dsmcFoam_supersonicCorner | DsmcFoam | Y | - | - |
| legacy_lagrangian_dsmcFoam_wedge15Ma5 | DsmcFoam | Y | 20 | - |
| legacy_lagrangian_mdEquilibrationFoam_periodicCubeArgon | MdFoam | Y | - | - |
| legacy_lagrangian_mdEquilibrationFoam_periodicCubeWater | MdFoam | Y | - | - |
| legacy_lagrangian_mdFoam_nanoNozzle | MdFoam | Y | - | - |

#### Mesh Generation (9 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| mesh_blockMesh_pipe |  | N | - | - |
| mesh_blockMesh_sphere |  | N | - | - |
| mesh_blockMesh_sphere7 |  | N | - | - |
| mesh_blockMesh_sphere7ProjectedEdges |  | N | - | - |
| mesh_refineMesh_refineFieldDirs |  | N | - | - |
| mesh_snappyHexMesh |  | N | - | - |
| mesh_snappyHexMesh_flange |  | N | 2 | - |
| mesh_snappyHexMesh_pipe |  | N | - | - |
| mesh_spiralPipe |  | N | - | - |

#### Moving Mesh (1 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| movingMesh_SnakeRiverCanyon | FluidFoam | Y | - | - |

#### Multi-Region CHT (20 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| multiRegion_CHT | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_VoFcoolingCylinder2D | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_circuitBoardCooling | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_coolingCylinder2D | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_coolingSphere | CHTMultiRegionFoam | Y | 1 | - |
| multiRegion_CHT_heatExchanger | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_heatedDuct | CHTMultiRegionFoam | Y | 1 | - |
| multiRegion_CHT_multiphaseCoolingCylinder2D | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_reverseBurner | CHTMultiRegionFoam | Y | 1 | - |
| multiRegion_CHT_shellAndTubeHeatExchanger | CHTMultiRegionFoam | Y | - | - |
| multiRegion_CHT_wallBoiling | CHTMultiRegionFoam | Y | 1 | - |
| multiRegion_film | FilmFoam | Y | - | - |
| multiRegion_film_VoFToFilm | FilmFoam | Y | - | - |
| multiRegion_film_cylinder | FilmFoam | Y | - | - |
| multiRegion_film_cylinderDripping | FilmFoam | Y | - | - |
| multiRegion_film_cylinderVoF | FilmFoam | Y | - | - |
| multiRegion_film_hotBoxes | FilmFoam | Y | - | - |
| multiRegion_film_rivuletBox | FilmFoam | Y | - | - |
| multiRegion_film_rivuletPanel | FilmFoam | Y | - | - |
| multiRegion_film_splashPanel | FilmFoam | Y | - | - |

#### Multicomponent Reacting (19 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| multicomponentFluid_DLR_A_LTS | MulticomponentFluidFoam | Y | 1 | - |
| multicomponentFluid_SandiaD_LTS |  | N | 37 | - |
| multicomponentFluid_aachenBomb | MulticomponentFluidFoam | Y | 15 | - |
| multicomponentFluid_counterFlowFlame2D | MulticomponentFluidFoam | Y | 11 | - |
| multicomponentFluid_counterFlowFlame2DLTS | MulticomponentFluidFoam | Y | 12 | - |
| multicomponentFluid_counterFlowFlame2DLTS_GRI_TDAC | MulticomponentFluidFoam | Y | 58 | - |
| multicomponentFluid_counterFlowFlame2D_GRI | MulticomponentFluidFoam | Y | 58 | - |
| multicomponentFluid_counterFlowFlame2D_GRI_TDAC | MulticomponentFluidFoam | Y | 58 | - |
| multicomponentFluid_filter | MulticomponentFluidFoam | Y | 8 | - |
| multicomponentFluid_lockExchange | MulticomponentFluidFoam | Y | 14 | - |
| multicomponentFluid_membrane | MulticomponentFluidFoam | Y | 10 | - |
| multicomponentFluid_nc7h16 | MulticomponentFluidFoam | Y | - | - |
| multicomponentFluid_parcelInBox | MulticomponentFluidFoam | Y | 7 | - |
| multicomponentFluid_simplifiedSiwek | MulticomponentFluidFoam | Y | 17 | - |
| multicomponentFluid_smallPoolFire2D | MulticomponentFluidFoam | Y | 25 | - |
| multicomponentFluid_smallPoolFire3D | MulticomponentFluidFoam | Y | - | - |
| multicomponentFluid_verticalChannel | MulticomponentFluidFoam | Y | 11 | - |
| multicomponentFluid_verticalChannelLTS | MulticomponentFluidFoam | Y | 12 | - |
| multicomponentFluid_verticalChannelSteady | MulticomponentFluidFoam | Y | 11 | - |

#### Multiphase Euler-Euler (26 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| multiphaseEuler_Grossetete | MultiphaseEulerFoam | Y | 15 | - |
| multiphaseEuler_LBend | MultiphaseEulerFoam | Y | 12 | - |
| multiphaseEuler_bed | MultiphaseEulerFoam | Y | 12 | - |
| multiphaseEuler_boilingBed | MultiphaseEulerFoam | Y | 2 | - |
| multiphaseEuler_bubbleColumn | MultiphaseEulerFoam | Y | 29 | - |
| multiphaseEuler_bubbleColumnEvaporating | MultiphaseEulerFoam | Y | 15 | - |
| multiphaseEuler_bubbleColumnEvaporatingDissolving | MultiphaseEulerFoam | Y | 17 | - |
| multiphaseEuler_bubbleColumnEvaporatingReacting | MultiphaseEulerFoam | Y | 23 | - |
| multiphaseEuler_bubbleColumnIATE | MultiphaseEulerFoam | Y | 18 | - |
| multiphaseEuler_bubbleColumnLES | MultiphaseEulerFoam | Y | 23 | - |
| multiphaseEuler_bubbleColumnLaminar | MultiphaseEulerFoam | Y | 17 | - |
| multiphaseEuler_bubblePipe | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_damBreak4phase | MultiphaseEulerFoam | Y | 20 | - |
| multiphaseEuler_fluidisedBed | MultiphaseEulerFoam | Y | 2 | - |
| multiphaseEuler_fluidisedBedLaminar | MultiphaseEulerFoam | Y | 17 | - |
| multiphaseEuler_hydrofoil | MultiphaseEulerFoam | Y | 3 | - |
| multiphaseEuler_injection | MultiphaseEulerFoam | Y | 13 | - |
| multiphaseEuler_mixerVessel2D | MultiphaseEulerFoam | Y | 25 | - |
| multiphaseEuler_mixerVessel2DMRF | MultiphaseEulerFoam | Y | 24 | - |
| multiphaseEuler_pipeBend | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_steamInjection | MultiphaseEulerFoam | Y | 4 | - |
| multiphaseEuler_titaniaSynthesis | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_titaniaSynthesisSurface | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_wallBoilingIATE | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_wallBoilingPolydisperse | MultiphaseEulerFoam | Y | - | - |
| multiphaseEuler_wallBoilingPolydisperseTwoGroups | MultiphaseEulerFoam | Y | - | - |

#### Multiphase VoF (4 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| incompressibleMultiphaseVoF_damBreak4phase | MultiphaseInterFoam | Y | 13 | - |
| incompressibleMultiphaseVoF_damBreak4phaseFineLaminar | MultiphaseInterFoam | Y | 4 | - |
| incompressibleMultiphaseVoF_damBreak4phaseLaminar | MultiphaseInterFoam | Y | 10 | - |
| incompressibleMultiphaseVoF_mixerVessel2DMRF | MultiphaseInterFoam | Y | 11 | - |

#### Potential Flow (2 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| potentialFoam_cylinder | PotentialFoam | Y | 6 | - |
| potentialFoam_pitzDaily | PotentialFoam | Y | 4 | - |

#### Solid Mechanics (2 cases)

| Case | pyFoam Solver | Validated | Ref Fields | L2 Data |
|------|---------------|-----------|------------|---------|
| solidDisplacement_beamEndLoad | SolidDisplacementFoam | Y | 3 | - |
| solidDisplacement_plateHole | SolidDisplacementFoam | Y | 9 | - |
