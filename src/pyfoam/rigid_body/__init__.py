"""
pyfoam.rigid_body — 6DOF rigid body dynamics solver.

Provides:

- :class:`SixDoFSolver` — six degree-of-freedom motion solver
- :class:`MotionSolver` — abstract base for motion solvers
- :class:`RigidBodySolver` — Newton-Euler rigid body solver
- :class:`Joint` hierarchy — revolute, prismatic, spherical, fixed
- :class:`Restraint` hierarchy — linear spring, linear/angular damper
"""

from pyfoam.rigid_body.six_dof_solver import SixDoFSolver
from pyfoam.rigid_body.motion_solver import MotionSolver
from pyfoam.rigid_body.solver import RigidBodySolver
from pyfoam.rigid_body.joints import (
    Joint,
    RevoluteJoint,
    PrismaticJoint,
    SphericalJoint,
    FixedJoint,
)
from pyfoam.rigid_body.restraints import (
    Restraint,
    LinearSpring,
    LinearDamper,
    AngularDamper,
)
from pyfoam.rigid_body.six_dof_solver_enhanced import (
    EnhancedSixDoFSolver,
    PositionConstraint,
    VelocityConstraint,
    ConstraintType,
)
from pyfoam.rigid_body.joints_enhanced import (
    CylindricalJoint,
    PlanarJoint,
    UniversalJoint,
    FreeJoint,
)
from pyfoam.rigid_body.restraints_enhanced import (
    TorsionSpring,
    NonlinearSpring,
    MotorRestraint,
    BushingRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_2 import (
    EnhancedSixDoFSolver2,
    BaumgarteParams,
)
from pyfoam.rigid_body.joints_enhanced_2 import (
    ScrewJoint,
    GimbalJoint,
    BushingJoint,
    RackPinionJoint,
)
from pyfoam.rigid_body.restraints_enhanced_2 import (
    CoulombFriction,
    HydraulicDamper,
    StopRestraint,
    PIDRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_3 import (
    EnhancedSixDoFSolver3,
    ContactParams,
    EnergyState,
)
from pyfoam.rigid_body.joints_enhanced_3 import (
    CamJoint,
    GearJoint,
    ConstantVelocityJoint,
    FlexibleJoint,
)
from pyfoam.rigid_body.restraints_enhanced_3 import (
    MagneticRestraint,
    BouyancyRestraint,
    ImpactRestraint,
    WindRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_4 import (
    EnhancedSixDoFSolver4,
    ForceHistoryEntry,
    StabilityInfo,
)
from pyfoam.rigid_body.joints_enhanced_4 import (
    ElasticJoint,
    ElectricalJoint,
    TelescopicJoint,
    PassiveJoint,
)
from pyfoam.rigid_body.restraints_enhanced_4 import (
    AerodynamicRestraint,
    ElasticFoundationRestraint,
    PressureRestraint,
    CentripetalRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_5 import (
    EnhancedSixDoFSolver5,
    EnergyTrackingState,
    AdaptiveSubstepConfig,
)
from pyfoam.rigid_body.joints_enhanced_5 import (
    MagnetorheologicalJoint,
    PneumaticJoint,
    HarmonicDriveJoint,
    RollingContactJoint,
)
from pyfoam.rigid_body.restraints_enhanced_5 import (
    ShapeMemoryAlloyRestraint,
    ElectrostaticRestraint,
    GeometricStiffnessRestraint,
    FluidInertiaRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_6 import (
    EnhancedSixDoFSolver6,
    AugmentedLagrangianConfig,
    MultiBodyCoupling,
    EnergyAdaptiveConfig,
)
from pyfoam.rigid_body.joints_enhanced_6 import (
    PiezoelectricJoint,
    VariableStiffnessJoint,
    FrictionJoint,
    MagneticLevitationJoint,
)
from pyfoam.rigid_body.restraints_enhanced_6 import (
    ViscoelasticRestraint,
    BistableSpringRestraint,
    ThermalExpansionRestraint,
    CreepRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_7 import (
    EnhancedSixDoFSolver7,
    ContactCouplingConfig,
    SensorModel,
    SLERPConfig,
)
from pyfoam.rigid_body.joints_enhanced_7 import (
    ShapeMemoryAlloyJoint,
    HydraulicJoint,
    SuperelasticJoint,
    TendonDrivenJoint,
)
from pyfoam.rigid_body.restraints_enhanced_7 import (
    MagnetorheologicalRestraint,
    FrictionPendulumRestraint,
    ParticleDamperRestraint,
    NegativeStiffnessRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_8 import (
    EnhancedSixDoFSolver8,
    MultiRateConfig,
    EnergyDriftConfig,
    ConstraintRelaxationConfig,
)
from pyfoam.rigid_body.joints_enhanced_8 import (
    MagnetostrictiveJoint,
    ElectroactivePolymerJoint,
    RotaryLinearJoint,
    GearedHarmonicJoint,
)
from pyfoam.rigid_body.restraints_enhanced_8 import (
    PneumaticHybridRestraint,
    ElectrorheologicalRestraint,
    InerterRestraint,
    QuasiZeroStiffnessRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_9 import (
    EnhancedSixDoFSolver9,
    ContactRestitutionConfig,
    AdaptiveSubstepConfig,
    ChartSwitchConfig,
)
from pyfoam.rigid_body.joints_enhanced_9 import (
    ShapeMemoryCompositeJoint,
    PneumaticArtificialMuscleJoint,
    TwistedStringJoint,
    HybridHydraulicJoint,
)
from pyfoam.rigid_body.restraints_enhanced_9 import (
    TunedMassDamperRestraint,
    MagnetorheologicalFluidRestraint,
    FrictionPendulumIsolator,
    ActiveTendonRestraint,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_10 import (
    EnhancedSixDoFSolver10,
    GeometricExactConfig,
    MultiRateConfig10,
    EnergyMomentumConfig,
)
from pyfoam.rigid_body.joints_enhanced_10 import (
    MagnetorheologicalCompositeJoint,
    ElectroactiveHydrogelJoint,
    DielectricElastomerJoint,
    ThermoplasticMemoryJoint,
)
from pyfoam.rigid_body.restraints_enhanced_10 import (
    ParticleImpactDamperRestraint,
    ElectrorheologicalFluidRestraint,
    NegativeStiffnessIsolator,
    ActiveMassDamperRestraint,
)

__all__ = [
    "SixDoFSolver",
    "MotionSolver",
    "RigidBodySolver",
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "SphericalJoint",
    "FixedJoint",
    "Restraint",
    "LinearSpring",
    "LinearDamper",
    "AngularDamper",
    # Enhanced
    "EnhancedSixDoFSolver",
    "PositionConstraint",
    "VelocityConstraint",
    "ConstraintType",
    "CylindricalJoint",
    "PlanarJoint",
    "UniversalJoint",
    "FreeJoint",
    "TorsionSpring",
    "NonlinearSpring",
    "MotorRestraint",
    "BushingRestraint",
    # V2 enhanced
    "EnhancedSixDoFSolver2",
    "BaumgarteParams",
    "ScrewJoint",
    "GimbalJoint",
    "BushingJoint",
    "RackPinionJoint",
    "CoulombFriction",
    "HydraulicDamper",
    "StopRestraint",
    "PIDRestraint",
    # V3 enhanced
    "EnhancedSixDoFSolver3",
    "ContactParams",
    "EnergyState",
    "CamJoint",
    "GearJoint",
    "ConstantVelocityJoint",
    "FlexibleJoint",
    "MagneticRestraint",
    "BouyancyRestraint",
    "ImpactRestraint",
    "WindRestraint",
    # V4 enhanced
    "EnhancedSixDoFSolver4",
    "ForceHistoryEntry",
    "StabilityInfo",
    "ElasticJoint",
    "ElectricalJoint",
    "TelescopicJoint",
    "PassiveJoint",
    "AerodynamicRestraint",
    "ElasticFoundationRestraint",
    "PressureRestraint",
    "CentripetalRestraint",
    # V5 enhanced
    "EnhancedSixDoFSolver5",
    "EnergyTrackingState",
    "AdaptiveSubstepConfig",
    "MagnetorheologicalJoint",
    "PneumaticJoint",
    "HarmonicDriveJoint",
    "RollingContactJoint",
    "ShapeMemoryAlloyRestraint",
    "ElectrostaticRestraint",
    "GeometricStiffnessRestraint",
    "FluidInertiaRestraint",
    # V6 enhanced
    "EnhancedSixDoFSolver6",
    "AugmentedLagrangianConfig",
    "MultiBodyCoupling",
    "EnergyAdaptiveConfig",
    "PiezoelectricJoint",
    "VariableStiffnessJoint",
    "FrictionJoint",
    "MagneticLevitationJoint",
    "ViscoelasticRestraint",
    "BistableSpringRestraint",
    "ThermalExpansionRestraint",
    "CreepRestraint",
    # V7 enhanced
    "EnhancedSixDoFSolver7",
    "ContactCouplingConfig",
    "SensorModel",
    "SLERPConfig",
    "ShapeMemoryAlloyJoint",
    "HydraulicJoint",
    "SuperelasticJoint",
    "TendonDrivenJoint",
    "MagnetorheologicalRestraint",
    "FrictionPendulumRestraint",
    "ParticleDamperRestraint",
    "NegativeStiffnessRestraint",
    # V8 enhanced
    "EnhancedSixDoFSolver8",
    "MultiRateConfig",
    "EnergyDriftConfig",
    "ConstraintRelaxationConfig",
    "MagnetostrictiveJoint",
    "ElectroactivePolymerJoint",
    "RotaryLinearJoint",
    "GearedHarmonicJoint",
    "PneumaticHybridRestraint",
    "ElectrorheologicalRestraint",
    "InerterRestraint",
    "QuasiZeroStiffnessRestraint",
    # V9 enhanced
    "EnhancedSixDoFSolver9",
    "ContactRestitutionConfig",
    "AdaptiveSubstepConfig",
    "ChartSwitchConfig",
    "ShapeMemoryCompositeJoint",
    "PneumaticArtificialMuscleJoint",
    "TwistedStringJoint",
    "HybridHydraulicJoint",
    "TunedMassDamperRestraint",
    "MagnetorheologicalFluidRestraint",
    "FrictionPendulumIsolator",
    "ActiveTendonRestraint",
    # V10 enhanced
    "EnhancedSixDoFSolver10",
    "GeometricExactConfig",
    "MultiRateConfig10",
    "EnergyMomentumConfig",
    "MagnetorheologicalCompositeJoint",
    "ElectroactiveHydrogelJoint",
    "DielectricElastomerJoint",
    "ThermoplasticMemoryJoint",
    "ParticleImpactDamperRestraint",
    "ElectrorheologicalFluidRestraint",
    "NegativeStiffnessIsolator",
    "ActiveMassDamperRestraint",
]
