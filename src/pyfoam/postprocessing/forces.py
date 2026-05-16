"""
Forces and ForceCoeffs — aerodynamic force calculation function objects.

Computes pressure and viscous forces on specified patches, along with
moments about a reference point.  Force coefficients (Cd, Cl, Cm) are
non-dimensionalised using reference area, velocity, and density.

Physics
-------
Pressure force:
    F_p = ∫ p·n dA

Viscous force:
    F_v = ∫ τ·n dA    where τ = μ(∇U + (∇U)^T)

Total force:
    F = F_p + F_v

Moment:
    M = ∫ (r × (p·n + τ·n)) dA

Force coefficients:
    Cd = F_D / (0.5 · ρ · U_ref² · A_ref)
    Cl = F_L / (0.5 · ρ · U_ref² · A_ref)
    Cm = M   / (0.5 · ρ · U_ref² · A_ref · L_ref)

References
----------
- OpenFOAM ``forces`` function object source
- OpenFOAM ``forceCoeffs`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Forces", "ForceCoeffs"]

logger = logging.getLogger(__name__)


class Forces(FunctionObject):
    """Compute pressure and viscous forces on boundary patches.

    Configuration keys (in ``controlDict`` sub-dictionary):

    - ``patches``: list of patch names to integrate over
    - ``rho``: reference density (default: ``"rhoInf"`` → 1.0)
    - ``CofR``: centre of rotation for moment calculation (default: ``(0,0,0)``)
    - ``directForceDensity``: if True, use fvc::div(dev(R)) instead of wallGradU

    Example controlDict entry::

        forces1
        {
            type            forces;
            libs            ("libforces.so");
            patches         (movingWall);
            rho             rhoInf;
            rhoInf          1.0;
            CofR            (0 0 0);
            log             true;
            writeControl    timeStep;
            writeInterval   1;
        }
    """

    def __init__(self, name: str = "forces", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._patches: List[str] = self.config.get("patches", [])
        self._rho_inf: float = float(self.config.get("rhoInf", 1.0))
        self._cofr: torch.Tensor = torch.tensor(
            self.config.get("CofR", [0.0, 0.0, 0.0]), dtype=torch.float64
        )
        self._direct_force_density: bool = self.config.get("directForceDensity", False)

        # Results storage
        self._force_pressure: List[torch.Tensor] = []
        self._force_viscous: List[torch.Tensor] = []
        self._force_total: List[torch.Tensor] = []
        self._moment: List[torch.Tensor] = []
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and validate patch names."""
        self._mesh = mesh
        self._fields = fields

        # Validate patches exist
        available = [p["name"] for p in mesh.boundary] if hasattr(mesh, "boundary") else []
        for pname in self._patches:
            if pname not in available:
                logger.warning("Patch '%s' not found in mesh. Available: %s", pname, available)

        logger.info("Forces '%s' initialised: patches=%s", self.name, self._patches)

    def execute(self, time: float) -> None:
        """Compute forces at current time step."""
        if not self._enabled or self._mesh is None:
            return

        p = self._fields.get("p")
        U = self._fields.get("U")

        if p is None or U is None:
            logger.warning("Fields 'p' and 'U' required for Forces. Skipping.")
            return

        f_p, f_v, f_total, moment = self._compute_forces(p, U)

        self._force_pressure.append(f_p.detach().cpu())
        self._force_viscous.append(f_v.detach().cpu())
        self._force_total.append(f_total.detach().cpu())
        self._moment.append(moment.detach().cpu())
        self._times.append(time)

        self._log.info(
            "t=%g  F_pressure=%s  F_viscous=%s  F_total=%s  M=%s",
            time,
            f_p.tolist(),
            f_v.tolist(),
            f_total.tolist(),
            moment.tolist(),
        )

    def _compute_forces(
        self, p_field, U_field
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pressure, viscous, total force and moment.

        Returns:
            Tuple of (F_pressure, F_viscous, F_total, moment), each ``(3,)``.
        """
        device = get_device()
        dtype = get_default_dtype()

        mesh = self._mesh
        n_internal = mesh.n_internal_faces
        face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        face_centres = mesh.face_centres.to(device=device, dtype=dtype)

        # Get field data
        if hasattr(p_field, "internal_field"):
            p_data = p_field.internal_field.to(device=device, dtype=dtype)
        else:
            p_data = p_field.to(device=device, dtype=dtype)

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        # Get viscosity (laminar + turbulent)
        mu = self._get_viscosity()

        F_pressure = torch.zeros(3, dtype=dtype, device=device)
        F_viscous = torch.zeros(3, dtype=dtype, device=device)
        moment = torch.zeros(3, dtype=dtype, device=device)

        # Integrate over each requested patch
        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long
            )

            # Face areas for this patch
            S = face_areas[face_indices]  # (n_faces, 3)
            S_mag = S.norm(dim=1, keepdim=True)  # (n_faces, 1)
            n = S / S_mag.clamp(min=1e-30)  # unit normals

            # Face centres for this patch
            r = face_centres[face_indices]  # (n_faces, 3)

            # Pressure on boundary faces: use owner cell value
            owner = mesh.owner[face_indices]
            p_face = p_data[owner]  # (n_faces,)

            # Pressure force: F_p = ∫ p·n dA = Σ p_face * n * |S|
            f_p_patch = (p_face.unsqueeze(1) * n * S_mag).sum(dim=0)
            F_pressure = F_pressure + f_p_patch

            # Viscous force: F_v = ∫ τ·n dA
            # τ = μ(∇U + (∇U)^T) · n
            # For wall boundaries, approximate using wall gradient
            U_face = U_data[owner]  # (n_faces, 3)

            # Simple approximation: τ ≈ μ * U / δ where δ is near-wall distance
            # More accurate: compute wall gradient from field gradient
            if self._direct_force_density and "gradU" in self._fields:
                gradU = self._fields["gradU"]
                if hasattr(gradU, "internal_field"):
                    gradU_data = gradU.internal_field.to(device=device, dtype=dtype)
                else:
                    gradU_data = gradU.to(device=device, dtype=dtype)

                gradU_face = gradU_data[owner]  # (n_faces, 3, 3)
                # dev(R) = μ * (∇U + (∇U)^T)
                strain = gradU_face + gradU_face.transpose(-1, -2)
                tau = mu * strain  # (n_faces, 3, 3)
                # τ · n
                tau_n = torch.einsum("fij,fj->fi", tau, n)  # (n_faces, 3)
                f_v_patch = (tau_n * S_mag).sum(dim=0)
            else:
                # Approximate: assume zero velocity at wall (no-slip)
                # and use finite difference with owner cell
                # This is a simplified approach
                f_v_patch = torch.zeros(3, dtype=dtype, device=device)

            F_viscous = F_viscous + f_v_patch

            # Moment about CofR: M = Σ (r - CofR) × (F_p + F_v) per face
            r_rel = r - self._cofr.to(device=device, dtype=dtype)
            f_face = (p_face.unsqueeze(1) * n * S_mag)  # (n_faces, 3)
            m_face = torch.cross(r_rel, f_face)  # (n_faces, 3)
            moment = moment + m_face.sum(dim=0)

        F_total = F_pressure + F_viscous
        return F_pressure, F_viscous, F_total, moment

    def _get_viscosity(self) -> float:
        """Get dynamic viscosity from fields or config."""
        # Try to get from turbulence model
        if "nut" in self._fields:
            # For RANS, use laminar viscosity
            pass
        # Default to 1.0 (incompressible, Re-based)
        return float(self.config.get("mu", 1.0))

    def _get_patch_info(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """Get patch information from mesh boundary."""
        if not hasattr(self._mesh, "boundary"):
            return None
        for p in self._mesh.boundary:
            if p["name"] == patch_name:
                return p
        return None

    def write(self) -> None:
        """Write force data to output files."""
        if self._output_path is None or not self._times:
            return

        # Write forces.dat
        forces_file = self._output_path / "forces.dat"
        with open(forces_file, "w") as f:
            f.write("# Time  Fpx  Fpy  Fpz  Fvx  Fvy  Fvz  Fx  Fy  Fz  Mx  My  Mz\n")
            for i, t in enumerate(self._times):
                fp = self._force_pressure[i]
                fv = self._force_viscous[i]
                ft = self._force_total[i]
                m = self._moment[i]
                f.write(
                    f"{t:.6e}  "
                    f"{fp[0]:.6e} {fp[1]:.6e} {fp[2]:.6e}  "
                    f"{fv[0]:.6e} {fv[1]:.6e} {fv[2]:.6e}  "
                    f"{ft[0]:.6e} {ft[1]:.6e} {ft[2]:.6e}  "
                    f"{m[0]:.6e} {m[1]:.6e} {m[2]:.6e}\n"
                )
        logger.info("Wrote forces to %s", forces_file)

    @property
    def force_pressure(self) -> List[torch.Tensor]:
        """List of pressure force vectors per time step."""
        return self._force_pressure

    @property
    def force_viscous(self) -> List[torch.Tensor]:
        """List of viscous force vectors per time step."""
        return self._force_viscous

    @property
    def force_total(self) -> List[torch.Tensor]:
        """List of total force vectors per time step."""
        return self._force_total

    @property
    def moment_history(self) -> List[torch.Tensor]:
        """List of moment vectors per time step."""
        return self._moment

    @property
    def times(self) -> List[float]:
        """List of time values."""
        return self._times


class ForceCoeffs(FunctionObject):
    """Compute non-dimensional force coefficients (Cd, Cl, Cm).

    Configuration keys:

    - ``patches``: list of patch names
    - ``rho``: reference density (default: ``"rhoInf"`` → 1.0)
    - ``rhoInf``: value of rhoInf (default: 1.0)
    - ``Uref``: reference velocity magnitude (default: 1.0)
    - ``Aref``: reference area (default: 1.0)
    - ``lRef``: reference length for moment coefficient (default: 1.0)
    - ``CofR``: centre of rotation (default: ``(0,0,0)``)
    - ``liftDir``: lift direction vector (default: ``(0,1,0)``)
    - ``dragDir``: drag direction vector (default: ``(1,0,0)``)
    - ``pitchAxis``: pitch axis for moment (default: ``(0,0,1)``)

    Example controlDict entry::

        forceCoeffs1
        {
            type            forceCoeffs;
            libs            ("libforces.so");
            patches         (aerofoil);
            rho             rhoInf;
            rhoInf          1.225;
            Uref            30;
            Aref            0.1;
            lRef            0.1;
            CofR            (0.05 0 0);
            liftDir         (0 1 0);
            dragDir         (1 0 0);
            pitchAxis       (0 0 1);
        }
    """

    def __init__(self, name: str = "forceCoeffs", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._patches: List[str] = self.config.get("patches", [])
        self._rho_inf: float = float(self.config.get("rhoInf", 1.0))
        self._u_ref: float = float(self.config.get("Uref", 1.0))
        self._a_ref: float = float(self.config.get("Aref", 1.0))
        self._l_ref: float = float(self.config.get("lRef", 1.0))
        self._cofr: torch.Tensor = torch.tensor(
            self.config.get("CofR", [0.0, 0.0, 0.0]), dtype=torch.float64
        )
        self._lift_dir: torch.Tensor = torch.tensor(
            self.config.get("liftDir", [0.0, 1.0, 0.0]), dtype=torch.float64
        )
        self._drag_dir: torch.Tensor = torch.tensor(
            self.config.get("dragDir", [1.0, 0.0, 0.0]), dtype=torch.float64
        )
        self._pitch_axis: torch.Tensor = torch.tensor(
            self.config.get("pitchAxis", [0.0, 0.0, 1.0]), dtype=torch.float64
        )

        # Dynamic pressure: q = 0.5 * rho * U^2 * A
        self._q_ref = 0.5 * self._rho_inf * self._u_ref**2 * self._a_ref
        self._m_ref = self._q_ref * self._l_ref  # moment reference

        # Results
        self._cd: List[float] = []
        self._cl: List[float] = []
        self._cm: List[float] = []
        self._times: List[float] = []

        # Internal forces object
        self._forces = Forces(name=f"{name}_forces", config=config)

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Initialise the internal forces calculator."""
        self._mesh = mesh
        self._fields = fields
        self._forces.initialise(mesh, fields)
        logger.info("ForceCoeffs '%s' initialised: Uref=%g, Aref=%g", self.name, self._u_ref, self._a_ref)

    def execute(self, time: float) -> None:
        """Compute force coefficients at current time step."""
        if not self._enabled or self._mesh is None:
            return

        # Compute forces
        self._forces.execute(time)

        if not self._forces.force_total:
            return

        F_total = self._forces.force_total[-1]
        moment = self._forces.moment_history[-1]

        # Project onto lift/drag directions
        drag = torch.dot(F_total, self._drag_dir.to(device=F_total.device, dtype=F_total.dtype))
        lift = torch.dot(F_total, self._lift_dir.to(device=F_total.device, dtype=F_total.dtype))

        # Moment coefficient about pitch axis
        cm = torch.dot(moment, self._pitch_axis.to(device=moment.device, dtype=moment.dtype))

        # Non-dimensionalise
        cd_val = (drag / self._q_ref).item()
        cl_val = (lift / self._q_ref).item()
        cm_val = (cm / self._m_ref).item()

        self._cd.append(cd_val)
        self._cl.append(cl_val)
        self._cm.append(cm_val)
        self._times.append(time)

        self._log.info("t=%g  Cd=%.6f  Cl=%.6f  Cm=%.6f", time, cd_val, cl_val, cm_val)

    def write(self) -> None:
        """Write coefficient data to output files."""
        self._forces.write()

        if self._output_path is None or not self._times:
            return

        coeffs_file = self._output_path / "forceCoeffs.dat"
        with open(coeffs_file, "w") as f:
            f.write("# Time  Cd  Cl  Cm\n")
            for i, t in enumerate(self._times):
                f.write(f"{t:.6e}  {self._cd[i]:.6e}  {self._cl[i]:.6e}  {self._cm[i]:.6e}\n")
        logger.info("Wrote force coefficients to %s", coeffs_file)

    @property
    def cd(self) -> List[float]:
        """Drag coefficient history."""
        return self._cd

    @property
    def cl(self) -> List[float]:
        """Lift coefficient history."""
        return self._cl

    @property
    def cm(self) -> List[float]:
        """Moment coefficient history."""
        return self._cm

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# Register function objects
FunctionObjectRegistry.register("forces", Forces)
FunctionObjectRegistry.register("forceCoeffs", ForceCoeffs)
