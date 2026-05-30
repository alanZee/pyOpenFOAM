"""
applyBoundaryLayer enhanced v4 — enhanced boundary layer application with
thermal BL model and pressure-gradient-aware fitting (fourth generation).

Extends :func:`apply_boundary_layer_enhanced_3` with:

- **Thermal boundary layer**: Compute temperature profile coupled with
  velocity BL using Reynolds analogy.
- **Pressure-gradient-aware fitting**: Improve BL profile fitting when
  adverse/favourable pressure gradients are present.
- **Wall heat flux estimation**: Estimate wall heat flux from the
  temperature gradient at the wall.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_4 import (
        apply_boundary_layer_enhanced_4, EnhancedBL4Properties,
    )

    bl = EnhancedBL4Properties(delta=0.1, nu=1e-5)
    result = apply_boundary_layer_enhanced_4(mesh, U, bl, wall_patches=["bottom"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL4Properties", "EnhancedBL4Result", "apply_boundary_layer_enhanced_4"]


@dataclass
class EnhancedBL4Properties:
    """Enhanced v4 boundary layer model parameters.

    Parameters
    ----------
    delta : float
        Boundary layer thickness (m).
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
    E : float
    u_star : float, optional
    z0_rough : float
    blend_width : float
    Cmu : float
    wall_function : str
    dp_dx : float
    reference_U : float, optional
    reference_y : float, optional
    Pr : float
        Prandtl number for thermal BL.
    T_wall : float
        Wall temperature (K).
    T_inf : float
        Freestream temperature (K).
    thermal_conductivity : float
        Fluid thermal conductivity (W/m/K).
    """

    delta: float = 0.1
    nu: float = 1e-5
    kappa: float = 0.41
    E: float = 9.8
    u_star: Optional[float] = None
    z0_rough: float = 0.0
    blend_width: float = 0.2
    Cmu: float = 0.09
    wall_function: str = "standard"
    dp_dx: float = 0.0
    reference_U: Optional[float] = None
    reference_y: Optional[float] = None
    Pr: float = 0.71
    T_wall: float = 300.0
    T_inf: float = 293.15
    thermal_conductivity: float = 0.025


@dataclass
class EnhancedBL4Result:
    """Result from :func:`apply_boundary_layer_enhanced_4`.

    Attributes
    ----------
    velocity, k, epsilon, omega : np.ndarray
    temperature : np.ndarray
        Temperature field (K) with thermal BL applied.
    u_star_used, max_y_plus, n_cells_modified
    displacement_thickness, momentum_thickness, shape_factor
    wall_heat_flux : float
        Estimated wall heat flux (W/m^2).
    thermal_thickness : float
        Thermal boundary layer thickness estimate.
    nusselt_number : float
        Mean Nusselt number at the wall.
    """

    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    u_star_used: float = 0.0
    max_y_plus: float = 0.0
    n_cells_modified: int = 0
    displacement_thickness: float = 0.0
    momentum_thickness: float = 0.0
    shape_factor: float = 0.0
    wall_heat_flux: float = 0.0
    thermal_thickness: float = 0.0
    nusselt_number: float = 0.0


def apply_boundary_layer_enhanced_4(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL4Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
    temperature_field: Optional[np.ndarray] = None,
) -> EnhancedBL4Result:
    """Apply enhanced v4 boundary layer correction with thermal model.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray ``(n_cells, 3)``
    bl : EnhancedBL4Properties
    wall_patches : sequence of str, optional
    k_field, epsilon_field, omega_field, temperature_field : np.ndarray, optional

    Returns
    -------
    EnhancedBL4Result
    """
    result_v = velocity.copy()
    n_cells = velocity.shape[0]
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    face_centres = mesh.face_centres.detach().cpu().numpy()

    # Collect wall face centres
    wall_fc_list = []
    for p in mesh.boundary:
        if wall_patches is not None and p["name"] not in wall_patches:
            continue
        if p.get("type", "") != "wall" and wall_patches is None:
            continue
        start = p["startFace"]
        for fi in range(start, start + p["nFaces"]):
            wall_fc_list.append(face_centres[fi])

    if not wall_fc_list:
        return EnhancedBL4Result(velocity=result_v, u_star_used=0.0)

    wall_fc = np.array(wall_fc_list, dtype=np.float64)

    # Fit u_star
    u_star = bl.u_star
    if u_star is None:
        if bl.reference_U is not None and bl.reference_y is not None:
            u_star = _fit_u_star(bl.reference_U, bl.reference_y, bl)
        else:
            u_star = _estimate_u_star(velocity, cell_centres, wall_fc, bl)

    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu
    delta = bl.delta
    blend_w = bl.blend_width
    z0 = max(bl.z0_rough, 1e-30)
    Cmu = bl.Cmu

    k_new = k_field.copy() if k_field is not None else None
    eps_new = epsilon_field.copy() if epsilon_field is not None else None
    omg_new = omega_field.copy() if omega_field is not None else None
    T_new = temperature_field.copy() if temperature_field is not None else np.full(n_cells, bl.T_inf)

    max_yp = 0.0
    n_mod = 0
    delta_star = 0.0
    theta_sum = 0.0
    n_integral = 0

    # Thermal BL tracking
    wall_heat_flux_sum = 0.0
    thermal_thickness_max = 0.0
    n_nusselt = 0

    for ci in range(n_cells):
        cc = cell_centres[ci]
        dists = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1)
        y = dists.min()

        if y > delta or y < 1e-30:
            continue

        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue

        U_dir = velocity[ci] / U_mag
        y_plus = y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10

        max_yp = max(max_yp, y_plus)

        # BL velocity
        if bl.wall_function == "rough":
            U_bl = _rough_wall_function(y, z0, u_star, kappa, nu)
        elif bl.wall_function == "multi_layer":
            U_bl = _multi_layer_function(y_plus, u_star, kappa, E_val)
        else:
            U_bl = _spalding_wall_function(y_plus, u_star, kappa, E_val)

        # Pressure gradient correction
        if bl.dp_dx != 0.0 and y < delta:
            rho = 1.0
            dp_corr = -bl.dp_dx * y / (rho * max(u_star, 1e-30))
            U_bl = max(U_bl + dp_corr, 0.0)

        # Blending
        blend = _blending_function(y, delta, blend_w)
        U_new = blend * U_bl + (1.0 - blend) * U_mag

        result_v[ci] = U_new * U_dir
        n_mod += 1

        # Integral quantities
        U_ref = max(U_bl, 1e-30)
        delta_star += (1.0 - U_new / U_ref) * y
        theta_sum += (U_new / U_ref) * (1.0 - U_new / U_ref) * y
        n_integral += 1

        if k_new is not None:
            k_new[ci] = u_star ** 2 / math.sqrt(Cmu)
        if eps_new is not None:
            eps_new[ci] = u_star ** 3 / (kappa * max(y, z0))
        if omg_new is not None:
            k_safe = max(k_new[ci] if k_new is not None else u_star ** 2 / math.sqrt(Cmu), 1e-30)
            omg_new[ci] = max(
                eps_new[ci] if eps_new is not None else u_star ** 3 / (kappa * y),
                1e-30,
            ) / (Cmu * k_safe)

        # Temperature (Reynolds analogy)
        if temperature_field is not None or True:
            T_new[ci] = _thermal_bl_profile(
                y, delta, bl.T_wall, bl.T_inf, bl.Pr, kappa, y_plus,
            )

            # Wall heat flux estimate (first cell)
            if y < delta * 0.1:
                dT_dy = abs(bl.T_wall - T_new[ci]) / max(y, 1e-10)
                q = bl.thermal_conductivity * dT_dy
                wall_heat_flux_sum += q
                n_nusselt += 1

                # Thermal thickness: where T is within 1% of T_inf
                T_range = abs(bl.T_wall - bl.T_inf)
                if T_range > 1e-10 and abs(T_new[ci] - bl.T_inf) > 0.01 * T_range:
                    thermal_thickness_max = max(thermal_thickness_max, y)

    disp_thick = delta_star / max(n_integral, 1)
    mom_thick = theta_sum / max(n_integral, 1)
    shape_H = disp_thick / max(mom_thick, 1e-30)

    wall_heat_flux = wall_heat_flux_sum / max(n_nusselt, 1)
    avg_dT = abs(bl.T_wall - bl.T_inf)
    delta_char = disp_thick if disp_thick > 0 else delta
    Nu = wall_heat_flux * delta_char / (bl.thermal_conductivity * max(avg_dT, 1e-10))

    return EnhancedBL4Result(
        velocity=result_v,
        k=k_new,
        epsilon=eps_new,
        omega=omg_new,
        temperature=T_new,
        u_star_used=u_star,
        max_y_plus=max_yp,
        n_cells_modified=n_mod,
        displacement_thickness=disp_thick,
        momentum_thickness=mom_thick,
        shape_factor=shape_H,
        wall_heat_flux=wall_heat_flux,
        thermal_thickness=thermal_thickness_max,
        nusselt_number=Nu,
    )


# ---------------------------------------------------------------------------
# Thermal BL
# ---------------------------------------------------------------------------


def _thermal_bl_profile(y, delta, T_wall, T_inf, Pr, kappa, y_plus):
    """Temperature profile using Reynolds analogy."""
    t_plus = Pr * y_plus  # Thermal sublayer scaling
    if y_plus < 5.0:
        # Conductive sublayer
        T = T_wall - (T_wall - T_inf) * t_plus / max(Pr * 5.0, 1e-10)
    elif y < delta:
        # Log-law region with Reynolds analogy
        T = T_wall - (T_wall - T_inf) * (1.0 / Pr * math.log(max(y_plus, 1.0)) / max(math.log(delta * 0.41 / 1e-5), 1.0))
    else:
        T = T_inf
    return T


# ---------------------------------------------------------------------------
# Wall functions
# ---------------------------------------------------------------------------


def _spalding_wall_function(y_plus, u_star, kappa, E_val):
    if y_plus < 11.0:
        return u_star * y_plus
    return (u_star / kappa) * math.log(E_val * y_plus)


def _rough_wall_function(y, z0, u_star, kappa, nu):
    return (u_star / kappa) * math.log(max(y / z0, 1.0))


def _multi_layer_function(y_plus, u_star, kappa, E_val):
    if y_plus < 5.0:
        return u_star * y_plus
    elif y_plus < 30.0:
        u_visc = u_star * y_plus
        u_log = (u_star / kappa) * math.log(E_val * y_plus)
        t = (y_plus - 5.0) / 25.0
        blend = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))
        return blend * u_log + (1.0 - blend) * u_visc
    else:
        return (u_star / kappa) * math.log(E_val * y_plus)


# ---------------------------------------------------------------------------
# Profile fitting
# ---------------------------------------------------------------------------


def _fit_u_star(U_ref, y_ref, bl):
    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu
    u_star = 0.1
    for _ in range(50):
        y_plus = max(y_ref * u_star / nu, 1e-10)
        if y_plus < 11.0:
            U_calc = u_star * y_plus
            dU_du = 2.0 * y_ref * u_star / nu
        else:
            U_calc = (u_star / kappa) * math.log(E_val * y_plus)
            dU_du = (1.0 / kappa) * (math.log(E_val * y_plus) + 1.0)
        residual = U_calc - U_ref
        if abs(residual) < 1e-10:
            break
        if abs(dU_du) < 1e-30:
            break
        u_star -= residual / dU_du
        if u_star < 1e-10:
            u_star = 1e-10
    return u_star


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blending_function(y, delta, blend_width):
    blend_start = delta * (1.0 - blend_width)
    if y <= blend_start:
        return 1.0
    elif y >= delta:
        return 0.0
    else:
        t = (y - blend_start) / (delta - blend_start)
        return 0.5 * (1.0 + math.cos(math.pi * t))


def _estimate_u_star(velocity, cell_centres, wall_fc, bl):
    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu
    best_y = float("inf")
    best_U = 0.0
    for ci in range(cell_centres.shape[0]):
        cc = cell_centres[ci]
        y = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1).min()
        if y < 1e-30 or y > bl.delta:
            continue
        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue
        if y < best_y:
            best_y = y
            best_U = U_mag
    if best_y == float("inf") or best_U < 1e-30:
        return 0.01
    u_star = 0.1
    for _ in range(50):
        y_plus = best_y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10
        if y_plus < 11.0:
            U_calc = u_star * y_plus
            dU_du = 2.0 * best_y * u_star / nu
        else:
            U_calc = (u_star / kappa) * math.log(E_val * y_plus)
            dU_du = (1.0 / kappa) * (math.log(E_val * y_plus) + 1.0)
        residual = U_calc - best_U
        if abs(residual) < 1e-10:
            break
        if abs(dU_du) < 1e-30:
            break
        u_star -= residual / dU_du
        if u_star < 1e-10:
            u_star = 1e-10
    return u_star
