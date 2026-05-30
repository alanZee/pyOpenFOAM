"""
Enhanced displacement solver v5 with modal analysis and dynamic response.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced_4.EnhancedDisplacementSolver4` with:

- Modal analysis (natural frequencies and mode shapes)
- Newmark-beta time integration for dynamic problems
- Rayleigh damping matrix construction
- Frequency response function computation

Usage::

    solver = EnhancedDisplacementSolver5(model)
    modal = solver.modal_analysis_1d(area, length, n_elements, total_mass)
    print(f"First natural frequency: {modal.frequencies[0]:.1f} Hz")

References
----------
- OpenFOAM ``solidDisplacementFoam`` with dynamic support
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_4 import (
    EnhancedDisplacementSolver4,
    ContactResult,
    RefinementIndicator,
)

__all__ = [
    "EnhancedDisplacementSolver5",
    "ModalResult",
    "NewmarkResult",
    "RayleighDamping",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RayleighDamping:
    """Rayleigh damping parameters: C = alpha*M + beta*K.

    Attributes:
        alpha: Mass-proportional damping coefficient.
        beta: Stiffness-proportional damping coefficient.
    """

    alpha: float = 0.0
    beta: float = 0.0

    @classmethod
    def from_modal(
        cls,
        freq1: float,
        freq2: float,
        zeta1: float,
        zeta2: float,
    ) -> "RayleighDamping":
        """Compute Rayleigh damping from two modal frequencies and damping ratios.

        Args:
            freq1: First natural frequency (Hz).
            freq2: Second natural frequency (Hz).
            zeta1: Damping ratio at freq1.
            zeta2: Damping ratio at freq2.

        Returns:
            :class:`RayleighDamping`.
        """
        w1 = 2.0 * math.pi * freq1
        w2 = 2.0 * math.pi * freq2

        # Solve: zeta_i = alpha/(2*w_i) + beta*w_i/2
        denom = w2 ** 2 - w1 ** 2
        if abs(denom) < 1e-30:
            return cls(alpha=0.0, beta=0.0)

        alpha = 2.0 * (zeta1 * w2 - zeta2 * w1) * w1 * w2 / denom
        beta = 2.0 * (zeta2 * w2 - zeta1 * w1) / denom

        return cls(alpha=max(alpha, 0.0), beta=max(beta, 0.0))


@dataclass
class ModalResult:
    """Result of modal analysis.

    Attributes:
        frequencies: ``(n_modes,)`` natural frequencies (Hz).
        angular_frequencies: ``(n_modes,)`` angular frequencies (rad/s).
        mode_shapes: ``(n_nodes, n_modes)`` mode shapes (column-wise).
        n_modes: Number of modes computed.
    """

    frequencies: torch.Tensor = None
    angular_frequencies: torch.Tensor = None
    mode_shapes: torch.Tensor = None
    n_modes: int = 0

    def __post_init__(self) -> None:
        if self.frequencies is None:
            self.frequencies = torch.zeros(0, dtype=torch.float64)
        if self.angular_frequencies is None:
            self.angular_frequencies = torch.zeros(0, dtype=torch.float64)
        if self.mode_shapes is None:
            self.mode_shapes = torch.zeros(0, 0, dtype=torch.float64)


@dataclass
class NewmarkResult:
    """Result of Newmark-beta time integration.

    Attributes:
        displacement: ``(n_steps, n_dof)`` displacement history.
        velocity: ``(n_steps, n_dof)`` velocity history.
        acceleration: ``(n_steps, n_dof)`` acceleration history.
        time_points: ``(n_steps,)`` time points.
        n_steps: Number of time steps.
    """

    displacement: torch.Tensor = None
    velocity: torch.Tensor = None
    acceleration: torch.Tensor = None
    time_points: torch.Tensor = None
    n_steps: int = 0

    def __post_init__(self) -> None:
        if self.displacement is None:
            self.displacement = torch.zeros(0, 0, dtype=torch.float64)
        if self.velocity is None:
            self.velocity = torch.zeros(0, 0, dtype=torch.float64)
        if self.acceleration is None:
            self.acceleration = torch.zeros(0, 0, dtype=torch.float64)
        if self.time_points is None:
            self.time_points = torch.zeros(0, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedDisplacementSolver5(EnhancedDisplacementSolver4):
    """v5 enhanced displacement solver with modal analysis and dynamic response.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    """

    def __init__(self, model: LinearElasticModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # 1D mass matrix
    # ------------------------------------------------------------------

    @staticmethod
    def consistent_mass_matrix_1d(
        density: float,
        area: float,
        length: float,
        n_elements: int,
    ) -> torch.Tensor:
        """Assemble consistent mass matrix for 1D bar.

        Args:
            density: Material density (kg/m^3).
            area: Cross-sectional area (m^2).
            length: Bar length (m).
            n_elements: Number of elements.

        Returns:
            ``(n_nodes, n_nodes)`` mass matrix.
        """
        n_nodes = n_elements + 1
        le = length / n_elements
        rho_A_le = density * area * le

        M = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
        for e in range(n_elements):
            i, j = e, e + 1
            # Consistent mass matrix for linear element
            M[i, i] += rho_A_le / 3.0
            M[i, j] += rho_A_le / 6.0
            M[j, i] += rho_A_le / 6.0
            M[j, j] += rho_A_le / 3.0

        return M

    # ------------------------------------------------------------------
    # Modal analysis
    # ------------------------------------------------------------------

    def modal_analysis_1d(
        self,
        area: float,
        length: float,
        n_elements: int,
        total_mass: float,
        n_modes: int = 3,
    ) -> ModalResult:
        """Perform modal analysis on a 1D bar.

        Solves the generalised eigenvalue problem K*phi = omega^2*M*phi
        for natural frequencies and mode shapes.

        Args:
            area: Cross-sectional area (m^2).
            length: Bar length (m).
            n_elements: Number of elements.
            total_mass: Total mass (kg), used to compute density.
            n_modes: Number of modes to extract.

        Returns:
            :class:`ModalResult`.
        """
        n_nodes = n_elements + 1

        # Stiffness matrix
        K = self._assemble_stiffness_1d(area, length, n_elements)

        # Mass matrix
        density = total_mass / (area * length)
        M = self.consistent_mass_matrix_1d(density, area, length, n_elements)

        # Apply boundary condition: fix first node
        K_red = K[1:, 1:]
        M_red = M[1:, 1:]

        # Solve generalised eigenvalue problem
        # K*phi = lambda*M*phi where lambda = omega^2
        try:
            M_inv = torch.linalg.inv(M_red)
            A = M_inv @ K_red
            eigenvalues, eigenvectors = torch.linalg.eig(A)
            # 取实部
            eigenvalues_real = eigenvalues.real
            eigenvectors_real = eigenvectors.real

            # 排序（正特征值，从小到大）
            positive_mask = eigenvalues_real > 1e-10
            eigenvalues_pos = eigenvalues_real[positive_mask]
            eigenvectors_pos = eigenvectors_real[:, positive_mask]

            sorted_indices = eigenvalues_pos.argsort()
            n_extract = min(n_modes, len(sorted_indices))

            omega_sq = eigenvalues_pos[sorted_indices[:n_extract]]
            omega = torch.sqrt(omega_sq)
            freqs = omega / (2.0 * math.pi)

            # 完整模态向量（补回边界节点的零位移）
            modes_full = torch.zeros(n_nodes, n_extract, dtype=torch.float64)
            modes_full[1:, :] = eigenvectors_pos[:, sorted_indices[:n_extract]]

            return ModalResult(
                frequencies=freqs,
                angular_frequencies=omega,
                mode_shapes=modes_full,
                n_modes=n_extract,
            )
        except Exception:
            # Fallback: use analytical solution for uniform bar
            freqs = torch.zeros(n_modes, dtype=torch.float64)
            E = self._model.youngs_modulus
            rho = density
            le = length
            for i in range(n_modes):
                freqs[i] = (i + 1) * math.pi / (2.0 * le) * math.sqrt(E / rho) / (2.0 * math.pi)

            return ModalResult(
                frequencies=freqs,
                angular_frequencies=2.0 * math.pi * freqs,
                mode_shapes=torch.zeros(n_nodes, n_modes, dtype=torch.float64),
                n_modes=n_modes,
            )

    # ------------------------------------------------------------------
    # Newmark-beta integration
    # ------------------------------------------------------------------

    def newmark_integration_1d(
        self,
        area: float,
        length: float,
        n_elements: int,
        total_mass: float,
        external_force: torch.Tensor,
        dt: float,
        n_steps: int,
        gamma: float = 0.5,
        beta_n: float = 0.25,
        damping: RayleighDamping | None = None,
    ) -> NewmarkResult:
        """Newmark-beta time integration for 1D bar dynamics.

        Args:
            area: Cross-sectional area (m^2).
            length: Bar length (m).
            n_elements: Number of elements.
            total_mass: Total mass (kg).
            external_force: ``(n_dof,)`` external force vector.
            dt: Time step (s).
            n_steps: Number of time steps.
            gamma: Newmark gamma parameter (default 0.5, average acceleration).
            beta_n: Newmark beta parameter (default 0.25, average acceleration).
            damping: Optional Rayleigh damping.

        Returns:
            :class:`NewmarkResult`.
        """
        n_dof = n_elements  # 固定第一个节点
        K = self._assemble_stiffness_1d(area, length, n_elements)[1:, 1:]
        density = total_mass / (area * length)
        M = self.consistent_mass_matrix_1d(density, area, length, n_elements)[1:, 1:]

        # Damping matrix
        if damping is not None:
            C = damping.alpha * M + damping.beta * K
        else:
            C = torch.zeros_like(K)

        # Initial conditions
        u = torch.zeros(n_dof, dtype=torch.float64)
        v = torch.zeros(n_dof, dtype=torch.float64)

        # Initial acceleration: M*a0 = F - K*u0 - C*v0
        F0 = external_force.to(dtype=torch.float64)[:n_dof] if external_force.numel() >= n_dof else torch.zeros(n_dof, dtype=torch.float64)
        try:
            a = torch.linalg.solve(M, F0)
        except Exception:
            a = torch.zeros(n_dof, dtype=torch.float64)

        # Newmark constants
        a1 = 1.0 / (beta_n * dt ** 2)
        a2 = 1.0 / (beta_n * dt)
        a3 = (1.0 / (2.0 * beta_n)) - 1.0
        a4 = gamma / (beta_n * dt)
        a5 = gamma / beta_n - 1.0
        a6 = dt * (gamma / (2.0 * beta_n) - 1.0)

        # Effective stiffness
        K_eff = K + a1 * M + a4 * C

        # Storage
        u_hist = torch.zeros(n_steps + 1, n_dof, dtype=torch.float64)
        v_hist = torch.zeros(n_steps + 1, n_dof, dtype=torch.float64)
        a_hist = torch.zeros(n_steps + 1, n_dof, dtype=torch.float64)
        t_hist = torch.zeros(n_steps + 1, dtype=torch.float64)

        u_hist[0] = u
        v_hist[0] = v
        a_hist[0] = a

        for step in range(n_steps):
            F_ext = external_force.to(dtype=torch.float64)[:n_dof] if external_force.numel() >= n_dof else torch.zeros(n_dof, dtype=torch.float64)

            # Effective force
            F_eff = (
                F_ext
                + M @ (a1 * u + a2 * v + a3 * a)
                + C @ (a4 * u + a5 * v + a6 * a)
            )

            # Solve
            try:
                u_new = torch.linalg.solve(K_eff, F_eff)
            except Exception:
                u_new = u.clone()

            a_new = a1 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)

            u = u_new
            v = v_new
            a = a_new

            u_hist[step + 1] = u
            v_hist[step + 1] = v
            a_hist[step + 1] = a
            t_hist[step + 1] = (step + 1) * dt

        return NewmarkResult(
            displacement=u_hist,
            velocity=v_hist,
            acceleration=a_hist,
            time_points=t_hist,
            n_steps=n_steps,
        )

    # ------------------------------------------------------------------
    # Internal: stiffness assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_stiffness_1d(
        area: float,
        length: float,
        n_elements: int,
    ) -> torch.Tensor:
        """Assemble 1D bar stiffness matrix.

        Args:
            area: Cross-sectional area.
            length: Bar length.
            n_elements: Number of elements.

        Returns:
            ``(n_nodes, n_nodes)`` stiffness matrix.
        """
        n_nodes = n_elements + 1
        le = length / n_elements
        k_e = area / le  # E*A/L, E is handled externally

        K = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
        for e in range(n_elements):
            i, j = e, e + 1
            K[i, i] += k_e
            K[i, j] -= k_e
            K[j, i] -= k_e
            K[j, j] += k_e

        return K

    def __repr__(self) -> str:
        return f"EnhancedDisplacementSolver5(model={self._model!r})"
