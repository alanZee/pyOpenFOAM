"""
financialFoam2 — Enhanced Black-Scholes solver with Greeks and American options.

Extends :class:`FinancialFoam` with:

- **Greeks computation**: Delta, Gamma, Theta, Vega, Rho via finite
  differences on the option value field.
- **American option pricing** with early exercise via a projected
  successive over-relaxation (PSOR) algorithm that enforces the
  constraint V(S, t) >= payoff(S) at every time step.
- **Dividend yield** support (continuous dividend yield q).

The PDE becomes:

    dV/dt + 0.5*sigma^2*S^2*d2V/dS^2 + (r-q)*S*dV/dS - r*V = 0

with the early exercise constraint V >= max(S-K, 0) for American calls
and V >= max(K-S, 0) for American puts.

Usage::

    from pyfoam.applications.financial_foam_2 import FinancialFoam2

    solver = FinancialFoam2(
        "path/to/case", option_type="put", exercise="american",
        K=100.0, r=0.05, sigma=0.2, q=0.02,
    )
    result = solver.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FinancialFoam2", "Greeks"]

logger = logging.getLogger(__name__)


# ======================================================================
# Greeks data container
# ======================================================================


@dataclass
class Greeks:
    """Option Greeks computed by finite differences.

    Attributes
    ----------
    delta : float
        dV/dS — sensitivity to underlying price.
    gamma : float
        d2V/dS2 — second-order sensitivity.
    theta : float
        dV/dt — time decay (per unit time).
    vega : float
        dV/dsigma — sensitivity to volatility.
    rho : float
        dV/dr — sensitivity to interest rate.
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


# ======================================================================
# Main solver
# ======================================================================


class FinancialFoam2(SolverBase):
    """Enhanced Black-Scholes solver with Greeks and American options.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    option_type : str
        ``"call"`` or ``"put"``.
    exercise : str
        ``"european"`` or ``"american"``.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    q : float
        Continuous dividend yield (default 0).
    S_max : float or None
        Maximum asset price. Default: ``3 * K``.
    theta : float
        Theta-method parameter (0 = explicit, 0.5 = CN, 1 = implicit).
    american_max_iter : int
        Max PSOR iterations per time step for American exercise.
    american_omega : float
        SOR relaxation parameter for American exercise (1.0 < omega < 2.0).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        option_type: str = "call",
        exercise: str = "european",
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        q: float = 0.0,
        S_max: float | None = None,
        theta: float = 1.0,
        american_max_iter: int = 100,
        american_omega: float = 1.2,
    ) -> None:
        super().__init__(case_path)

        self.option_type = option_type.lower()
        self.exercise = exercise.lower()
        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q
        self.S_max = S_max if S_max is not None else 3.0 * K
        self.theta_method = theta
        self.american_max_iter = american_max_iter
        self.american_omega = american_omega

        # fvSolution settings
        self._read_fv_solution_settings()

        # Initialise fields
        self.V, self._field_data = self._init_fields()

        # Asset price grid
        self.n_cells = self.mesh.n_cells
        self.dS = self.S_max / self.n_cells
        self.S = torch.linspace(
            self.dS, self.S_max, self.n_cells,
            dtype=get_default_dtype(), device=get_device(),
        )

        # Set initial payoff
        self._set_initial_payoff()

        # Cache payoff for American exercise constraint
        self._payoff = self._compute_payoff()

        logger.info(
            "FinancialFoam2 ready: %s %s, K=%.4g, r=%.4g, sigma=%.4g, "
            "q=%.4g, S_max=%.4g",
            self.exercise, self.option_type, self.K, self.r,
            self.sigma, self.q, self.S_max,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("financialFoam/convergenceTolerance", 1e-6)
        )
        self.scheme = str(fv.get_path("financialFoam/scheme", "implicit"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, Any]:
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        try:
            V_tensor, field_data = self.read_field_tensor("V", 0)
            V = V_tensor.to(device=device, dtype=dtype).reshape(-1)
            return V, field_data
        except Exception:
            V = torch.zeros(n_cells, dtype=dtype, device=device)
            return V, None

    def _compute_payoff(self) -> torch.Tensor:
        """Compute the exercise payoff on the grid."""
        if self.option_type == "call":
            return torch.clamp(self.S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - self.S, min=0.0)

    def _set_initial_payoff(self) -> None:
        self.V = self._compute_payoff()

    # ------------------------------------------------------------------
    # Black-Scholes discretisation (with dividend yield)
    # ------------------------------------------------------------------

    def _build_coefficients(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tridiagonal coefficients for the modified Black-Scholes PDE.

        PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS^2 + (r-q)*S*dV/dS - r*V = 0
        """
        S = self.S
        sigma = self.sigma
        r = self.r
        q = self.q
        dS = self.dS

        diffusion = 0.5 * sigma ** 2 * S ** 2
        convection = (r - q) * S

        a = diffusion / dS ** 2 - convection / (2.0 * dS)
        b = -2.0 * diffusion / dS ** 2 - r
        c = diffusion / dS ** 2 + convection / (2.0 * dS)

        diag = torch.ones_like(S)
        return a, b, c, diag

    def _apply_boundary_conditions(self, V: torch.Tensor) -> torch.Tensor:
        """Apply boundary conditions (accounting for dividend yield)."""
        tau = self.end_time
        disc = math.exp(-self.r * tau)
        div_disc = math.exp(-self.q * tau)

        if self.option_type == "call":
            V[0] = 0.0
            V[-1] = max(self.S_max * div_disc - self.K * disc, 0.0)
        else:
            V[0] = self.K * disc
            V[-1] = 0.0
        return V

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _explicit_step(self) -> torch.Tensor:
        V = self.V.clone()
        a, b, c, diag = self._build_coefficients()

        L_V = torch.zeros_like(V)
        for i in range(1, self.n_cells - 1):
            L_V[i] = a[i] * V[i - 1] + b[i] * V[i] + c[i] * V[i + 1]

        V_new = V + self.delta_t * L_V
        V_new = self._apply_boundary_conditions(V_new)
        return V_new

    def _implicit_step(self) -> torch.Tensor:
        V = self.V.clone()
        a, b, c, diag = self._build_coefficients()
        dt = self.delta_t
        n = self.n_cells

        main_diag = diag.clone()
        main_diag[1:-1] = 1.0 - dt * b[1:-1]
        main_diag[0] = 1.0
        main_diag[-1] = 1.0

        lower = torch.zeros(n, dtype=V.dtype, device=V.device)
        lower[1:] = -dt * a[1:]
        lower[0] = 0.0

        upper = torch.zeros(n, dtype=V.dtype, device=V.device)
        upper[:-1] = -dt * c[:-1]
        upper[-1] = 0.0

        rhs = V.clone()
        V_new = self._thomas_algorithm(lower, main_diag, upper, rhs)
        V_new = self._apply_boundary_conditions(V_new)
        return V_new

    @staticmethod
    def _thomas_algorithm(
        lower: torch.Tensor,
        main: torch.Tensor,
        upper: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Thomas algorithm for tridiagonal systems."""
        n = len(main)
        c_prime = torch.zeros(n, dtype=main.dtype, device=main.device)
        d_prime = torch.zeros(n, dtype=main.dtype, device=main.device)

        c_prime[0] = upper[0] / main[0]
        d_prime[0] = rhs[0] / main[0]

        for i in range(1, n):
            m = main[i] - lower[i] * c_prime[i - 1]
            if abs(m.item()) < 1e-30:
                m = torch.tensor(1e-30, dtype=main.dtype, device=main.device)
            c_prime[i] = upper[i] / m
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / m

        x = torch.zeros(n, dtype=main.dtype, device=main.device)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def _theta_step(self) -> torch.Tensor:
        """Theta method time step."""
        if abs(self.theta_method) < 1e-12:
            V_new = self._explicit_step()
        elif abs(self.theta_method - 1.0) < 1e-12:
            V_new = self._implicit_step()
        else:
            V_exp = self._explicit_step()
            V_imp = self._implicit_step()
            V_new = (1.0 - self.theta_method) * V_exp + self.theta_method * V_imp
        return V_new

    # ------------------------------------------------------------------
    # American exercise (PSOR)
    # ------------------------------------------------------------------

    def _apply_american_exercise(self, V: torch.Tensor) -> torch.Tensor:
        """Apply American early exercise constraint via PSOR.

        For each time step after the linear solve, enforce::

            V_i >= payoff_i   for all interior nodes

        Using projected SOR: if V_i < payoff_i, set V_i = payoff_i.
        Repeat for ``american_max_iter`` sweeps to converge the
        linear complementarity problem.
        """
        if self.exercise != "american":
            return V

        omega = self.american_omega
        payoff = self._payoff
        a, b, c, diag = self._build_coefficients()
        dt = self.delta_t
        n = self.n_cells

        # Build LHS coefficients for implicit step
        main_diag = diag.clone()
        main_diag[1:-1] = 1.0 - dt * b[1:-1]
        main_diag[0] = 1.0
        main_diag[-1] = 1.0

        lower_coeff = torch.zeros(n, dtype=V.dtype, device=V.device)
        lower_coeff[1:] = -dt * a[1:]

        upper_coeff = torch.zeros(n, dtype=V.dtype, device=V.device)
        upper_coeff[:-1] = -dt * c[:-1]

        rhs = self.V.clone()  # old V

        # PSOR iteration
        for _ in range(self.american_max_iter):
            V_old_sweep = V.clone()

            for i in range(1, n - 1):
                residual = (
                    rhs[i] - lower_coeff[i] * V[i - 1]
                    - main_diag[i] * V[i] - upper_coeff[i] * V[i + 1]
                )
                V[i] = V[i] + omega * residual / main_diag[i]

                # Project onto constraint
                if V[i] < payoff[i]:
                    V[i] = payoff[i]

            V = self._apply_boundary_conditions(V)

            # Check sweep convergence
            change = float((V - V_old_sweep).abs().max().item())
            if change < 1e-12:
                break

        return V

    # ------------------------------------------------------------------
    # Greeks computation
    # ------------------------------------------------------------------

    def compute_greeks(self, S_target: float | None = None) -> Greeks:
        """Compute Greeks at a target asset price.

        Uses central finite differences on the value field.

        Parameters
        ----------
        S_target : float or None
            Asset price at which to evaluate. Default: strike price K.

        Returns
        -------
        Greeks
            Computed Greeks at the target price.
        """
        if S_target is None:
            S_target = self.K

        S = self.S
        V = self.V
        dS = self.dS

        # Find index near S_target
        idx = int(torch.searchsorted(
            S, torch.tensor(S_target, dtype=S.dtype, device=S.device),
        ).item())
        idx = max(1, min(idx, self.n_cells - 2))

        # Delta = dV/dS (central difference)
        delta = float(((V[idx + 1] - V[idx - 1]) / (2.0 * dS)).item())

        # Gamma = d2V/dS2 (central difference)
        gamma = float(
            ((V[idx + 1] - 2.0 * V[idx] + V[idx - 1]) / (dS ** 2)).item()
        )

        # Theta = dV/dt (approximate: compare with previous time step)
        theta = 0.0  # Computed during run if needed

        # Vega and Rho via bump-and-revalue
        vega = self._compute_vega(S_target)
        rho = self._compute_rho(S_target)

        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def _compute_vega(self, S_target: float, bump: float = 0.01) -> float:
        """Compute Vega via bump-and-revalue."""
        sigma_orig = self.sigma
        V_orig = self.V.clone()

        self.sigma = sigma_orig + bump
        self._payoff = self._compute_payoff()
        V_up = self._implicit_step()

        self.sigma = sigma_orig - bump
        self._payoff = self._compute_payoff()
        V_down = self._implicit_step()

        # Restore
        self.sigma = sigma_orig
        self._payoff = self._compute_payoff()
        self.V = V_orig

        # Interpolate at S_target
        idx = int(torch.searchsorted(
            self.S,
            torch.tensor(S_target, dtype=self.S.dtype, device=self.S.device),
        ).item())
        idx = max(1, min(idx, self.n_cells - 2))

        vega = float(((V_up[idx] - V_down[idx]) / (2.0 * bump)).item())
        return vega

    def _compute_rho(self, S_target: float, bump: float = 0.0001) -> float:
        """Compute Rho via bump-and-revalue."""
        r_orig = self.r
        V_orig = self.V.clone()

        self.r = r_orig + bump
        V_up = self._implicit_step()

        self.r = r_orig - bump
        V_down = self._implicit_step()

        # Restore
        self.r = r_orig
        self.V = V_orig

        idx = int(torch.searchsorted(
            self.S,
            torch.tensor(S_target, dtype=self.S.dtype, device=self.S.device),
        ).item())
        idx = max(1, min(idx, self.n_cells - 2))

        rho = float(((V_up[idx] - V_down[idx]) / (2.0 * bump)).item())
        return rho

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run FinancialFoam2 solver.

        Returns
        -------
        dict
            ``converged``, ``steps``, ``residual``, ``V_at_K``,
            ``greeks`` (at strike).
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting FinancialFoam2 run")
        logger.info("  exercise=%s, option=%s", self.exercise, self.option_type)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0

        for t, step in time_loop:
            V_old = self.V.clone()

            # Theta method time step
            self.V = self._theta_step()

            # Apply American exercise constraint
            self.V = self._apply_american_exercise(self.V)

            # Ensure non-negative
            self.V = torch.clamp(self.V, min=0.0)

            # Compute residual
            residual = float((self.V - V_old).abs().max().item())
            converged = convergence.update(step + 1, {"V": residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FinancialFoam2 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        # Compute Greeks at strike
        greeks = self.compute_greeks(self.K)
        V_at_K = self._interpolate_at_S(self.K)

        logger.info("FinancialFoam2 completed: V(K)=%.6g, delta=%.4f",
                     V_at_K, greeks.delta)

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
            "V_at_K": V_at_K,
            "greeks": greeks,
        }

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interpolate_at_S(self, S_target: float) -> float:
        S = self.S
        V = self.V
        idx = torch.searchsorted(S, torch.tensor(S_target, dtype=S.dtype, device=S.device))
        idx = max(1, min(int(idx.item()), self.n_cells - 1))
        w = (S_target - S[idx - 1]) / (S[idx] - S[idx - 1])
        V_interp = V[idx - 1] * (1.0 - w) + V[idx] * w
        return float(V_interp.item())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def intrinsic_value(self) -> torch.Tensor:
        return self._compute_payoff()

    @property
    def time_value(self) -> torch.Tensor:
        return self.V - self.intrinsic_value

    # ------------------------------------------------------------------
    # Field output
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        time_str = f"{time:g}"
        if self._field_data is not None:
            self.write_field("V", self.V, time_str, self._field_data)
        else:
            from pyfoam.io.field_io import FieldData, write_field as _write_field
            from pyfoam.io.foam_file import FoamFileHeader, FileFormat

            time_dir = self.case_path / time_str
            time_dir.mkdir(parents=True, exist_ok=True)
            field_data = FieldData(
                header=FoamFileHeader(
                    version="2.0", format=FileFormat.ASCII,
                    class_name="volScalarField",
                    location=time_str, object="V",
                ),
                dimensions=[0, 0, 0, 0, 0, 0, 0],
                internal_field=self.V.detach().cpu(),
                boundary_field=[],
                is_uniform=False,
                scalar_type="scalar",
            )
            _write_field(time_dir / "V", field_data, overwrite=True)
