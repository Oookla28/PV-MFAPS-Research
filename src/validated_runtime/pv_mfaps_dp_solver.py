#!/usr/bin/env python3
"""PV-MFAPS finite-action risk-envelope and exact budget DP.

This module is deliberately independent from the running A100 pilot.  It does
not launch training and it never reads HELD.  The DP solves a *finite,
discretized* PV-MFAPS surrogate exactly after the risk envelope and costs have
been supplied.  It does not claim that proxy A/sigma values are a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, log, sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    k: int
    alpha: float = 0.0


@dataclass
class DPSolution:
    status: str
    actions: List[Action]
    cost: float
    sum_risk: float
    tail_risk: float
    objective: float
    candidate_r: float
    diagnostics: Dict[str, object]


def stability_bound(
    n: Sequence[float],
    *,
    lipschitz: float,
    grad_bound: float,
    smoothness: float,
    step_size: float | Sequence[float],
    k_grid: Sequence[int],
) -> np.ndarray:
    """Uniform-stability envelope for a k-step gradient inner solver.

    Assumptions are the standard bounded-gradient, Lipschitz-loss and smooth
    update assumptions.  Constants are intentionally explicit so callers can
    record them in a manifest.  This is a bound for the specified solver, not a
    claim about arbitrary Adam/AdamW trajectories.
    """
    n_arr = np.maximum(np.asarray(n, dtype=float), 1.0)
    steps = np.atleast_1d(np.asarray(step_size, dtype=float))
    out = np.zeros((n_arr.size, len(k_grid)), dtype=float)
    for j, k in enumerate(k_grid):
        if k <= 0:
            continue
        eta = np.resize(steps, k)
        s = 0.0
        for t in range(k):
            prod_after = float(np.prod(1.0 + smoothness * eta[t + 1 :]))
            s += float(eta[t]) * prod_after
        out[:, j] = (2.0 * lipschitz * grad_bound / n_arr) * s
    return out


def empirical_bernstein_bound(
    sigma: Sequence[float],
    n: Sequence[float],
    *,
    delta: float,
    n_classes: int,
    n_actions: int,
    loss_bound: float,
) -> np.ndarray:
    """Simultaneous empirical-Bernstein term for a finite class/action set."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 0.0)
    n_arr = np.maximum(np.asarray(n, dtype=float), 1.0)
    log_term = log(2.0 * n_classes * n_actions / delta)
    denom = np.maximum(n_arr - 1.0, 1.0)
    return np.sqrt(2.0 * (sigma_arr[:, None] ** 2) * log_term / n_arr[:, None]) + (
        7.0 * loss_bound * log_term / (3.0 * denom[:, None])
    )


def build_risk_envelope(
    rho: Sequence[float],
    A: Sequence[float],
    sigma: Sequence[float],
    n: Sequence[float],
    *,
    k_grid: Sequence[int] = (0, 1, 2, 4),
    xi: float = 1.0,
    delta: float = 0.05,
    loss_bound: float = 1.0,
    lipschitz: float = 0.0,
    grad_bound: float = 0.0,
    smoothness: float = 0.0,
    step_size: float | Sequence[float] = 0.0,
) -> np.ndarray:
    """Build U[c,k] = bias + stability + empirical-Bernstein envelope.

    ``A`` is interpreted as the nonnegative bias scale b_c.  If a caller uses
    A as an amplitude whose square is required, it must square it before this
    function and record that choice.  ``xi`` is a sensitivity parameter; only
    xi=1 backed by the explicit stability term is theorem-grounded here.
    """
    rho_arr = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0)
    A_arr = np.maximum(np.asarray(A, dtype=float), 0.0)
    n_arr = np.maximum(np.asarray(n, dtype=float), 1.0)
    k_arr = np.asarray(k_grid, dtype=int)
    if xi < 1.0:
        raise ValueError("xi must be >= 1")
    bias = A_arr[:, None] * (rho_arr[:, None] ** (2 * k_arr[None, :]))
    stab = stability_bound(
        n_arr,
        lipschitz=lipschitz,
        grad_bound=grad_bound,
        smoothness=smoothness,
        step_size=step_size,
        k_grid=k_arr,
    )
    eb = empirical_bernstein_bound(
        sigma,
        n_arr,
        delta=delta,
        n_classes=A_arr.size,
        n_actions=len(k_arr),
        loss_bound=loss_bound,
    )
    # xi>1 is retained only as an explicitly marked sensitivity envelope.
    complexity = (np.asarray(sigma, dtype=float)[:, None] ** 2 / n_arr[:, None]) * (
        xi ** k_arr[None, :]
    )
    return bias + stab + eb + (complexity if xi > 1.0 else 0.0)


def _integer_cost(cost: float, scale: int) -> int:
    if cost < 0:
        raise ValueError("cost must be nonnegative")
    # Ceil is conservative: a returned solution cannot exceed the real budget
    # because quantized cost is rounded upward.
    return int(ceil(float(cost) * scale - 1e-12))


def solve_pv_mfaps_dp(
    risk: np.ndarray,
    q: Sequence[float],
    actions: Sequence[Action],
    *,
    budget: float,
    tail_mask: Sequence[bool],
    macro_cap: Optional[float] = None,
    mu: float = 1.0,
    nu: Optional[Sequence[float]] = None,
    cost: Optional[np.ndarray] = None,
    cost_scale: int = 10000,
) -> DPSolution:
    """Exact finite-action DP for the discretized PV-MFAPS problem.

    The solver enumerates candidate tail risk levels r and minimizes total
    envelope risk under the cost budget and optional macro-risk cap.  ``nu``
    is accepted as a dual tie-break weight; primal minimax feasibility itself
    is represented by the candidate-r enumeration.  A returned solution is
    exact for the supplied quantized costs and risk table, not for the
    continuous relaxation.
    """
    U = np.asarray(risk, dtype=float)
    if U.ndim != 2 or U.shape[0] == 0:
        raise ValueError("risk must have shape [n_classes, n_actions]")
    C, A = U.shape
    if A != len(actions):
        raise ValueError("risk/action dimension mismatch")
    q_arr = np.asarray(q, dtype=float)
    tail = np.asarray(tail_mask, dtype=bool)
    if q_arr.shape != (C,) or tail.shape != (C,):
        raise ValueError("q and tail_mask must have one value per class")
    if np.any(q_arr < 0) or not np.isfinite(q_arr).all():
        raise ValueError("q must be finite and nonnegative")
    if abs(float(q_arr.sum()) - 1.0) > 1e-6:
        raise ValueError("q must sum to one")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if cost is None:
        # The default budget is the registered expected action budget
        # sum_c q_c k_c.  Callers supplying measured FLOPs/wall costs should
        # pass a full [C,A] cost matrix instead.
        cost_arr = q_arr[:, None] * np.tile(
            np.asarray([a.k for a in actions], dtype=float), (C, 1)
        )
    else:
        cost_arr = np.asarray(cost, dtype=float)
        if cost_arr.shape != U.shape:
            raise ValueError("cost must have the same shape as risk")
    if np.any(cost_arr < 0) or not np.isfinite(cost_arr).all():
        raise ValueError("cost must be finite and nonnegative")
    if nu is not None:
        nu_arr = np.asarray(nu, dtype=float)
        if nu_arr.shape != (C,) or np.any(nu_arr < 0):
            raise ValueError("nu must be a nonnegative vector of length C")
        if np.any(nu_arr[~tail] != 0) or (tail.any() and abs(float(nu_arr[tail].sum()) - 1.0) > 1e-5):
            raise ValueError("nu must be supported on tail and sum to one there")
    else:
        nu_arr = np.zeros(C, dtype=float)

    budget_i = int(np.floor(float(budget) * cost_scale + 1e-12))
    cost_i = np.vectorize(lambda z: _integer_cost(z, cost_scale))(cost_arr)
    tail_values = sorted(set(float(v) for v in U[tail].ravel())) if tail.any() else [0.0]
    macro_limit = None if macro_cap is None else float(macro_cap) * C
    best: Optional[DPSolution] = None

    for r in tail_values:
        # DP map: quantized cost -> (sum envelope risk, action indices, dual sum)
        states: Dict[int, Tuple[float, List[int], float]] = {0: (0.0, [], 0.0)}
        feasible = True
        for c in range(C):
            next_states: Dict[int, Tuple[float, List[int], float]] = {}
            for b_used, (sum_u, chosen, dual_sum) in states.items():
                for a_idx in range(A):
                    if tail[c] and U[c, a_idx] > r + 1e-12:
                        continue
                    b_new = b_used + int(cost_i[c, a_idx])
                    if b_new > budget_i:
                        continue
                    s_new = sum_u + float(U[c, a_idx])
                    if macro_limit is not None and s_new > macro_limit + 1e-12:
                        continue
                    d_new = dual_sum + float((mu / C + (nu_arr[c] if tail[c] else 0.0)) * U[c, a_idx])
                    prev = next_states.get(b_new)
                    if prev is None or d_new < prev[2] - 1e-12:
                        next_states[b_new] = (s_new, chosen + [a_idx], d_new)
            states = next_states
            if not states:
                feasible = False
                break
        if not feasible:
            continue
        for b_used, (sum_u, chosen, dual_sum) in states.items():
            real_cost = float(sum(cost_arr[c, a] for c, a in enumerate(chosen)))
            tail_risk = float(max(U[c, a] for c, a in enumerate(chosen) if tail[c])) if tail.any() else 0.0
            objective = float(r + mu * sum_u / C)
            candidate = DPSolution(
                status="EXECUTED",
                actions=[actions[a] for a in chosen],
                cost=real_cost,
                sum_risk=float(sum_u),
                tail_risk=tail_risk,
                objective=objective,
                candidate_r=float(r),
                diagnostics={
                    "cost_scale": cost_scale,
                    "quantized_budget": budget_i,
                    "quantized_cost": b_used,
                    "dual_weighted_sum": float(dual_sum),
                    "macro_cap": macro_cap,
                    "nu_tail_sum": float(nu_arr[tail].sum()),
                },
            )
            if best is None or candidate.objective < best.objective - 1e-12:
                best = candidate
    if best is None:
        return DPSolution(
            status="INFEASIBLE",
            actions=[],
            cost=float("inf"),
            sum_risk=float("inf"),
            tail_risk=float("inf"),
            objective=float("inf"),
            candidate_r=float("inf"),
            diagnostics={"cost_scale": cost_scale, "quantized_budget": budget_i},
        )
    return best


__all__ = [
    "Action",
    "DPSolution",
    "build_risk_envelope",
    "empirical_bernstein_bound",
    "solve_pv_mfaps_dp",
    "stability_bound",
]
