#!/usr/bin/env python3
"""PV-MFAPS finite-action constrained solver.

This module is intentionally independent of the running A100 pilot.  It solves
an integer action allocation over a finite action set using a Pareto-frontier
DP.  The solver is exact for the quantised-cost problem and returns an explicit
certificate; it does not claim that the proxy risk envelope is a generalisation
theorem.  Callers must provide a separately justified risk envelope.

The default envelope is
    U_c(k) = A_c * rho_c**(2*k) + (sigma_c**2 / n_c) * xi**k.
``xi`` is deliberately distinct from a CVaR tail level.  Alpha actions are
optional and are required to be zero when k=0.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor, isfinite
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    """One class-specific discrete action."""

    k: int
    alpha: float
    cost: float
    risk: float
    score: float


@dataclass(frozen=True)
class _State:
    cost_i: int
    risk_sum: float
    objective: float
    actions: Tuple[Tuple[int, float], ...]


def risk_envelope(
    A: Sequence[float],
    rho: Sequence[float],
    sigma: Sequence[float],
    n: Sequence[float],
    k: int,
    xi: float = 1.01,
) -> np.ndarray:
    """Compute the explicit proxy envelope used by the offline branch.

    This is a proxy formula, not a theorem by itself.  A caller that has a
    proven stability envelope should pass its resulting risk values directly
    to ``solve_pv_mfaps_dp`` rather than relying on this helper.
    """
    if xi <= 0:
        raise ValueError("xi must be positive")
    A = np.asarray(A, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = np.asarray(n, dtype=float)
    if not (A.shape == rho.shape == sigma.shape == n.shape):
        raise ValueError("A, rho, sigma, n must have identical shapes")
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if np.any(rho < 0) or np.any(rho > 1):
        raise ValueError("rho must lie in [0, 1]")
    return A * np.power(rho, 2 * int(k)) + (sigma * sigma / n) * (xi**int(k))


def _as_vector(value: Sequence[float] | float, C: int, name: str) -> np.ndarray:
    if np.isscalar(value):
        return np.full(C, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (C,):
        raise ValueError(f"{name} must have shape ({C},)")
    return arr


def _class_cost(
    cost: Optional[Sequence[float] | np.ndarray],
    q_c: float,
    c: int,
    k: int,
) -> float:
    """Resolve base cost then multiply by q_c.

    ``cost=None`` means expected action cost q_c*k.  A vector means a class
    base cost independent of k (the common measured fixed-k calibration).  A
    matrix may provide a measured cost for each class and k-index; callers
    should pass ``cost_k`` together with ``k_grid`` in that case.
    """
    if cost is None:
        return float(q_c * k)
    arr = np.asarray(cost, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] <= c:
            raise ValueError("cost vector has too few classes")
        return float(q_c * arr[c] * max(k, 1 if k > 0 else 0))
    raise ValueError("cost matrix requires cost_k in solve_pv_mfaps_dp")


def _make_actions(
    c: int,
    A: np.ndarray,
    rho: np.ndarray,
    sigma: np.ndarray,
    n: np.ndarray,
    q: np.ndarray,
    k_grid: Sequence[int],
    alpha_grid: Sequence[float],
    xi: float,
    cost: Optional[np.ndarray],
    cost_k: Optional[np.ndarray],
    weights: np.ndarray,
    tail_mask: np.ndarray,
    tail_cap: Optional[float],
) -> List[Action]:
    actions: List[Action] = []
    for k in sorted(set(int(v) for v in k_grid)):
        if k < 0:
            raise ValueError("k_grid must be nonnegative")
        alphas = [0.0] if k == 0 else sorted(set(float(v) for v in alpha_grid))
        for alpha in alphas:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must lie in [0, 1]")
            risk = float(risk_envelope(A, rho, sigma, n, k, xi)[c])
            if tail_cap is not None and tail_mask[c] and risk > tail_cap + 1e-12:
                continue
            if cost_k is not None:
                ck = np.asarray(cost_k, dtype=float)
                if ck.ndim != 2 or ck.shape[0] != len(A) or ck.shape[1] != len(k_grid):
                    raise ValueError("cost_k must have shape (C, len(k_grid))")
                # k_grid is sorted and de-duplicated above; use original lookup.
                kk = list(sorted(set(int(v) for v in k_grid))).index(k)
                action_cost = float(q[c] * ck[c, kk])
            else:
                action_cost = _class_cost(cost, float(q[c]), c, k)
            score = float(weights[c] * risk)
            actions.append(Action(k=k, alpha=alpha, cost=action_cost, risk=risk, score=score))
    if not actions:
        raise ValueError(f"no feasible action for class {c}; tail_cap too small")
    return actions


def _prune(states: Iterable[_State], tol: float = 1e-12) -> List[_State]:
    """Pareto-prune states with identical quantised cost.

    A state is dominated when another state has no larger risk sum and no larger
    weighted objective.  Keeping only the Pareto frontier is exact for the
    discretised-cost DP.
    """
    by_cost: Dict[int, List[_State]] = {}
    for s in states:
        by_cost.setdefault(s.cost_i, []).append(s)
    kept: List[_State] = []
    for cost_i in sorted(by_cost):
        arr = sorted(by_cost[cost_i], key=lambda s: (s.risk_sum, s.objective, s.actions))
        frontier: List[_State] = []
        best_obj = float("inf")
        for s in arr:
            if s.objective < best_obj - tol:
                frontier.append(s)
                best_obj = s.objective
            elif s.objective <= best_obj + tol:
                # Same objective but lower/equal risk is already represented by
                # the first state; retain deterministic tie-breaking only.
                continue
        kept.extend(frontier)
    return kept


def solve_pv_mfaps_dp(
    *,
    rho: Sequence[float],
    A: Sequence[float],
    sigma: Sequence[float],
    n: Sequence[float],
    mu: float,
    nu: Optional[Sequence[float]] = None,
    q: Optional[Sequence[float]] = None,
    budget: float,
    k_grid: Sequence[int] = (0, 1, 2, 4),
    alpha_grid: Sequence[float] = (0.25,),
    xi: float = 1.01,
    tail_mask: Optional[Sequence[bool]] = None,
    tail_cap: Optional[float] = None,
    macro_cap: Optional[float] = None,
    cost: Optional[Sequence[float] | np.ndarray] = None,
    cost_k: Optional[np.ndarray] = None,
    budget_scale: int = 1000,
    return_frontier_stats: bool = False,
) -> Dict[str, Any]:
    """Solve the finite-action PV-MFAPS problem with exact budget DP.

    The objective is ``sum_c w_c U_c(a_c)`` where
    ``w_c = mu/C + nu_c``.  ``nu`` is the tail max-constraint subgradient and
    should be nonnegative.  ``macro_cap`` is an average risk cap.  ``tail_cap``
    filters actions for tail classes.  Costs are conservatively quantised with
    ``ceil(cost * budget_scale)`` so any returned solution is feasible for the
    original floating-point budget.
    """
    A = np.asarray(A, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = np.asarray(n, dtype=float)
    C = len(A)
    if not (rho.shape == sigma.shape == n.shape == (C,)):
        raise ValueError("rho, sigma, n, A must have equal one-dimensional shape")
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    if budget < 0 or budget_scale <= 0:
        raise ValueError("budget and budget_scale must be nonnegative/positive")
    qv = np.full(C, 1.0 / C) if q is None else np.asarray(q, dtype=float)
    if qv.shape != (C,) or np.any(qv < 0) or qv.sum() <= 0:
        raise ValueError("q must be nonnegative with positive sum")
    qv = qv / qv.sum()
    nuv = np.zeros(C, dtype=float) if nu is None else np.asarray(nu, dtype=float)
    if nuv.shape != (C,) or np.any(nuv < 0):
        raise ValueError("nu must be a nonnegative vector with shape (C,)")
    tail = np.zeros(C, dtype=bool) if tail_mask is None else np.asarray(tail_mask, dtype=bool)
    if tail.shape != (C,):
        raise ValueError("tail_mask must have shape (C,)")
    weights = mu / C + nuv
    if not np.any(weights > 0):
        raise ValueError("all objective weights are zero")
    if macro_cap is not None and macro_cap < 0:
        raise ValueError("macro_cap must be nonnegative")
    actions_by_class: List[List[Action]] = []
    k_values = sorted(set(int(k) for k in k_grid))
    for c in range(C):
        actions_by_class.append(_make_actions(
            c, A, rho, sigma, n, qv, k_values, alpha_grid, xi,
            None if cost is None else np.asarray(cost, dtype=float),
            cost_k, weights, tail, tail_cap,
        ))
    budget_i = int(floor(float(budget) * budget_scale + 1e-12))
    macro_total_cap = None if macro_cap is None else float(macro_cap) * C
    states: List[_State] = [_State(0, 0.0, 0.0, tuple())]
    frontier_sizes: List[int] = [1]
    for c, actions in enumerate(actions_by_class):
        expanded: List[_State] = []
        for s in states:
            for a in actions:
                ci = s.cost_i + int(ceil(max(0.0, a.cost) * budget_scale - 1e-12))
                if ci > budget_i:
                    continue
                rs = s.risk_sum + a.risk
                if macro_total_cap is not None and rs > macro_total_cap + 1e-10:
                    continue
                expanded.append(_State(
                    ci,
                    rs,
                    s.objective + a.score,
                    s.actions + ((a.k, a.alpha),),
                ))
        states = _prune(expanded)
        frontier_sizes.append(len(states))
        if not states:
            return {
                "status": "INFEASIBLE",
                "reason": f"no state after class {c}",
                "n_classes": C,
                "budget": float(budget),
                "macro_cap": macro_cap,
            }
    best = min(states, key=lambda s: (s.objective, s.risk_sum, s.cost_i, s.actions))
    ks = [int(k) for k, _ in best.actions]
    alphas = [float(a) for _, a in best.actions]
    per_class = []
    for c, (k, alpha), actions in zip(range(C), best.actions, actions_by_class):
        chosen = next(a for a in actions if a.k == k and abs(a.alpha - alpha) <= 1e-12)
        per_class.append({
            "class": c, "k": int(k), "alpha": float(alpha),
            "cost": float(chosen.cost), "risk": float(chosen.risk),
            "weight": float(weights[c]), "weighted_risk": float(chosen.score),
            "is_tail": bool(tail[c]),
        })
    actual_cost = float(sum(x["cost"] for x in per_class))
    risk_sum = float(sum(x["risk"] for x in per_class))
    tail_max = float(max((x["risk"] for x in per_class if x["is_tail"]), default=0.0))
    result: Dict[str, Any] = {
        "status": "OPTIMAL_DISCRETE_QUANTIZED_COST",
        "certificate": {
            "budget_scale": int(budget_scale),
            "quantized_budget": int(budget_i),
            "quantized_cost": int(best.cost_i),
            "actual_cost": actual_cost,
            "budget_slack": float(budget - actual_cost),
            "macro_risk": float(risk_sum / C),
            "tail_max_risk": tail_max,
            "objective": float(best.objective),
            "macro_cap_satisfied": macro_cap is None or risk_sum / C <= macro_cap + 1e-10,
            "tail_cap_satisfied": tail_cap is None or tail_max <= tail_cap + 1e-10,
            "dp_frontier_final": len(states),
        },
        "n_classes": C,
        "budget": float(budget),
        "macro_cap": macro_cap,
        "tail_cap": tail_cap,
        "xi": float(xi),
        "mu": float(mu),
        "nu": nuv.tolist(),
        "q": qv.tolist(),
        "k_grid": k_values,
        "k": ks,
        "alpha": alphas,
        "per_class": per_class,
    }
    if return_frontier_stats:
        result["diagnostics"] = {"frontier_sizes": frontier_sizes}
    return result


def solve_minimax_dp(
    *,
    tail_classes: Sequence[int],
    tail_cap_candidates: Sequence[float],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Solve the discrete tail-minimax problem by enumerating risk caps.

    Candidate caps can be all distinct action risks for tail classes.  The first
    feasible cap is the exact minimum tail envelope for the quantised budget;
    the weighted objective breaks ties.  This is an integer enumeration over a
    finite action set, not a claim of continuous KKT equivalence.
    """
    C = len(kwargs["A"])
    mask = np.zeros(C, dtype=bool)
    mask[list(map(int, tail_classes))] = True
    last = None
    for cap in sorted(set(float(x) for x in tail_cap_candidates)):
        out = solve_pv_mfaps_dp(tail_mask=mask, tail_cap=cap, **kwargs)
        last = out
        if out.get("status") == "OPTIMAL_DISCRETE_QUANTIZED_COST":
            out["objective_type"] = "discrete_tail_minimax_with_weighted_tie_break"
            out["minimax_cap_candidates"] = sorted(set(float(x) for x in tail_cap_candidates))
            out["selected_tail_cap"] = cap
            return out
    if last is None:
        return {"status": "INFEASIBLE", "reason": "empty tail_cap_candidates"}
    return {"status": "INFEASIBLE", "reason": "no candidate tail cap feasible", "last_attempt": last}


if __name__ == "__main__":
    # Tiny deterministic smoke, useful for remote preflight without touching
    # the A100 runner.
    out = solve_pv_mfaps_dp(
        A=[2.0, 1.0, 3.0], rho=[0.9, 0.8, 0.95], sigma=[.1, .2, .3], n=[10, 5, 20],
        mu=1.0, nu=[.5, 0.0, .5], q=[.2, .3, .5], budget=.4,
        k_grid=(0, 1, 2), xi=1.01, tail_mask=[True, False, True],
        macro_cap=3.0, return_frontier_stats=True,
    )
    print(out["status"])
    print(out.get("k"), out.get("certificate"))
