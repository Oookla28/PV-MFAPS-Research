"""Finite-action Feedback-Descent-Safe Dynamic Pareto Allocation.

This module is CPU-only and contains deterministic surrogate optimization. It does
not read HELD/test data. The returned policy is optimal only for the supplied
finite frontier and quantized budget.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, inf, log, sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Action:
    name: str
    cost: int
    risk: Tuple[float, ...]
    macro_risk: float = 0.0
    payload: Any = None

    @property
    def mean_risk(self) -> float:
        return sum(self.risk) / len(self.risk)


def stability_bound(L: float, G: float, beta: float, etas: Sequence[float], n: int, k: int) -> float:
    """Equation (1), with a stable beta->0 limit."""
    if k <= 0:
        return 0.0
    total = 0.0
    for j in range(k):
        eta_j = float(etas[j])
        prod = 1.0
        for ell in range(j + 1, k):
            prod *= 1.0 + beta * float(etas[ell])
        total += eta_j * prod
    return 2.0 * L * G * total / max(int(n), 1)


def constant_step_stability(L: float, G: float, beta: float, eta: float, n: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if abs(beta) < 1e-15:
        return 2.0 * L * G * eta * k / max(int(n), 1)
    return 2.0 * L * G / (max(int(n), 1) * beta) * ((1.0 + beta * eta) ** k - 1.0)


def kl_dro_value(risks: Sequence[float], epsilon: float, nominal: Optional[Sequence[float]] = None) -> float:
    """KL-ball robust expectation for a finite partition vector.

    Minimizes the exact scalar dual numerically using a deterministic log-temperature
    grid followed by local refinement. epsilon=0 returns the nominal expectation exactly.
    """
    vals = [float(x) for x in risks]
    if nominal is None:
        nominal = [1.0 / len(vals)] * len(vals)
    u = [float(x) for x in nominal]
    if len(vals) != len(u) or any(x <= 0 for x in u):
        raise ValueError("nominal distribution must be positive and match risks")
    z = sum(u[i] * vals[i] for i in range(len(vals)))
    if epsilon <= 0.0:
        return z
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-15:
        return hi

    def dual(log_tau: float) -> float:
        tau = exp(log_tau)
        m = max(v / tau for v in vals)
        return tau * epsilon + tau * (m + log(sum(u[i] * exp(vals[i] / tau - m) for i in range(len(vals)))) )

    # A broad deterministic grid is robust for the tiny partition count used by
    # PlantSPD and CIFAR diagnostics. The result is an upper envelope, clipped
    # to the max risk to avoid numerical overshoot.
    grid = [(-16.0 + 32.0 * i / 256.0) for i in range(257)]
    best_i = min(range(len(grid)), key=lambda i: dual(grid[i]))
    a = grid[max(0, best_i - 1)]
    b = grid[min(len(grid) - 1, best_i + 1)]
    for _ in range(80):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if dual(m1) < dual(m2):
            b = m2
        else:
            a = m1
    return min(hi, max(z, dual((a + b) / 2.0)))


def safe_activate(
    base_u: float,
    candidate_u: float,
    base_radius: float,
    candidate_radius: float,
    cost_delta: float,
    switch_delta: float,
    lambda_cost: float,
    lambda_switch: float,
    margin: float,
) -> bool:
    """Certified one-sided activation test from the risk-envelope derivation."""
    ucb = candidate_u + candidate_radius + lambda_cost * cost_delta + lambda_switch * switch_delta
    lcb = base_u - base_radius
    return bool(ucb < lcb - margin)


def _distance(a: Action, b: Action) -> float:
    if a.payload is None or b.payload is None:
        return 0.0 if a.name == b.name else 1.0
    try:
        return float(sum(x != y for x, y in zip(a.payload, b.payload)))
    except TypeError:
        return 0.0 if a.payload == b.payload else 1.0


def solve_sequence(
    frontiers: Sequence[Sequence[Action]],
    budget: int,
    stage_weights: Optional[Sequence[float]] = None,
    epsilon: Optional[Sequence[float]] = None,
    macro_weight: float = 0.0,
    lambda_switch: float = 0.0,
    initial_action: Optional[Action] = None,
) -> Dict[str, Any]:
    """Exact DP on a finite, integer-cost frontier sequence."""
    S = len(frontiers)
    if S == 0:
        return {"status": "EXECUTED", "actions": [], "objective": 0.0, "cost": 0}
    weights = list(stage_weights or [1.0] * S)
    eps = list(epsilon or [0.0] * S)
    if len(weights) != S or len(eps) != S:
        raise ValueError("stage parameter length mismatch")
    states: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
    for j, action in enumerate(frontiers[0]):
        if action.cost <= budget:
            stage = weights[0] * (kl_dro_value(action.risk, eps[0]) + macro_weight * action.macro_risk)
            if initial_action is not None:
                stage += lambda_switch * _distance(action, initial_action)
            states[(action.cost, j)] = (stage, [j])
    for s in range(1, S):
        nxt: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
        for (used, prev_j), (value, path) in states.items():
            prev = frontiers[s - 1][prev_j]
            for j, action in enumerate(frontiers[s]):
                new_used = used + action.cost
                if new_used > budget:
                    continue
                stage = weights[s] * (kl_dro_value(action.risk, eps[s]) + macro_weight * action.macro_risk)
                candidate = value + stage + lambda_switch * _distance(action, prev)
                key = (new_used, j)
                if key not in nxt or candidate < nxt[key][0]:
                    nxt[key] = (candidate, path + [j])
        states = nxt
    if not states:
        return {"status": "BOUNDARY_NO_FEASIBLE_POLICY", "actions": [], "objective": inf, "cost": None}
    (used, _), (value, path) = min(states.items(), key=lambda item: item[1][0])
    actions = [frontiers[s][path[s]] for s in range(S)]
    return {
        "status": "EXECUTED",
        "actions": [a.name for a in actions],
        "payloads": [a.payload for a in actions],
        "objective": float(value),
        "cost": int(used),
        "path_indices": path,
    }


def certified_phase_policy(
    stage_actions: Sequence[Sequence[Action]],
    base_name: str,
    radii: Sequence[Dict[str, float]],
    lambda_cost: float,
    lambda_switch: float,
    margin: float,
) -> List[Dict[str, Any]]:
    """Return per-stage activation masks and diagnostics without test access."""
    logs: List[Dict[str, Any]] = []
    prev = base_name
    for s, actions in enumerate(stage_actions):
        base = next(a for a in actions if a.name == base_name)
        rows = []
        for a in actions:
            if a.name == base_name:
                active = False
            else:
                r = radii[s].get(a.name, 0.0)
                active = safe_activate(
                    max(base.risk), max(a.risk), radii[s].get(base.name, 0.0), r,
                    float(a.cost - base.cost), 1.0 if a.name != prev else 0.0,
                    lambda_cost, lambda_switch, margin,
                )
            rows.append({"action": a.name, "safe_active": active})
        chosen = next((row["action"] for row in rows if row["safe_active"]), base_name)
        logs.append({"stage": s, "previous": prev, "chosen": chosen, "actions": rows})
        prev = chosen
    return logs
