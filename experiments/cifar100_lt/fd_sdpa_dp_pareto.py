#!/usr/bin/env python3
"""FD-SDPA Pareto DP over the finite action grid.

Kept in a separate module from ``fd_sdpa_dp.py`` (which contains the sequence /
certificate helpers written in another session) so that two concurrent writers
do not overwrite each other.

Two solvers:

* ``solve_stage_bruteforce`` enumerates the full finite action grid. It is the
  ground truth for the offline equivalence test and is only usable for small C.
* ``solve_stage_pareto`` is a Pareto-frontier DP over quantised cost. It is
  exact for the quantised problem while the frontier stays under the cap; when
  the cap is hit it coarsens the risk grid and reports ``approximate=True``.

Only the finite, quantised surrogate is optimised. No statement is made about
the continuous training problem.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fd_sdpa_policy import K_GRID, kl_dro


@dataclass
class StageProblem:
    """Risk and cost tables for one stage.

    U:    (P, C, K) risk envelope values, K aligned with ``k_grid``.
    cost: (C, K)    measured cost per class and action index.
    """
    U: np.ndarray
    cost: np.ndarray
    k_grid: Sequence[int] = K_GRID

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U, dtype=float)
        self.cost = np.asarray(self.cost, dtype=float)
        P, C, K = self.U.shape
        if self.cost.shape != (C, K):
            raise ValueError(f"cost must be {(C, K)}, got {self.cost.shape}")
        if np.any(~np.isfinite(self.U)) or np.any(~np.isfinite(self.cost)):
            raise ValueError("U and cost must be finite")
        if np.any(self.cost < 0):
            raise ValueError("cost must be nonnegative")

    @property
    def n_classes(self) -> int:
        return int(self.U.shape[1])

    @property
    def n_partitions(self) -> int:
        return int(self.U.shape[0])


@dataclass
class StageSolution:
    k: np.ndarray
    cost: float
    risk: np.ndarray
    objective: float
    status: str
    approximate: bool = False
    tolerance: float = 0.0
    detail: Optional[Dict[str, object]] = None


def stage_objective(*, risk: Sequence[float], ubar: Sequence[float], epsilon: float,
                    cost_total: float, budget: float, macro_weight: float,
                    lambda_cost: float, switch_penalty: float,
                    switch_distance: float, dro_grid: int = 160) -> Dict[str, float]:
    dro = kl_dro(risk, ubar, epsilon, grid=dro_grid)
    macro = float(np.mean(np.asarray(risk, dtype=float)))
    cost_term = lambda_cost * float(cost_total)
    switch_term = switch_penalty * float(switch_distance)
    over = max(0.0, float(cost_total) - float(budget))
    j = dro["value"] + macro_weight * macro + cost_term + switch_term + 1e6 * over
    return {"objective": float(j), "dro": float(dro["value"]),
            "macro": macro, "cost_term": float(cost_term),
            "switch_term": float(switch_term), "budget_violation": float(over),
            "tau": float(dro["tau"])}


def solve_stage_bruteforce(problem: StageProblem, *, ubar: Sequence[float],
                           epsilon: float, budget: float, macro_weight: float,
                           lambda_cost: float, switch_penalty: float = 0.0,
                           prev_k: Optional[Sequence[int]] = None,
                           max_combinations: int = 4_000_000,
                           dro_grid: int = 48) -> StageSolution:
    C = problem.n_classes
    K = problem.U.shape[2]
    total = K ** C
    if total > max_combinations:
        raise ValueError(f"bruteforce would enumerate {total} combinations")
    best: Optional[StageSolution] = None
    for combo in itertools.product(range(K), repeat=C):
        idx = np.asarray(combo, dtype=np.int64)
        cost = float(problem.cost[np.arange(C), idx].sum())
        if cost > budget + 1e-12:
            continue
        risk = problem.U[:, np.arange(C), idx].sum(axis=1) / C
        sw = 0.0 if prev_k is None else float(np.sum(idx != np.asarray(prev_k)))
        obj = stage_objective(risk=risk, ubar=ubar, epsilon=epsilon, cost_total=cost,
                              budget=budget, macro_weight=macro_weight,
                              lambda_cost=lambda_cost, switch_penalty=switch_penalty,
                              switch_distance=sw, dro_grid=dro_grid)
        if best is None or obj["objective"] < best.objective - 1e-12:
            best = StageSolution(
                k=np.asarray([problem.k_grid[i] for i in idx], dtype=np.int64),
                cost=cost, risk=np.asarray(risk, dtype=float),
                objective=obj["objective"], status="OPTIMAL_DISCRETE_BRUTEFORCE",
                detail={"switch_distance": sw, **obj})
    if best is None:
        raise ValueError("bruteforce found no feasible action vector")
    return best


def solve_stage_pareto(problem: StageProblem, *, ubar: Sequence[float],
                       epsilon: float, budget: float, macro_weight: float,
                       lambda_cost: float, switch_penalty: float = 0.0,
                       prev_k: Optional[Sequence[int]] = None,
                       cost_scale: Optional[int] = None,
                       risk_scale: Optional[int] = None,
                       target_risk_levels: int = 800,
                       target_cost_levels: int = 4000,
                       frontier_cap: int = 25_000) -> StageSolution:
    """Pareto-frontier DP over the quantised cost.

    Scales are chosen automatically so that the attainable risk range and the
    budget map onto a bounded number of levels. The envelope can span several
    orders of magnitude because the squared feedback-mismatch term dominates,
    and a fixed scale would either overflow the frontier or destroy resolution.
    """
    P = problem.n_partitions
    if P != 2:
        raise NotImplementedError("the frontier DP currently supports P=2")
    prev = None if prev_k is None else np.asarray(prev_k, dtype=np.int64)

    if risk_scale is None:
        shift = problem.U.min(axis=2).min(axis=0)
        total = float(np.max((problem.U - shift[None, :, None]).sum(axis=1)))
        risk_scale = max(1, int(round(target_risk_levels / max(total, 1e-12))))
    if cost_scale is None:
        cost_scale = max(1, int(round(target_cost_levels / max(float(budget), 1e-12))))
    requested = int(risk_scale)

    eff = int(risk_scale)
    for _ in range(6):
        out = _dp_attempt(problem, ubar=ubar, epsilon=epsilon, budget=budget,
                          macro_weight=macro_weight, lambda_cost=lambda_cost,
                          switch_penalty=switch_penalty, prev=prev,
                          cost_scale=int(cost_scale), risk_scale=eff,
                          frontier_cap=frontier_cap, requested_risk_scale=requested)
        if out is not None:
            return out
        eff = max(1, eff // 2)
    raise RuntimeError("frontier stayed above the cap after 6 coarsening rounds")


def _dp_attempt(problem: StageProblem, *, ubar, epsilon, budget, macro_weight,
                lambda_cost, switch_penalty, prev, cost_scale, risk_scale,
                frontier_cap, requested_risk_scale) -> Optional[StageSolution]:
    C = problem.n_classes
    K = problem.U.shape[2]
    budget_i = int(math.floor(budget * cost_scale + 1e-9))
    cost_i = np.array([[int(math.ceil(max(0.0, problem.cost[c, j]) * cost_scale - 1e-9))
                        for j in range(K)] for c in range(C)], dtype=np.int64)

    # Per-class shift applied to both partitions. Phi_eps is translation
    # equivariant along the all-ones direction and the macro term shifts by the
    # same constant, so the argmin over actions is unchanged. Centring keeps the
    # quantised spread small when the squared mismatch term dominates.
    shift = problem.U.min(axis=2).min(axis=0)
    risk_i = np.rint((problem.U - shift[None, :, None]) * risk_scale).astype(np.int64)

    # The switching count must be part of the state key: two paths with identical
    # cost and risk can differ in how many classes changed action.
    states: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {(0, 0, 0, 0): ()}
    for c in range(C):
        new: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}
        for (b, r0, r1, sw), path in states.items():
            for j in range(K):
                nb = b + int(cost_i[c, j])
                if nb > budget_i:
                    continue
                nsw = sw + (0 if prev is None or int(prev[c]) == j else 1)
                key = (nb, r0 + int(risk_i[0, c, j]), r1 + int(risk_i[1, c, j]), nsw)
                if key not in new:
                    new[key] = path + (j,)
        if not new:
            return None
        states = _prune_states(new, cap=frontier_cap)
        if states is None:
            return None

    best = None
    for (b, r0, r1, _sw), path in states.items():
        idx = np.asarray(path, dtype=np.int64)
        cost = float(problem.cost[np.arange(C), idx].sum())
        if cost > budget + 1e-9:
            continue
        risk = problem.U[:, np.arange(C), idx].sum(axis=1) / C
        sw = 0.0 if prev is None else float(np.sum(idx != prev))
        obj = stage_objective(risk=risk, ubar=ubar, epsilon=epsilon,
                              cost_total=cost, budget=budget,
                              macro_weight=macro_weight, lambda_cost=lambda_cost,
                              switch_penalty=switch_penalty, switch_distance=sw)
        if best is None or obj["objective"] < best[0] - 1e-12:
            best = (obj["objective"], idx, cost, risk, sw, obj)
    if best is None:
        return None
    obj_value, idx, cost, risk, sw, obj = best
    tol = 1.0 / max(risk_scale, 1) + 1.0 / max(cost_scale, 1)
    return StageSolution(
        k=np.asarray([problem.k_grid[i] for i in idx], dtype=np.int64),
        cost=cost, risk=np.asarray(risk, dtype=float), objective=obj_value,
        status="OPTIMAL_DISCRETE_QUANTIZED_COST",
        approximate=bool(risk_scale < requested_risk_scale), tolerance=float(tol),
        detail={"switch_distance": sw, "risk_scale": risk_scale,
                "cost_scale": cost_scale, "budget_quantized": budget_i,
                "requested_risk_scale": requested_risk_scale,
                "relative_tolerance": float(tol / max(abs(obj_value), 1e-12)), **obj})


def _prune_states(states: Dict[Tuple[int, int, int, int], Tuple[int, ...]],
                  cap: int) -> Optional[Dict[Tuple[int, int, int, int], Tuple[int, ...]]]:
    """Keep the (cost, r0, r1) Pareto frontier separately for each switch count."""
    if len(states) > cap * 12:
        return None
    by_sw: Dict[int, Dict[Tuple[int, int, int, int], Tuple[int, ...]]] = {}
    for key, path in states.items():
        by_sw.setdefault(key[3], {})[key] = path

    kept: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}
    for sw, group in by_sw.items():
        items = sorted(group.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))
        r1_values = sorted({k[2] for k, _ in items})
        if not r1_values:
            continue
        r1_index = {v: i for i, v in enumerate(r1_values)}
        last_r0 = None
        r0_group: List[Tuple[Tuple[int, int, int, int], Tuple[int, ...]]] = []

        def flush(grp, kept=kept):
            cur = np.iinfo(np.int64).max
            for key, path in grp:
                if key[0] < cur:
                    kept[key] = path
                    cur = key[0]

        for key, path in items:
            if last_r0 is None or key[1] == last_r0:
                r0_group.append((key, path))
            else:
                flush(r0_group)
                r0_group = [(key, path)]
            last_r0 = key[1]
        if r0_group:
            flush(r0_group)

        sub = {k: v for k, v in kept.items() if k[3] == sw}
        if len(sub) > cap // max(len(by_sw), 1):
            items2 = sorted(sub.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))
            running = np.full(len(r1_values), np.iinfo(np.int64).max, dtype=np.int64)
            drop = []
            for key, _ in items2:
                i = r1_index[key[2]]
                prefix = int(running[: i + 1].min())
                if key[0] >= prefix:
                    drop.append(key)
                running[i] = min(running[i], key[0])
            for k in drop:
                kept.pop(k, None)

    if len(kept) > cap:
        return None
    return kept


@dataclass
class StageRecord:
    stage: int
    k: np.ndarray
    cost: float
    risk: List[float]
    objective: float
    status: str
    approximate: bool
    tolerance: float
    switch_distance: float
    params: Dict[str, float]


def solve_sequence(stages: List[StageProblem], *, ubar: Sequence[float],
                   params_per_stage: List[Dict[str, float]],
                   budget_per_stage: float, solver: str = "pareto",
                   cost_scale: Optional[int] = None,
                   risk_scale: Optional[int] = None,
                   frontier_cap: int = 25_000) -> List[StageRecord]:
    """Stage-by-stage solve with an explicit switching penalty.

    Exact per stage *given* the previous stage's action. The joint problem over
    all stages is not enumerated for C>6; ``solve_sequence_exact_small`` exists
    for verification only.
    """
    records: List[StageRecord] = []
    prev_k: Optional[np.ndarray] = None
    for s, (prob, params) in enumerate(zip(stages, params_per_stage)):
        if solver == "bruteforce":
            sol = solve_stage_bruteforce(
                prob, ubar=ubar, epsilon=params["epsilon_s"],
                budget=budget_per_stage, macro_weight=params["macro_weight"],
                lambda_cost=params["lambda_s"],
                switch_penalty=params["switch_penalty"], prev_k=prev_k)
        else:
            sol = solve_stage_pareto(
                prob, ubar=ubar, epsilon=params["epsilon_s"],
                budget=budget_per_stage, macro_weight=params["macro_weight"],
                lambda_cost=params["lambda_s"],
                switch_penalty=params["switch_penalty"], prev_k=prev_k,
                cost_scale=cost_scale, risk_scale=risk_scale,
                frontier_cap=frontier_cap)
        sw = 0.0 if prev_k is None else float(np.sum(sol.k != prev_k))
        records.append(StageRecord(
            stage=s, k=sol.k, cost=sol.cost, risk=list(map(float, sol.risk)),
            objective=sol.objective, status=sol.status,
            approximate=sol.approximate, tolerance=sol.tolerance,
            switch_distance=sw, params=dict(params)))
        prev_k = sol.k
    return records


def solve_sequence_exact_small(stages: List[StageProblem], *, ubar: Sequence[float],
                               params_per_stage: List[Dict[str, float]],
                               budget_per_stage: float) -> Dict[str, object]:
    """Exhaustive joint optimum for tiny instances (verification only)."""
    C = stages[0].n_classes
    K = stages[0].U.shape[2]
    total = (K ** C) ** len(stages)
    if total > 2_000_000:
        raise ValueError(f"joint enumeration too large: {total}")
    per_stage: List[List[Tuple[Tuple[int, ...], float, np.ndarray]]] = []
    for prob in stages:
        opts = []
        for combo in itertools.product(range(K), repeat=C):
            idx = np.asarray(combo, dtype=np.int64)
            cost = float(prob.cost[np.arange(C), idx].sum())
            if cost > budget_per_stage + 1e-12:
                continue
            risk = prob.U[:, np.arange(C), idx].sum(axis=1) / C
            opts.append((combo, cost, risk))
        per_stage.append(opts)
    best = None
    for choice in itertools.product(*[range(len(o)) for o in per_stage]):
        total_obj = 0.0
        prev = None
        for s, ci in enumerate(choice):
            combo, cost, risk = per_stage[s][ci]
            sw = 0.0 if prev is None else float(np.sum(np.asarray(combo) != np.asarray(prev)))
            p = params_per_stage[s]
            obj = stage_objective(risk=risk, ubar=ubar, epsilon=p["epsilon_s"],
                                  cost_total=cost, budget=budget_per_stage,
                                  macro_weight=p["macro_weight"],
                                  lambda_cost=p["lambda_s"],
                                  switch_penalty=p["switch_penalty"],
                                  switch_distance=sw)
            total_obj += obj["objective"]
            prev = combo
        if best is None or total_obj < best[0] - 1e-12:
            best = (total_obj, choice)
    if best is None:
        raise ValueError("no feasible joint sequence")
    return {"objective": float(best[0]), "choice": list(best[1]),
            "n_stage_options": [len(o) for o in per_stage]}
