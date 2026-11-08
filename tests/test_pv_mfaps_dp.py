#!/usr/bin/env python3
"""Deterministic unit tests for the independent PV-MFAPS DP solver."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "validated_runtime"))
from pv_mfaps_dp import risk_envelope, solve_pv_mfaps_dp


def brute_force():
    A = np.array([2.0, 1.0, 3.0])
    rho = np.array([0.9, 0.8, 0.95])
    sigma = np.array([0.1, 0.2, 0.3])
    n = np.array([10.0, 5.0, 20.0])
    q = np.array([0.2, 0.3, 0.5])
    mu = 1.0
    nu = np.array([0.5, 0.0, 0.5])
    k_grid = (0, 1, 2)
    xi = 1.01
    tail = np.array([True, False, True])
    macro_cap = 3.0
    budget = 0.4
    weights = mu / 3.0 + nu
    best = None
    for ks in product(k_grid, repeat=3):
        costs = sum(q[c] * ks[c] for c in range(3))
        risks = np.array([risk_envelope(A, rho, sigma, n, k, xi)[c] for c, k in enumerate(ks)])
        if costs > budget + 1e-12 or risks.mean() > macro_cap + 1e-12:
            continue
        # The DP smoke does not set a tail cap; this is the same feasible set.
        obj = float(np.dot(weights, risks))
        key = (obj, risks.sum(), costs, ks)
        if best is None or key < best[0]:
            best = (key, ks)
    assert best is not None
    out = solve_pv_mfaps_dp(
        A=A, rho=rho, sigma=sigma, n=n, mu=mu, nu=nu, q=q, budget=budget,
        k_grid=k_grid, xi=xi, tail_mask=tail, macro_cap=macro_cap,
    )
    assert out["status"] == "OPTIMAL_DISCRETE_QUANTIZED_COST"
    assert tuple(out["k"]) == best[1], (out["k"], best)
    cert = out["certificate"]
    assert cert["actual_cost"] <= budget + 1e-12
    assert cert["macro_cap_satisfied"]
    return out


def smoke_tail_cap():
    out = solve_pv_mfaps_dp(
        A=[2.0, 1.0, 3.0], rho=[0.9, 0.8, 0.95], sigma=[.1, .2, .3], n=[10, 5, 20],
        mu=1.0, nu=[.5, 0.0, .5], q=[.2, .3, .5], budget=.4,
        k_grid=(0, 1, 2), xi=1.01, tail_mask=[True, False, True], tail_cap=3.1,
        macro_cap=3.0,
    )
    assert out["status"] == "OPTIMAL_DISCRETE_QUANTIZED_COST"
    assert out["certificate"]["tail_cap_satisfied"]
    return out


if __name__ == "__main__":
    a = brute_force()
    b = smoke_tail_cap()
    print("PV_MFAPS_DP_TEST_PASS", a["k"], b["k"])
