#!/usr/bin/env python3
"""Contract tests for the parameterized PC-FA action space (no Plant data).

Checks that k_grid / gamma / inner_budget are honored, registered, and
enforced, and that unsupported configurations fail explicitly instead of
being silently clipped.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "validated_runtime"))

import numpy as np

from pcfa_policy import choose_pcfa, normalize_k_grid
from tfgps_gpu_lib import cvar_terms, omega_from_counts


def synthetic_inputs(C=8):
    counts = np.asarray([100, 60, 40, 20, 10, 6, 3, 2], dtype=np.int64)[:C]
    rng = np.random.default_rng(3407)
    A = rng.uniform(0.5, 2.0, size=C)
    rho = rng.uniform(0.3, 0.7, size=C)
    sigma = rng.uniform(0.05, 0.2, size=C)
    q = omega_from_counts(counts, 0.75)
    m = 8.0
    r_fixed = sigma ** 2 / (m * np.maximum(q, 1e-15))
    cvar_fixed, _, _ = cvar_terms(r_fixed, 0.8)
    class_costs = {
        str(c): {
            "native_cost_per_sample": 1.0e6,
            "group_native_cost_per_sample": 1.0e6,
            "reference_k0_cost_per_sample": 5.0e5,
            "reference_action_cost_per_sample": 5.0e5,
            "solver_step_cost_per_sample": 1.5e5,
        }
        for c in range(C)
    }
    cost_row = {
        "q": q.tolist(), "m": int(m), "class_costs": class_costs,
        "fixed_k": {"per_sample_flops": 1.5e6},
        "mean_risk_fixed_k": float(np.sum(q * r_fixed)),
        "cvar_fixed_k": float(cvar_fixed),
    }
    cal = {"counts": counts.tolist(), "A": A.tolist(), "rho": rho.tolist(),
           "sigma": sigma.tolist()}
    return cost_row, cal


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    cost_row, cal = synthetic_inputs()
    q = np.asarray(cost_row["q"])
    results = {}

    pol = choose_pcfa(cost_row, cal, gamma=0.8, k_grid=(0, 1, 2, 4), inner_budget=100.0,
                      variant="a100_fp32_fused")
    check(set(pol["k_discrete"]) <= {0, 1, 2, 4}, "default grid violated")
    check(pol["registered_config"]["k_grid"] == [0, 1, 2, 4], "k_grid not registered")
    check(pol["registered_config"]["inner_budget"] == 100.0, "inner_budget not registered")
    check(pol["registered_config"]["variant"] == "a100_fp32_fused", "variant not registered")
    check(pol["registered_config"]["tail_cvar_gamma"] == 0.8, "gamma not registered")
    results["grid_default"] = pol["k_unique"]

    wide = choose_pcfa(cost_row, cal, gamma=0.9, k_grid=[0, 1, 2, 4, 8, 16],
                       inner_budget=100.0, variant="a100_fp32_fused")
    check(set(wide["k_discrete"]) <= {0, 1, 2, 4, 8, 16}, "wide grid violated")
    check(wide["registered_config"]["k_grid"] == [0, 1, 2, 4, 8, 16], "wide grid not registered")
    check(wide["registered_config"]["tail_cvar_gamma"] == 0.9, "wide gamma not registered")
    results["grid_wide"] = wide["k_unique"]

    for pol_any in (pol, wide):
        k = np.asarray(pol_any["k_discrete"]); a = np.asarray(pol_any["alpha_discrete"])
        check(not np.any((k == 0) & (a > 0)), "k=0 must imply alpha=0")
        check(float(np.sum(q * k)) <= float(pol_any["inner_budget"]) + 1e-9,
              "inner k budget violated")

    tight = choose_pcfa(cost_row, cal, gamma=0.8, k_grid=[0, 1, 2, 4, 8, 16],
                        inner_budget=0.01, variant="a100_fp32_fused")
    check(tight["valid_nondegenerate"] is False, "tiny inner budget must be infeasible")
    check("BOUNDARY_ACTION_SPACE" in tight["boundary_reason"],
          f"expected BOUNDARY_ACTION_SPACE, got {tight['boundary_reason']}")

    budget_ok = choose_pcfa(cost_row, cal, gamma=0.8, k_grid=[0, 1, 2, 4, 8, 16],
                            inner_budget=0.35, variant="a100_fp32_fused")
    if budget_ok["valid_nondegenerate"]:
        check(float(np.sum(q * np.asarray(budget_ok["k_discrete"]))) <= 0.35 + 1e-9,
              "selected policy exceeds registered inner budget")

    for bad in ([], [1, 2], [2, 4], [-1, 0, 1]):
        try:
            normalize_k_grid(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"grid {bad} should be rejected")
    try:
        normalize_k_grid("0,1,abc")
    except ValueError:
        pass
    else:
        raise AssertionError("non-integer grid should be rejected")

    check(math.isfinite(pol["cost_per_sample"]), "non-finite cost")
    print({"status": "PASS", "cases": results,
           "tiny_budget_boundary": tight["boundary_reason"],
           "tiny_budget_status": tight["valid_nondegenerate"]})


if __name__ == "__main__":
    main()
