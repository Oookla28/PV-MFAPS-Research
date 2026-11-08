#!/usr/bin/env python3
"""CPU-only exactness and phase-policy smoke for FD-SDPA."""
from __future__ import annotations
import json
import itertools
from pathlib import Path
from fd_sdpa_dp import Action, certified_phase_policy, constant_step_stability, kl_dro_value, safe_activate, solve_sequence, stability_bound


def main(out: str):
    # Stability identity and constant-step closed form agree.
    seq = stability_bound(1.0, 2.0, 0.4, [0.1] * 4, 10, 4)
    closed = constant_step_stability(1.0, 2.0, 0.4, 0.1, 10, 4)
    assert abs(seq - closed) < 1e-12
    # KL-DRO endpoints and monotonicity.
    risks = [1.0, 3.0]
    d0 = kl_dro_value(risks, 0.0)
    d1 = kl_dro_value(risks, 0.1)
    assert abs(d0 - 2.0) < 1e-12 and 2.0 <= d1 <= 3.0
    # Safe gate rejects a candidate whose confidence upper bound is worse.
    assert not safe_activate(1.0, 0.99, 0.02, 0.05, 1.0, 0.0, 0.01, 0.0, 0.01)
    assert safe_activate(1.0, 0.7, 0.02, 0.01, 1.0, 0.0, 0.01, 0.0, 0.01)
    base = Action("k0", 1, (1.0, 1.0), payload=(0,))
    k1 = Action("k1", 2, (0.7, 0.8), payload=(1,))
    k4 = Action("k4", 5, (0.6, 0.7), payload=(4,))
    f = [[base, k1], [base, k1, k4], [base, k1, k4]]
    dp = solve_sequence(f, budget=6, epsilon=[0.0, 0.1, 0.1], lambda_switch=0.1)
    assert dp["status"] == "EXECUTED" and dp["cost"] <= 6
    brute = []
    for choice in itertools.product(*f):
        cost = sum(a.cost for a in choice)
        if cost > 6:
            continue
        value = sum(kl_dro_value(a.risk, eps) for a, eps in zip(choice, [0.0, 0.1, 0.1]))
        value += 0.1 * sum(float(a.payload != b.payload) for a, b in zip(choice[1:], choice[:-1]))
        brute.append((value, cost, [a.name for a in choice]))
    brute_value, brute_cost, brute_actions = min(brute)
    assert abs(dp["objective"] - brute_value) < 1e-10
    assert dp["cost"] == brute_cost and dp["actions"] == brute_actions
    phase_frontiers = [
        [base, Action("k1", 2, (1.2, 1.2), payload=(1,))],
        [base, Action("k1", 2, (0.7, 0.8), payload=(1,)), k4],
        [base, Action("k1", 2, (0.6, 0.7), payload=(1,)), k4],
    ]
    logs = certified_phase_policy(phase_frontiers, "k0", [{"k0": 0.01, "k1": 0.01}, {"k0": 0.01, "k1": 0.01, "k4": 0.01}, {"k0": 0.01, "k1": 0.01, "k4": 0.01}], 0.01, 0.01, 0.01)
    assert len(logs) == 3 and logs[0]["chosen"] == "k0" and logs[1]["chosen"] == "k1"
    result = {
        "status": "EXECUTED",
        "held_reads": 0,
        "stability_identity_abs_error": abs(seq - closed),
        "kl_dro": {"epsilon_0": d0, "epsilon_0_1": d1},
        "safe_gate_checks": 2,
        "dp": dp,
        "brute_force": {"objective": brute_value, "cost": brute_cost, "actions": brute_actions},
        "phase_logs": logs,
        "claims": {
            "finite_grid_exactness": "verified on synthetic CPU smoke",
            "generalization": "UNVERIFIED",
            "held_accuracy": "UNVERIFIED",
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(args.output)
