#!/usr/bin/env python3
"""Deterministic offline checks for the FTRR policy core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ftrr_policy import (cone_feasible, empirical_bernstein, feedback_proxy,
                         legal_actions, noise_floor, trust_alpha_max)


def main() -> int:
    g0 = np.array([1.0, 0.0, 0.0])
    gk = np.array([0.8, 0.3, 0.0])
    alpha = trust_alpha_max(g0, gk, radius=0.25, xi=0.2)
    ga = g0 + alpha * (gk - g0)
    samples = np.linspace(0.05, 0.45, 32)
    eb = empirical_bernstein(samples, delta=0.05, bound=1.0,
                             multiplicity=32)
    result = {
        "status": "EXECUTED",
        "held_reads": 0,
        "action_count": len(legal_actions()),
        "noise_floor": noise_floor(0.05, 0.4, 8, 0.8),
        "proxy_k0": feedback_proxy(0, 1.2, 0.8, 1.0, 0.05, 0.4, 8),
        "proxy_k4": feedback_proxy(4, 1.2, 0.8, 1.0, 0.05, 0.4, 8),
        "alpha_max": alpha,
        "cone_ok": cone_feasible(g0, ga, 0.2),
        "displacement": float(np.linalg.norm(ga - g0)),
        "eb": eb,
        "claim_status": "OFFLINE_POLICY_ONLY",
    }
    out = Path(__file__).resolve().parents[2] / "out" / "ftrr_policy_smoke_20261108"
    out.mkdir(parents=True, exist_ok=True)
    (out / "FTRR_POLICY_OFFLINE.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["cone_ok"] and result["held_reads"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
