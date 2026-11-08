#!/usr/bin/env python3
"""Deterministic FTRR policy smoke. No dataset or HELD access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ftrr_policy import (admit_action, cone_project_alpha, noise_floor,
                         paired_eb_radius)


def main() -> None:
    rng = np.random.default_rng(20261108)
    g0 = np.array([1.0, 0.0, 0.0])
    gr = np.array([0.8, 0.4, 0.0])
    alpha, c, margin = cone_project_alpha(g0, gr, xi=0.8, alpha_max=1.0)
    assert 0.0 <= alpha <= 1.0 and c >= 0.8 - 1e-9

    paired = rng.normal(0.2, 0.03, 64).clip(-1, 1)
    radius = paired_eb_radius(paired, delta=0.05, bound=1.0, simultaneous=8)
    assert radius > 0
    assert admit_action(float(paired.mean()), radius, 0.0, 0.0, 0.0) is False

    floor_early = noise_floor(0.05, 0.25, 32, 0.99)
    floor_late = noise_floor(0.05, 0.25, 32, 0.60)
    assert floor_early > floor_late

    out = {
        "status": "EXECUTED",
        "kind": "FTRR_OFFLINE_POLICY_SMOKE",
        "held_reads": 0,
        "cone": {"alpha": alpha, "cosine": c, "margin": margin},
        "paired_eb": {"n": 64, "radius": radius, "delta": 0.05},
        "noise_floor": {"early": floor_early, "late": floor_late},
        "claim_status": "OFFLINE_POLICY_ONLY",
    }
    out_dir = Path(__file__).resolve().parents[2] / "out" / "ftrr_offline_20261108"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FTRR_OFFLINE_SMOKE.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
