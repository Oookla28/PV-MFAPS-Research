#!/usr/bin/env python3
"""Synthetic pooled-certificate smoke. It never reads CIFAR or HELD."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ftrr_pooled import admit_pooled_gain, pooled_gain_certificate
from ftrr_policy import paired_eb_radius


def main() -> None:
    rng = np.random.default_rng(20261108)
    bound = 1.0
    delta = 0.05
    classes = 100
    samples_per_class = 12
    N = classes * samples_per_class
    gains = np.clip(rng.normal(0.20, 0.06, N), -bound, bound)
    pooled = admit_pooled_gain(gains, delta=delta, bound=bound,
                               cost=0.02, simultaneous=5)
    per_class_radius = [paired_eb_radius(gains[c * samples_per_class:(c + 1) * samples_per_class],
                                         delta, bound, simultaneous=5)
                        for c in range(classes)]
    cert = pooled_gain_certificate(gains, delta=delta, bound=bound,
                                   simultaneous=5)
    assert cert["n"] == N
    assert cert["radius"] < float(np.median(per_class_radius))
    assert pooled["lower_confidence_gain"] <= pooled["mean_gain"]
    out = {
        "status": "EXECUTED",
        "kind": "FTRR_POOLED_PAIRED_CERTIFICATE_SMOKE",
        "held_reads": 0,
        "classes": classes,
        "samples_per_class": samples_per_class,
        "pooled": pooled,
        "median_per_class_radius": float(np.median(per_class_radius)),
        "mean_per_class_radius": float(np.mean(per_class_radius)),
        "radius_reduction_vs_median": float(
            1.0 - cert["radius"] / float(np.median(per_class_radius))),
        "claim_status": "OFFLINE_POLICY_ONLY",
        "interpretation": "stage-level eligibility only; class-conditional risk remains unverified",
    }
    out_dir = Path(__file__).resolve().parents[2] / "out" / "ftrr_pooled_smoke_20261108"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FTRR_POOLED_CERTIFICATE_OFFLINE.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
