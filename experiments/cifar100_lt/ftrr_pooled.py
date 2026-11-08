"""Pooled paired certificate for FTRR stage admission.

The pooled certificate is a stage-level eligibility test. It uses every FIT
calibration sample for a candidate refinement depth and does not claim a
class-conditional confidence guarantee. Class allocation remains a separate
finite-action surrogate problem.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ftrr_policy import paired_eb_radius


def pooled_gain_certificate(gains: Sequence[float], *, delta: float,
                            bound: float, simultaneous: int = 1) -> dict:
    """Return a bounded paired-Empirical-Bernstein certificate over all FIT rows."""
    x = np.asarray(gains, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("pooled gains must be a one-dimensional sample vector")
    # Re-clip at the certificate boundary to absorb floating-point overshoot
    # from upstream subtraction/clip operations. The declared bounded proxy
    # remains exactly [-bound, bound].
    preclip = x.copy()
    x = np.clip(x, -bound, bound)
    n_clipped_before_clip = int(np.count_nonzero(preclip != x))
    clip_fraction = float(n_clipped_before_clip / x.size)
    clipped = np.abs(x) > bound
    clip_count = int(np.sum(clipped))
    x = np.clip(x, -bound, bound)
    radius = paired_eb_radius(x, delta, bound, simultaneous)
    return {
        "n": int(x.size),
        "mean_gain": float(x.mean()),
        "variance": float(x.var(ddof=1)),
        "radius": float(radius),
        "lower_confidence_gain": float(x.mean() - radius),
        "bound": float(bound),
        "delta": float(delta),
        "simultaneous": int(simultaneous),
        "clip_count": clip_count,
        "clip_fraction": float(clip_count / x.size),
        "n_clipped_before_clip": n_clipped_before_clip,
        "clip_fraction": clip_fraction,
    }


def admit_pooled_gain(gains: Sequence[float], *, delta: float, bound: float,
                      cost: float = 0.0, switch_cost: float = 0.0,
                      margin: float = 0.0, simultaneous: int = 1) -> dict:
    cert = pooled_gain_certificate(gains, delta=delta, bound=bound,
                                   simultaneous=simultaneous)
    threshold = float(cost + switch_cost + margin)
    cert["threshold"] = threshold
    cert["admitted"] = bool(cert["lower_confidence_gain"] >= threshold)
    return cert
