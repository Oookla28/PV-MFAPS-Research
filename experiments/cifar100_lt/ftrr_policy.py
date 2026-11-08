"""Finite-action Feedback Trust-Region Readout Refinement.

This module is a policy core only. It never reads the held/test split. The
returned certificates apply to bounded FIT calibration proxies and must not be
presented as test-risk guarantees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Action:
    k: int
    alpha: float


def cosine(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / max(den, eps))


def cone_feasible(g0: Sequence[float], ga: Sequence[float], xi: float) -> bool:
    a = np.asarray(g0, dtype=float)
    b = np.asarray(ga, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12:
        return False
    return bool(float(a @ b) >= float(xi * na * nb) - 1e-10)


def trust_alpha_max(g0: Sequence[float], gk: Sequence[float], radius: float,
                    xi: float = 0.0, grid: int = 128) -> float:
    """Largest sampled/bisected alpha satisfying displacement and cone tests."""
    a = np.asarray(g0, dtype=float)
    d = np.asarray(gk, dtype=float) - a
    if radius < 0 or xi < 0 or xi >= 1:
        raise ValueError("radius >= 0 and xi in [0,1) are required")
    if np.linalg.norm(d) <= 1e-12:
        return 1.0

    def ok(x: float) -> bool:
        g = a + x * d
        return np.linalg.norm(x * d) <= radius + 1e-10 and cone_feasible(a, g, xi)

    xs = np.linspace(0.0, 1.0, max(int(grid), 8) + 1)
    good = [bool(ok(float(x))) for x in xs]
    if not good[0]:
        return 0.0
    last = max(i for i, v in enumerate(good) if v)
    if last == len(xs) - 1:
        return 1.0
    lo, hi = float(xs[last]), float(xs[last + 1])
    for _ in range(32):
        mid = (lo + hi) / 2.0
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def cone_project_alpha(g0: Sequence[float], gr: Sequence[float],
                       xi: float = 0.2, alpha_max: float = 1.0,
                       iterations: int = 60):
    """Return the largest trust-cone coefficient and its margin."""
    if not (0.0 <= xi < 1.0) or alpha_max < 0:
        raise ValueError("invalid cone parameters")
    x = np.asarray(g0, dtype=float)
    d = np.asarray(gr, dtype=float) - x
    radius = float(np.linalg.norm(d) * alpha_max)
    alpha = trust_alpha_max(x, np.asarray(gr, dtype=float), radius,
                            xi=xi, grid=max(64, iterations))
    mixed = x + alpha * d
    return float(alpha), cosine(x, mixed), float(cosine(x, mixed) - xi)


def noise_floor(eta: float, sigma: float, n: float, rho: float) -> float:
    if eta < 0 or sigma < 0 or n <= 0 or not (0 <= rho < 1):
        raise ValueError("invalid inner-SGD parameters")
    return float(eta * eta * sigma * sigma / (n * (1.0 - rho * rho)))


def feedback_proxy(k: int, kappa: float, rho: float, distance: float,
                   eta: float, sigma: float, n: float) -> float:
    """Squared feedback-risk proxy with a non-vanishing stochastic floor."""
    if k < 0 or kappa < 0 or distance < 0:
        raise ValueError("invalid feedback proxy inputs")
    if not (0 <= rho < 1):
        raise ValueError("rho must lie in [0,1)")
    return float(kappa * kappa * (rho ** (2 * int(k)) * distance * distance
                                  + noise_floor(eta, sigma, n, rho)))


def empirical_bernstein(samples: Sequence[float], delta: float, bound: float,
                        multiplicity: int = 1) -> dict:
    x = np.asarray(samples, dtype=float)
    if x.size < 2 or not (0 < delta < 1) or bound <= 0 or multiplicity < 1:
        raise ValueError("invalid bounded calibration sample contract")
    if np.any(x < 0) or np.any(x > bound):
        raise ValueError("samples must be clipped to [0, bound]")
    log_term = math.log(2.0 * multiplicity / delta)
    radius = math.sqrt(2.0 * float(x.var(ddof=1)) * log_term / x.size)
    radius += 7.0 * bound * log_term / (3.0 * (x.size - 1))
    return {"mean": float(x.mean()), "variance": float(x.var(ddof=1)),
            "radius": float(radius), "n": int(x.size), "bound": float(bound)}


def paired_eb_radius(samples: Sequence[float], delta: float, bound: float,
                     simultaneous: int = 1) -> float:
    """Empirical-Bernstein radius for bounded paired differences."""
    x = np.asarray(samples, dtype=float)
    if x.size < 2 or not (0.0 < delta < 1.0) or bound <= 0:
        raise ValueError("invalid paired EB inputs")
    # Enforce the declared bounded proxy at the certificate boundary. This
    # absorbs floating-point overshoot from upstream subtraction/clip calls
    # without weakening the bound.
    x = np.clip(x, -bound, bound)
    x = np.clip(x, -bound, bound)
    log_term = math.log(2.0 * max(1, int(simultaneous)) / delta)
    return float(math.sqrt(2.0 * float(x.var(ddof=1)) * log_term / x.size)
                 + 7.0 * bound * log_term / (3.0 * (x.size - 1)))


def admit_action(gain_mean: float, radius: float, cost: float,
                 switch_cost: float, margin: float) -> bool:
    return bool(float(gain_mean) - float(radius)
                >= float(cost) + float(switch_cost) + float(margin))


def legal_actions(k_grid=(0, 1, 2, 4, 8), alpha_grid=(0.0, 0.125, 0.25, 0.5)):
    out = []
    for k in sorted(set(int(v) for v in k_grid)):
        if k < 0:
            raise ValueError("k must be nonnegative")
        if k == 0:
            out.append(Action(0, 0.0))
        else:
            out.extend(Action(k, float(a)) for a in sorted(set(alpha_grid))
                        if 0.0 <= float(a) <= 1.0)
    return out


def cone_safe_action(g0: Sequence[float], gk: Sequence[float], radius: float,
                     k: int, alpha: float, xi: float = 0.0) -> bool:
    if k == 0:
        return alpha == 0.0
    max_alpha = trust_alpha_max(g0, gk, radius, xi=xi)
    return bool(0.0 <= alpha <= max_alpha + 1e-10)
