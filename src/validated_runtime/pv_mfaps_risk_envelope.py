#!/usr/bin/env python3
"""Create a FIT-only PV-MFAPS risk-envelope diagnostic figure.

The figure is deliberately labelled as a proxy diagnostic. It uses the
completed pilot's class proxy fields and an explicit growth sensitivity to
visualize how bias, stability, and empirical-Bernstein terms compete. It is
not a reported generalization bound until the stability constants and
per-action empirical variances are measured and certified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_proxy(A, rho, sigma, n, *, k_grid, growth, eta, beta, loss_bound, delta, C, action_count):
    effective_growth = 1.0 + beta * eta if beta is not None else growth
    log_term = np.log(max(2.0 * C * action_count / delta, 2.0))
    out = {"k": list(k_grid), "bias": [], "stability": [], "eb": [], "total": []}
    for k in k_grid:
        geom_sum = sum(effective_growth**j for j in range(k))
        bias = A * rho ** (2 * k)
        stability = 2.0 * A * eta * geom_sum / max(n, 1.0)
        eb = np.sqrt(max(0.0, 2.0 * sigma**2 * effective_growth ** (2 * k) * log_term / max(n, 1.0)))
        eb += 7.0 * loss_bound * log_term / (3.0 * max(n - 1.0, 1.0))
        out["bias"].append(float(bias))
        out["stability"].append(float(stability))
        out["eb"].append(float(eb))
        out["total"].append(float(bias + stability + eb))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--growth", type=float, default=1.05)
    ap.add_argument("--class-index", type=int, default=None)
    args = ap.parse_args()

    x = json.loads(Path(args.input).read_text())
    if int(x.get("held_reads", -1)) != 0:
        raise RuntimeError("input contains held reads")
    summary = x["class_proxy_summary"]
    A = np.asarray(summary["A"], dtype=float)
    rho = np.asarray(summary["rho"], dtype=float)
    sigma = np.asarray(summary["sigma"], dtype=float)
    n = np.asarray(summary["support"], dtype=float)
    c = int(args.class_index) if args.class_index is not None else int(np.lexsort((-A, n))[0])
    if c < 0 or c >= len(A):
        raise ValueError("class index out of range")
    k_grid = (0, 1, 2, 4)
    curve = compute_proxy(A[c], np.clip(rho[c], 0.0, 1.0), sigma[c], n[c],
                          k_grid=k_grid, growth=args.growth, eta=1.0, beta=None,
                          loss_bound=1.0, delta=0.05, C=len(A), action_count=8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    for key, color, label, marker in [
        ("bias", "#1f77b4", r"bias $A_c\rho_c^{2k}$", "o"),
        ("stability", "#d62728", r"stability proxy", "s"),
        ("eb", "#2ca02c", r"empirical-Bernstein proxy", "^"),
        ("total", "#111111", r"total envelope", "D"),
    ]:
        ax.semilogy(curve["k"], np.maximum(curve[key], 1e-12), marker=marker,
                    linewidth=2.0, label=label, color=color)
    ax.set_xlabel("inner solver steps k")
    ax.set_ylabel("proxy risk scale (log)")
    ax.set_title(f"Class {c}: proxy envelope, n={int(n[c])}, growth={args.growth:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    base = max(curve["total"][0], 1e-12)
    for key, color, label, marker in [
        ("bias", "#1f77b4", "bias / total(k=0)", "o"),
        ("stability", "#d62728", "stability / total(k=0)", "s"),
        ("eb", "#2ca02c", "EB / total(k=0)", "^"),
        ("total", "#111111", "total / total(k=0)", "D"),
    ]:
        ax.plot(curve["k"], np.asarray(curve[key]) / base, marker=marker,
                linewidth=2.0, label=label, color=color)
    ax.set_xlabel("inner solver steps k")
    ax.set_ylabel("relative proxy scale")
    ax.set_title("Decomposition relative to k=0")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("PV-MFAPS risk-envelope diagnostic (UNVERIFIED_OFFLINE)", fontsize=13)
    fig.savefig(args.output, dpi=180)
    sidecar = {
        "status": "UNVERIFIED_OFFLINE",
        "held_reads": 0,
        "input": str(Path(args.input).resolve()),
        "class_index": c,
        "support": int(n[c]),
        "A": float(A[c]),
        "rho": float(rho[c]),
        "sigma": float(sigma[c]),
        "growth": float(args.growth),
        "k_grid": list(k_grid),
        "curve": curve,
        "note": "Illustrative proxy. Stability constants and per-action variances are not yet certified; do not use as a theorem or HELD result.",
    }
    Path(args.sidecar).write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": sidecar["status"], "class_index": c, "support": int(n[c]), "output": args.output}))


if __name__ == "__main__":
    main()

