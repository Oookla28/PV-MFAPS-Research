#!/usr/bin/env python3
"""Synthetic SPD mechanism benchmark for the feedback-risk line.

CPU-only, data-free.  It does not read PlantSPD, does not touch HELD and does
not run on the GPU used by the A100 pilot.

Model: e_c = J_c^T D_c dW_c with an SPD readout Jacobian J_c (rank-limited,
geometric spectrum rho^k).  Two task-orientation regimes are registered:

  random : the class task direction is a random unit vector (orientation is
           unidentified, so spectral condition may be swamped);
  aligned: the task direction is the top left-singular direction of J_c
           (spectral condition is identifiable).

For each regime the benchmark reports the Schur bound ratio, gradient cosine,
margin change, surrogate class risk, FIT/HELD-proxy tail recall, the M3
predictive comparison (kappa vs 1/n, loss, gradient norm), and the registered
A/B counterfactual.  All of this is synthetic mechanism evidence; A_c and
sigma_c remain proxies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def partial_spearman(x, y, controls):
    x = np.asarray(x, float); y = np.asarray(y, float)
    Z = np.column_stack([np.ones_like(x)] + [np.asarray(c, float) for c in controls])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return spearman(x - Z @ bx, y - Z @ by)


def tail_recall(margins, threshold=0.0, sigma=0.35):
    z = (np.asarray(margins, float) - threshold) / max(sigma, 1e-9)
    return float(np.mean(0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))))


def class_geometry(rng, d=64, rank=8, rho=0.999, align=0.7, residual=0.05,
                   grad_norm=1.0, n=25, loss=0.5, task_mode="random"):
    sv = rho ** np.arange(rank)
    U, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    V, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    J = (U * sv) @ V.T
    if task_mode == "aligned":
        mix = U[:, 0].copy()
    else:
        raw = rng.normal(size=d); raw /= np.linalg.norm(raw)
        mix = align * U[:, 0] + math.sqrt(max(1.0 - align ** 2, 0.0)) * raw
        mix /= np.linalg.norm(mix)
    D = np.eye(d) + 0.5 * np.diag(np.linspace(1.0, 0.5, d))
    dW = residual * rng.normal(size=(d, 1)) @ mix[None, :]
    e = J.T @ D @ dW[:, 0]
    e_norm = float(np.linalg.norm(e))
    fro = float(np.linalg.norm(J, "fro") * np.linalg.norm(D, "fro") * np.linalg.norm(dW, "fro"))
    max_entry = float(np.linalg.norm(J, np.inf) * np.linalg.norm(D, np.inf) *
                      np.linalg.norm(dW, np.inf) * d)
    g_ref = grad_norm * mix
    cos = float(e @ g_ref / max(e_norm * np.linalg.norm(g_ref), 1e-30))
    margin_change = float(e @ mix)
    kappa = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {"rho": rho, "kappa": kappa, "align": align, "n": int(n), "loss": float(loss),
            "grad_norm": float(grad_norm), "e_norm": e_norm, "bound_fro": fro,
            "bound_max_entry": max_entry, "bound_ratio_fro": e_norm / max(fro, 1e-30),
            "bound_ratio_max_entry": e_norm / max(max_entry, 1e-30),
            "cos_e_grad": cos, "margin_change": margin_change,
            "risk_proxy": float(e_norm ** 2 / max(n, 1))}


def run_regime(rng, mode, repeats):
    rows = []
    for rep in range(repeats):
        for n in (5, 10, 25, 100):
            for rho in (0.95, 0.99, 0.999, 0.9999):
                for align in (0.3, 0.9):
                    r = class_geometry(rng, rho=rho, align=align, n=n,
                                       residual=0.05 * (1 + rng.normal() * 0.1),
                                       grad_norm=1.0 + 0.2 * rng.normal(),
                                       loss=0.4 + 0.2 * rng.random(), task_mode=mode)
                    r["rep"] = rep; r["mode"] = mode
                    r["later_degradation"] = r["margin_change"] + 0.02 * rng.normal()
                    base_margin = 1.0 + 0.5 * rng.normal(size=n)
                    r["fit_recall_before"] = tail_recall(base_margin)
                    r["fit_recall_after"] = tail_recall(base_margin - r["margin_change"])
                    held_margin = 1.0 + 0.5 * rng.normal(size=n)
                    r["held_recall_after"] = tail_recall(held_margin - r["margin_change"])
                    rows.append(r)
    kappa = [r["kappa"] for r in rows]; deg = [r["later_degradation"] for r in rows]
    invn = [1.0 / r["n"] for r in rows]; loss = [r["loss"] for r in rows]
    gn = [r["grad_norm"] for r in rows]
    m3 = {
        "spearman_kappa_margin_change": spearman(kappa, [r["margin_change"] for r in rows]),
        "spearman_kappa_degradation": spearman(kappa, deg),
        "spearman_inv_n_degradation": spearman(invn, deg),
        "spearman_loss_degradation": spearman(loss, deg),
        "spearman_gradnorm_degradation": spearman(gn, deg),
        "partial_spearman_kappa_given_n_loss_grad": partial_spearman(kappa, deg, [invn, loss, gn]),
        "partial_spearman_invn_given_kappa_loss_grad": partial_spearman(invn, deg, [kappa, loss, gn]),
    }
    ab = []
    for _ in range(8):
        a = class_geometry(rng, rho=0.9999, align=0.9, n=25, loss=0.5, grad_norm=1.0, task_mode=mode)
        b = class_geometry(rng, rho=0.95, align=0.9, n=25, loss=0.5, grad_norm=1.0, task_mode=mode)
        ab.append({"kappa_A_gt_B": a["kappa"] > b["kappa"],
                   "degradation_A_gt_B": a["margin_change"] > b["margin_change"],
                   "consistent": (a["kappa"] > b["kappa"]) == (a["margin_change"] > b["margin_change"])})
    return rows, m3, ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    regimes = {}
    all_rows = []
    for mode in ("random", "aligned"):
        rows, m3, ab = run_regime(rng, mode, args.repeats)
        all_rows.extend(rows)
        regimes[mode] = {"m3": m3, "counterfactual_AB": {
            "n_trials": len(ab),
            "consistent_fraction": float(np.mean([x["consistent"] for x in ab])),
            "kappa_A_gt_B_fraction": float(np.mean([x["kappa_A_gt_B"] for x in ab])),
            "degradation_A_gt_B_fraction": float(np.mean([x["degradation_A_gt_B"] for x in ab])),
            "rows": ab}}
    with open(out / "MECHANISM_SYNTHETIC_CLASSES.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    # Tail permutation and equal-logit interventions (orientation regime: aligned).
    perm = []
    margins = np.array([1.2, 1.0, 0.8, 0.6, 0.4])
    base = tail_recall(margins)
    for _ in range(5):
        p = rng.permutation(margins)
        perm.append({"recall_before": base, "recall_after_permutation": tail_recall(p),
                     "delta": abs(base - tail_recall(p))})
    eq = []
    for _ in range(5):
        g1 = class_geometry(rng, rho=0.9999, align=0.9, n=25, task_mode="aligned")
        g2 = class_geometry(rng, rho=0.99, align=0.3, n=25, task_mode="aligned")
        eq.append({"e_norm_ratio": g1["e_norm"] / max(g2["e_norm"], 1e-30),
                   "cos_gap": abs(g1["cos_e_grad"] - g2["cos_e_grad"]),
                   "margin_gap": abs(g1["margin_change"] - g2["margin_change"])})
    ratios = [r["bound_ratio_fro"] for r in all_rows]
    result = {
        "status": "EXECUTED",
        "kind": "MECHANISM_SYNTHETIC_SPD",
        "scope": "synthetic SPD readout geometry; no PlantSPD data, no HELD, CPU only",
        "held_reads": 0,
        "registered_config": {
            "seed": args.seed, "repeats": args.repeats,
            "n_grid": [5, 10, 25, 100], "rho_grid": [0.95, 0.99, 0.999, 0.9999],
            "align_grid": [0.3, 0.9], "rank": 8, "dim": 64,
            "task_modes": ["random", "aligned"],
            "mechanism_model": "e_c = J_c^T D_c dW_c; margin change = <e_c, task direction>",
            "claim_boundary": "synthetic mechanism evidence only; A_c/sigma_c remain proxies"},
        "execution_manifest": {
            "n_rows": len(all_rows),
            "bound_ratio_fro": {"median": float(np.median(ratios)),
                                "p95": float(np.percentile(ratios, 95)),
                                "min": float(np.min(ratios)), "max": float(np.max(ratios))},
            "bound_type": "Frobenius / max-entry Schur upper bounds; ratio << 1 means loose bound",
            "regimes": regimes,
            "tail_permutation": {"delta_mean": float(np.mean([x["delta"] for x in perm])), "rows": perm},
            "equal_logit_different_gradient": {"rows": eq,
                                               "interpretation": "forward margin equality does not imply "
                                                                 "equal feedback gradient"},
            "mean_fit_recall_drop": float(np.mean([r["fit_recall_before"] - r["fit_recall_after"] for r in all_rows])),
            "mean_held_proxy_recall_drop": float(np.mean([r["fit_recall_before"] - r["held_recall_after"] for r in all_rows])),
        },
        "coverage_status": {"status": "synthetic_only",
                            "note": "not a PlantSPD, HELD, SOTA or generalisation result"},
    }
    write_json(out / "MECHANISM_SYNTHETIC_RESULT.json", result)
    md = ["# Synthetic SPD mechanism benchmark", "",
          f"- rows: {len(all_rows)}",
          f"- bound/observed (Frobenius) median: {np.median(ratios):.6f} (very loose)",
          "", "## M3: does kappa predict later degradation?", "",
          "| regime | Spearman(kappa, degradation) | partial(kappa | n,loss,grad) | Spearman(1/n, degradation) | A/B consistent |",
          "|---|---|---|---|---|"]
    for mode, d in regimes.items():
        md.append(f"| {mode} | {d['m3']['spearman_kappa_degradation']:+.4f} | "
                  f"{d['m3']['partial_spearman_kappa_given_n_loss_grad']:+.4f} | "
                  f"{d['m3']['spearman_inv_n_degradation']:+.4f} | "
                  f"{d['counterfactual_AB']['consistent_fraction']:.3f} |")
    md += ["", "Status: EXECUTED (synthetic only). Not a PlantSPD/HELD/SOTA result."]
    (out / "MECHANISM_SYNTHETIC_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(json.dumps({"status": "EXECUTED", "rows": len(all_rows),
                      "bound_median": float(np.median(ratios)),
                      "regimes": {k: {"spearman_kappa_deg": v["m3"]["spearman_kappa_degradation"],
                                      "partial_kappa": v["m3"]["partial_spearman_kappa_given_n_loss_grad"],
                                      "ab_consistent": v["counterfactual_AB"]["consistent_fraction"]}
                                  for k, v in regimes.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
