#!/usr/bin/env python3
"""Plot the PV-MFAPS risk envelope for one completed FIT arm.

The plot is deliberately marked as an *offline proxy*: current artifacts expose
A, rho, sigma and support, but do not provide a proved per-action generalisation
bound.  Stability uses a standard constant-step upper-bound form and EB uses an
empirical-Bernstein proxy.  The script never reads HELD.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np


def compute_envelope(
    arm: Dict[str, Any],
    class_index: int | None = None,
    k_grid=(0, 1, 2, 4),
    xi: float = 1.01,
    eta: float = 0.01,
    smoothness: float = 1.0,
    lipschitz: float = 1.0,
    grad_bound: float = 1.0,
    delta: float = 0.05,
    loss_bound: float = 1.0,
    normalize: bool = True,
) -> Dict[str, Any]:
    summary = arm["class_proxy_summary"]
    A = np.asarray(summary["A"], dtype=float)
    rho = np.asarray(summary["rho"], dtype=float)
    sigma = np.asarray(summary["sigma"], dtype=float)
    support = np.asarray(summary["support"], dtype=float)
    C = len(A)
    if class_index is None:
        class_index = int(np.argmin(support))
    if not (0 <= class_index < C):
        raise ValueError(f"class_index must be in [0, {C})")
    n = max(float(support[class_index]), 1.0)
    r = float(np.clip(rho[class_index], 0.0, 1.0))
    a = max(float(A[class_index]), 0.0)
    sig = max(float(sigma[class_index]), 0.0)
    ks = np.asarray(list(k_grid), dtype=int)
    if np.any(ks < 0) or xi <= 0 or eta < 0 or smoothness < 0:
        raise ValueError("invalid k_grid/xi/eta/smoothness")
    log_term = math.log(max(2.0 * C * max(len(ks), 1) / delta, 1.0000001))
    bias = a * np.power(r, 2 * ks)
    variance = (sig * sig / n) * np.power(xi, ks)
    stability = []
    for k in ks:
        if k == 0 or eta == 0:
            stability.append(0.0)
        else:
            # 2 L G/n * sum_{j=0}^{k-1} eta (1 + beta eta)^(k-1-j)
            gsum = sum(eta * (1.0 + smoothness * eta) ** j for j in range(int(k)))
            stability.append(2.0 * lipschitz * grad_bound * gsum / n)
    stability = np.asarray(stability, dtype=float)
    eb = np.sqrt(2.0 * variance * log_term / n) + (
        7.0 * loss_bound * log_term / max(3.0 * (n - 1.0), 1.0)
    )
    total = bias + stability + eb
    if normalize:
        scale = max(float(total[0]), 1e-12)
        bias, stability, eb, total = [x / scale for x in (bias, stability, eb, total)]
    out = {
        "status": "UNVERIFIED_OFFLINE_PROXY",
        "held_reads": 0,
        "source_arm": arm.get("state_path") or arm.get("checkpoint"),
        "class_index": int(class_index),
        "support_n": n,
        "rho": r,
        "A": a,
        "sigma": sig,
        "k_grid": [int(k) for k in ks],
        "xi": float(xi),
        "eta": float(eta),
        "smoothness": float(smoothness),
        "lipschitz": float(lipschitz),
        "grad_bound": float(grad_bound),
        "delta": float(delta),
        "loss_bound": float(loss_bound),
        "normalization": "divide by total(k=0)" if normalize else "none",
        "components": {
            "bias": bias.tolist(),
            "stability_proxy": stability.tolist(),
            "empirical_bernstein_proxy": eb.tolist(),
            "risk_envelope_proxy": total.tolist(),
        },
        "warning": "No claim of a proved generalisation bound; stability and EB constants must be justified on FIT before use in a paper.",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--class-index", type=int, default=None)
    ap.add_argument("--xi", type=float, default=1.01)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--smoothness", type=float, default=1.0)
    ap.add_argument("--lipschitz", type=float, default=1.0)
    ap.add_argument("--grad-bound", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--loss-bound", type=float, default=1.0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()
    arm = json.loads(Path(args.arm_json).read_text(encoding="utf8"))
    result = compute_envelope(
        arm,
        class_index=args.class_index,
        xi=args.xi,
        eta=args.eta,
        smoothness=args.smoothness,
        lipschitz=args.lipschitz,
        grad_bound=args.grad_bound,
        delta=args.delta,
        loss_bound=args.loss_bound,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        # Deterministic SVG fallback keeps the artifact usable on bare
        # containers where matplotlib is not installed.
        result["plot_status"] = "EXECUTED_SVG_FALLBACK"
        result["plot_error"] = repr(exc)
        out_png = Path(args.out_png)
        out_svg = out_png.with_suffix(".svg")
        ks = result["k_grid"]
        comp = result["components"]
        names = ["bias", "stability_proxy", "empirical_bernstein_proxy", "risk_envelope_proxy"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        W, H, L, R, T, B = 760, 500, 70, 25, 35, 65
        vals = [v for name in names for v in comp[name]]
        vmax = max(max(vals), 1e-12)
        xmax = max(ks) if ks else 1
        def xy(k, v):
            x = L + (W - L - R) * (float(k) / max(xmax, 1))
            y = H - B - (H - T - B) * (float(v) / vmax)
            return x, y
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>',
                 f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>',
                 f'<text x="{W/2:.1f}" y="25" text-anchor="middle" font-size="16">PV-MFAPS offline risk envelope (proxy)</text>',
                 f'<text x="{W/2:.1f}" y="{H-15}" text-anchor="middle" font-size="13">inner solver steps k</text>',
                 f'<text x="16" y="{H/2:.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 16 {H/2:.1f})">normalized risk</text>']
        for name, color in zip(names, colors):
            pts = ' '.join(f'{xy(k,v)[0]:.1f},{xy(k,v)[1]:.1f}' for k,v in zip(ks, comp[name]))
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for k in ks:
            x, _ = xy(k, 0)
            lines.append(f'<text x="{x:.1f}" y="{H-B+22}" text-anchor="middle" font-size="12">{k}</text>')
        legend_y = T + 18
        for i, (name, color) in enumerate(zip(names, colors)):
            x = L + i * 160
            lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+22}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{x+27}" y="{legend_y+4}" font-size="11">{name}</text>')
        lines.append('</svg>')
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text('\n'.join(lines) + '\n', encoding='utf8')
        result["svg_path"] = str(out_svg)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"status": result["plot_status"], "json": str(out_json), "svg": str(out_svg)}, ensure_ascii=False))
        return
    ks = result["k_grid"]
    comp = result["components"]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(ks, comp["bias"], "o-", label=r"bias $A_c\rho_c^{2k}$")
    plt.plot(ks, comp["stability_proxy"], "s-", label="stability proxy")
    plt.plot(ks, comp["empirical_bernstein_proxy"], "^-", label="EB proxy")
    plt.plot(ks, comp["risk_envelope_proxy"], "D--", linewidth=2.2, label="total envelope proxy")
    plt.xlabel("inner solver steps k")
    plt.ylabel("normalized risk component")
    plt.title(f"PV-MFAPS offline risk envelope — class {result['class_index']} (n={result['support_n']:.0f})")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
    result["plot_status"] = "EXECUTED"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["plot_status"], "json": str(out_json), "png": str(out_png)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
