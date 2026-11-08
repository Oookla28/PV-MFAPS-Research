#!/usr/bin/env python3
"""FD-SDPA metrics, provenance helpers and dependency-free SVG figures.

Local macOS has no matplotlib, so figures are emitted as plain SVG. That keeps
the offline validator runnable anywhere and matches the repository's existing
``.svg`` figure convention.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REQUIRED_KEYS = (
    "status", "code_sha256", "config_sha256", "data_sha256", "held_reads",
    "calibration_split", "action_sequence", "action_histogram",
    "feedback_mismatch", "safe_gate_statistics", "actual_flops",
    "cuda_wall_seconds", "physical_encoder_forwards", "logical_action_count",
)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_tree(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(Path(p).name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def build_result_record(**kwargs) -> Dict[str, object]:
    missing = [k for k in REQUIRED_KEYS if k not in kwargs]
    if missing:
        raise ValueError(f"result record missing required keys: {missing}")
    return {k: kwargs[k] for k in REQUIRED_KEYS} | {
        k: v for k, v in kwargs.items() if k not in REQUIRED_KEYS
    }


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return p


# --------------------------------------------------------------------------- #
# action / policy metrics
# --------------------------------------------------------------------------- #
def action_histogram(k: Sequence[int]) -> Dict[str, int]:
    vals, counts = np.unique(np.asarray(k, dtype=int), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def active_classes(k: Sequence[int]) -> List[int]:
    arr = np.asarray(k, dtype=int)
    return [int(i) for i in np.flatnonzero(arr > 0)]


def switch_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def oscillation_metrics(k_by_stage: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Count action changes and detect 0 -> positive -> 0 reversals per class."""
    mats = [np.asarray(k, dtype=int) for k in k_by_stage]
    if not mats:
        return {"switches": [], "total_switches": 0, "oscillating_classes": 0}
    switches = [switch_distance(mats[i - 1], mats[i]) for i in range(1, len(mats))]
    C = mats[0].size
    osc = 0
    for c in range(C):
        traj = [int(m[c]) for m in mats]
        positive = [v > 0 for v in traj]
        flips = sum(1 for i in range(1, len(positive)) if positive[i] != positive[i - 1])
        if flips >= 2:
            osc += 1
    return {"switches": switches, "total_switches": int(sum(switches)),
            "oscillating_classes": int(osc),
            "max_single_stage_switch": int(max(switches)) if switches else 0}


def safe_gate_statistics(gates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    n = len(gates)
    act = [g for g in gates if g.get("activated")]
    margins = [float(g.get("margin", 0.0)) for g in gates]
    return {
        "evaluated": int(n),
        "activated": int(len(act)),
        "rejected": int(n - len(act)),
        "activation_rate": float(len(act) / n) if n else 0.0,
        "median_margin": float(np.median(margins)) if margins else 0.0,
        "min_margin": float(np.min(margins)) if margins else 0.0,
        "max_margin": float(np.max(margins)) if margins else 0.0,
        "violations_of_safety_condition": int(
            sum(1 for g in gates if g.get("activated") and float(g.get("margin", 0)) <= 0)),
    }


def macro_tail_recall(per_class_recall: Sequence[float], counts: Sequence[int],
                      tail_fraction: float = 0.1) -> Dict[str, float]:
    r = np.asarray(per_class_recall, dtype=float)
    c = np.asarray(counts, dtype=int)
    C = r.size
    k = max(1, int(math.ceil(tail_fraction * C)))
    tail = np.argsort(c)[:k]
    return {"macro_recall": float(r.mean()),
            "tail_recall": float(r[tail].mean()),
            "tail_classes": [int(x) for x in tail],
            "n_classes": int(C)}


def paired_bootstrap_ci(diff: Sequence[float], n_boot: int = 2000,
                        seed: int = 20261108) -> Dict[str, float]:
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        raise ValueError("empty diff")
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "n": int(d.size)}


def tail_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


# --------------------------------------------------------------------------- #
# SVG figures (no third-party dependency)
# --------------------------------------------------------------------------- #
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def _svg_frame(width: int, height: int, title: str, xlabel: str, ylabel: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15" '
        f'font-weight="bold">{_esc(title)}</text>',
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12">'
        f'{_esc(xlabel)}</text>',
        f'<text x="14" y="{height/2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {height/2})">{_esc(ylabel)}</text>',
    ]


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi - lo < 1e-12:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def svg_rho_vs_threshold(stages: Sequence[str], median_rho: Sequence[float],
                         threshold: float, path: str | Path,
                         width: int = 720, height: int = 420) -> Path:
    L, R, T, B = 70, 30, 50, 60
    xs = list(range(len(stages)))
    lo = min(min(median_rho), threshold) - 1e-5
    hi = max(max(median_rho), threshold) + 1e-5
    out = _svg_frame(width, height, "median rho per stage vs activation threshold",
                     "stage", "median rho")
    x = lambda i: _scale(i, 0, max(len(stages) - 1, 1), L, width - R)
    y = lambda v: _scale(v, lo, hi, height - B, T)
    out.append(f'<rect x="{L}" y="{T}" width="{width-L-R}" height="{height-T-B}" '
               f'fill="none" stroke="#888"/>')
    out.append(f'<line x1="{L}" y1="{y(threshold):.2f}" x2="{width-R}" '
               f'y2="{y(threshold):.2f}" stroke="#d62728" stroke-dasharray="6,4"/>')
    out.append(f'<text x="{L+6}" y="{y(threshold)-6:.2f}" font-size="11" fill="#d62728">'
               f'threshold {threshold:.6f}</text>')
    pts = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in zip(xs, median_rho))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{_PALETTE[0]}" stroke-width="2"/>')
    for i, v in zip(xs, median_rho):
        out.append(f'<circle cx="{x(i):.2f}" cy="{y(v):.2f}" r="4" fill="{_PALETTE[0]}"/>')
        out.append(f'<text x="{x(i):.2f}" y="{height-B+18}" text-anchor="middle" '
                   f'font-size="11">{_esc(stages[i])}</text>')
    out.append("</svg>")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def svg_action_histogram_by_stage(stages: Sequence[str],
                                  histograms: Sequence[Dict[str, int]],
                                  path: str | Path,
                                  width: int = 760, height: int = 420) -> Path:
    keys = sorted({k for h in histograms for k in h}, key=lambda s: int(s))
    L, R, T, B = 70, 30, 55, 60
    maxv = max([v for h in histograms for v in h.values()] + [1])
    out = _svg_frame(width, height, "action histogram by stage", "k", "class count")
    gw = (width - L - R) / max(len(stages), 1)
    bw = gw / (len(keys) + 1)
    y = lambda v: _scale(v, 0, maxv, height - B, T)
    for si, (s, h) in enumerate(zip(stages, histograms)):
        x0 = L + si * gw
        for ki, k in enumerate(keys):
            v = h.get(k, 0)
            xb = x0 + bw * (ki + 0.5)
            out.append(f'<rect x="{xb:.2f}" y="{y(v):.2f}" width="{bw*0.8:.2f}" '
                       f'height="{height-B-y(v):.2f}" fill="{_PALETTE[ki % len(_PALETTE)]}"/>')
            if v:
                out.append(f'<text x="{xb+bw*0.4:.2f}" y="{y(v)-3:.2f}" '
                           f'text-anchor="middle" font-size="9">{v}</text>')
        out.append(f'<text x="{x0+gw/2:.2f}" y="{height-B+18}" text-anchor="middle" '
                   f'font-size="11">{_esc(s)}</text>')
    for ki, k in enumerate(keys):
        lx = L + ki * 90
        out.append(f'<rect x="{lx}" y="{height-28}" width="12" height="12" '
                   f'fill="{_PALETTE[ki % len(_PALETTE)]}"/>')
        out.append(f'<text x="{lx+17}" y="{height-18}" font-size="11">k={k}</text>')
    out.append("</svg>")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def svg_cumulative_regret(stages: Sequence[str], cumulative: Sequence[float],
                          path: str | Path, width: int = 720, height: int = 420) -> Path:
    L, R, T, B = 75, 30, 50, 60
    lo, hi = 0.0, max(max(cumulative), 1e-9)
    out = _svg_frame(width, height, "cumulative surrogate regret vs static policy",
                     "stage", "cumulative regret")
    x = lambda i: _scale(i, 0, max(len(stages) - 1, 1), L, width - R)
    y = lambda v: _scale(v, lo, hi, height - B, T)
    out.append(f'<rect x="{L}" y="{T}" width="{width-L-R}" height="{height-T-B}" '
               f'fill="none" stroke="#888"/>')
    pts = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in enumerate(cumulative))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{_PALETTE[1]}" stroke-width="2"/>')
    for i, v in enumerate(cumulative):
        out.append(f'<circle cx="{x(i):.2f}" cy="{y(v):.2f}" r="3.5" fill="{_PALETTE[1]}"/>')
        out.append(f'<text x="{x(i):.2f}" y="{height-B+18}" text-anchor="middle" '
                   f'font-size="11">{_esc(stages[i])}</text>')
        out.append(f'<text x="{x(i):.2f}" y="{y(v)-8:.2f}" text-anchor="middle" '
                   f'font-size="9">{v:.4g}</text>')
    out.append("</svg>")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p
