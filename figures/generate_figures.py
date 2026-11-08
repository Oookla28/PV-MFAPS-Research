#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)

c100 = json.loads((ROOT / "artifacts/results/TRR_AUDITED_PREFIX10_VS_LA_SUMMARY.json").read_text())
c10 = json.loads((ROOT / "artifacts/results/TRR_AUDITED_C10_PREFIX10_VS_LA_SUMMARY.json").read_text())
h50 = json.loads((ROOT / "artifacts/results/TRR_AUDITED_HORIZON50_SUMMARY.json").read_text())
d100 = json.loads((ROOT / "artifacts/results/TRR_AUDITED_PREFIX10_PAIRED_SEED_DELTAS.json").read_text())

datasets = ["CIFAR-100-LT\n30 ep", "CIFAR-10-LT\n30 ep"]
macro = [c100["rows"][0]["held_macro_mean"], c10["mean_std"]["prefix_macro"]["mean"]]
macro_la = [c100["rows"][1]["held_macro_mean"], c10["mean_std"]["la_macro"]["mean"]]
tail = [c100["rows"][0]["held_tail10_mean"], c10["mean_std"]["prefix_tail10"]["mean"]]
tail_la = [c100["rows"][1]["held_tail10_mean"], c10["mean_std"]["la_tail10"]["mean"]]

x = np.arange(len(datasets))
w = 0.18
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.bar(x - 1.5*w, macro_la, w, label="LA Macro", color="#9ecae1")
ax.bar(x - .5*w, macro, w, label="Stage-TRR Macro", color="#2171b5")
ax.bar(x + .5*w, tail_la, w, label="LA Tail-10%", color="#fdae6b")
ax.bar(x + 1.5*w, tail, w, label="Stage-TRR Tail-10%", color="#e6550d")
ax.set_xticks(x, datasets)
ax.set_ylim(0, 0.9)
ax.set_ylabel("F1 / recall")
ax.legend(frameon=False, ncol=2, loc="upper left")
save(fig, "stage_trr_cross_dataset.pdf")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
horizon = ["C100 30 ep", "C100 50 ep"]
dm = [c100["comparison"]["macro_delta"], h50["mean_std"]["delta_macro"]["mean"]]
dt = [c100["comparison"]["tail_delta"], h50["mean_std"]["delta_tail10"]["mean"]]
x = np.arange(2)
ax.axhline(0, color="black", linewidth=.8)
ax.plot(x, dm, marker="o", linewidth=2, label="Macro-F1", color="#2171b5")
ax.plot(x, dt, marker="s", linewidth=2, label="Tail-10%", color="#e6550d")
ax.set_xticks(x, horizon)
ax.set_ylabel("Stage-TRR minus LA")
ax.legend(frameon=False)
save(fig, "stage_trr_horizon_boundary.pdf")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
seeds = [str(r["seed"]) for r in d100["rows"]]
dm = [r["delta_macro"] for r in d100["rows"]]
dt = [r["delta_tail10"] for r in d100["rows"]]
x = np.arange(len(seeds))
ax.axhline(0, color="black", linewidth=.8)
ax.scatter(x - .08, dm, s=42, label="Macro-F1", color="#2171b5")
ax.scatter(x + .08, dt, s=42, label="Tail-10%", color="#e6550d")
ax.set_xticks(x, seeds)
ax.set_xlabel("Training seed")
ax.set_ylabel("Paired gain")
ax.legend(frameon=False)
save(fig, "stage_trr_paired_seed_deltas.pdf")

fused = json.loads((OUT / "fused_system_data.json").read_text())
fig, ax = plt.subplots(figsize=(5.2, 3.0))
labels = ["Legacy", "Fused"]
fwd = [fused["legacy"]["physical_forwards"], fused["fused"]["physical_forwards"]]
wall = [fused["legacy"]["cuda_seconds"], fused["fused"]["cuda_seconds"]]
x = np.arange(2)
ax.bar(x - .18, fwd, .36, label="Physical encoder forwards", color="#756bb1")
ax2 = ax.twinx()
ax2.bar(x + .18, wall, .36, label="CUDA wall-clock (s)", color="#31a354")
ax.set_xticks(x, labels)
ax.set_ylabel("Encoder forwards")
ax2.set_ylabel("CUDA seconds")
ax.set_ylim(0, max(fwd) * 1.25)
ax2.set_ylim(0, max(wall) * 1.25)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center")
save(fig, "fused_system_equivalence.pdf")
