#!/usr/bin/env python3
"""One-time post-freeze HELD evaluation for audited TRR artifacts."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from cifar100_lt_runner import CifarLT, CifarResNet, ensure_data, read_batch


def eval_one(model, loader, device, counts):
    model.eval()
    per = np.zeros((len(counts), 2), dtype=np.float64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in range(len(counts)):
                m = y == c
                per[c, 0] += int(m.sum())
                per[c, 1] += int(((pred == c) & m).sum())
    rec = per[:, 1] / np.maximum(per[:, 0], 1)
    tail = np.argsort(counts)[:max(1, int(np.ceil(.1 * len(counts))))]
    return {
        "accuracy": float(per[:, 1].sum() / max(per[:, 0].sum(), 1)),
        "macro_recall": float(rec.mean()),
        "tail10_recall": float(rec[tail].mean()),
        "per_class_recall": rec.tolist(),
        "n": int(per[:, 0].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    base = ensure_data(Path(args.data_root))
    test_x, test_y = read_batch(base / "test")
    rows = []
    held_reads = 0
    for path in sorted(run_dir.glob("trr_audited_seed*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "EXECUTED" or row.get("held_reads") != 0:
            raise RuntimeError(f"invalid pre-HELD artifact {path}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu")
        model = CifarResNet(len(state["counts"])).to(device)
        model.load_state_dict(state["model"])
        loader = DataLoader(
            CifarLT(test_x, test_y, False, int(row["seed"])),
            batch_size=int(row.get("batch_size", 128)) * 2,
            shuffle=False,
            num_workers=0,
        )
        metrics = eval_one(model, loader, device, np.asarray(state["counts"], dtype=np.int64))
        row["held_metrics"] = metrics
        row["held_reads"] = metrics["n"]
        path.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        held_reads += metrics["n"]
    groups = {}
    for row in rows:
        key = row["arm"]
        groups.setdefault(key, []).append(row)
    summary = []
    for arm, rs in sorted(groups.items()):
        summary.append({
            "arm": arm,
            "n": len(rs),
            "held_macro_mean": statistics.fmean(r["held_metrics"]["macro_recall"] for r in rs),
            "held_tail10_mean": statistics.fmean(r["held_metrics"]["tail10_recall"] for r in rs),
            "held_accuracy_mean": statistics.fmean(r["held_metrics"]["accuracy"] for r in rs),
            "wall_mean": statistics.fmean(r["wall_seconds"] for r in rs),
            "seeds": [r["seed"] for r in rs],
        })
    out = {
        "status": "EXECUTED",
        "scope": "single post-freeze HELD evaluation",
        "held_reads": held_reads,
        "rows": summary,
        "source_json_count": len(rows),
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "rows": len(rows), "held_reads": held_reads}))


if __name__ == "__main__":
    main()
