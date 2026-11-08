#!/usr/bin/env python3
"""CIFAR-100-LT preflight without test/HELD access.

The preflight verifies the official CIFAR-100 train archive, the IR=100
construction, class-count contract, model output shape, and all baseline loss
paths. It never opens the test batch and records ``held_reads=0``.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cifar100_lt_runner import (
    CifarLT,
    CifarResNet,
    ensure_data,
    loss_for,
    make_lt_indices,
    read_batch,
    seed_all,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--imbalance-ratio", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--augmentation", choices=["reflect", "zero"], default="zero")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.imbalance_ratio <= 1:
        raise ValueError("imbalance ratio must be greater than one")
    seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    base = ensure_data(Path(args.data_root))
    train_x, train_y = read_batch(base / "train")
    if train_x.shape != (50000, 32, 32, 3):
        raise ValueError(f"unexpected CIFAR-100 train shape: {train_x.shape}")
    if int(train_y.min()) != 0 or int(train_y.max()) != 99:
        raise ValueError("CIFAR-100 fine labels must span 0..99")

    selected, counts = make_lt_indices(train_y, args.imbalance_ratio, args.seed)
    expected_min = int(round(500 * args.imbalance_ratio ** (-99 / 99)))
    expected_max = 500
    if len(counts) != 100 or int(counts.min()) != expected_min or int(counts.max()) != expected_max:
        raise ValueError(
            f"unexpected IR={args.imbalance_ratio} counts: classes={len(counts)}, "
            f"min={counts.min()}, max={counts.max()}"
        )

    dataset = CifarLT(train_x[selected], train_y[selected], True, args.seed, args.augmentation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    model = CifarResNet(100, depth=args.depth, width=args.width).to(device)
    model.eval()

    loss_rows = []
    for arm in ("erm", "balanced_softmax", "logit_adjustment", "ldam_drw"):
        model.zero_grad(set_to_none=True)
        logits = model(images.to(device))
        if tuple(logits.shape) != (len(labels), 100):
            raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
        loss = loss_for(
            arm,
            logits,
            labels.to(device),
            counts,
            epoch=0,
            device=device,
            ldam_switch_epoch=100,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss for {arm}")
        loss.backward()
        loss_rows.append({"arm": arm, "loss": float(loss.detach().cpu())})

    result = {
        "status": "EXECUTED",
        "dataset": "CIFAR100-LT",
        "source": "official CIFAR-100 python archive, train split only",
        "imbalance_ratio": args.imbalance_ratio,
        "class_count": int(len(counts)),
        "train_n_full": int(len(train_y)),
        "train_n_long_tail": int(len(selected)),
        "counts_min": int(counts.min()),
        "counts_max": int(counts.max()),
        "counts_sum": int(counts.sum()),
        "count_formula": "round(500 * IR ** (-class_id / 99))",
        "seed": args.seed,
        "device": str(device),
        "model_depth": model.model_depth,
        "model_width": model.model_width,
        "augmentation": args.augmentation,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "loss_smoke": loss_rows,
        "held_reads": 0,
        "test_batch_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
