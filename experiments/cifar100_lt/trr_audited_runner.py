#!/usr/bin/env python3
"""Audited FIT-only TRR runner.

This runner is deliberately separate from the exploratory ``trr_runner.py``.
It implements the slow-head bilevel path with a nested k=1 readout sequence,
an independent FIT calibration split, code/data/config metadata, checkpoint
output, and no test/HELD access.
"""
from __future__ import annotations

import argparse, hashlib, json, math, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cifar100_lt_runner import CifarLT, CifarResNet, ensure_data, make_lt_indices, read_batch, seed_all


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def calibration_split(labels: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)
    cal, train = [], []
    for c in np.unique(labels):
        cls = np.flatnonzero(labels == c).copy()
        rng.shuffle(cls)
        n_cal = max(1, int(math.ceil(len(cls) * fraction)))
        cal.extend(cls[:n_cal].tolist())
        train.extend(cls[n_cal:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(cal, dtype=np.int64), [int(labels[cal].tolist().count(c)) for c in np.unique(labels)]


def nested_head(W: torch.Tensor, b: torch.Tensor, feats_det: torch.Tensor,
                y: torch.Tensor, prior: torch.Tensor, lr: float):
    Wk = W.detach().clone().requires_grad_(True)
    bk = b.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(feats_det @ Wk.T + bk + prior, y)
    gW, gb = torch.autograd.grad(loss, (Wk, bk))
    return (Wk - lr * gW).detach(), (bk - lr * gb).detach()


def evaluate_fit(model, loader, device, counts):
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
    tail = np.argsort(counts)[:max(1, int(math.ceil(.1 * len(counts))))]
    return {'accuracy': float(per[:, 1].sum() / max(per[:, 0].sum(), 1)),
            'macro_recall': float(rec.mean()), 'tail10_recall': float(rec[tail].mean()),
            'per_class_recall': rec.tolist(), 'n': int(per[:, 0].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr-encoder', type=float, default=.1)
    ap.add_argument('--lr-head', type=float, default=.05)
    ap.add_argument('--calibration-fraction', type=float, default=.1)
    ap.add_argument('--trr-epochs', type=int, default=-1,
                    help='number of initial epochs using slow-head TRR. '
                         '0 is LA-only, -1 means TRR for all epochs')
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--limit-batches', type=int, default=0)
    ap.add_argument('--holdout-calibration', action='store_true',
                    help='exclude the diagnostic calibration subset from training. '
                         'Default keeps all FIT samples for fair comparison with full-data baselines.')
    ap.add_argument('--synthetic', action='store_true')
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.synthetic:
        raw_x = np.random.default_rng(args.seed).integers(0, 255, (600, 32, 32, 3), dtype=np.uint8)
        raw_y = np.tile(np.arange(10), 60).astype(np.int64)
        counts = np.bincount(raw_y)
        selected = np.arange(len(raw_y), dtype=np.int64)
    else:
        base = ensure_data(Path(args.data_root))
        full_x, full_y = read_batch(base / 'train')
        selected, counts = make_lt_indices(full_y, 100, args.seed)
        raw_x, raw_y = full_x[selected], full_y[selected]
    train_pos, cal_pos, cal_counts = calibration_split(raw_y, args.calibration_fraction, args.seed)
    train_idx = train_pos if args.holdout_calibration else np.arange(len(raw_y), dtype=np.int64)
    train_ds = CifarLT(raw_x[train_idx], raw_y[train_idx], True, args.seed)
    fit_ds = CifarLT(raw_x[train_idx], raw_y[train_idx], False, args.seed)
    cal_ds = CifarLT(raw_x[cal_pos], raw_y[cal_pos], False, args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    model = CifarResNet(len(counts)).to(device)
    prior = torch.as_tensor(counts / counts.sum(), device=device, dtype=torch.float32).log()
    # One outer optimizer matches the audited reference implementation. During
    # TRR epochs the live head is updated by an explicit inner step and its
    # outer gradients are cleared. During LA epochs the same optimizer updates
    # encoder and head jointly. This avoids silently carrying momentum from a
    # different head-update semantics across the stage boundary.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_encoder, momentum=.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    start = time.time()

    trr_epochs = args.epochs if args.trr_epochs < 0 else min(args.trr_epochs, args.epochs)
    for epoch in range(args.epochs):
        model.train()
        use_trr = epoch < trr_epochs
        for bi, (x, y) in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if not use_trr:
                la_loss = F.cross_entropy(model(x) + prior, y)
                la_loss.backward()
                opt.step()
                continue

            # One train-mode encoder forward supplies both the live head
            # update and the encoder feedback. No duplicate feature forward
            # or BatchNorm state mutation is introduced.
            feats = model.features(x)
            Wk, bk = nested_head(model.fc.weight, model.fc.bias,
                                 feats.detach(), y, prior, args.lr_head)
            with torch.no_grad():
                model.fc.weight.copy_(Wk)
                model.fc.bias.copy_(bk)
            enc_loss = F.cross_entropy(feats @ Wk.T + bk + prior, y)
            enc_loss.backward()
            model.fc.weight.grad = None
            model.fc.bias.grad = None
            opt.step()
        sched.step()

    fit_metrics = evaluate_fit(model, fit_loader, device, counts)
    cal_metrics = evaluate_fit(model, cal_loader, device, counts)
    out_root = Path(args.output); out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = out_root / f'trr_audited_seed{args.seed}.pt'
    torch.save({'model': model.state_dict(), 'counts': counts.tolist(),
                'seed': args.seed, 'epochs': args.epochs, 'held_reads': 0}, checkpoint)
    code_files = [Path(__file__), Path(__file__).with_name('cifar100_lt_runner.py')]
    code_sha = hashlib.sha256(b''.join(p.read_bytes() for p in code_files)).hexdigest()
    data_sha = 'synthetic' if args.synthetic else sha256_file(Path(args.data_root) / 'cifar-100-python' / 'train')
    row = {
        'status': 'EXECUTED', 'kind': 'TRR_AUDITED_FIT_ONLY',
        'arm': ('bilevel_slow_head_k1' if trr_epochs == args.epochs else
                ('stage_trr_then_la' if trr_epochs > 0 else 'logit_adjustment')),
        'seed': args.seed, 'epochs': args.epochs, 'trr_epochs': trr_epochs,
        'batch_size': args.batch_size,
        'schedule': {'trr_feedback_epochs': list(range(trr_epochs)),
                     'la_epochs': list(range(trr_epochs, args.epochs))},
        'held_reads': 0,
        'calibration_split': ('independent_fit_stratified' if args.holdout_calibration
                              else 'stratified_diagnostic_subset_overlapping_fit'),
        'holdout_calibration': bool(args.holdout_calibration),
        'calibration_fraction': args.calibration_fraction,
        'calibration_n': len(cal_ds), 'calibration_class_counts': cal_counts,
        'train_n': len(train_ds), 'fit_metrics': fit_metrics,
        'calibration_metrics': cal_metrics, 'checkpoint': str(checkpoint),
        'code_sha256': code_sha, 'data_sha256': data_sha,
        'lr_encoder': args.lr_encoder, 'lr_head': args.lr_head,
        'stage_policy': ('TRR_all_epochs' if trr_epochs == args.epochs else
                         ('LA_all_epochs' if trr_epochs == 0 else 'TRR_then_joint_LA')),
        'wall_seconds': time.time() - start,
        'boundary': 'FIT-only; test/HELD never loaded',
    }
    (out_root / f'trr_audited_seed{args.seed}.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
