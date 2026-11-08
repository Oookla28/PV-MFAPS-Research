#!/usr/bin/env python3
"""Minimal FIT-only Feedback Trust-Region Refinement runner.

This is a new runner. It does not read test/HELD and does not modify TRR or
FD-SDPA results. It is intended for a one-batch numerical smoke before a
registered multi-seed run.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cifar100_lt_runner import (CifarLT, CifarResNet, ensure_data, make_lt_indices,
                                read_batch, seed_all)
from ftrr_policy import cone_project_alpha, paired_eb_radius, noise_floor
from ftrr_pooled import admit_pooled_gain


def stratified_indices(labels: np.ndarray, fraction: float) -> List[int]:
    out: List[int] = []
    for c in np.unique(labels):
        cls = np.flatnonzero(labels == c)
        out.extend(cls[:max(1, int(math.ceil(len(cls) * fraction)))].tolist())
    return out


def flat_grads(grads) -> torch.Tensor:
    return torch.cat([g.detach().flatten() for g in grads if g is not None])


def nested_heads(model, feats_det, y, prior, max_k, inner_lr):
    W0 = model.fc.weight.detach().clone()
    b0 = model.fc.bias.detach().clone()
    seq = [(W0, b0)]
    for _ in range(max_k):
        W, b = seq[-1]
        W = W.detach().requires_grad_(True)
        b = b.detach().requires_grad_(True)
        loss = F.cross_entropy(feats_det @ W.T + b + prior, y)
        gW, gb = torch.autograd.grad(loss, (W, b))
        seq.append(((W - inner_lr * gW).detach(),
                    (b - inner_lr * gb).detach()))
    return seq


def calibration(model, loader, counts, prior, device, args):
    batches = list(loader)
    if not batches:
        raise RuntimeError("empty FIT calibration loader")
    xb = torch.cat([b[0] for b in batches], dim=0).to(device)
    yb = torch.cat([b[1] for b in batches], dim=0).to(device)
    if args.max_calibration_samples > 0 and xb.shape[0] > args.max_calibration_samples:
        # The stratified subset is class ordered. Prefix truncation would keep
        # only head classes. Round-robin selection preserves class coverage.
        labels = yb.detach().cpu().numpy()
        by_class = {int(c): np.flatnonzero(labels == c).tolist()
                    for c in np.unique(labels)}
        selected = []
        depth = 0
        classes = sorted(by_class)
        while len(selected) < args.max_calibration_samples:
            progressed = False
            for c in classes:
                rows = by_class[c]
                if depth < len(rows):
                    selected.append(rows[depth])
                    progressed = True
                    if len(selected) >= args.max_calibration_samples:
                        break
            if not progressed:
                break
            depth += 1
        idx = torch.as_tensor(selected, dtype=torch.long)
        xb, yb = xb[idx], yb[idx]
    model.eval()
    feats = model.features(xb)
    fd = feats.detach()
    seq = nested_heads(model, fd, yb, prior, args.max_k, args.inner_lr)
    # Feedback is measured in encoder parameter space. The readout parameters
    # are deliberately excluded because W_k is a detached nested copy.
    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith('fc.')]
    native = F.cross_entropy(model.fc(feats) + prior, yb)
    g0 = flat_grads(torch.autograd.grad(native, params, retain_graph=True,
                                        allow_unused=True))
    rows = []
    pooled_certs = []
    chosen = np.zeros(len(counts), dtype=np.int64)
    for k in range(1, args.max_k + 1):
        Wk, bk = seq[k]
        loss_k = F.cross_entropy(feats @ Wk.T + bk + prior, yb)
        gk = flat_grads(torch.autograd.grad(loss_k, params, retain_graph=True,
                                            allow_unused=True))
        alpha, cos, margin = cone_project_alpha(g0.cpu().numpy(), gk.cpu().numpy(),
                                                args.xi, args.alpha_max)
        per0 = F.cross_entropy(model.fc(feats.detach()) + prior, yb,
                               reduction='none').detach().cpu().numpy()
        perk = F.cross_entropy(fd @ Wk.T + bk + prior, yb,
                               reduction='none').detach().cpu().numpy()
        d = np.clip(per0 - perk, -args.clip_bound, args.clip_bound)
        pooled = admit_pooled_gain(
            d, delta=args.delta, bound=args.clip_bound,
            cost=args.cost_per_step * k, margin=args.margin,
            simultaneous=args.max_k)
        pooled_certs.append({'k': int(k), **pooled})
        radius = paired_eb_radius(d, args.delta, args.clip_bound,
                                   simultaneous=len(counts) * args.max_k)
        for c in range(len(counts)):
            m = yb.detach().cpu().numpy() == c
            if int(m.sum()) >= 2:
                dc = d[m]
                rc = paired_eb_radius(dc, args.delta, args.clip_bound,
                                       simultaneous=len(counts) * args.max_k)
                gain = float(dc.mean())
                class_admit = bool(
                    gain - rc >= args.cost_per_step * k + args.margin)
                pooled_admit = bool(
                    args.certificate_mode == 'off' or
                    (args.certificate_mode == 'pooled' and pooled['admitted']))
                admitted = bool(margin >= 0 and
                                (pooled_admit if args.certificate_mode in ('pooled', 'off')
                                 else class_admit))
                rows.append({'class': int(c), 'k': int(k), 'gain': gain,
                             'radius': float(rc), 'alpha': alpha, 'cosine': cos,
                             'admitted': admitted})
                alloc_gain = (gain - rc if args.certificate_mode == 'classwise'
                              else gain)
                if admitted and alloc_gain > args.cost_per_step * k and k > chosen[c]:
                    chosen[c] = k
        if not rows:
            rows.append({'class': -1, 'k': int(k), 'gain': float(d.mean()),
                         'radius': radius, 'alpha': alpha, 'cosine': cos,
                         'admitted': False})
    # Enforce an average finite-action budget by retaining highest admitted gains.
    budget = int(args.budget_fraction * len(counts))
    active = np.flatnonzero(chosen > 0)
    if len(active) > budget:
        scores = {c: max((r['gain'] - r['radius'] for r in rows
                          if r['class'] == int(c) and r['admitted']), default=-1e9)
                  for c in active}
        if args.allocation_mode == 'tail':
            keep = sorted(active, key=lambda c: (int(counts[int(c)]),
                                                 -scores[int(c)]))[:budget]
        else:
            keep = sorted(active, key=lambda c: scores[int(c)], reverse=True)[:budget]
        chosen[:] = 0
        chosen[keep] = 1
    return chosen, {'rows': rows, 'pooled_certificates': pooled_certs,
                    'certificate_mode': args.certificate_mode,
                    'calibration_n': int(len(yb)),
                    'calibration_class_count': int(torch.unique(yb).numel()),
                    'calibration_missing_classes': sorted(set(range(len(counts))) - set(int(v) for v in torch.unique(yb).detach().cpu().tolist())),
                    'noise_floor': noise_floor(args.inner_lr, 0.25,
                                               max(len(yb), 2), args.rho),
                    'held_reads': 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--stages', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--head-lr', type=float, default=.05)
    ap.add_argument('--head-update-mode', choices=['native_step', 'copy_refined'],
                    default='native_step')
    ap.add_argument('--head-update-k', type=int, default=1)
    ap.add_argument('--limit-batches', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--calibration-fraction', type=float, default=.1)
    ap.add_argument('--max-k', type=int, default=2)
    ap.add_argument('--inner-lr', type=float, default=.05)
    ap.add_argument('--xi', type=float, default=.2)
    ap.add_argument('--alpha-max', type=float, default=.5)
    ap.add_argument('--clip-bound', type=float, default=10.)
    ap.add_argument('--delta', type=float, default=.05)
    ap.add_argument('--cost-per-step', type=float, default=.0)
    ap.add_argument('--margin', type=float, default=.0)
    ap.add_argument('--budget-fraction', type=float, default=.2)
    ap.add_argument('--certificate-mode', choices=['classwise', 'pooled', 'off'],
                    default='classwise')
    ap.add_argument('--allocation-mode', choices=['score', 'tail'], default='score')
    ap.add_argument('--rho', type=float, default=.9)
    ap.add_argument('--max-calibration-samples', type=int, default=128)
    ap.add_argument('--synthetic', action='store_true')
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.synthetic:
        x = np.random.default_rng(args.seed).integers(0, 255, (600, 32, 32, 3), dtype=np.uint8)
        y = np.tile(np.arange(10), 60).astype(np.int64)
        counts = np.bincount(y)
    else:
        base = ensure_data(Path(args.data_root)); raw_x, raw_y = read_batch(base / 'train')
        idx, counts = make_lt_indices(raw_y, 100, args.seed); x, y = raw_x[idx], raw_y[idx]
    train = CifarLT(x, y, True, args.seed); ev = CifarLT(x, y, False, args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    cal = Subset(ev, stratified_indices(y, args.calibration_fraction))
    cal_loader = DataLoader(cal, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CifarResNet(len(counts)).to(device)
    prior = torch.as_tensor(counts / counts.sum(), device=device, dtype=torch.float32).log()
    enc = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
    opt = torch.optim.SGD(enc, lr=.1, momentum=.9, weight_decay=5e-4)
    head_opt = torch.optim.SGD(model.fc.parameters(), lr=args.head_lr,
                               momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(head_opt, args.epochs)
    k = np.zeros(len(counts), dtype=np.int64); stages=[]; start=time.time()
    base_epochs, remainder = divmod(args.epochs, max(args.stages, 1))
    stage_lengths = [base_epochs + (1 if i < remainder else 0)
                     for i in range(args.stages)]
    for stage in range(args.stages):
        k, diag = calibration(model, cal_loader, counts, prior, device, args)
        stages.append({'stage': stage, 'action_hist': {str(v): int((k == v).sum())
                                                         for v in np.unique(k)},
                       'active_class_ids': np.flatnonzero(k > 0).astype(int).tolist(),
                       'active_class_counts': counts[np.flatnonzero(k > 0)].astype(int).tolist(), **diag})
        for _ in range(stage_lengths[stage]):
            for bi, (xb, yb) in enumerate(loader):
                if args.limit_batches > 0 and bi >= args.limit_batches:
                    break
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                feats = model.features(xb); fd = feats.detach()
                head_k = max(1, min(int(args.head_update_k), int(args.max_k)))
                seq = nested_heads(model, fd, yb, prior,
                                   max(int(max(k)), head_k), args.inner_lr)
                loss = feats.new_zeros(())
                for kk in np.unique(k):
                    m = torch.as_tensor(k, device=device)[yb] == int(kk)
                    if bool(m.any()):
                        Wk, bk = seq[int(kk)]
                        loss = loss + F.cross_entropy(
                            feats[m] @ Wk.T + bk + prior, yb[m]) * m.float().mean()
                loss.backward(); opt.step()
                if args.head_update_mode == 'copy_refined':
                    with torch.no_grad():
                        W_head, b_head = seq[head_k]
                        model.fc.weight.copy_(W_head)
                        model.fc.bias.copy_(b_head)
                else:
                    head_opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        head_feats = model.features(xb).detach()
                    head_loss = F.cross_entropy(model.fc(head_feats) + prior, yb)
                    head_loss.backward(); head_opt.step()
            sched.step()
            if args.head_update_mode == 'native_step':
                head_sched.step()
    out = {'status':'EXECUTED','kind':'FTRR_FIT_ONLY_SMOKE','held_reads':0,
           'epochs':args.epochs,'stages':args.stages,
           'stage_lengths':stage_lengths,'seed':args.seed,
           'action_histogram':{str(v):int((k==v).sum()) for v in np.unique(k)},
           'stage_history':stages,'wall_seconds':time.time()-start}
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'counts':counts.tolist(),'held_reads':0}, root/'ftrr_seed.pt')
    (root/'ftrr_smoke.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__': main()
