#!/usr/bin/env python3
"""CPU smoke for the algebraic PC-FA fused-gradient implementation.

This test does not use Plant data and makes no efficacy claim.  It checks that
the fused single-graph correction agrees with the registered native/reference
gradient decomposition on a deterministic toy encoder.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

import tfgps_gpu_lib as lib


class ToyRunner:
    def normalized_batch(self, x, ii, device):
        return x.index_select(0, torch.as_tensor(ii, dtype=torch.long, device=device))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.classifier = nn.Linear(4, 2, bias=True)
        self.method = "raw"

    def forward(self, x):
        h = self.encoder(x)
        return {"dual_logits": self.classifier(h), "covariance": h.reshape(-1, 2, 2)}


def flatten(grads):
    return torch.cat([g.detach().reshape(-1) for g in grads])


def main():
    torch.manual_seed(3407)
    x = torch.randn(4, 4)
    yy = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    ii = np.arange(4, dtype=np.int64)
    masks = [np.asarray([0, 1], dtype=np.int64),
             np.asarray([2, 3], dtype=np.int64)]
    alphas = [0.25, 0.50]
    ks = [1, 2]
    weights = torch.tensor([1.1, 0.9, 1.2, 0.8], dtype=torch.float32)

    old_model = ToyModel()
    fused_model = copy.deepcopy(old_model)
    old_head = nn.Linear(4, 2)
    torch.manual_seed(991)
    nn.init.normal_(old_head.weight, std=0.1)
    nn.init.normal_(old_head.bias, std=0.1)
    fused_head = copy.deepcopy(old_head)
    runner = ToyRunner()

    old_params = [old_model.encoder.weight]
    old_counter = lib.FlopCounter(); old_counter.attach(old_model)
    old_audit = lib.make_audit()
    gn, _ = lib.native_gradient(old_model, runner, x, ii, yy, old_params,
                                torch.device("cpu"), old_counter, old_audit,
                                weights=weights, pass_kind="smoke_native")
    old_parts = list(gn)
    for mask, alpha, k in zip(masks, alphas, ks):
        gi = ii[mask]
        gy = yy[torch.as_tensor(mask)]
        gw = weights[torch.as_tensor(mask)]
        gr, _, _ = lib.reference_gradient(
            old_model, runner, old_head, x, gi, gy, "raw", old_params, k,
            torch.device("cpu"), old_counter, old_audit, weights=gw,
            role="action", denom=len(ii))
        gnc, _ = lib.native_gradient(
            old_model, runner, x, gi, gy, old_params, torch.device("cpu"),
            old_counter, old_audit, weights=gw,
            pass_kind="smoke_group_native", denom=len(ii))
        old_parts = [a + alpha * (b - c) for a, b, c in zip(old_parts, gr, gnc)]
    old_grad = flatten(tuple(old_parts))
    old_counter.close()

    fused_params = [fused_model.encoder.weight]
    fused_counter = lib.FlopCounter(); fused_counter.attach(fused_model)
    fused_audit = lib.make_audit()
    fused_grad, _, meta = lib.pcfa_fused_gradient(
        fused_model, runner, fused_head, x, ii, yy, "raw", fused_params,
        torch.device("cpu"), fused_counter, fused_audit,
        [{"class": 0, "mask": masks[0], "k": ks[0], "alpha": alphas[0]},
         {"class": 1, "mask": masks[1], "k": ks[1], "alpha": alphas[1]}],
        weights=weights, denom=len(ii))
    fused_grad = flatten(fused_grad)
    fused_counter.close()

    abs_err = float(torch.max(torch.abs(old_grad - fused_grad)))
    rel_err = float(torch.linalg.vector_norm(old_grad - fused_grad) /
                    (torch.linalg.vector_norm(old_grad) + 1e-12))
    counts_ok = (
        fused_audit["native_passes"] == 3 and
        fused_audit["reference_passes"] == 2 and
        fused_audit["backward_passes"] == 5 and
        fused_audit["native_group_passes"] == 2 and
        fused_audit["reference_action_passes"] == 2 and
        fused_audit["solver_steps"] == 3 and
        fused_audit["sample_solver_steps"] == 6 and
        fused_audit["physical_encoder_forwards"] == 1 and
        fused_audit["physical_encoder_backwards"] == 1 and
        fused_audit["physical_reference_actions"] == 2 and
        old_audit["physical_encoder_forwards"] == 5 and
        old_audit["physical_encoder_backwards"] == 5 and
        old_audit["physical_reference_actions"] == 2
    )
    ok = abs_err < 2e-6 and rel_err < 2e-6 and len(meta) == 2 and counts_ok
    print({"status": "PASS" if ok else "FAIL", "max_abs_error": abs_err,
           "relative_error": rel_err, "group_count": len(meta),
           "logical_counts_pass": counts_ok})
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
