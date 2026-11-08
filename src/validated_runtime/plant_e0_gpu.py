"""E0 readout-vs-feedback diagnostic on an existing Plant_SPD checkpoint.

This is deliberately a diagnostic, not a new paired-training run.  It loads
one frozen checkpoint, fits an affine probe on the same mapped SPD features,
and compares gradients of the native distance head with the probe reference
head on a fixed FIT-only diagnostic subset.  Results are labelled pilot/E0
and never touch HELD data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


import os
ROOT = Path(os.environ.get("PLANT_E01_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "E01")))
SRC = ROOT / "source/runner_v2.py"
BACKBONE = ROOT / "assets/resnet18_torch_only_v1.py"
WEIGHTS = ROOT / "assets/resnet18-f37072fd.pth"
INPUT = ROOT / "input/rgb_views_uint8.npy"
META = ROOT / "input/meta.npz"
SPLITS = ROOT / "history_origin/prepared/SPLITS.npz"
INIT_DIR = ROOT / "initializations"


def import_runner():
    spec = importlib.util.spec_from_file_location("plant_runner_v2_e0", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def affine_logits_from_mapped(z, proto, scales, temperature):
    """Equivalent class-affine logits for the native squared-distance dual head.

    The omitted -||z||^2 term is common to all classes and therefore leaves
    cross-entropy and its parameter gradient unchanged.
    z: [B,2,1024], proto: [2,69,1024], scales: [2]
    """
    # Native dual logits average the two view-specific normalized distances.
    w = 2.0 * proto / scales[:, None, None] / temperature
    b = -(proto.square().sum(-1)) / scales[:, None] / temperature
    branch = torch.einsum("bvd,vcd->bvc", z, w) + b[None]
    return branch.mean(1)


def mapped_features(runner, out, method):
    cov = out["covariance"]
    if method == "log":
        mapped = runner.stable_logm(cov)
    else:
        mapped = cov
    return mapped.reshape(mapped.shape[0], 2, -1)


def load_checkpoint(method: str, seed: int, partition: int, device: str | None = None):
    runner = import_runner()
    init = {k: v for k, v in np.load(INIT_DIR / f"INIT_p{partition}_s{seed}.npz", allow_pickle=False).items()}
    ckpt = ROOT / "runs" / f"pst1r1_p{partition}_s{seed}_{method}" / "FINAL_CHECKPOINT.pt"
    payload = torch.load(ckpt, map_location="cpu")
    model = runner.build_model(
        str(BACKBONE), str(WEIGHTS),
        runner.sha_file(BACKBONE), runner.sha_file(WEIGHTS),
        init["prototype_factors"], init["scales_raw"], init["scales_log"],
        method=method, seed=seed,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if device is not None:
        model = model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return runner, model, ckpt, payload


def load_data(partition: int):
    x = np.load(INPUT, allow_pickle=False, mmap_mode="r")
    with np.load(META, allow_pickle=False) as z:
        labels = np.asarray(z["labels"], dtype=np.int64)
    with np.load(SPLITS, allow_pickle=False) as z:
        fit = np.flatnonzero(z["fit_mask"][partition])
        held = np.flatnonzero(z["held_mask"][partition])
    if len(fit) + len(held) != len(labels):
        raise ValueError("fit/held partition is not a complement")
    return x, labels, fit, held


@torch.no_grad()
def collect_features(runner, model, x, indices, device, batch_size):
    chunks = []
    ys = []
    for start in range(0, len(indices), batch_size):
        ii = np.asarray(indices[start:start + batch_size], dtype=np.int64)
        out = model(runner.normalized_batch(x, ii, device))
        z = mapped_features(runner, out, model.method)
        proto = out["prototype_covariance"]
        if model.method == "log":
            proto = runner.stable_logm(proto)
        chunks.append(z.detach().cpu())
        ys.append(torch.as_tensor(labels_global[ii], dtype=torch.long))
    return torch.cat(chunks, 0), torch.cat(ys, 0)


REFERENCE_L2 = 1.17e-8


def fit_affine(z, y, n_classes, init_weight=None, init_bias=None,
               steps=100, lr=0.2, weight_decay=REFERENCE_L2):
    # LBFGS on the fixed feature table provides a deterministic convex probe.
    z = z.float()
    y = y.long()
    head = nn.Linear(z.shape[1] * z.shape[2], n_classes, bias=True)
    if init_weight is None:
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    else:
        head.weight.data.copy_(init_weight.float())
        head.bias.data.copy_(init_bias.float())
    opt = torch.optim.LBFGS(head.parameters(), lr=lr, max_iter=steps,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                            tolerance_change=1e-10)
    def closure():
        opt.zero_grad()
        logits = head(z.flatten(1))
        loss = F.cross_entropy(logits, y) + 0.5 * weight_decay * head.weight.square().sum()
        loss.backward()
        return loss
    final = float(opt.step(closure).detach())
    return head, final


def gradient_vector(model):
    """Return only encoder feedback, with a fixed parameter ordering."""
    out = []
    for name, p in model.named_parameters():
        if not name.startswith("encoder."):
            continue
        if p.grad is None:
            out.append(torch.zeros_like(p, dtype=torch.float64).reshape(-1).cpu())
        else:
            out.append(p.grad.detach().reshape(-1).double().cpu())
    return torch.cat(out) if out else torch.zeros(1, dtype=torch.float64)


def run_one(method, seed, partition, fit_limit, grad_limit, batch_size, device_name):
    global labels_global
    runner, model, ckpt, payload = load_checkpoint(method, seed, partition)
    x, labels, fit, held = load_data(partition)
    labels_global = labels
    if fit_limit > 0:
        probe_indices = fit[:fit_limit]
    else:
        probe_indices = fit
    # Keep the gradient comparison fixed and FIT-only. No HELD sample is read.
    grad_indices = probe_indices[:min(len(probe_indices), grad_limit)]
    device = torch.device(device_name)
    torch.set_num_threads(4)
    z, y = collect_features(runner, model, x, probe_indices, device, batch_size)
    # The native squared-distance dual head is itself an affine observer of
    # mapped covariance up to a sample-wise common logit term.  Initializing
    # from that exact observer avoids mistaking an under-optimized convex
    # probe for a readout/feedback effect.
    with torch.no_grad():
        proto = model.prototype_factors @ model.prototype_factors.transpose(-1, -2) + 1e-4 * model.identity
        if method == "log":
            proto = runner.stable_logm(proto)
        scale = model.distance_scales[1 if method == "log" else 0]
        temp = model.log_temperature.exp()
        branch_w = 2.0 * proto.reshape(2, model.n_classes, -1) / scale[:, None, None] / temp
        branch_b = -(proto.square().sum((-2, -1))) / scale[:, None] / temp
        init_w = (0.5 * branch_w.reshape(2, model.n_classes, -1).permute(1, 0, 2).reshape(model.n_classes, -1)).cpu()
        init_b = (0.5 * branch_b.sum(0)).cpu()
    head, probe_loss = fit_affine(z, y, model.n_classes, init_w, init_b)

    # Native current head and affine reference on a common graph.
    model.zero_grad(set_to_none=True)
    ii = np.asarray(grad_indices, dtype=np.int64)
    out = model(runner.normalized_batch(x, ii, device))
    yy = torch.as_tensor(labels[ii], dtype=torch.long, device=device)
    current_loss = F.cross_entropy(out["dual_logits"], yy)
    current_loss.backward()
    g_actual = gradient_vector(model)

    model.zero_grad(set_to_none=True)
    out2 = model(runner.normalized_batch(x, ii, device))
    z2 = mapped_features(runner, out2, model.method)
    ref_logits = head(z2.flatten(1))
    reference_loss = F.cross_entropy(ref_logits, yy)
    reference_loss.backward()
    g_ref = gradient_vector(model)
    diff = g_actual - g_ref
    denom = (torch.linalg.vector_norm(g_actual) * torch.linalg.vector_norm(g_ref)).item()
    rho = float(torch.dot(g_actual, g_ref).item() / (denom + 1e-12))
    kappa = float(torch.linalg.vector_norm(diff).item() / (torch.linalg.vector_norm(g_ref).item() + 1e-12))
    pred_current = out["dual_logits"].argmax(-1).detach().cpu().numpy()
    pred_ref = ref_logits.argmax(-1).detach().cpu().numpy()
    result = {
        "status": "PASS",
        "kind": "real_checkpoint_E0_pilot",
        "scope": "FIT_only_same_checkpoint_affine_probe_reference",
        "method": method,
        "seed": seed,
        "partition": partition,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_step": int(payload.get("step", -1)),
        "device": str(device),
        "probe_n": int(len(probe_indices)),
        "gradient_n": int(len(grad_indices)),
        "fit_total_n": int(len(fit)),
        "held_n_not_used": int(len(held)),
        "probe_regularized_loss": probe_loss,
        "reference_l2": REFERENCE_L2,
        "current_dual_loss": float(current_loss.detach()),
        "reference_affine_loss": float(reference_loss.detach()),
        "current_accuracy_on_grad_subset": float(np.mean(pred_current == labels[ii])),
        "reference_accuracy_on_grad_subset": float(np.mean(pred_ref == labels[ii])),
        "actual_gradient_l2": float(torch.linalg.vector_norm(g_actual)),
        "reference_gradient_l2": float(torch.linalg.vector_norm(g_ref)),
        "feedback_difference_l2": float(torch.linalg.vector_norm(diff)),
        "rho_actual_reference": rho,
        "kappa_feedback": kappa,
        "interpretation": "pilot mechanism diagnostic; affine probe is a reference observer, not native-head optimum; no generalization claim",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["raw", "log"], required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--fit-limit", type=int, default=256)
    ap.add_argument("--grad-limit", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = run_one(args.method, args.seed, args.partition, args.fit_limit,
                     args.grad_limit, args.batch_size, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
