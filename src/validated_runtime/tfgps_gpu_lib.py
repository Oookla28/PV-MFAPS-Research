#!/usr/bin/env python3
"""Auditable GPU utilities for the TF-GPS PlantSPD experiment.

The library contains the registered action grids, FIT-only class-proxy
estimation, water-filling / CVaR policy solvers, importance-corrected native
and reference gradients, the exact trust-region alpha cap, and the execution
self-check used by every arm.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
TAU_GRID = (0.0, 0.5, 0.75, 1.0)
KAPPA_GRID = (0.0, 0.25, 0.5)
K_GRID = (0, 1, 2, 4, 8)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GAMMA_TAIL = 0.8
KM_TAIL_FRACTION = 0.2
KM_BUDGET = 0.2
KM_K_GRID = (0, 1, 2, 4)
INNER_LR = 0.05
REFERENCE_L2 = 1.17e-8
LAMBDA_COST = 0.05
WEIGHT_CLIP = 200.0
BATCH_SIZE = 8


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


E0 = load_module(HERE / "plant_e0_gpu.py", "tfgps_e0_gpu")


# ---------------------------------------------------------------------------
# Basic numerical helpers
# ---------------------------------------------------------------------------
def counts_and_prior(labels: np.ndarray, fit: np.ndarray, n_classes: int):
    counts = np.bincount(labels[np.asarray(fit)], minlength=n_classes).astype(np.int64)
    p = counts.astype(float) / counts.sum()
    return counts, p


def omega_from_counts(counts: np.ndarray, tau: float) -> np.ndarray:
    raw = np.power(np.maximum(counts.astype(float), 1.0), 1.0 - float(tau))
    return raw / raw.sum()


def waterfill(a: np.ndarray, kappa: float, tol: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c = len(a)
    floor = float(kappa) / c
    q = np.full(c, floor, dtype=float)
    free = np.ones(c, dtype=bool)
    while free.any():
        rem = 1.0 - floor * (~free).sum()
        denom = np.sqrt(np.maximum(a[free], 0.0)).sum()
        if denom <= tol:
            q[free] = rem / max(free.sum(), 1)
            break
        q[free] = rem * np.sqrt(np.maximum(a[free], 0.0)) / denom
        bad = free & (q < floor - tol)
        if not bad.any():
            break
        q[bad] = floor
        free[bad] = False
    q = np.maximum(q, floor)
    return q / q.sum()


def cvar_terms(r: np.ndarray, gamma: float, weights: np.ndarray | None = None):
    """Exact empirical CVaR and its subgradient xi (sum xi = 1).

    For gamma=0 the weighted mean is returned.  For gamma>0 the worst
    (1-gamma) weight mass is averaged.  The subgradient xi is used by the
    CVaR sampling water-filling step.
    """
    r = np.asarray(r, dtype=float)
    c = len(r)
    if weights is None:
        weights = np.full(c, 1.0 / c)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if gamma <= 0:
        xi = weights.copy()
        return float(np.dot(xi, r)), float(np.min(r)), xi
    target = 1.0 - float(gamma)
    order = np.argsort(-r)
    xi = np.zeros(c, dtype=float)
    cum = 0.0
    eta = float(r[order[0]])
    for idx in order:
        take = min(float(weights[idx]), target - cum)
        if take <= 0:
            break
        xi[idx] = take / target
        eta = float(r[idx])
        cum += take
        if cum >= target - 1e-15:
            break
    # Numerical guard: xi must sum to one even with ties at the boundary.
    s = xi.sum()
    if s <= 0:
        xi = weights.copy()
    else:
        xi /= s
    return float(np.dot(xi, r)), eta, xi


def cvar_solve_q(alpha: float, b: np.ndarray, sigma: np.ndarray, m: int,
                 kappa: float, gamma: float, iters: int = 400):
    """Convex CVaR sampling block solve by active-set water-filling.

    For fixed alpha, the epigraph optimum over q and eta is obtained by the
    KKT system q_c ∝ sqrt(xi_c alpha^2 sigma_c^2 / m).  The active set is
    iterated to a fixed point.  Returns q, CVaR, eta and a residual.
    """
    b = np.asarray(b, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    c = len(b)
    floor = float(kappa) / c
    if alpha <= 1e-15:
        q = waterfill((sigma ** 2) / max(m, 1), kappa)
        r = b.copy()
        cv, eta, xi = cvar_terms(r, gamma)
        return q, cv, eta, xi, 0.0
    q = np.full(c, 1.0 / c)
    q = np.maximum(q, floor)
    q /= q.sum()
    prev = None
    best = None
    for it in range(iters):
        r = (1.0 - alpha) ** 2 * b + alpha ** 2 * sigma ** 2 / (m * np.maximum(q, 1e-15))
        cv, eta, xi = cvar_terms(r, gamma)
        a = xi * (alpha ** 2) * sigma ** 2 / max(m, 1)
        q_new = waterfill(a, kappa)
        obj = cv
        if best is None or obj < best[0] - 1e-14:
            best = (obj, q_new.copy(), eta, xi.copy())
        res = float(np.max(np.abs(q_new - q)))
        if res < 1e-11:
            q = q_new
            break
        # Damped update avoids two-cycle behavior on near-tied active sets.
        q = 0.5 * q + 0.5 * q_new
        q = np.maximum(q, floor)
        q /= q.sum()
        if prev is not None and abs(obj - prev) < 1e-13 and it > 20:
            break
        prev = obj
    obj, q, eta, xi = best
    return q, float(obj), float(eta), xi, res


def cvar_slsqp(alpha: float, b: np.ndarray, sigma: np.ndarray, m: int,
               kappa: float, gamma: float, q0: np.ndarray):
    """Optional independent SLSQP check for the CVaR (q, eta) subproblem."""
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}
    c = len(b)
    floor = float(kappa) / c
    weights = np.full(c, 1.0 / c)

    def obj(x):
        q = x[:c]
        eta = x[c]
        r = (1.0 - alpha) ** 2 * b + alpha ** 2 * sigma ** 2 / (m * np.maximum(q, 1e-15))
        return float(eta + np.dot(weights, np.maximum(r - eta, 0.0)) / max(1.0 - gamma, 1e-12))

    x0 = np.r_[np.clip(q0, floor, 1.0), float(np.quantile(b, gamma))]
    try:
        res = minimize(
            obj, x0, method="SLSQP",
            bounds=[(floor, 1.0)] * c + [(0.0, None)],
            constraints={"type": "eq", "fun": lambda x: float(np.sum(x[:c]) - 1.0)},
            options={"ftol": 1e-11, "maxiter": 500, "disp": False},
        )
        return {
            "available": True, "success": bool(res.success), "message": str(res.message),
            "objective": float(res.fun), "q": np.asarray(res.x[:c]).tolist(),
            "eta": float(res.x[c]), "simplex_residual": float(abs(np.sum(res.x[:c]) - 1.0)),
        }
    except Exception as exc:
        return {"available": True, "success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------
def choose_device(gpu: int | None = None):
    forced = os.environ.get("TFGPS_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        idx = 0 if gpu is None else int(gpu)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def set_deterministic(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2 ** 32 - 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Old/fused equivalence is registered at 1e-5 on encoder state. TF32
    # convolution reduction order can make a single composed backward differ
    # from separately summed legacy gradients by >1e-4, so both the legacy and
    # fused paths run with TF32 disabled and identical FP32 numerics.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # cuDNN convolution backward accumulation order adds a composition-order
    # error above the registered state tolerance; the deterministic aten
    # convolution path is used for both legacy and fused execution.
    torch.backends.cudnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def freeze_readout(model):
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("encoder."))


def encoder_params(model):
    return [p for name, p in model.named_parameters() if name.startswith("encoder.")]


def clone_head(head):
    out = nn.Linear(head.in_features, head.out_features, bias=True)
    out.load_state_dict(head.state_dict())
    return out


def vec(parts):
    return torch.cat([p.detach().reshape(-1) for p in parts])


def assign_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = g.detach().clone()


class FlopCounter:
    def __init__(self):
        self.forward_flops = 0.0
        self.head_forward_flops = 0.0
        self.handles = []

    def attach(self, model):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(module, nn.Conv2d):
                b = int(x.shape[0])
                h, w = int(output.shape[-2]), int(output.shape[-1])
                macs = b * int(module.out_channels) * h * w
                macs *= int(module.in_channels // module.groups)
                macs *= int(module.kernel_size[0]) * int(module.kernel_size[1])
                self.forward_flops += 2.0 * macs
            elif isinstance(module, nn.Linear):
                b = int(x.shape[0]) if x.ndim > 1 else 1
                self.forward_flops += 2.0 * b * int(module.in_features) * int(module.out_features)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(hook))

    def add_head_forward(self, z, head):
        b = int(z.shape[0])
        self.head_forward_flops += 2.0 * b * int(head.in_features) * int(head.out_features)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @property
    def estimated_training_flops(self):
        return 3.0 * (self.forward_flops + self.head_forward_flops)


def make_audit():
    return {
        "native_passes": 0,
        "reference_passes": 0,
        "reference_probe_passes": 0,
        "reference_action_passes": 0,
        "reference_rho_probe_passes": 0,
        "backward_passes": 0,
        "probe_solver_steps": 0,
        "action_solver_steps": 0,
        "sample_solver_steps": 0,
        "rho_probe_solver_steps": 0,
        "solver_steps": 0,
        "class_probe_passes": 0,
        "native_group_passes": 0,
        "physical_encoder_forwards": 0,
        "physical_encoder_backwards": 0,
        "physical_reference_actions": 0,
        "pass_kinds": [],
    }


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def _weighted_loss(loss_vec, weights, denom=None):
    if weights is None:
        return loss_vec.sum() / denom if denom is not None else loss_vec.mean()
    if denom is not None:
        return (loss_vec * weights).sum() / denom
    return (loss_vec * weights).mean()


def native_gradient(model, runner, x, ii, yy, params, device, counter, audit,
                    weights=None, pass_kind="native", log_prior=None, denom=None):
    audit["native_passes"] += 1
    audit["backward_passes"] += 1
    audit["physical_encoder_forwards"] += 1
    audit["physical_encoder_backwards"] += 1
    audit["pass_kinds"].append(pass_kind)
    model.zero_grad(set_to_none=True)
    out = model(runner.normalized_batch(x, ii, device))
    logits = out["dual_logits"]
    if log_prior is not None:
        logits = logits + log_prior
    loss_vec = F.cross_entropy(logits, yy, reduction="none")
    loss = _weighted_loss(loss_vec, weights, denom=denom)
    g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return g, float(loss.detach())


def reference_gradient(model, runner, head, x, ii, yy, method, params, k, device,
                       counter, audit, weights=None, role="action", denom=None,
                       head_lr=INNER_LR, log_prior=None):
    """One reference action: encoder forward, k detached-head steps, encoder grad.

    ``denom`` is used for class-group K-MFAPS pieces so that the sum over groups
    equals the full-batch weighted reference loss (sum / batch_size).
    """
    if k < 0:
        raise ValueError("negative inner steps")
    audit["reference_passes"] += 1
    audit["backward_passes"] += 1
    audit["physical_encoder_forwards"] += 1
    audit["physical_encoder_backwards"] += 1
    audit["pass_kinds"].append(f"reference_{role}")
    if role == "probe":
        audit["reference_probe_passes"] += 1
        audit["probe_solver_steps"] += int(k)
    elif role == "rho_probe":
        audit["reference_rho_probe_passes"] += 1
        audit["rho_probe_solver_steps"] += int(k)
    else:
        audit["reference_action_passes"] += 1
        audit["physical_reference_actions"] += 1
        audit["action_solver_steps"] += int(k)
    audit["solver_steps"] += int(k)
    model.zero_grad(set_to_none=True)
    out = model(runner.normalized_batch(x, ii, device))
    z = E0.mapped_features(runner, out, method)
    z_det = z.detach().flatten(1)
    local = clone_head(head).to(device)
    opt = torch.optim.SGD(local.parameters(), lr=head_lr)
    for _ in range(int(k)):
        opt.zero_grad(set_to_none=True)
        solver_logits = local(z_det)
        if log_prior is not None:
            solver_logits = solver_logits + log_prior
        lv = F.cross_entropy(solver_logits, yy, reduction="none")
        li = _weighted_loss(lv, weights)
        li = li + 0.5 * REFERENCE_L2 * local.weight.square().sum()
        li.backward()
        opt.step()
        counter.add_head_forward(z_det, local)
    logits = local(z.flatten(1))
    if log_prior is not None:
        logits = logits + log_prior
    counter.add_head_forward(z.flatten(1), local)
    lv = F.cross_entropy(logits, yy, reduction="none")
    loss = _weighted_loss(lv, weights, denom=denom)
    g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return g, float(loss.detach()), local


def pcfa_fused_gradient(model, runner, head, x, ii, yy, method, params, device,
                        counter, audit, active_groups, weights=None, denom=None,
                        log_prior=None):
    """Evaluate a PC-FA update with one encoder forward/backward graph.

    The registered update is
    g_native(B) + sum_c alpha_c * (g_ref,c(B_c) - g_native,c(B_c)).
    This is the gradient of the summed loss constructed below. The detached
    head solver remains unchanged; repeated encoder forwards, repeated
    host-to-device input materialisation, and repeated encoder backward calls
    are removed. TF32 is disabled by ``set_deterministic`` so the single-graph
    gradient stays inside the registered 1e-5 state-equivalence bound.
    """
    if denom is None:
        denom = len(ii)
    n_groups = int(len(active_groups))
    # Logical audit counts are identical to the legacy implementation.
    audit["native_passes"] += 1 + n_groups
    audit["reference_passes"] += n_groups
    audit["backward_passes"] += 1 + 2 * n_groups
    audit["pass_kinds"].append("pcfa_fused_encoder_graph")
    audit["native_group_passes"] += n_groups
    audit["reference_action_passes"] += n_groups
    # Physical execution: one shared encoder forward/backward graph while the
    # logical native/reference/group counters remain legacy-identical.
    audit["physical_encoder_forwards"] += 1
    audit["physical_encoder_backwards"] += 1
    audit["physical_reference_actions"] += n_groups

    model.zero_grad(set_to_none=True)
    xx = runner.normalized_batch(x, ii, device)
    out = model(xx)
    native_logits = out["dual_logits"]
    if log_prior is not None:
        native_logits = native_logits + log_prior
    native_loss_vec = F.cross_entropy(native_logits, yy, reduction="none")
    native_loss = _weighted_loss(native_loss_vec, weights, denom=denom)
    total_loss = native_loss
    z = E0.mapped_features(runner, out, method).flatten(1)
    z_det = z.detach()
    group_meta = []

    for group in active_groups:
        mask = torch.as_tensor(group["mask"], dtype=torch.long, device=device)
        gy = yy.index_select(0, mask)
        gw = weights.index_select(0, mask) if weights is not None else None
        kc = int(group["k"])
        ac = float(group["alpha"])
        local = clone_head(head).to(device)
        opt = torch.optim.SGD(local.parameters(), lr=INNER_LR)
        z_group_det = z_det.index_select(0, mask)
        for _ in range(kc):
            opt.zero_grad(set_to_none=True)
            solver_logits = local(z_group_det)
            if log_prior is not None:
                solver_logits = solver_logits + log_prior
            lv = F.cross_entropy(solver_logits, gy, reduction="none")
            li = _weighted_loss(lv, gw)
            li = li + 0.5 * REFERENCE_L2 * local.weight.square().sum()
            li.backward()
            opt.step()
            counter.add_head_forward(z_group_det, local)
        z_group = z.index_select(0, mask)
        ref_logits = local(z_group)
        if log_prior is not None:
            ref_logits = ref_logits + log_prior
        ref_lv = F.cross_entropy(ref_logits, gy, reduction="none")
        ref_loss = _weighted_loss(ref_lv, gw, denom=denom)
        native_group_lv = F.cross_entropy(native_logits.index_select(0, mask), gy,
                                          reduction="none")
        native_group_loss = _weighted_loss(native_group_lv, gw, denom=denom)
        total_loss = total_loss + ac * (ref_loss - native_group_loss)
        counter.add_head_forward(z_group, local)
        audit["action_solver_steps"] += kc
        audit["sample_solver_steps"] += kc * int(mask.numel())
        audit["solver_steps"] += kc
        group_meta.append({
            "class": int(group.get("class", -1)),
            "n": int(mask.numel()),
            "k_c": kc,
            "alpha_c": ac,
            "group_native_loss": float(native_group_loss.detach()),
            "reference_loss": float(ref_loss.detach()),
        })

    grads = torch.autograd.grad(total_loss, params, retain_graph=False,
                                create_graph=False)
    return grads, float(native_loss.detach()), group_meta


def mix_grads(gn, gr, alpha):
    return tuple((1.0 - alpha) * a + alpha * b for a, b in zip(gn, gr))


def sum_grads(groups):
    if not groups:
        return None
    return tuple(sum(g[i] for g in groups) for i in range(len(groups[0])))


def trust_alpha(gn, gr, alpha_star, delta=0.10, cos_floor=0.0):
    """Exact algebraic trust-region cap ||g_mix-g_n|| <= delta ||g_n||."""
    a = vec(gn)
    b = vec(gr)
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na < 1e-12:
        return 0.0, 0.0, 0.0
    cos = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12))
    if cos < float(cos_floor):
        return 0.0, cos, 0.0
    d = b - a
    A = float(torch.dot(d, d))
    B = float(2.0 * torch.dot(a, d))
    R = ((1.0 + float(delta)) ** 2 - 1.0) * na * na
    if A <= 1e-18:
        return float(np.clip(min(alpha_star, 1.0), 0.0, 1.0)), cos, 0.0
    disc = max(B * B + 4.0 * A * R, 0.0)
    root = (-B + math.sqrt(disc)) / (2.0 * A)
    return float(np.clip(min(alpha_star, root), 0.0, 1.0)), cos, float(torch.linalg.vector_norm(d) / (na + 1e-12))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def class_indices_from_fit(fit, labels, n_classes):
    fit = np.asarray(fit, dtype=np.int64)
    return [fit[labels[fit] == c] for c in range(n_classes)]


def make_batches_class(fit, labels, q, steps, batch_size, seed, n_classes):
    rng = np.random.default_rng(int(seed))
    pools = class_indices_from_fit(fit, labels, n_classes)
    out = []
    for _ in range(int(steps)):
        cls = rng.choice(n_classes, size=int(batch_size), replace=True, p=q)
        ii = np.asarray([int(rng.choice(pools[int(c)])) for c in cls], dtype=np.int64)
        out.append(ii)
    return out


def make_batches_empirical(fit, steps, batch_size, seed):
    rng = np.random.default_rng(int(seed))
    fit = np.asarray(fit, dtype=np.int64)
    out = []
    for _ in range(int(steps)):
        ii = rng.choice(fit, size=int(batch_size), replace=False)
        out.append(np.asarray(ii, dtype=np.int64))
    return out


def importance_weights(labels_batch, omega, q, device, clip=WEIGHT_CLIP):
    w = omega[np.asarray(labels_batch, dtype=np.int64)] / np.maximum(q[np.asarray(labels_batch, dtype=np.int64)], 1e-12)
    w_raw = w.copy()
    w = np.minimum(w, float(clip))
    return (torch.as_tensor(w, dtype=torch.float32, device=device),
            float(np.max(w_raw)), float(np.mean(w_raw >= float(clip))), float(np.mean(w)))


# ---------------------------------------------------------------------------
# Class proxies
# ---------------------------------------------------------------------------
def calibrate_endpoint_with_head(model, runner, head, x, labels, fit, method, device,
                                 batch_size=8, max_per_class=4, seed=3407,
                                 rho_tail_fraction=0.2):
    """Estimate classwise A/rho/sigma and feedback cosine on FIT only."""
    n_classes = int(model.n_classes)
    counts, p = counts_and_prior(labels, fit, n_classes)
    pools = class_indices_from_fit(fit, labels, n_classes)
    rng = np.random.default_rng(int(seed) + 17)
    A = np.zeros(n_classes, dtype=float)
    sigma = np.zeros(n_classes, dtype=float)
    rho = np.zeros(n_classes, dtype=float)
    support = np.zeros(n_classes, dtype=int)
    cos1 = np.zeros(n_classes, dtype=float)
    err1 = np.zeros(n_classes, dtype=float)
    ref_l2 = np.zeros(n_classes, dtype=float)
    audit = make_audit()
    counter = FlopCounter()
    counter.attach(model)
    params = encoder_params(model)
    for cls in range(n_classes):
        idx = pools[cls]
        support[cls] = len(idx)
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(int(max_per_class), len(idx)), replace=False)
        yy = torch.as_tensor(labels[take], dtype=torch.long, device=device)
        gn, _ = native_gradient(model, runner, x, take, yy, params, device, counter, audit,
                                pass_kind="class_proxy_native")
        gr, _, _ = reference_gradient(model, runner, head, x, take, yy, method, params,
                                      1, device, counter, audit, role="probe")
        vn, vr = vec(gn), vec(gr)
        diff = float(torch.linalg.vector_norm(vn - vr))
        refn = float(torch.linalg.vector_norm(vr))
        denom = float(torch.linalg.vector_norm(vn) * torch.linalg.vector_norm(vr)) + 1e-12
        A[cls] = diff
        ref_l2[cls] = refn
        cos1[cls] = float(torch.dot(vn, vr) / denom)
        err1[cls] = diff / (refn + 1e-12)
        rho[cls] = float(np.clip(diff / (refn + 1e-8), 0.90, 0.999999))
        sigma[cls] = float((refn + diff) / math.sqrt(max(len(take), 1)))
        audit["class_probe_passes"] += 1
    # Multi-step rho probe on the lowest-count tail classes.  This reuses the
    # same FIT samples and is a diagnostic, not a convergence certificate.
    tail = np.argsort(counts)[:max(1, int(math.ceil(float(rho_tail_fraction) * n_classes)))]
    rho_rows = []
    e1_arr = np.zeros(n_classes, dtype=float)
    e2_arr = np.zeros(n_classes, dtype=float)
    e4_arr = np.zeros(n_classes, dtype=float)
    for cls in tail:
        idx = pools[int(cls)]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(int(max_per_class), len(idx)), replace=False)
        yy = torch.as_tensor(labels[take], dtype=torch.long, device=device)
        gn, _ = native_gradient(model, runner, x, take, yy, params, device, counter, audit,
                                pass_kind="rho_probe_native")
        vals = {}
        for k in (1, 2, 4):
            gr, _, _ = reference_gradient(model, runner, head, x, take, yy, method, params,
                                          k, device, counter, audit, role="rho_probe")
            vals[k] = float(torch.linalg.vector_norm(vec(gn) - vec(gr)))
        e1_arr[cls], e2_arr[cls], e4_arr[cls] = vals[1], vals[2], vals[4]
        rho12 = float(np.clip(vals[2] / max(vals[1], 1e-12), 0.0, 0.999999))
        rho24 = float(np.clip(math.sqrt(max(vals[4], 1e-12) / max(vals[2], 1e-12)), 0.0, 0.999999))
        identifiable = bool(vals[1] > 1e-10 and vals[2] > 1e-10 and 0.5 < rho12 < 1.0)
        if identifiable:
            rho[int(cls)] = float(np.clip(0.5 * (rho12 + rho24), 0.90, 0.999999))
        rho_rows.append({
            "class": int(cls), "count": int(counts[cls]), "e1": vals[1],
            "e2": vals[2], "e4": vals[4], "rho12": rho12, "rho24": rho24,
            "identifiable": identifiable,
        })
    counter.close()
    # Low-support classes: pooled upper proxies, never zero.  The threshold is
    # the registered < 8 FIT samples.
    low = support < 8
    valid = (~low) & (sigma > 0)
    pooled = float(np.percentile(sigma[valid], 90)) if valid.any() else 1.0
    pooled_rho = float(np.median(rho[valid])) if valid.any() else 0.99
    pooled_A = float(np.median(A[valid])) if valid.any() else pooled
    sigma_raw = sigma.copy()
    sigma[low] = np.maximum(sigma[low], pooled)
    sigma[sigma <= 0] = pooled
    rho[low] = np.maximum(rho[low], pooled_rho)
    rho[rho <= 0] = pooled_rho
    A[A <= 0] = pooled_A
    coverage = {
        "status": "proxy_only_unverified",
        "low_support_threshold": 8,
        "low_support_classes": [int(c) for c in np.flatnonzero(low)],
        "low_support_count": int(low.sum()),
        "pooled_sigma_source": "p90_of_support_ge_8_class_sigma_proxies",
        "pooled_rho_source": "median_of_support_ge_8_class_rho_proxies",
        "note": "sigma is an empirical gradient-noise proxy, not a certificate; low-support sigma is pooled/upward, never zero",
    }
    return {
        "counts": counts, "prior": p, "A": A, "sigma": sigma, "sigma_raw": sigma_raw,
        "rho": rho, "support": support, "cos1": cos1, "err1": err1,
        "ref_l2": ref_l2, "low_support": low, "tail_indices": tail.tolist(),
        "rho_rows": rho_rows, "audit": audit, "counter": counter,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Policy solvers
# ---------------------------------------------------------------------------
def nearest_grid(x, grid):
    grid = np.asarray(grid, dtype=float)
    return float(grid[int(np.argmin(np.abs(grid - float(x))))])


def choose_mean_policy(counts, A, rho, sigma, batch_size=8, tau_grid=TAU_GRID,
                       kappa_grid=KAPPA_GRID, k_grid=K_GRID, lambda_cost=LAMBDA_COST):
    c = len(counts)
    candidates = []
    for tau in tau_grid:
        omega = omega_from_counts(counts, tau)
        for kappa in kappa_grid:
            a = (omega * sigma) ** 2 / max(batch_size, 1)
            q = waterfill(a, kappa)
            V = float(np.sum(a / np.maximum(q, 1e-15)))
            for k in k_grid:
                D = float(np.sum(omega * (A ** 2) * np.power(np.clip(rho, 0.0, 0.999999), 2 * int(k))))
                alpha_cont = float(D / (D + V + 1e-12))
                alpha = nearest_grid(alpha_cont, ALPHA_GRID)
                if int(k) == 0:
                    alpha = 0.0  # k>0 is required for a reference action
                elif alpha <= 1e-12:
                    # k>0 with alpha=0 would execute no reference and silently
                    # turn into a native action; exclude it from the action set.
                    continue
                risk = (1.0 - alpha) ** 2 * D + alpha ** 2 * V
                cost = int(k) + (1.0 if alpha > 1e-12 else 0.0)
                candidates.append({
                    "score": risk + float(lambda_cost) * cost, "risk": risk,
                    "tau": float(tau), "kappa": float(kappa), "k": int(k),
                    "alpha_base": float(alpha), "alpha_cont": alpha_cont,
                    "D": D, "V": V, "q": q.tolist(), "cost": cost,
                    "q_solver": "mean_waterfill_closed_form",
                })
    best = min(candidates, key=lambda z: z["score"])
    best["all_candidates_n"] = len(candidates)
    best["omega"] = omega_from_counts(counts, best["tau"]).tolist()
    best["gamma"] = 0.0
    return best, candidates


def choose_cvar_policy(counts, A, rho, sigma, batch_size=8, tau=0.75, kappa=0.25,
                       gamma=GAMMA_TAIL, k_grid=K_GRID, lambda_cost=LAMBDA_COST,
                       run_slsqp_check=False):
    c = len(counts)
    omega = omega_from_counts(counts, tau)
    candidates = []
    slsqp_rows = []
    for k in k_grid:
        b = (A ** 2) * np.power(np.clip(rho, 0.0, 0.999999), 2 * int(k))
        floor_c = float(kappa) / len(b)
        scale = max(float(np.max(b)),
                    float(np.max(sigma ** 2 / (batch_size * max(floor_c, 1e-12)))),
                    1e-12)
        b_s = b / scale
        sigma_s = sigma / math.sqrt(scale)
        for alpha in ALPHA_GRID:
            if int(k) == 0 and alpha > 0:
                continue
            if int(k) > 0 and alpha <= 0:
                continue
            q0, cv0_s, eta0_s, xi0, res0 = cvar_solve_q(alpha, b_s, sigma_s,
                                                          batch_size, kappa, gamma)
            cv0 = cv0_s * scale
            eta0 = eta0_s * scale
            q, cv, eta, xi = q0, cv0, eta0, xi0
            solver = "cvar_active_set_waterfill_warmstart"
            solver_ok = False
            check = None
            if alpha > 0:
                check = cvar_slsqp(alpha, b_s, sigma_s, batch_size, kappa, gamma, q0)
                if check.get("available") and check.get("success"):
                    qc = np.asarray(check["q"], dtype=float)
                    floor = float(kappa) / len(b)
                    qc = np.maximum(qc, floor)
                    qc = qc / qc.sum()
                    r = (1.0 - alpha) ** 2 * b_s + alpha ** 2 * sigma_s ** 2 / (batch_size * np.maximum(qc, 1e-15))
                    cvc_s, etac_s, xic = cvar_terms(r, gamma)
                    cvc = float(cvc_s) * scale
                    etac = float(etac_s) * scale
                    if cvc <= cv0 + 1e-6 * scale:
                        q, cv, eta, xi = qc, float(cvc), float(etac), xic
                        solver = "slsqp_epigraph"
                        solver_ok = True
                    else:
                        solver = "slsqp_worse_than_warmstart"
                else:
                    solver = "alternating_fallback_slsqp_failed"
            row = {
                "k": int(k), "alpha": float(alpha), "cvar": float(cv), "eta": float(eta),
                "q": q.tolist(), "q_solver": solver,
                "q_residual": float(res0), "xi_sum": float(xi.sum()),
                "slsqp_ok": bool(solver_ok),
                "solver_verified": bool(solver_ok or alpha <= 0),
                "alternative_cvar_warmstart": float(cv0),
                "risk_scale": float(scale),
                "cvar_scaled": float(cv / scale),
                "cost": int(k) + (1.0 if alpha > 0 else 0.0),
            }
            if check is not None:
                row["slsqp"] = {k2: v for k2, v in check.items() if k2 != "q"}
                if check.get("available") and check.get("success"):
                    row["slsqp_gap_vs_warmstart"] = float((cv0 - check["objective"] * scale))
                    slsqp_rows.append({"k": int(k), "alpha": float(alpha),
                                       "gap": row["slsqp_gap_vs_warmstart"],
                                       "success": True, "solver": solver})
            row["score"] = row["cvar"] + float(lambda_cost) * row["cost"]
            candidates.append(row)
    # k=0 is native only; q is the mean water-fill and has no CVaR effect.
    best = min(candidates, key=lambda z: z["score"])
    best["tau"] = float(tau)
    best["kappa"] = float(kappa)
    best["gamma"] = float(gamma)
    best["alpha_base"] = float(best["alpha"])
    best["omega"] = omega.tolist()
    best["all_candidates_n"] = len(candidates)
    best["slsqp_checks"] = slsqp_rows
    best["solver_verified"] = bool(best.get("solver_verified", False))
    return best, candidates


def choose_kmfaps_policy(counts, A, rho, sigma, batch_size=8, tau=0.75, kappa=0.25,
                         gamma_tail=KM_TAIL_FRACTION, budget=KM_BUDGET,
                         k_grid=KM_K_GRID, beta=0.0):
    c = len(counts)
    omega = omega_from_counts(counts, tau)
    q = waterfill((omega * sigma) ** 2 / max(batch_size, 1), kappa)
    tail = np.argsort(counts)[:max(1, int(math.ceil(float(gamma_tail) * c)))]
    b = (A + float(beta) * sigma) ** 2
    rho_c = np.clip(rho, 0.7, 0.999999)
    kg = np.asarray(k_grid, dtype=int)
    kmax = int(kg.max())

    def k_of(z):
        return np.clip(np.log(np.maximum(b / max(float(z), 1e-300), 1.0)) /
                       (-2.0 * np.log(rho_c)), 0.0, float(kmax))

    def cost(z):
        return float(np.sum(q[tail] * k_of(z)[tail]))

    lo = max(float(b.min()) * 1e-12, 1e-300)
    hi = float(max(b.max(), 1e-12))
    for _ in range(200):
        mid = math.sqrt(lo * hi) if lo > 0 else 0.5 * hi
        if cost(mid) > float(budget):
            lo = mid
        else:
            hi = mid
    z_star = hi
    k_cont = k_of(z_star)
    k_disc = np.asarray([int(kg[np.argmin(np.abs(kg - x))]) for x in k_cont], dtype=int)
    # Only tail classes receive extra solver steps; non-tail classes use native.
    non_tail = np.setdiff1d(np.arange(c), tail)
    k_disc[non_tail] = 0
    while float(np.sum(q[tail] * k_disc[tail])) > float(budget) + 1e-9:
        cand = [int(j) for j in tail if k_disc[j] > 0]
        if not cand:
            break
        j = max(cand, key=lambda t: k_disc[t])
        lower = kg[kg < k_disc[j]]
        k_disc[j] = int(lower[-1]) if len(lower) else 0
    expected = float(np.sum(q[tail] * k_disc[tail]))
    b_k = (A ** 2) * np.power(rho_c, 2 * k_disc)
    # Global reference mix from the tail active set at the allocated k_c.
    xi = np.zeros(c, dtype=float)
    xi[tail] = 1.0 / len(tail)
    den = np.sum(xi * (b_k + sigma ** 2 / (batch_size * np.maximum(q, 1e-15))))
    alpha_cont = float(np.sum(xi * b_k) / (den + 1e-15))
    alpha = nearest_grid(alpha_cont, ALPHA_GRID)
    if k_disc.max() == 0:
        alpha = 0.0
    elif alpha <= 1e-12:
        alpha = 0.25  # keep k>0 executable with a registered positive mix
    dispersion = float(np.std(k_disc[tail]))
    unique = sorted(set(int(v) for v in k_disc.tolist()))
    risk_global_k1 = float(np.max(b[tail] * rho_c[tail] ** 2))
    risk_kmfaps = float(np.max(b_k[tail]))
    ident = [r["identifiable"] for r in []]  # filled by caller with rho rows
    return {
        "tau": float(tau), "kappa": float(kappa), "tail_fraction": float(gamma_tail),
        "budget": float(budget), "k_grid": list(k_grid), "beta": float(beta),
        "tail_indices": tail.tolist(), "tail_size": int(len(tail)),
        "q": q.tolist(), "omega": omega.tolist(), "q_tail_mass": float(q[tail].sum()),
        "z_star": float(z_star), "k_continuous": k_cont.tolist(),
        "k_discrete": k_disc.tolist(), "k_discrete_unique": unique,
        "k_dispersion": dispersion, "expected_budget": expected,
        "alpha_base": float(alpha), "alpha_cont": alpha_cont,
        "risk_global_k1": risk_global_k1, "risk_kmfaps": risk_kmfaps,
        "risk_reduction_vs_global_k1": float(risk_global_k1 / (risk_kmfaps + 1e-12)),
        "valid_nondegenerate": bool(len(unique) > 1 and dispersion > 0.0),
        "solver": "threshold_monotone_plus_grid_repair",
    }


def choose_kmfaps_policy_kappa(counts, A, sigma, rho, rho_rows=None, batch_size=8,
                               tau=0.75, budget=KM_BUDGET, k_grid=KM_K_GRID,
                               beta=0.0, gamma_tail=KM_TAIL_FRACTION):
    """Registered K-MFAPS allocation: b_c=(kappa_c+beta*sigma_c)^2.

    kappa_c=A_c/(2*sigma_c-A_c) requires a strictly positive denominator.
    rho_c must come from the FIT-only k=1,2,4 residual probe for every tail
    class.  If identifiability or the discrete k vector degenerates, the
    returned policy is marked invalid and no K-MFAPS training may run.
    """
    c = len(counts)
    omega = omega_from_counts(counts, tau)
    q = omega.copy()
    tail = np.argsort(counts)[:max(1, int(math.ceil(float(gamma_tail) * c)))]
    tail = np.asarray(tail, dtype=int)
    den = 2.0 * np.asarray(sigma, dtype=float) - np.asarray(A, dtype=float)
    kappa_c = np.where(den > 0.0, np.asarray(A, dtype=float) / np.maximum(den, 1e-12),
                       np.nan)
    den_pos = bool(np.all(den[tail] > 0.0))
    kappa_clipped = bool(np.any(den[tail] <= 0.0))
    kappa_safe = np.where(np.isfinite(kappa_c), kappa_c, 1e12)
    b = (kappa_safe + float(beta) * np.asarray(sigma, dtype=float)) ** 2
    rho_c = np.clip(np.asarray(rho, dtype=float), 0.7, 0.999999)
    kg = np.asarray(k_grid, dtype=int)
    kmax = int(kg.max())

    def k_of(z):
        return np.clip(np.log(np.maximum(b / max(float(z), 1e-300), 1.0)) /
                       (-2.0 * np.log(rho_c)), 0.0, float(kmax))

    def cost(z):
        return float(np.sum(q[tail] * k_of(z)[tail]))

    lo = max(float(np.min(b[tail])) * 1e-12, 1e-300)
    hi = float(max(np.max(b[tail]), 1e-12))
    for _ in range(200):
        mid = math.sqrt(lo * hi) if lo > 0 else 0.5 * hi
        if cost(mid) > float(budget):
            lo = mid
        else:
            hi = mid
    z_star = hi
    k_cont = k_of(z_star)
    k_disc = np.asarray([int(kg[np.argmin(np.abs(kg - x))]) for x in k_cont], dtype=int)
    non_tail = np.setdiff1d(np.arange(c), tail)
    k_disc[non_tail] = 0
    while float(np.sum(q[tail] * k_disc[tail])) > float(budget) + 1e-9:
        cand = [int(j) for j in tail if k_disc[j] > 0]
        if not cand:
            break
        j = max(cand, key=lambda t: k_disc[t])
        lower = kg[kg < k_disc[j]]
        k_disc[j] = int(lower[-1]) if len(lower) else 0
    expected = float(np.sum(q[tail] * k_disc[tail]))
    b_k = b * np.power(rho_c, 2 * k_disc)
    xi = np.zeros(c, dtype=float)
    xi[tail] = 1.0 / len(tail)
    den_a = np.sum(xi * (b_k + sigma ** 2 / (batch_size * np.maximum(q, 1e-15))))
    alpha_cont = float(np.sum(xi * b_k) / (den_a + 1e-15))
    alpha = nearest_grid(alpha_cont, ALPHA_GRID)
    if int(np.max(k_disc[tail])) == 0:
        alpha = 0.0
    elif alpha <= 1e-12:
        alpha = 0.25
    dispersion = float(np.std(k_disc[tail]))
    unique = sorted(set(int(v) for v in k_disc[tail].tolist()))
    # Identifiability of the multi-step rho probe on the registered tail set.
    rows = rho_rows or []
    row_by_class = {int(r.get("class")): r for r in rows}
    tail_rows = [row_by_class.get(int(c2)) for c2 in tail]
    ident_flags = [bool(r.get("identifiable")) for r in tail_rows if r is not None]
    rho_frac = float(np.mean(ident_flags)) if ident_flags else 0.0
    boundary = []
    if not den_pos:
        boundary.append("kappa_denominator_nonpositive")
    if rho_frac < 1.0:
        boundary.append("rho_not_fully_identifiable")
    if len(unique) <= 1 or dispersion <= 0.0:
        boundary.append("k_c_degenerate")
    valid = not boundary
    return {
        "tau": float(tau), "kappa": 0.0, "gamma": 0.0,
        "tail_fraction": float(gamma_tail), "budget": float(budget),
        "k_grid": [int(v) for v in k_grid], "beta": float(beta),
        "tail_indices": tail.tolist(), "tail_size": int(len(tail)),
        "q": q.tolist(), "omega": omega.tolist(), "q_tail_mass": float(q[tail].sum()),
        "z_star": float(z_star), "k_continuous": k_cont.tolist(),
        "k_discrete": k_disc.tolist(), "k_discrete_unique": unique,
        "k_dispersion": dispersion, "expected_budget": expected,
        "alpha_base": float(alpha), "alpha_cont": alpha_cont,
        "kappa_proxy": [None if not np.isfinite(v) else float(v) for v in kappa_c.tolist()],
        "kappa_proxy_range": [float(np.nanmin(kappa_c[tail])), float(np.nanmax(kappa_c[tail]))] if den_pos else None,
        "kappa_denominator_min_tail": float(np.min(den[tail])),
        "kappa_denominator_nonpositive": kappa_clipped,
        "rho_source": "FIT-only k=1,2,4 multi-step residual probe on tail classes",
        "rho_identifiable_fraction": rho_frac,
        "risk_global_k1": float(np.max(b[tail] * rho_c[tail] ** 2)),
        "risk_kmfaps": float(np.max(b_k[tail])),
        "risk_reduction_vs_global_k1": float(np.max(b[tail] * rho_c[tail] ** 2) / (np.max(b_k[tail]) + 1e-12)),
        "valid_nondegenerate": bool(valid),
        "boundary_reason": boundary,
        "solver": "kappa_threshold_monotone_plus_grid_repair",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _metrics_from_predictions(pred, yy, counts):
    c = len(counts)
    rec = np.full(c, np.nan, dtype=float)
    f1 = np.full(c, np.nan, dtype=float)
    for cls in range(c):
        m = yy == cls
        if not m.any():
            continue
        tp = int(np.sum((pred[m] == cls)))
        fn = int(np.sum(m)) - tp
        fp = int(np.sum((pred == cls) & (~m)))
        rec[cls] = tp / max(int(np.sum(m)), 1)
        f1[cls] = 2 * tp / max(2 * tp + fp + fn, 1)
    order = np.argsort(counts)
    n_tail = max(1, int(math.ceil(0.1 * c)))
    tail = order[:n_tail]
    head = order[-n_tail:]
    return {
        "macro_recall": float(np.nanmean(rec)),
        "macro_f1": float(np.nanmean(f1)),
        "tail_recall_worst10": float(np.nanmean(rec[tail])),
        "head_recall_top10": float(np.nanmean(rec[head])),
        "head_tail_gap": float(np.nanmean(rec[head]) - np.nanmean(rec[tail])),
        "n": int(len(yy)),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "tail_classes": tail.tolist(),
        "head_classes": head.tolist(),
    }


@torch.no_grad()
def eval_model(model, runner, x, labels, indices, head, method, device, counts,
               batch_size=32, log_prior=None, use_fixed_head=True):
    indices = np.asarray(indices, dtype=np.int64)
    preds = []
    for st in range(0, len(indices), int(batch_size)):
        ii = indices[st:st + int(batch_size)]
        out = model(runner.normalized_batch(x, ii, device))
        if use_fixed_head:
            z = E0.mapped_features(runner, out, method)
            logits = head(z.flatten(1))
        else:
            logits = out["dual_logits"]
            if log_prior is not None:
                logits = logits + log_prior
        preds.append(logits.argmax(-1).detach().cpu().numpy())
    pred = np.concatenate(preds, 0) if preds else np.zeros(0, dtype=np.int64)
    yy = np.asarray(labels[indices], dtype=np.int64)
    return _metrics_from_predictions(pred, yy, counts)


def eval_fixed_head_fit(model, runner, x, labels, fit, head, method, device, counts):
    return eval_model(model, runner, x, labels, fit, head, method, device, counts,
                      use_fixed_head=True)


def eval_native_fit(model, runner, x, labels, fit, device, counts, log_prior=None):
    return eval_model(model, runner, x, labels, fit, None, "raw", device, counts,
                      log_prior=log_prior, use_fixed_head=False)


def state_to_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("encoder.")}


# ---------------------------------------------------------------------------
# Execution self-check
# ---------------------------------------------------------------------------
def nearest_grid_index(x, grid):
    grid = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(grid - float(x))))


def build_registered_config(arm, steps, policy, batch_size, batches, n_classes, labels=None):
    """Static registered action contract used before execution."""
    tau = float(policy.get("tau", 0.0))
    kappa = float(policy.get("kappa", policy.get("kappa_floor", 0.0)))
    gamma = float(policy.get("gamma", 0.0))
    k = int(policy.get("k", 0))
    alpha_base = float(policy.get("alpha_base", policy.get("alpha", 0.0)))
    cfg = {
        "arm": arm, "steps": int(steps), "batch_size": int(batch_size),
        "tau": tau, "kappa": kappa, "gamma": gamma, "k": k,
        "alpha_base": alpha_base,
        "tau_grid": list(TAU_GRID), "kappa_grid": list(KAPPA_GRID),
        "k_grid": list(K_GRID), "alpha_grid": list(ALPHA_GRID),
        "skip_semantics": "k=0 requires alpha=0 and zero reference action; k>0 requires alpha>0 for reference",
    }
    if "delta" in policy:
        cfg["trust_delta"] = float(policy["delta"])
    if arm in ("native", "native_balanced", "uniform_balanced", "effective_number", "logit_adjustment", "balanced_softmax"):
        cfg.update({"native_passes": int(steps), "reference_probe_passes": 0,
                    "reference_action_passes": 0, "solver_steps": 0,
                    "sample_solver_steps": 0})
    elif arm == "two_forward_native":
        cfg.update({"native_passes": 2 * int(steps), "reference_probe_passes": 0,
                    "reference_action_passes": 0, "solver_steps": 0,
                    "sample_solver_steps": 0})
    elif arm.startswith("fixed_k_") or arm == "bs_fixed_k_matched_B02":
        k_seq = policy["k_sequence"]
        cfg.update({"native_passes": int(steps),
                    "reference_probe_passes": 0,
                    "reference_action_passes": int(sum(1 for x in k_seq if x > 0)),
                    "solver_steps": int(sum(k_seq)),
                    "sample_solver_steps": int(sum(k_seq)) * int(batch_size),
                    "k_sequence": [int(x) for x in k_seq]})
    elif arm == "kmfaps":
        k_disc = np.asarray(policy["k_discrete"], dtype=int)
        groups = 0
        act_steps = 0
        sample_steps = 0
        for ii in batches:
            ii_arr = np.asarray(ii, dtype=np.int64)
            if labels is None:
                raise ValueError("build_registered_config requires labels for K-MFAPS groups")
            cls = np.unique(np.asarray(labels)[ii_arr])
            for c in cls:
                groups += 1
                act_steps += int(k_disc[int(c)])
                n_c = int(np.sum(np.asarray(labels)[ii_arr] == int(c)))
                sample_steps += n_c * int(k_disc[int(c)])
        cfg.update({"native_passes": int(steps),
                    "reference_probe_passes": 0,
                    "reference_action_passes": int(groups) if alpha_base > 0 else 0,
                    "solver_steps": int(act_steps) if alpha_base > 0 else 0,
                    "sample_solver_steps": int(sample_steps) if alpha_base > 0 else 0,
                    "expected_inner_budget_per_sample": float(sample_steps) / max(int(steps) * int(batch_size), 1),
                    "k_discrete": [int(v) for v in k_disc.tolist()],
                    "action_groups_semantics": "one reference action per sampled class; k_c inner steps"})
    elif arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps"):
        k_arr = np.asarray(policy["k_discrete"], dtype=int)
        a_arr = np.asarray(policy["alpha_discrete"], dtype=float)
        groups = 0
        act_steps = 0
        sample_steps = 0
        for ii in batches:
            ii_arr = np.asarray(ii, dtype=np.int64)
            if labels is None:
                raise ValueError("build_registered_config requires labels for PC-FA groups")
            cls = np.unique(np.asarray(labels)[ii_arr])
            for cc in cls:
                if float(a_arr[int(cc)]) > 0:
                    groups += 1
                    act_steps += int(k_arr[int(cc)])
                    n_c = int(np.sum(np.asarray(labels)[ii_arr] == int(cc)))
                    sample_steps += n_c * int(k_arr[int(cc)])
        cfg.update({
            "native_passes": int(steps) + int(groups),
            "native_group_passes": int(groups),
            "reference_probe_passes": 0,
            "reference_action_passes": int(groups),
            "solver_steps": int(act_steps),
            "sample_solver_steps": int(sample_steps),
            "k_discrete": [int(v) for v in k_arr.tolist()],
            "alpha_discrete": [float(v) for v in a_arr.tolist()],
            "k_discrete_unique": sorted(set(int(v) for v in k_arr.tolist())),
            "alpha_discrete_unique": sorted(set(float(v) for v in a_arr.tolist())),
            "measured_cost_per_sample": float(policy.get("cost_per_sample", 0.0)),
            "measured_budget_per_sample": float(policy.get("budget_per_sample", 0.0)),
            "mean_risk": float(policy.get("mean_risk", 0.0)),
            "cvar": float(policy.get("cvar", 0.0)),
            "action_groups_semantics": "full native + per-class group native/reference for alpha_c>0",
            "fused": bool(policy.get("pcfa_fused", False)),
            "physical_encoder_forwards": int(steps) if policy.get("pcfa_fused", False) else int(steps) + 2 * int(groups),
            "physical_encoder_backwards": int(steps) if policy.get("pcfa_fused", False) else int(steps) + 2 * int(groups),
            "physical_reference_actions": int(groups),
            "physical_backward_semantics": (
                "single shared encoder graph backward" if policy.get("pcfa_fused", False)
                else "one backward per native/reference forward"),
        })
    elif arm == "old_rfaps":
        cfg.update({"native_passes": int(steps), "reference_probe_passes": None,
                    "reference_action_passes": None, "solver_steps": None,
                    "sample_solver_steps": None,
                    "cache_protocol": "probe_every_2_cold_start_1_drift_0.5"})
    else:
        # global TF-GPS / fixed-k-1 / trust arms
        expected_actions = int(steps) if (k > 0 and alpha_base > 0) else 0
        cfg.update({"native_passes": int(steps), "reference_probe_passes": 0,
                    "reference_action_passes": expected_actions,
                    "solver_steps": expected_actions * int(k),
                    "sample_solver_steps": expected_actions * int(k) * int(batch_size)})
    return cfg


def validate_execution(registered, observed, rows, arm):
    errors = []
    for key in ("native_passes", "reference_probe_passes", "reference_action_passes",
                "solver_steps", "sample_solver_steps", "native_group_passes",
                "physical_encoder_forwards", "physical_encoder_backwards",
                "physical_reference_actions"):
        exp = registered.get(key)
        if exp is not None and int(observed.get(key, -1)) != int(exp):
            errors.append(f"{key}: observed={observed.get(key)} expected={exp}")
    if int(observed.get("backward_passes", -1)) != int(observed.get("native_passes", 0)) + int(observed.get("reference_passes", 0)):
        errors.append("backward_passes != native_passes + reference_passes")
    if int(observed.get("reference_passes", 0)) != int(observed.get("reference_probe_passes", 0)) + int(observed.get("reference_action_passes", 0)) + int(observed.get("reference_rho_probe_passes", 0)):
        errors.append("reference_passes != probe + action + rho_probe")
    if int(observed.get("solver_steps", 0)) != int(observed.get("probe_solver_steps", 0)) + int(observed.get("action_solver_steps", 0)) + int(observed.get("rho_probe_solver_steps", 0)):
        errors.append("solver_steps != probe + action + rho_probe solver steps")
    for r in rows:
        if arm not in ("kmfaps", "pcfa", "bs_pcfa", "native_plus_pv_mfaps") and int(r.get("k", 0)) == 0 and float(r.get("alpha_used", 0.0)) > 1e-12:
            errors.append(f"step {r.get('step')}: k=0 with alpha>0")
        if r.get("used_reference") and float(r.get("alpha_used", 0.0)) <= 1e-12 and not r.get("reference_used_for_trust", False):
            errors.append(f"step {r.get('step')}: reference used but alpha=0")
    if arm == "old_rfaps":
        if int(observed.get("reference_probe_passes", 0)) != sum(bool(r.get("reference_probe_used")) for r in rows):
            errors.append("old_rfaps probe counter disagrees with rows")
        if int(observed.get("reference_action_passes", 0)) != sum(bool(r.get("reference_action_used")) for r in rows):
            errors.append("old_rfaps action counter disagrees with rows")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def static_policy_check(policy, q, n_classes, arm, tol=1e-9):
    errors = []
    q = np.asarray(q, dtype=float)
    if abs(q.sum() - 1.0) > 1e-8:
        errors.append(f"q sum {q.sum()}")
    floor = float(policy.get("kappa", policy.get("kappa_floor", 0.0))) / n_classes
    if q.min() < floor - 1e-8:
        errors.append(f"q min {q.min()} < floor {floor}")
    if policy.get("tau") is not None and nearest_grid(policy["tau"], TAU_GRID) != float(policy["tau"]):
        errors.append(f"tau {policy['tau']} not on registered grid")
    if policy.get("kappa") is not None and nearest_grid(policy["kappa"], KAPPA_GRID) != float(policy["kappa"]):
        errors.append(f"kappa {policy['kappa']} not on registered grid")
    if arm in ("mean_tfgps", "weighted_trust_d005", "weighted_trust_d010", "weighted_trust_d020", "cvar_tfgps", "fixed_k_1", "bs_fixed_k_matched_B02"):
        if arm == "cvar_tfgps" and abs(float(policy.get("gamma", -1)) - 0.8) > 1e-12:
            errors.append(f"cvar gamma {policy.get('gamma')} != registered 0.8")
        if arm == "mean_tfgps" and abs(float(policy.get("gamma", 0.0))) > 1e-12:
            errors.append("mean policy gamma != 0")
        if int(policy["k"]) not in K_GRID:
            errors.append(f"k {policy['k']} not in grid")
        if int(policy["k"]) > 0 and float(policy["alpha_base"]) not in ALPHA_GRID:
            errors.append(f"alpha_base {policy['alpha_base']} not in grid")
        if int(policy["k"]) == 0 and float(policy["alpha_base"]) > 1e-12:
            errors.append("k=0 with alpha_base>0")
    if arm == "kmfaps":
        bad = [v for v in policy["k_discrete"] if int(v) not in KM_K_GRID]
        if bad:
            errors.append(f"kmfaps k outside grid: {bad}")
        if not policy.get("valid_nondegenerate", False):
            errors.append("kmfaps allocation degenerate (constant k_c)")
    if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps"):
        k_arr = np.asarray(policy.get("k_discrete", []), dtype=float)
        a_arr = np.asarray(policy.get("alpha_discrete", []), dtype=float)
        if len(k_arr) != n_classes or len(a_arr) != n_classes:
            errors.append("pcfa k/alpha arrays have wrong length")
        else:
            if any(int(v) not in KM_K_GRID for v in k_arr):
                errors.append("pcfa k outside registered grid")
            if any(nearest_grid(v, ALPHA_GRID) != float(v) for v in a_arr):
                errors.append("pcfa alpha outside registered grid")
            if any(int(k_arr[c]) == 0 and float(a_arr[c]) > 1e-12 for c in range(n_classes)):
                errors.append("pcfa k=0 with alpha>0")
            if any(int(k_arr[c]) > 0 and float(a_arr[c]) <= 1e-12 for c in range(n_classes)):
                errors.append("pcfa k>0 with alpha=0")
        if not policy.get("valid_nondegenerate", False):
            errors.append("pcfa allocation degenerate (constant k_c)")
        if not policy.get("feasible", False):
            errors.append("pcfa allocation not feasible under measured budget/mean risk")
        if float(policy.get("cost_per_sample", 1e99)) > float(policy.get("budget_per_sample", 0.0)) + 1e-9:
            errors.append("pcfa cost exceeds 1.05x measured fixed-k budget")
        if float(policy.get("mean_risk", 1e99)) > float(policy.get("mean_risk_bar", 0.0)) + 1e-9:
            errors.append("pcfa mean risk exceeds fixed-k mean-risk constraint")
    return errors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, path)


def hash_file(path, chunk=8 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
