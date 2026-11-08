"""Explicit long-tail losses for the post-pilot baseline branch.

This module is isolated from the running A100 pilot. Balanced Softmax must be
applied consistently to native, reference, and fused reference losses; merely
adding a prior to the native pass is an invalid comparison.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _log_counts(counts: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=dtype, device=device)
    if counts.ndim != 1 or torch.any(counts <= 0):
        raise ValueError("counts must be a one-dimensional positive tensor")
    return counts.log()


def balanced_softmax_logits(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Return logits used by Balanced Softmax cross-entropy.

    The class-count prior is applied inside the denominator for every native
    and reference loss. Evaluation must use the declared protocol separately;
    this function never changes the fixed-head evaluation rule.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    return logits + _log_counts(counts, dtype=logits.dtype, device=logits.device)


def balanced_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    counts: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.cross_entropy(
        balanced_softmax_logits(logits, counts), target, reduction=reduction
    )


def logit_adjusted_logits(logits: torch.Tensor, counts: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    if tau < 0:
        raise ValueError("tau must be non-negative")
    return logits + float(tau) * _log_counts(counts, dtype=logits.dtype, device=logits.device)

