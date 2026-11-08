from __future__ import annotations

import torch

from longtail_losses import balanced_softmax_logits, balanced_softmax_loss, logit_adjusted_logits


def main() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2, 3, 1])
    counts = torch.tensor([5, 10, 20, 40])
    z = balanced_softmax_logits(logits, counts)
    assert z.shape == logits.shape
    assert torch.allclose(z - logits, counts.float().log().expand_as(logits))
    loss = balanced_softmax_loss(logits, target, counts)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.allclose(logit_adjusted_logits(logits.detach(), counts, 0.0), logits.detach())
    try:
        balanced_softmax_logits(logits.detach(), torch.tensor([1, 0, 2, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-count contract did not fail")
    print("LONGTAIL_LOSSES_TEST_PASS")


if __name__ == "__main__":
    main()

