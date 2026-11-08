# FD-SDPA novelty and baseline constraints

The dynamic allocation line is not novel merely because it uses a stage schedule,
CVaR/KL-DRO, empirical-Bernstein bounds, or finite-action dynamic programming.
Those are established ingredients. The defensible contribution must be the
combination of a class-conditional feedback-risk signal, a paired confidence
gate for an approximate inner solver, and an exact finite resource allocation
under measured cost.

## Required comparisons

- pure Logit Adjustment, because it is the strongest current CIFAR-100-LT
  baseline in this project;
- fixed-k and oracle finite-action DP;
- independent UCB/LCB gate, paired-difference gate, and no gate;
- frequency, loss, and gradient-norm signals as control schedules;
- official LDAM-DRW implementation before using LDAM as a performance claim.

## Claim boundary

Paired empirical-Bernstein coverage is a calibration-split proxy result. It is
not a held/test generalization theorem. A held gain must be demonstrated with a
frozen FIT protocol, one-time held evaluation, and matched measured cost.
Without that evidence the result remains a mechanism or boundary result.
