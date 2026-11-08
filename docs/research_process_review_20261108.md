# Research process review

This review follows the chronological artifacts in docs/, the PlantSPD and CIFAR-LT
manifests, the selected claims audits, and the final figure inputs. It is a
source review for the public release. It does not replace a fresh training run.

## 1. Starting point

The first experiments compared Raw and Log SPD covariance mappings. The important
separation was between forward readout quality and the encoder feedback generated
by an approximate readout. The Loewner divided-difference representation of a
spectral derivative supplied the algebraic link.

## 2. Feedback-risk measurements

Fixed FIT checkpoints were used to compare native and sufficiently refined
readout responses. The resulting quantities A, rho, sigma, support n, and
feedback discrepancy were treated as FIT-calibrated proxies. The review keeps
these quantities tied to their calibration protocol and does not reinterpret
them as population constants.

## 3. Allocation prototypes

Mean-risk scheduling, CVaR scheduling, importance correction, trust-region
controls, and K-MFAPS were run as progressively more structured prototypes.
Negative or incomplete branches remain in the historical docs. The selected
finite-action DP is exact only for its supplied quantized action table and
measured budget.

## 4. PlantSPD partition robustness

The PlantSPD E01 protocol uses two paired FIT partitions with identical
per-class counts. The PR-PV-MFAPS epsilon sweep therefore tests partition-risk
geometry while holding frequency counts fixed. The public summaries contain the
action sweep, the two-class action changes, and the one-time post-freeze
evaluation record.

## 5. Fused execution

Legacy and Fused were paired on a real checkpoint. Action sequences and logical
counters agree, physical encoder work is reduced, and the state error is
reported separately from model quality. FLOPs, physical forwards, CUDA event
time, and end-to-end wall-clock are kept as different quantities.

## 6. Plugin branch

Balanced Softmax is exposed as a separate runner and output family. Native,
reference, solver, and fused correction all receive the same long-tail
objective. The plugin is not merged into the PV-MFAPS implementation and its
arm list is independently reproducible.

## 7. Statistical-gate diagnosis and repair

The FTRR/FD-SDPA branch exposed the statistical bottleneck. Several intermediate
runs were repaired or excluded after finding calibration-prefix bias, stale
slow-head updates, missing scheduler semantics, optimizer-path mismatch, or a
wrong CIFAR-10 data hash. The public package keeps the corrected code and the
selected audit summaries, while invalid artifacts are not used by the figure
scripts.

## 8. Stage-TRR scale extension

The final Stage-TRR-10 runner uses one explicit detached readout refinement for
the first ten epochs and then returns to joint Logit Adjustment. Three seeds were
run on CIFAR-100-LT and CIFAR-10-LT at imbalance ratio 100. Every group was
frozen before one post-training evaluator pass. A 50-epoch CIFAR-100-LT run
was kept as a horizon boundary check.

## 9. Public-release review

The public package was scanned for local paths, SSH hosts, passwords, tokens,
raw server logs, checkpoints, and large data files. It contains source code,
small JSON summaries, manifests, vector figures, protocol documents, tests,
and the AAAI-style LaTeX draft. External data and model states remain external
inputs. The release integrity list excludes the Git metadata directory and
records SHA256 for the public files.

## Reproduction order

1. Read README.md and docs/data_sources.md.
2. Run the contract tests and preflight.
3. Reproduce the PlantSPD chain or the CIFAR-LT Stage-TRR chain.
4. Freeze all training artifacts.
5. Run the evaluator exactly once for each frozen group.
6. Generate figures from the JSON summaries.
7. Read the evidence map and the boundary notes before reusing any number.
