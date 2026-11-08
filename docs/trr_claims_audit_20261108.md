# TRR claims audit

The supplied TRR report is a useful mechanism and stage-boundary record. Its
raw run artifacts were not present in the current repository mirror, so the
numeric rows remain `USER_SUPPLIED_REPORT_UNVERIFIED` until the original JSON,
checkpoints, code/config/data SHA files, and freeze/HELD manifests are copied.

## Supported after artifact recovery

- Parameter-space native-versus-refined encoder feedback can be a mechanism
  measurement if the same encoder, features, objective, and FIT calibration
  samples are used.
- The exact feedback decomposition in the report is algebraically valid.
- A nested detached-feature readout sequence is a distinct implementation
  object from the detached FD-SDPA head-only control.
- The 30/50/200 horizon comparison is a useful boundary design, provided the
  rows are recovered and independently recomputed.
- Finite-action DP is exact only for the registered proxy objective and budget.

## Required corrections before using the TRR numbers as primary evidence

1. `trr_runner.py --eval-test` reads test directly per run and has no freeze,
   one-time HELD manifest, or code/config/data SHA binding. Re-run or wrap it
   with the current freeze/evaluation protocol.
2. The runner does not create an independent calibration split. Any policy or
   action selection must use a declared FIT calibration subset and must record
   its size and class counts.
3. The paper mentions a CIFAR-10 matrix while the experiment section labels it
   not executed. Keep it `REGISTERED_NOT_EXECUTED` until artifacts exist.
4. The 30-epoch headline row must be checked per seed. A single-seed result
   cannot be written as a multi-seed strict improvement.
5. The empirical-Bernstein theorem requires bounded feedback-risk samples. Add
   an explicit clip value and clip-rate evidence, or state the boundedness
   assumption and its empirical verification.
6. The SGD noise-floor recursion needs independence or a bounded conditional
   second-moment assumption. It is not automatic for a general nonquadratic
   readout objective.
7. Replace class-specific shared-readout notation (W_c,W_c^*) with a
   class-conditional residual or row-conditioned quantity unless separate
   readouts are actually trained.
8. Keep the forward-offset lag statement explicitly linearized and conditional
   on fixed drift and controlled curvature changes.
9. Remove or weaken the `Omega(1)` Gaussian separation claim unless constants
   and parameter ranges are supplied.
10. Align the action domain with the implementation. If `k=0` is legal, the
    method definition must include it.
11. The bilevel inner loop currently reuses the training batch. Treat the
    resulting certificate as a proxy or add cross-fitting.
12. Resolve the 200-epoch status contradiction between the experiment section
    and discussion.

The recovered inventory also contains a `main200b/bilevel_k1_s3409.json` row
with test Macro/Tail `0.4977/0.2280`, which does not match the supplied report's
`0.5028/0.1780` seed-3409 row. This is a configuration or run-identity
discrepancy that must be resolved from code/config manifests before any
200-epoch aggregate is recomputed.

The current FD-SDPA audits remain separate. They establish the implementation
and cost history, while TRR should become the mechanism line only after this
artifact-level reconciliation.
