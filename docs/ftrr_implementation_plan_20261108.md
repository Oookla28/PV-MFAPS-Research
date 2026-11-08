# FTRR implementation plan

## Target

Turn TRR's short-horizon refinement effect into a constrained training
algorithm whose refinement strength is selected by a measurable feedback
certificate. The new object is **Feedback Trust-Region Refinement (FTRR)**.

## Algorithm

For a FIT calibration batch, let (g_0) be the native encoder gradient and
(g_k) the gradient induced by the (k)-step nested readout. The update uses

\[
g_{\alpha,k}=g_0+\alpha(g_k-g_0),
\]

where (alpha) is the largest value in ([0,\bar\alpha]) satisfying

\[
\frac{\langle g_0,g_{\alpha,k}\rangle}
{\|g_0\|\|g_{\alpha,k}\|}\ge\xi .
\]

The candidate is admitted only when a lower confidence bound on its paired
feedback-risk reduction exceeds measured compute and switching costs. If no
candidate is admitted, the stage uses (k=0,alpha=0). A stage also stops
refinement when the estimated stochastic noise floor dominates the remaining
contraction benefit.

## Training path

1. Compute encoder features once and detach a calibration copy.
2. Build a nested readout sequence on the full calibration batch.
3. Measure native and refined encoder gradients on the same calibration samples.
4. Solve the finite class-action allocation with the existing exact quantized
   DP under a stage budget.
5. Project every accepted feedback update into the native-gradient cone.
6. Update the slow readout from its detached objective and the encoder from the
   accepted FTRR gradient.

The head-only path remains an ablation. The algorithm claim uses the encoder
feedback path. FIT calibration is disjoint from the training reporting subset,
and test/HELD is read only after all FIT checkpoints are frozen.

The current smoke caps the parameter-space calibration batch at 128 samples to
keep gradient certificate cost bounded. This cap is part of the smoke
configuration and must be recorded with calibration size and data hash. The
calibration is FIT-proxy evidence and is not cross-fitting.

## Required experiments

- Offline numerical smoke for cone feasibility, alpha monotonicity, finite-action
  DP exactness, and stage stopping at a synthetic noise floor.
- One-batch real-checkpoint gradient smoke with `held_reads=0`.
- 30/50/200 epoch CIFAR-100-LT matched runs with LA, TRR, and FTRR.
- A matched-cost timing run without unrelated GPU jobs.

## Evidence boundary

The cone and allocation statements are conditional on the calibration estimates
and smoothness assumptions. They do not imply held generalization. The first
algorithm gate requires a held Tail-10% non-inferiority margin, a positive
feedback-risk reduction, and no wall-clock increase under an uncontended
matched protocol.
