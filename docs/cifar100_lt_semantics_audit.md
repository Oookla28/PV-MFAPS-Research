# CIFAR-100-LT execution-path audit (2026-11-08)

This note records a semantic issue found after the first 30-epoch CIFAR-100-LT matrix. It is intentionally separate from the result summary so that the old artifacts remain traceable.

## Issue

For an arm with `active` correction groups, the legacy branch accumulated

\[
L_{\mathrm{legacy}}=L_0+0.05\sum_{g=1}^{\mathrm{active}}L_g,
\]

whereas the fused branch evaluated one auxiliary loss and accumulated

\[
L_{\mathrm{fused}}=L_0+0.05L_1.
\]

The fused implementation incremented `logical_actions` by `active`, but it did not multiply the auxiliary loss by `active`. In addition, the legacy branch recomputed the encoder (and its training-mode BatchNorm state), while the fused branch reused a detached feature tensor. Therefore the two paths are not mathematically equivalent training procedures.

## Consequence

The first CIFAR-100-LT fused/legacy runs are valid executed measurements of their respective code paths, but they are **not a fair same-update equivalence comparison**. They must not be used to claim that fusion preserves the algorithmic trajectory. The legacy path's wall-clock measurement is still useful as a diagnostic of duplicate encoder work, subject to this scope.

## Required repair before a system claim

Define the action objective first. If all active groups share the same correction head, the shared-feature implementation must use

\[
L_{\mathrm{fused}}=L_0+0.05\,\mathrm{active}\,L_1,
\]

with the same detach convention as the intended algorithm. Then run a real-checkpoint one-step equivalence test (actions, counters, state delta, BatchNorm buffers, and held reads) before any corrected matched matrix. Keep old outputs in an `invalid_for_equivalence` archive and never overwrite them.
