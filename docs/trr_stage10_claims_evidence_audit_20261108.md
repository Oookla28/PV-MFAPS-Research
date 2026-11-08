# TRR Stage-10 claims and evidence audit (2026-11-08)

## Executed evidence

| Claim | Evidence | Status |
|---|---|---|
| Stage-TRR-10 improves both Macro-F1 and Tail-10% over matched LA on CIFAR-100-LT at 30 epochs | Two groups of three frozen checkpoints, one evaluator read per group, 60,000 test reads total, paired deltas +0.0176 and +0.0547 | **PARTIAL / EXPLORATORY** |
| The same fixed stage policy transfers to CIFAR-10-LT IR=100 | Three paired seeds, correct five-batch data hash, one evaluator read per group, paired deltas +0.0372 and +0.0397 | **PARTIAL / CROSS-DOMAIN EXPLORATORY** |
| The mechanism is horizon-stable | At 50 epochs Macro delta +0.0046, Tail delta -0.0033 | **NOT SUPPORTED for universal improvement** |
| Stage-aware TRR is a distinct bilevel update | The runner performs an explicit one-step detached readout refinement for the first N=10 epochs, then joint logit-adjusted training | **EXECUTED IMPLEMENTATION** |
| The finite-action switching bound is valid under its certificate event | Conditional surrogate-risk proposition in docs/ftrr_stage_switching_bound_20261108.md | **CONDITIONAL THEORY ONLY** |

## Protocol state

- 30-epoch CIFAR-100-LT prefix10 and LA groups each passed FIT freeze and one-time evaluator.
- 50-epoch CIFAR-100-LT prefix10 and LA groups each passed FIT freeze and one-time evaluator.
- CIFAR-10-LT prefix10 and LA groups each passed FIT freeze and one-time evaluator.
- All invalid optimizer-semantics runs are excluded.
- All old direct-test pilot rows remain exploratory and were not merged into the frozen summaries.
- The 30-epoch prefix length was selected after a direct-test pilot. Therefore the 30-epoch CIFAR-100 result is exploratory even though the final groups were frozen correctly.
- CIFAR-10 uses a distinct five-batch data hash b937bd...4e1a6. The earlier smoke with the CIFAR-100 hash is invalid and excluded.

## Claims that remain unverified

- SOTA against the full long-tail literature.
- Universal generalization beyond CIFAR-100-LT and CIFAR-10-LT.
- A confirmatory result with a stage length fixed before any test read.
- A positive Tail-10% result at long horizons.
- A non-exploratory statistical significance claim with more than three seeds.

## Safe paper-level wording

The evidence supports a stage-aware feedback mechanism that consistently improves both
metrics in the 30-epoch exploratory setting on two long-tailed CIFAR benchmarks.
It does not support a universal or horizon-independent improvement claim.
