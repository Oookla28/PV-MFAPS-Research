# ARIS Result-to-Claim Gate — 2026-11-07

**Overall: WARN / narrow claims**

The clean PR chain supports one aggregate PlantSPD matched-budget result against fixed-k. It does not support a multi-budget empirical Pareto frontier, per-seed dominance, universal long-tail improvement, SOTA, cross-task transfer, or a population generalization theorem.

| Claim | Verdict | What is supported | Boundary |
|---|---|---|---|
| C1 | **partial** | Clean final: held Macro 0.4841 vs 0.4553 (+0.0288), tail 0.2976 vs 0.2659 (+0.0317), max wall 0.9974, max FLOPs 0.8889; 2 partitions x 3 seeds. | Aggregate point only; per pair Macro deltas are mixed, no multi-budget frontier or universal gain. |
| C2 | **no** | Aggregate PR is higher by +0.0080 Macro and +0.0198 Tail. | Wall ratio 1.0768 and FLOPs ratio ~1.0000, so it is not within the fixed-k 1.05 budget; per pair deltas are mixed. |
| C3 | **partial** | Real-checkpoint smoke: actions/counters equal, max state delta 5.77e-6 < 1e-5, physical encoder forwards 3 to 1. | Two one-step pairs and protocol-specific latency only; old A100 matched wall gate failed at 1.1913 and the 8-step profile is not a valid generic speed claim. |
| C4 | **no** | Finite quantized DP exactness and synthetic mechanism proxy are executed. | PlantSPD predictive mechanism M3, calibrated constants, Macro-F1 theorem, alpha-aware optimality, robust-cost theorem, and cross-domain validation are unverified. |
| C5 | **partial** | Exploratory frozen prediction capture: 6276 pairs, PR-only 773, fixed-only 537, exact pooled p=7.51e-11. | Post-hoc same endpoints; endpoint sign p=0.6875 and per-endpoint deltas are mixed. No confirmatory/generalization claim. |
| C6 | **no** | Available summary reports p0/seed3407/batch32 pilot with PR fixed-head Macro 0.2337 vs fixed 0.2687 and tied tail 0.1429. | Small pilot and local raw matched/state artifacts are missing; it is a boundary observation at most, not efficacy confirmation. |
| C7 | **no** | None. | CIFAR-LT/ImageNet-LT, second SPD and independent semantic review are registered-not-executed; community SOTA not measured. |
| C8 | **partial** | The manifest-linked PR clean chain has 24 FIT artifacts with held_reads=0 and one post-freeze HELD read of 25104 samples. | McNemar and 50-step made additional reads; sibling VALID_EPS001 provenance differs; wording must be local to the clean chain. |

## Required wording

- Say **aggregate matched-budget PlantSPD point** for the clean PR result.
- Say **exploratory paired prediction difference** for McNemar; include endpoint clustering/mixed deltas.
- Say **Fused equivalence and duplicate-work reduction in a real-checkpoint smoke** for the system result.
- Keep the finite-action DP as a solver certificate, not a neural-network generalization theorem.
- Treat the 50-step pilot as a negative/limited stage-boundary observation until raw artifacts are restored.
- Keep `REGISTERED_NOT_EXECUTED`, `UNVERIFIED`, and `INVALID/QUARANTINED` labels visible.
