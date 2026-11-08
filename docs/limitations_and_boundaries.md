# Limitations and boundaries

Date: 2026-11-06. All statuses below are strict artifact statuses, not quality
judgements.

## 1. What is EXECUTED

| artifact | path | what it certifies |
|---|---|---|
| A100 pilot (in progress) | `out/pcfa_a100_actionspace_2026-11-06/` | registered grid executions; FIT only |
| fused equivalence | `.../PCFA_FUSED_EQUIVALENCE_SMOKE.json` | action/logical equality, state delta 5.77e-6 |
| fused 8-step profile | `.../preflight/FUSED_PROFILE_8STEP.json` | encoder/head attribution, peak memory |
| PV-MFAPS DP benchmark | `out/pv_mfaps_theory_2026-11-06/dp/PV_MFAPS_DP_BENCH.json` | DP exactness vs brute force, scale curve |
| synthetic mechanism benchmark | `out/mechanism_synthetic_2026-11-06/MECHANISM_SYNTHETIC_RESULT.json` | synthetic bounds/identifiability |
| Balanced Softmax local smoke | `pv_mfaps_artifacts/bs_pcfa_real_smoke_p1_s3407_v3/` | execution contract only (local MPS) |

## 2. What is REGISTERED_NOT_EXECUTED

- Balanced Softmax + PV-MFAPS plugin matrix (2 partitions x 3 seeds x 8 arms).
- Plugin HELD evaluation.
- Per-checkpoint legacy/fused profiler pairs for the new branch (the pilot owns the GPU).
- PlantSPD M3 test with per-class FIT recall.
- CIFAR10-LT / CIFAR100-LT / ImageNet-LT transfer.

## 3. What is UNVERIFIED or BOUNDARY

- **A100 pilot algorithm gate**: as of 15/36 configurations, every partition-1
  cell has Macro-F1 `-1.06pp` to `-1.72pp` below fixed-k, outside the
  registered `0.01` tolerance. If this persists, the selector will emit
  `NO_FIT_PARETO_CONFIG` and HELD will not be read.
- **Mechanism M3**: synthetic design is not identifiable (Spearman ~0 in both
  orientation regimes); no PlantSPD M3 number exists yet.
- **Risk-envelope constants**: `A_c`, `rho_c`, `sigma_c`, `xi` remain proxies.
  Only the stability and empirical-Bernstein terms have explicit assumptions.
- **Macro-F1**: never implied by the surrogate envelope.

## 4. Hard prohibitions honoured

- No HELD read before FIT freeze; all pilot artifacts report `held_reads=0`.
- No reuse of the pilot output directory; new branches write to new directories
  and a separate code directory (`code_pv_2026-11-06/`).
- No offline DP action fed into the running selector.
- No deletion of negative results: the partition-1 Macro deficit, the negative
  synthetic M3 result, and the pilot boundary status are all retained.
- No community SOTA, universal improvement, Macro-F1 guarantee, or
  Plant-to-CIFAR/ImageNet transfer claim.

## 5. Decision rule at the end of the pilot

```text
if selector emits NO_FIT_PARETO_CONFIG:
    mechanism/DP/system lines keep their own evidence;
    algorithm-tail claim stays UNVERIFIED; HELD is not read.
if selector emits configs:
    final matched + freeze + one-time HELD per the frozen protocol;
    any post-HELD tuning is INVALID_CONFIG.
```
