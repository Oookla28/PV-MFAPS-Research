# Fused system line

Status: `EXECUTED` for the real-checkpoint equivalence gate and the 8-step
profile; per-checkpoint profiler attribution for the new branch is
`REGISTERED_NOT_EXECUTED` (it must run after the A100 pilot releases the GPU).

## 1. What is claimed

Fused PC-FA reuses a single encoder graph per step and applies the reference
correction on the detached head, instead of re-running the encoder once per
class group. Under the same logical action sequence and within the state
tolerance, this reduces physical encoder forwards/backwards and measured
FLOPs.

This is a **system implementation** claim. It does not imply algorithmic tail
gain, and it does not imply that a faster device makes the algorithm better.

## 2. Evidence already frozen (pilot branch, read-only)

From `out/pcfa_a100_actionspace_2026-11-06/PCFA_FUSED_EQUIVALENCE_SMOKE.json`
(2-step, real checkpoint `pst1r1_p0_s3407_raw`, batch 8):

| check | value |
|---|---|
| action sequence equal | true |
| logical counters equal | true |
| states ok | true |
| state max absolute delta | 5.77e-6 (limit 1e-5) |
| physical encoder forwards (legacy / fused) | 3 / 1 |
| held_reads | 0 |

From `preflight/FUSED_PROFILE_8STEP.json` (8 steps, same checkpoint):

| metric | value |
|---|---|
| fused wall mean | 0.160 s |
| native-only wall mean | 0.073 s |
| wall ratio mean / max | 2.27 / 3.63 |
| FLOPs ratio mean / max | 1.0001 / 1.0001 |
| peak allocated memory | 463 MB |
| op attribution (encoder / head-solver / other) | 44.5% / 24.4% / 31.0% |

**Important reading.** The profile compares a fused PC-FA step (which performs
reference/solver work) with a native-only step, so ratios above 1 are expected
and are **not** the matched cost gate. The matched gate compares PC-FA against
the fixed-k arm at the same logical action budget; in the completed pilot
configurations that ratio is 0.80 (FLOPs) and 0.62-0.99 (wall).

## 3. What the new branch still must produce

Registered for the post-pilot branch:

1. per-checkpoint legacy/fused pairs (both partitions, seeds 3407-3409);
2. `torch.profiler` attribution with a summary CSV: encoder kernel time,
   head/solver kernel time, H2D memcpy, CPU launch/synchronization, duplicate
   encoder forwards, kernel launches, CUDA-event wall, FLOPs;
3. exported trace + configuration SHA;
4. a two-panel figure: legacy multi-branch graph vs fused single graph.

Until (1)-(4) exist, the system claim is limited to the frozen equivalence and
profile above: **"fused execution preserves update semantics and reduces
duplicate encoder work"**, with no wall-clock generalisation beyond the
measured protocol.
