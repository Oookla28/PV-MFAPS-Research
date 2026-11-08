# TRR and FD-SDPA synthesis

The supplied TRR report changes the preferred method line. TRR implements the
feedback object directly: a nested readout sequence is built on detached
features, the encoder receives the selected refined-readout feedback, and the
live readout is treated as a slow variable. The current FD-SDPA LA runs are a
useful implementation and certificate study, but their detached main arm
changes the readout update without changing encoder feedback. The encoder
feedback arm fixes that semantic issue, improves held Macro-F1 by 0.39 pp over
LA, and loses 0.57 pp on Tail-10% while suffering GPU-contention wall overhead.

The TRR report supplies a stronger mechanism narrative if its raw artifacts are
available and independently bound to code/data SHA. Its reported 30-epoch
gain, 50-epoch reversal, and 200-epoch seed dependence define a stage boundary.
They should be kept separate from the FD-SDPA artifacts until the original
JSON/checkpoint/freeze files are verified.

## Evidence classification

| item | current status |
|---|---|
| parameter-space feedback cosine 0.388 and tail correlation | supplied TRR report, raw artifact verification pending |
| TRR 30/50/200 metrics | supplied TRR report, raw artifact verification pending |
| FD-SDPA detached 30x3 | EXECUTED, claims audit complete, accuracy gate failed |
| FD-SDPA encoder-feedback 30x3 | EXECUTED, Macro trade-off, Tail and wall gates failed |
| paired gate | offline/smoke boundary only |
| tail-constrained variant | UNVERIFIED_BLOCKED_BY_CONTENTION |

## Unified method direction

The defensible unified object is the class-conditional feedback error

\[
e_{c,t}(k)=g_{c,t}(\psi,W_t^{(k)})-g_{c,t}(\psi,W_t^*),
\]

with a nested readout sequence and a finite allocation policy. The paper should
make the stage boundary explicit. Refinement is useful while the readout lag is
large and can be harmful after the lag reaches its noise floor. A stage-aware
policy must therefore be triggered by a measured feedback certificate and stop
refinement when the certificate no longer supports a positive action.

The TRR evidence is stronger for the mechanism because it uses a slow readout
and true encoder feedback. FD-SDPA contributes the allocation/certificate
infrastructure and the negative boundary analysis. Neither line currently
supports a converged-stage SOTA claim.

## Verification needed before merging

1. Recover the original TRR JSON, checkpoints, code SHA, data SHA, and freeze/
   HELD manifests.
2. Confirm that the 30/50/200 runs used identical test access semantics and
   that no HELD result was used for selecting `inner-k`, stage length, or
   allocation.
3. Recompute paired differences and confidence intervals from per-seed rows.
4. Check that the TRR nested sequence uses the same prior, data order,
   augmentation, optimizer, and backbone as the LA baseline.
5. Compare TRR against the current encoder-feedback FD-SDPA under a common
   uncontended timing protocol.
