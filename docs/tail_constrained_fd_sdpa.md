# Tail-constrained feedback allocation

The encoder-feedback run improved held Macro-F1 but reduced Tail-10%. A
tail-aware variant is therefore registered as a separate hypothesis, not a
post-hoc reinterpretation of that result.

Let `T` be the lowest-count 10% classes. At stage `s`, the finite-action DP
minimizes the partition-robust surrogate with weights

`w[c,s] = 1/C + nu_s * 1[c in T] / |T|`,

where `nu_s >= 0` is a FIT-only dual variable. Given a registered tail-risk
target `tau_T`, update it only from the calibration split:

`nu[s+1] = max(0, nu[s] + eta_nu * (R_hat[T,s] - tau_T))`.

The DP still enforces the measured compute budget and the same safe activation
gate. This is a constrained surrogate allocation, not a held guarantee. The
variant is convincing only if it improves held Tail-10% without violating the
measured wall-clock budget and without choosing `tau_T` from HELD.
