# Stage-Aware Annealing

The long-horizon issue is represented by a registered strategy specification. Let `s=0,...,S-1` index training stages, let `F_s(B_s)` be the finite Pareto frontier produced from FIT risk and measured cost tables, and let `J_s(A)` be the KL-robust risk of a shared policy `A` on that frontier. We define

\[
\pi_s^* = \arg\min_{\pi\in\Delta(F_s)}
\mathbb{E}_{A\sim\pi}[J_s(A)]
+\tau_s\,\mathrm{KL}(\pi\|\mathrm{Unif}(F_s)).
\]

The finite-problem solution is

\[
\pi_s^*(A)=
\frac{\exp[-J_s(A)/\tau_s]}
{\sum_{A'\in F_s(B_s)}\exp[-J_s(A')/\tau_s]},
\qquad
\tau_s=\tau_{\max}
\left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{s/(S-1)}.
\]

A hard shared action vector is sampled once at the beginning of each stage and reused through that stage. The total measured budget obeys `sum_s L_s B_s <= B_total`. Every sampled action remains on the registered finite grid and satisfies `k=0 => alpha=0`. Stage manifests record the frontier hash, sampled policy hash, temperature, stage length, RNG seed, entropy and physical native/reference/solver counters.

This strategy has not been trained or evaluated. It is a follow-up specification for the 50-step boundary and is not part of the executed PlantSPD result chain.
