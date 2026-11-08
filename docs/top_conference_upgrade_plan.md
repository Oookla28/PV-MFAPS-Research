# Stage-Aware Robust Pareto Allocation: research upgrade plan

**Status:** registered proposal; theorem statements below are conditional on the stated assumptions. No new accuracy or generalization result is claimed by this document.

## 1. What the current evidence does and does not show

The 30-epoch CIFAR-100-LT/IF=100 run is a short-horizon, budgeted benchmark. It is not a community-standard SOTA comparison: published CIFAR-LT protocols commonly use 180--200 epochs and a scheduled learning-rate decay. The executed Fused matrix currently contains ERM, Fixed-k, Logit Adjustment, LDAM-DRW, and PV-MFAPS. The current PV-MFAPS row does not dominate Fixed-k or Logit Adjustment on both macro accuracy and tail-10%. The LDAM rows collapsed under the present custom recipe and are therefore a baseline-integration failure until the official recipe is reproduced.

The system result and the algorithm result must stay separate:

- **System:** Fused and Legacy implement the same logical actions; the comparison measures physical encoder work and wall-clock overhead.
- **Algorithm:** the allocation policy must improve a held-out class-risk objective under a matched budget. A speedup alone is not an algorithmic accuracy claim.

## 2. Why a hard stage floor is not a valid algorithm

The rule “after 20% of training, force every class to use (k\ge 1)” has no safety or optimality guarantee. Let

\[
U_{c,p,s}(k)=b_{c,p,s}\rho_{c,p,s}^{2k}+S_{c,p,s}(k)+E_{c,p,s}(k)
\]

be a class/partition/stage risk envelope and let (C_s(k)) be measured cost. The move (k\to k+1) is admissible only when the certified marginal gain is positive:

\[
\Delta_{c,s}(k)=U_{c,s}(k)-U_{c,s}(k+1)
  > \lambda_s\bigl(C_s(k+1)-C_s(k)\bigr)
  + \lambda_{\rm sw}d_s(k+1,k_s^{\rm prev})
  + 2r_{c,s}.
\]

Here (r_{c,s}) is a confidence radius obtained on a calibration split independent of the split used to select the action. If this inequality is not certified, the policy retains (k=0) (or the previous action). Thus a stage transition can repair long-horizon underfitting without forcing a variance-increasing action.

For a constant-step inner update with smoothness β and gradient sensitivity (LG/n_c), a standard stability recursion gives

\[
d_{j+1}\le(1+\beta\eta)d_j+\frac{2LG\eta}{n_c},\qquad
S_c(k)\le\frac{2LG}{n_c\beta}\left[(1+\beta\eta)^k-1\right],
\]

for β>0 and (d_0=0). The frequently used γ^k variance term is therefore a model-specific proxy with γ=1+βη; it is not a general theorem without these assumptions.

## 3. Proposed algorithmic object

We define a finite action at stage (s) as

\[
A_s=\{(k_{c,s},\alpha_{c,s})\}_{c=1}^C,
\qquad k_{c,s}\in\mathcal K_s,
\quad \alpha_{c,s}\in\mathcal A_s,
\quad k=0\Rightarrow\alpha=0.
\]

The stage envelope is

\[
U_{p,s}(A_s)=\frac1C\sum_{c=1}^{C}
\omega_{c,s}\left[b_{c,p,s}\rho_{c,p,s}^{2k_{c,s}}
+S_{c,p,s}(k_{c,s})+E_{c,p,s}(k_{c,s},\alpha_{c,s})\right]
+\eta_s\left\|g^{\rm enc}_{p,s}(A_s)-g^{\rm ref}_{p,s}\right\|_2^2.
\]

The partition-robust objective uses the KL ambiguity set

\[
\mathcal R_{\varepsilon_s}(A_s)=
\sup_{q\in\Delta_P:\,D_{\rm KL}(q\|u)\le\varepsilon_s}
\sum_{p\in P}q_pU_{p,s}(A_s).
\]

For finite (P), its dual form is

\[
\mathcal R_{\varepsilon_s}(A_s)=
\inf_{\tau>0}\left\{
\tau\varepsilon_s+\tau\log\sum_{p\in P}u_p
\exp\left(\frac{U_{p,s}(A_s)}{\tau}\right)\right\}.
\]

The sequence-level problem is

\[
\min_{A_0,\ldots,A_{S-1}}
\sum_{s=0}^{S-1}\omega_s
\left[\mathcal R_{\varepsilon_s}(A_s)+\mu_s\bar U_s(A_s)+\lambda_s C_s(A_s)\right]
+\lambda_{\rm sw}\sum_{s>0}d(A_s,A_{s-1})
\]

subject to

\[
\sum_{s=0}^{S-1}L_sC_s(A_s)\le B_{\rm total},
\qquad A_s\in\mathcal F_s,
\]

where (mathcal F_s) contains only legal actions and certified activations. The macro term (ar U_s) prevents the minimax objective from starving intermediate classes.

## 4. Exact discrete solver and guarantee

Quantize cost into (B_q) integer units. Define the Bellman value

\[
V_s(b,A_{s-1})=\min_{A_s\in\mathcal F_s:\,L_sC_s(A_s)\le b}
\left[J_s(A_s)+V_{s+1}(b-L_sC_s(A_s),A_s)\right],
\]

with (J_s) equal to the bracketed stage objective plus switching cost. Backtracking returns a globally optimal sequence over the finite frontier (mathcal F_0\times\cdots\times\mathcal F_{S-1}), up to cost quantization. Its complexity is (O(SB_q|\mathcal F|^2)) in the dense implementation.

If the cross-fitted envelope estimates satisfy a simultaneous error bound

\[
|\widehat J_s(A)-J_s(A)|\le \xi_s
\quad\text{for all }s,A
\]

with probability at least (1-\delta), the sequence returned by estimated DP has true objective at most

\[
J(\widehat\pi)-J(\pi^*)\le2\sum_s\omega_s\xi_s
\]

above the best finite-grid sequence, plus the explicitly reported cost-quantization error. If Gibbs sampling is used instead of deterministic backtracking, the additional optimization gap is bounded by

\[
\sum_s\omega_s\tau_s\log|\mathcal F_s|,
\qquad
\pi_s(A)\propto\exp[-\widehat J_s(A)/\tau_s].
\]

This is an optimization approximation bound, not a generalization theorem. Generalization requires the independent calibration bound and its confidence assumptions.

## 5. Stage-aware continuation rule

Stage-aware annealing changes the cost price, risk ambiguity, and solver temperature using only FIT/calibration information:

\[
\lambda_s=\lambda_0\exp(-a s/S),\quad
\varepsilon_s=\varepsilon_0+b s/S,\quad
\tau_s=\tau_{\max}(\tau_{\min}/\tau_{\max})^{s/(S-1)}.
\]

The schedule is not allowed to override the certified marginal-gain gate. In particular, a later stage may activate (k>0) only if the inequality in Section 2 is satisfied. This separates stage adaptation from the already-known deferred re-weighting schedule in LDAM-DRW.

## 6. Required experiments

### Must-run A: baseline integrity

1. Reproduce LDAM-DRW using the official implementation or a line-by-line equivalent recipe, including normalized margin, DRW switch, optimizer, and scheduler. The current collapsed rows remain a failed integration attempt.
2. Finish the current Legacy PV run and compare it with Fused PV on identical seeds, checkpoints, actions, and batch order.
3. Add a `torch.compile` or CUDA-Graph control if available; otherwise record it as registered but unavailable. A custom fused path is not a novelty claim without a modern compiler baseline.

### Must-run B: algorithm isolation

Run the same total update budget and seeds for:

- Fixed-k;
- current PV-MFAPS;
- hard-floor stage rule (negative control);
- risk-envelope gate without stage schedule;
- full stage-aware robust allocation (proposed);
- Logit Adjustment and a verified LDAM-DRW baseline.

Report macro accuracy, tail-10%, per-class risk, action histograms, risk-envelope calibration error, wall-clock, and measured FLOPs. Use a fresh calibration split to choose actions and a frozen test/HELD split only once.

### Must-run C: horizon test

Use 8/30/50 epochs as a controlled horizon study. This supports a stage-dependence claim only. For community benchmark claims, add a standard 180--200 epoch protocol; 30/50 epochs alone are budgeted short-horizon evidence.

### Must-run D: cross-benchmark boundary

Run CIFAR-10-LT/IF=100 with the same code and a pre-registered seed set. Treat a 50-epoch run as a budgeted transfer check; do not call it standard SOTA without the standard long schedule.

### Must-run E: statistical protocol

Use five seeds only after the method and hyperparameters are frozen. Report mean, standard deviation, paired per-example confidence intervals, and the full per-class tail overlap. Do not select ε, λ, or the stage schedule from HELD/test results.

## 7. Decision gates

- **Algorithm gate:** the proposed allocation must improve the pre-registered held-out risk objective over Fixed-k and the verified long-tail baselines at matched wall-clock, or the claim is reduced to a mechanism/boundary result.
- **Stage gate:** the stage-aware policy must outperform the hard-floor negative control on at least two horizons without violating the certified budget or activation inequality.
- **System gate:** Fused must preserve action sequence and state within the equivalence tolerance and reduce physical encoder work against Legacy; any accuracy improvement is reported separately.
- **Transfer gate:** CIFAR-10-LT and standard-length CIFAR-100-LT are independent evidence; a short-horizon transfer run cannot be used as SOTA evidence.

## 8. Current execution status

- **EXECUTED:** CIFAR-100-LT 30-epoch Fused matrix; one-time test evaluation; smoke and data checksum; LDAM schedule rerun (failed integration); Fused action/cost counters.
- **RUNNING:** Legacy PV 30-epoch comparison.
- **REGISTERED_NOT_EXECUTED:** verified official LDAM recipe, stage-aware robust allocation, 50-epoch horizon matrix, CIFAR-10-LT, compiler/CUDA-Graph control, five-seed confirmation.
- **UNVERIFIED:** generalization, SOTA, universal tail gain, cross-task transfer.
