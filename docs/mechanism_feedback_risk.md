# Mechanism line: feedback risk and class-level tail degradation

Status: `UNVERIFIED` for PlantSPD; `EXECUTED` only for the synthetic benchmark
(`out/mechanism_synthetic_2026-11-06/`). HELD was not read.

## 0. Objects and notation

- Encoder feature map `f_theta(x) in R^d`; for class `c`, `z = vec(phi(C_c))`
  where `C_c` is the view-averaged second-moment (covariance) statistic and
  `phi` is either identity (`raw`) or matrix logarithm (`log`).
- Native readout: squared-distance head with prototype `p_c`, temperature
  `tau_c`; its feature-space gradient is `2(z - p_c)/tau_c`.
- Reference readout: affine probe `W z + b` fitted on FIT only.
- Feedback error: `e_c = g_actual,c - g_ref,c`, where both gradients are taken
  with respect to the encoder parameters on the same FIT batch.
- Class proxies: `A_c` (bias scale), `rho_c` (spectral decay), `sigma_c`
  (dispersion), `n_c` (support). These are **proxies** unless a separate
  calibration certifies them as bound constants.

## 1. M1: norm bound for the feedback error

**Assumption M1-a (matrix-function regularity).** `phi` is Fréchet
differentiable on the spectrum of `C_c`, with Jacobian `J_c = D phi(C_c)`.

**Assumption M1-b (readout residual).** The native and reference readouts
differ by a bounded linear residual `Delta W_c` on the class subspace, and the
task weighting `D_c = diag(1/tau_c)` is positive definite with
`lambda_max(D_c) <= 1/tau_min`.

**Proposition M1.** For every class `c`,

```text
||e_c||_F  <=  ||J_c||_op * ||D_c phi(C_c)||_F * ||Delta W_c||_F
```

*Proof.* Write `z = vec(phi(C))`. The chain rule gives the feature-space
error `delta z_c = J_c vec(E_c)` for a perturbation `E_c` induced by the
readout residual, and the parameter-space error is the readout Jacobian applied
to `delta z_c`. Submultiplicativity of the Frobenius norm and
`||A B||_F <= ||A||_op ||B||_F` give the claim. The temperature enters through
`D_c`, whose largest eigenvalue is `1/tau_min`. ∎

**Spectral Jacobian, Raw vs Log.** In the eigenbasis `C_c = sum_i lambda_i u_i u_i^T`
the Fréchet derivative acts as

```text
D phi(C_c)[E] = sum_{i,j} phi^[1](lambda_i, lambda_j) u_i u_j^T E u_j u_i^T
phi^[1](a,b) = (phi(a) - phi(b)) / (a - b)   (a != b),   phi'(a)  (a = b)
```

- `raw`: `phi = id`, `phi^[1] = 1`, so `J_c = I` and `||J_c||_op = 1`.
- `log`: `phi^[1](a,b) = log(a/b)/(a-b)`, hence
  `||J_c||_op <= 1/lambda_min(C_c)` and the bound inflates by the condition
  number of `C_c` relative to `raw`.

This is the precise sense in which the Log variant converts small-eigenvalue
directions into large feedback sensitivity. It is a statement about the
**Loewner divided difference**, not about classification accuracy.

**Schur bounds.** The Frobenius bound above is usually loose. Two tighter
entrywise variants are:

```text
||A||_2 <= sqrt(||A||_1 ||A||_inf) <= sqrt(m n) * max_{ij} |A_ij|
```

The max-entry Schur bound is what a task-direction proxy can tighten further:
if the residual is concentrated on a known task direction `u_c`, then
`||Delta W_c||_F` is replaced by `||Delta W_c u_c||`, which is the quantity the
synthetic benchmark measures as the **task-direction proxy**. The proxy is not
a theorem constant; it is a measurable quantity, and it is reported separately.

**Bound looseness (synthetic).** In `MECHANISM_SYNTHETIC_RESULT.json` the
Frobenius bound / observed ratio has median `~1.3e-3`, i.e. the Schur bound
overestimates by roughly three orders of magnitude under random orientation.
This is expected: a rank-one residual inside a full-rank Frobenius envelope is
a worst-case bound. Any claim that uses the bound must therefore report the
ratio, not just the inequality.

## 2. M2: perturbation of margin and surrogate risk

**Assumption M2-a (margin Lipschitz).** For the class margin
`m_c(w) = s_c(w) - max_{c' != c} s_{c'}(w)`, the logit map is locally
`L_m`-Lipschitz in the encoder parameters on the FIT neighbourhood.

**Assumption M2-b (surrogate loss).** The class surrogate loss
`ell_c(m)` is `L_ell`-Lipschitz and bounded in `[0, M]`.

**Assumption M2-c (inner step).** One feedback update is
`w' = w - eta * e_c` with `eta > 0` and `||e_c||` bounded by Proposition M1.

**Proposition M2.**

```text
|m_c(w') - m_c(w)| <= L_m * eta * ||e_c||
|R_c(w') - R_c(w)| <= L_ell * L_m * eta * ||e_c||
```

where `R_c(w) = E[ell_c(m_c(w))]` is the **surrogate class risk**.

*Proof.* Apply M2-a to the parameter displacement `eta e_c`, then M2-b to the
one-dimensional Lipschitz composition. ∎

**Scope limit (explicit).** Proposition M2 bounds a surrogate class risk. It
does **not** bound Macro-F1, which is a non-decomposable ranking/argmax
functional. No step in this derivation implies a Macro-F1 generalisation
theorem; Macro-F1 remains an independent FIT/HELD gate. Any text that says
otherwise is out of scope for this artifact.

## 3. M3: is kappa predictive after controlling for confounds?

**Registered test.** Regress/rank-correlate `kappa_c` with the later
degradation of class `c`, controlling for `1/n_c`, class loss, and class
gradient norm. If the partial correlation collapses, the mechanism claim must
be re-scoped: kappa would be a re-description of frequency/optimisation state
rather than an independent mechanism variable.

**Synthetic result (EXECUTED, synthetic only).**

| orientation regime | Spearman(kappa, degradation) | partial(kappa \| n, loss, grad) | Spearman(1/n, degradation) | A/B consistency |
|---|---|---|---|---|
| random | -0.051 | -0.034 | -0.011 | 0.625 |
| aligned | +0.039 | +0.000 | -0.011 | 0.500 |

Interpretation, stated conservatively:

1. In this synthetic construction the mechanism is **not identifiable** from
   spectral condition alone; random orientation of the task direction swamps
   the `rho` effect, and A/B consistency is at chance.
2. This is a **design limitation**, not evidence against the PlantSPD
   mechanism. It says the synthetic benchmark cannot certify M3.
3. Therefore M3 remains `UNVERIFIED` and must be tested on PlantSPD FIT
   artifacts with per-class recall, not on this simulator.

**PlantSPD-side test status.** `REGISTERED_NOT_EXECUTED`. The currently frozen
A100 pilot artifacts store class proxies (`A`, `rho`, `sigma`, support) and
aggregate FIT metrics, but not per-class post-training recall for every arm.
A proper M3 test needs either (a) per-class FIT recall written by a new
registered arm, or (b) an independent FIT split with per-class evaluation.
Neither is available in the current frozen artifacts, so no PlantSPD M3 number
is reported here.

## 4. Counterfactual dataset (synthetic)

Registered criterion: two classes A/B with equal `n`, equal loss, equal
gradient norm, but different spectral condition; mechanism is supported only if
`kappa_A > kappa_B` **and** degradation A > B.

Observed: consistency `0.625` (random orientation) and `0.500` (aligned), i.e.
the criterion is **not met**. Status: `UNVERIFIED` (counterfactual design did
not isolate the spectral variable).

Additional interventions executed:

- **Tail permutation**: permuting class identity leaves mean recall unchanged
  by construction (`delta_mean` ~ 0), confirming that recall alone cannot
  identify which class carries the risk.
- **Equal logits / different gradient**: forward margin equality does not
  imply equal `e_c`; the forward readout is not a sufficient statistic for
  backward feedback usability.

## 5. What this line can and cannot claim

Can claim (once PlantSPD FIT evidence exists):

- a measured class-level risk envelope from FIT proxies;
- an explicit, checkable norm bound with reported looseness;
- a negative/positive predictive test of `kappa` under stated controls.

Cannot claim:

- Macro-F1 guarantees;
- universal tail improvement;
- that `A_c`, `rho_c`, `sigma_c` are theorem constants;
- any statement about CIFAR-LT/ImageNet-LT from PlantSPD evidence.
