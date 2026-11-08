# Theory and module map

This document is the mathematical companion to README.md. It describes the
objects implemented in the public package and the assumptions used by each
entry point.

## 1. SPD spectral map

For a symmetric positive-definite matrix
```math
C=U\mathrm{diag}(\lambda_1,\ldots,\lambda_d)U^\top,
```
a scalar spectral map $\phi$ acts as
```math
\phi(C)=U\mathrm{diag}(\phi(\lambda_1),\ldots,\phi(\lambda_d))U^\top.
```

For a symmetric perturbation $H$, its Frechet derivative is
```math
D\phi_C[H]
=
U\left(G_\phi(\lambda)\odot(U^\top H U)\right)U^\top,
```
where
```math
(G_\phi)_{ij}
=
\begin{cases}
\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}
{\lambda_i-\lambda_j}, & i\ne j,\\[6pt]
\phi'(\lambda_i), & i=j.
\end{cases}
```

The forward representation depends on $\phi(\lambda)$, while the feedback
Jacobian depends on the Loewner matrix $G_\phi$.

Relevant modules:

- `src/model/mechanism_synthetic.py`
- `src/model/pv_mfaps_risk_envelope.py`
- `docs/stage_trr_derivation.md`

## 2. Feedback gap

Let $z_\psi(x)$ be the representation and $W_t$ the current readout. Let
$\widetilde W_t$ be a more thoroughly optimized reference response on FIT.
The encoder feedback gap is
```math
e_t
=
\nabla_\psi L(\psi_t,W_t)
-
\nabla_\psi L(\psi_t,\widetilde W_t).
```

The relative diagnostic is
```math
\kappa_t
=
\frac{\lVert e_t\rVert}
{\lVert\nabla_\psi L(\psi_t,\widetilde W_t)\rVert+\delta}.
```

These quantities are measured on FIT batches. They are diagnostics for update
distortion and are not population generalization constants.

## 3. Finite-action feedback risk

For class $c$, partition $p$, inner readout depth $k$, and reference mixing
coefficient $\alpha$, the basic proxy is
```math
r_{c,p}(k,\alpha)
=
(1-\alpha)^2A_{c,p}^2\rho_{c,p}^{2k}
+
\alpha^2\frac{\sigma_{c,p}^2}{m q_c}.
```

Here $A_{c,p}$ is a FIT-calibrated residual scale, $\rho_{c,p}$ is a
readout contraction proxy, $\sigma_{c,p}$ is the batch-variation scale,
$n_c$ is the class support, $q_c$ is the class sampling quality, and $m$ is
the batch size.

## 4. KL partition robustness

For the paired PlantSPD partitions, let $\bar u=(1/2,1/2)$ and define
```math
\mathcal U_\varepsilon
=
\{u\in\Delta^2:
\mathrm{KL}(u\|\bar u)\le\varepsilon\}.
```

For a candidate action $a$,
```math
\mathcal E_c(a)
=
\sup_{u\in\mathcal U_\varepsilon}
\sum_{p=0}^{1}u_p\widehat R_{c,p}(a).
```

The KL dual used by the policy code is
```math
\mathcal E_c(a)
=
\inf_{\eta>0}
\left[
\eta\varepsilon+
\eta\log
\left(
\sum_{p=0}^{1}\bar u_p
\exp\left(\frac{\widehat R_{c,p}(a)}{\eta}\right)
\right)
\right].
```

The epsilon sweep changes the ambiguity set while keeping paired class
frequencies fixed.

Relevant modules:

- src/model/pcfa_policy.py
- src/validated_runtime/pcfa_policy.py
- configs/policies/

## 5. Quantized Pareto dynamic program

Let $\mathcal A$ be the finite action set and let $d_c(a)$ be the measured
cost of action $a$. With cost quantum $\Delta$, define
$\bar d_c(a)=\lceil d_c(a)/\Delta\rceil$.

The DP stores nondominated partition-risk vectors:
```math
\mathcal P_0(0)=\{(0,0;\varnothing)\},
```
and
```math
\mathcal P_j(b)
=
\mathrm{Pareto}
\left(
\bigcup_{a\in\mathcal A}
\left\{
(r_0+R_{j,0}(a),r_1+R_{j,1}(a);\pi\cup\{a\})
\right\}
\right).
```

The DP is exact for the supplied finite risk table, action grid, and cost
quantization. It is not a global continuous optimization theorem.

Relevant modules:

- src/model/pv_mfaps_dp.py
- src/model/pv_mfaps_dp_solver.py
- tests/test_pv_mfaps_dp.py

## 6. Stage-aware readout refinement

For long-tailed CIFAR experiments, Stage-TRR-$N$ uses one detached readout
refinement for the first $N$ epochs:
```math
W_t^{(1)}
=
W_t-\eta_h\nabla_W L_{\mathrm{LA}}(\psi_t,W_t),
```
followed by
```math
\psi_{t+1}
=
\psi_t-
\eta_\psi
\nabla_\psi
L_{\mathrm{LA}}(\psi_t,\mathrm{stopgrad}(W_t^{(1)})).
```

After epoch $N$, the standard joint Logit Adjustment objective is used:
```math
(\psi_{t+1},W_{t+1})
=
(\psi_t,W_t)
-
\eta_t\nabla_{\psi,W}
L_{\mathrm{LA}}(\psi_t,W_t).
```

A conditional finite-action stage bound is
```math
J(\widehat k_{1:T})-J(k^*_{1:T})
\le
2\sum_{t,c}\omega_c\epsilon_{t,c}
+
\epsilon_{\mathrm{quant}},
```
provided the simultaneous FIT certificate
$|\widehat U_{t,c}(k)-U_{t,c}(k)|\le\epsilon_{t,c}$ holds.

The bound is conditional on the certificate event. It does not imply a
test-set improvement or a universal horizon-independent gain.

Relevant modules:

- experiments/cifar100_lt/trr_audited_runner.py
- experiments/cifar100_lt/trr_audited_cifar10_runner.py
- experiments/cifar100_lt/trr_audited_evaluate.py

## 7. Fused execution

The Legacy path recomputes the encoder for correction groups. The Fused path
computes one feature tensor and reuses it for detached correction. The
equivalence smoke checks action identity, logical counters, and state error.

Relevant modules:

- src/validated_runtime/pcfa_equivalence_smoke.py
- src/validated_runtime/pcfa_cost_model.py
- src/validated_runtime/pcfa_cost_smoke.py

## 8. Evaluation contract

Training writes FIT-only JSON with held_reads=0. A freeze manifest records
code, data, configuration, JSON, and checkpoint hashes. The evaluator is
invoked only after all checkpoints in a group exist. It reads the evaluation
split once and writes metrics to a separate evaluation summary.
