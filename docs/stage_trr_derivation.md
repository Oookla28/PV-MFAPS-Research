# From SPD feedback to Stage-TRR

This note explains the bridge between the SPD diagnostic and the stage-aware
CIFAR intervention. The two experiments use different data representations,
so the connection is a mechanism transfer rather than an assertion that a
CIFAR feature is itself an SPD matrix.

## 1. The local feedback object

For a symmetric positive definite matrix

$$
C=U\operatorname{diag}(\lambda_1,\ldots,\lambda_d)U^\top,
$$

a spectral map is

$$
\phi(C)=U\operatorname{diag}(\phi(\lambda_1),\ldots,\phi(\lambda_d))U^\top.
$$

For a symmetric perturbation $H$, its Frechet derivative is

$$
D\phi_C[H]
=U\left(G_\phi(\Lambda)\odot U^\top H U\right)U^\top,
$$

with

$$
(G_\phi)_{ij}=\begin{cases}
\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}{\lambda_i-\lambda_j}, & i\ne j,\\[6pt]
\phi'(\lambda_i), & i=j.
\end{cases}
$$

The forward representation therefore depends on the values $\phi(\lambda_i)$,
while the backward transmission of a residual depends on the divided differences
$G_\phi$. This separation is the reason a representation can look readable
under a frozen probe and still send a different encoder update during joint
training.

## 2. Readout error becomes an encoder error

Let $W_t$ be the readout used by the training step and let $W_t^\star$ be a
reference readout obtained by a small inner solve on the same FIT features.
For encoder parameters $\psi$ define

$$
e_t=\nabla_\psi L(\psi_t,W_t)-\nabla_\psi L(\psi_t,W_t^\star).
$$

A dimensionless diagnostic is

$$
\kappa_t=\frac{\lVert e_t\rVert_2}
{\lVert\nabla_\psi L(\psi_t,W_t^\star)\rVert_2+\delta}.
$$

In the SPD branch, $e_t$ is modulated by the Loewner matrix. In the CIFAR
branch, the same object is measured directly by retaining the feature tensor,
refining a detached head, and comparing the encoder-side gradients. The
Stage-TRR update is therefore a controlled intervention on the feedback
channel, not another prior correction.

## 3. Why an early stage is natural

Assume the readout inner update has a local contraction factor $\rho_t$ and
let $A_t$ bound the initial readout-induced encoder error. After $k$ inner
steps, the deterministic part of the feedback error is bounded by

$$
B_t(k)\le A_t\rho_t^k.
$$

With noisy class-conditional estimates, an empirical variance term remains.
A simple finite-action surrogate is

$$
R_t(k,\alpha)
=(1-\alpha)^2A_t^2\rho_t^{2k}
+\alpha^2\frac{\sigma_t^2}{m_tq_t}
+\lambda k,
$$

where $m_t$ is the number of calibration observations, $q_t$ is the
sampling allocation, and $\lambda$ is measured inner-loop cost. A static policy
uses one action for the whole horizon. Stage-TRR instead uses a prefix policy
$k_t=1$ for $t<N$ and returns to the ordinary joint update afterwards.

The difference between the two policies is visible when $\rho_t$ changes over
time. Early training has a large readout gap and a useful refinement step. Later
training has a smaller deterministic gap, so the same extra correction can be
mostly cost or noise. This explains why the short-horizon experiment and the
50/200 epoch boundary are both part of the reproducible story.

## 4. Stage-TRR-N update

During the prefix, compute features once,

$$
h_t=f_{\psi_t}(x_t),
$$

update a detached copy of the readout,

$$
W_t^{(1)}=W_t-\eta_h\nabla_W
L_{\mathrm{LA}}(\operatorname{stopgrad}(h_t),W_t),
$$

copy that refined readout into the live head, and update only the encoder
through

$$
\psi_{t+1}=\psi_t-\eta_\psi\nabla_\psi
L_{\mathrm{LA}}(h_t,\operatorname{stopgrad}(W_t^{(1)})).
$$

After epoch $N$, the runner uses the ordinary joint update

$$
(\psi_{t+1},W_{t+1})=(\psi_t,W_t)
-\eta_t\nabla_{\psi,W}L_{\mathrm{LA}}(\psi_t,W_t).
$$

The detached copy is essential. It makes the intervention explicit and keeps
the encoder gradient from silently differentiating through the inner solve.
The runner records the stage schedule and keeps the evaluation split unopened
until all FIT checkpoints are frozen.

## 5. What the experiments test

The short CIFAR-100-LT and CIFAR-10-LT comparisons test whether the measured
feedback intervention changes the early feature trajectory relative to the
same train-time Logit Adjustment objective. The 50 and 200 epoch runs test
whether that benefit persists after the readout has had time to catch up.
The PlantSPD branch supplies the Loewner-derivative motivation and a separate
Fused/Legacy execution comparison. No single number is used to substitute for
these distinct mechanisms.
