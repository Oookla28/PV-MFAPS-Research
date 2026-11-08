# Feedback Trust-Region Readout Refinement (FTRR)

## Purpose and scope

TRR showed that a refined readout can improve the early training trajectory but can
hurt the tail after the readout has largely tracked the moving encoder. FTRR turns
that observation into a constrained finite-action algorithm. The method controls
three quantities jointly: readout refinement depth, the amount of refined feedback
mixed into the native feedback, and the measured computational budget.

The guarantees below concern a FIT-calibrated feedback-risk surrogate. They do not
by themselves imply test accuracy or community-wide SOTA.

## Setup

For a sample ((x,y)), let (z=f_\psi(x)) be the encoder feature and (W_t) the
live readout. Let (W_t^{(0)}=W_t), and construct a detached-feature inner sequence

\[
W_t^{(j+1)}=W_t^{(j)}-\eta_w\nabla_W\ell(W_t^{(j)}z,y),
\qquad j=0,\ldots,K-1 .
\]

The encoder gradients induced by the live and refined readouts are

\[
g_{0}=\nabla_\psi\ell(f_\psi(x),W_t),
\qquad
g_{k}=\nabla_\psi\ell(f_\psi(x),W_t^{(k)}).
\]

For action $a=(k,\alpha)$, define the mixed feedback

\[
g_a=g_0+\alpha(g_k-g_0),
\qquad 0\leq\alpha\leq1,
\]

and the reference feedback (g_* = \nabla_\psi\ell(f_\psi(x),W_t^*)), where (W_t^*)
is a sufficiently solved readout on a FIT calibration split. The feedback error is
(e_a=g_a-g_*).

## Trust region and safety cone

The action must satisfy two constraints measured on the calibration split.

### Feedback displacement

\[
\|g_a-g_0\|\leq r_t . \tag{1}
\]

Here (r_t) is a registered stage radius. It can be estimated from a held-out
portion of FIT calibration or from a deterministic time-uniform confidence sequence.

### Native descent cone

For a prescribed \(\xi\in[0,1)), require

\[
\langle g_0,g_a\rangle
\geq \xi\|g_0\|\,\|g_a\| . \tag{2}
\]

The feasible set always contains \(\alpha=0\) when $g_0\ne0$. Since
$g_a=g_0+\alpha d_k$, with $d_k=g_k-g_0$, the largest feasible coefficient can be found
by one-dimensional bisection. For the weaker cone \(\xi=0\), a sufficient closed form is

\[
\alpha_{\mathrm{cone}}=
\begin{cases}
1,&\langle g_0,d_k\rangle\geq0,\\
\min\left(1,\dfrac{\|g_0\|^2}{-\langle g_0,d_k\rangle}\right),&\text{otherwise.}
\end{cases} \tag{3}
\]

The trust-region coefficient is then

\[
\alpha_{\max}(k)=\min\left(1,\alpha_{\mathrm{cone}},
\frac{r_t}{\|d_k\|+\varepsilon}\right). \tag{4}
\]

This is a first-order update-direction constraint. It is not a generalization bound.

## Stage noise floor

Assume that, conditional on the current encoder, the inner readout objective is
$\mu_c$-strongly convex and $L_c$-smooth and that the stochastic readout gradient
has conditional second moment bounded by $\sigma_{c,p}^2/n_{c,p}$. For a stable inner
step $0<\eta_w<2/L_c$, write

\[
\rho_c=\max\{|1-\eta_w\mu_c|,|1-\eta_wL_c|\}<1.
\]

The standard recursion gives

\[
\mathbb E\|W_{c,p}^{(k)}-W_{c,p}^*\|^2
\leq
\rho_c^{2k}D_{c,p,0}^2
+\frac{\eta_w^2\sigma_{c,p}^2}
{n_{c,p}(1-\rho_c^2)}. \tag{5}
\]

Combining (5) with a local feedback amplification bound
\(|g(W)-g(W^*)\|\leq\kappa_{c,p}\|W-W^*\|) gives the proxy

\[
U_{c,p,t}(k)=
\kappa_{c,p,t}^2\left[
\rho_{c,p,t}^{2k}D_{c,p,t}^2+
\frac{\eta_w^2\sigma_{c,p,t}^2}
{n_{c,p,t}(1-\rho_{c,p,t}^2)}
\right]. \tag{6}
\]

The second term is a non-vanishing noise floor. Therefore increasing $k$ is not
monotonically beneficial. The proxy is only valid under the stated local convexity,
step-size, and moment assumptions.

For an action $a=(k,\alpha)$, a conservative mixed-feedback proxy is

\[
U_{c,p,t}(a)
\leq (1-\alpha)^2U_{c,p,t}(0)
+\alpha^2U_{c,p,t}(k)
+2\alpha(1-\alpha)\widehat C_{c,p,t}(k), \tag{7}
\]

where \(\widehat C\) is an empirical cross term. If no valid cross-term estimate is
available, use the safe upper bound
\(\widehat C\leq\sqrt{U(0)U(k)}\). This bound is conservative and should be
reported as a proxy certificate.

## Calibration certificate

For a fixed class, partition, stage, and legal action, compute bounded paired samples
$q_i(a)\in[0,M]$ on a FIT calibration split. Let \(\widehat U\) and \(\widehat V\) be
their mean and sample variance. With a union allocation
\(\delta_{c,p,t,a}=\delta/(CPS|\mathcal A|)\), the empirical-Bernstein radius is

\[
r_{c,p,t}(a)=
\sqrt{\frac{2\widehat V_{c,p,t}(a)\log(2/\delta_{c,p,t,a})}{n_{c,p,t}}}
+\frac{7M\log(2/\delta_{c,p,t,a})}{3(n_{c,p,t}-1)}. \tag{8}
\]

The upper certificate is \(\overline U=\widehat U+r\). The samples must be clipped
or otherwise provably bounded. Using an unbounded cross-entropy or gradient norm
without recording $M$ is not a valid application of (8).

## Finite-action allocation

At stage $t$, define the legal action set

\[
\mathcal A_t=\{(k,\alpha): k\in\{0,1,2,4,8\},
\alpha\in\mathcal G_\alpha,
\alpha\leq\alpha_{\max}(k),
\|g_a-g_0\|\leq r_t\}.
\]

Let \(c_t(a)\) be measured or calibrated cost, and let \(s_t(a,a_{t-1})\) be a switching cost. For partition risks $v_p(a)$, define the KL-robust aggregate

\[
\Phi_\varepsilon(v)=
\sup_{q:D_{\mathrm{KL}}(q\|\bar q)\leq\varepsilon}q^\top v . \tag{9}
\]

The stage problem is

\[
\min_{a_1,\ldots,a_C}
\Phi_\varepsilon\left(
\sum_c w_c\overline U_{c,1,t}(a_c),
\ldots,
\sum_c w_c\overline U_{c,P,t}(a_c)
\right)
+\lambda_t\sum_c c_t(a_c)
+\gamma_t\sum_c s_t(a_c,a_{c,t-1}) 
\]
\[
\text{s.t.}\qquad
\sum_c c_t(a_c)\leq B_t,
\qquad a_c\in\mathcal A_{c,t}. \tag{10}
\]

After quantizing cost and risk to registered integer grids, (10) is a finite
multiple-choice allocation problem. A dynamic program over class index, budget,
partition-risk bins, and switching count returns the exact optimum on that grid.
The solver must retain switching count in its state key. Its optimality claim is
therefore restricted to the finite quantized problem.

## Conditional safety proposition

Assume the outer objective $F$ has $L_s$-Lipschitz gradient and
\(\eta_\psi L_s\leq1/4\). Let $g_*$ be the reference encoder gradient and let the
selected action satisfy the certified error condition

\[
\|g_a-g_*\|\leq\xi\|g_*\|,
\qquad \xi<\sqrt{2/5}. \tag{11}
\]

Then the update \(\psi^+=\psi-\eta_\psi g_a\) obeys

\[
F(\psi^+)
\leq F(\psi)-\eta_\psi\left(\frac12-\\frac{5}{4}\xi^2\right)\|g_*\|^2. \tag{12}
\]

Thus the update is strictly descending whenever (11) holds. The proof follows from
smoothness and the decomposition $g_a=g_*+e_a$:

\[
F(\psi^+)\leq F(\psi)-\frac{\eta_\psi}{2}\|g_*\|^2
+\frac{5\eta_\psi}{4}\|e_a\|^2.
\]

This proposition is conditional on a valid reference-gradient certificate. The
native cone constraint (2) alone does not imply (11).

## Allocation regret proposition

Suppose the calibration event simultaneously satisfies
\( |U_{c,p,t}(a)-\widehat U_{c,p,t}(a)|\leq r_{c,p,t}(a) \) for all legal indices and the
DP quantization error is at most \(\varepsilon_q\). Let \(\widehat a_t\) be the FTRR
DP solution and $a_t^*$ the best feasible action sequence for the true robust proxy.
Then, on this event,

\[
\mathcal R(\widehat a_t)
\leq
\mathcal R(a_t^*)
+2\sum_{c}w_c\max_{p,a}r_{c,p,t}(a)
+\varepsilon_q. \tag{13}
\]

This follows by sandwiching the true objective between the upper certificate and its
lower counterpart before applying DP optimality. Equation (13) is an allocation
statement for the calibrated feedback proxy. It is not a test-accuracy theorem.

## Algorithm sketch

1. Freeze a deterministic FIT calibration split and measure $g_0$, refined
   gradients, paired bounded feedback samples, and the stage noise-floor terms.
2. Build the legal action set using (4) and the empirical certificate (8).
3. Solve the finite quantized KL-DRO allocation problem (10).
4. Apply $g_a$ only when the cone, displacement, and certified reference-error
   conditions pass. Otherwise use $g_0$.
5. Recalibrate at registered stage boundaries. Stop refinement when the certified
   action set contains only $k=0$, which is a mathematically meaningful boundary
   rather than a forced floor.
6. Freeze all FIT artifacts before one-time HELD evaluation.

## Non-claims and required experiments

FTRR does not claim global bilevel convergence, universal long-tail gains, or SOTA
from (11)--(13). The required experiment is a common audited 30/50/200-epoch matrix
with LA, TRR k=1, FTRR, fixed-k, and a zero-refinement control. Report parameter-space
gradient cosine, action histograms, certificate coverage, Macro-F1, Tail-10%, wall-clock,
and FLOPs. The 200-epoch s3409 artifact must exist before making a three-seed claim.
