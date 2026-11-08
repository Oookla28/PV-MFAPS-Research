# Conditional stage-switching proposition for FTRR

This note gives a conditional comparison between a static refinement depth and
a stage policy. It is a surrogate-risk statement. It does not claim held
accuracy or that $\rho$ must improve monotonically during training.

## Assumptions

At stage $t\in\{1,\ldots,T\}$, class $c$ and action $k\in\mathcal K$ have a
feedback-risk proxy
\[
U_{t,c}(k)=A_{t,c}\rho_{t,c}^{2k}+F_{t,c}(k)+C_{t,c}(k),
\]
where $F$ contains the readout noise floor and $C$ contains measured compute
cost. Let $d(k,k')$ be the switching distance and $\lambda\ge0$ its penalty.
The dynamic oracle minimizes
\[
J(k_{1:T})=
\sum_{t=1}^T\sum_c \omega_c U_{t,c}(k_{t,c})
+\lambda\sum_{t=1}^T d(k_t,k_{t-1}).
\]

Suppose a FIT calibration certificate satisfies, simultaneously for all
$(t,c,k)$,
\[
\left|\widehat U_{t,c}(k)-U_{t,c}(k)\right|
\le \varepsilon_{t,c}.
\]
The certificate may be empirical-Bernstein, paired, or another valid bounded
confidence construction. The bound below is conditional on this event.

## Proposition

Let $\widehat k_{1:T}$ minimize the finite-action estimated objective with the
same budget and switching penalty, and let $k^*_{1:T}$ minimize the true
objective. Then
\[
J(\widehat k_{1:T})-J(k^*_{1:T})
\le
2\sum_{t=1}^T\sum_c \omega_c\varepsilon_{t,c}
+\varepsilon_{\rm quant},
\]
where $\varepsilon_{\rm quant}$ is the finite budget-quantization error.
The switching penalty is already part of both objectives, so a policy cannot
claim a gain by changing actions more often without paying for it.

For a static action $\bar k$, define its oracle phase gap
\[
\Gamma_T(\bar k)
=
J(\bar k,\ldots,\bar k)-J(k^*_{1:T}).
\]
The estimated stage policy therefore satisfies
\[
J(\bar k,\ldots,\bar k)-J(\widehat k_{1:T})
\ge
\Gamma_T(\bar k)
-2\sum_{t=1}^T\sum_c\omega_c\varepsilon_{t,c}
-\varepsilon_{\rm quant}.
\]
Thus a dynamic policy is guaranteed to beat a static action only when the
oracle phase gap exceeds the confidence and quantization terms.

## Transition condition

For an adjacent transition $k\to k+1$, define
\[
\Delta_{t,c}(k)=U_{t,c}(k+1)-U_{t,c}(k).
\]
The transition is preferred by the penalized oracle when
\[
\Delta_{t,c}(k)+\lambda d(k+1,k)<0.
\]
For $k=0$, if
\[
F_{t,c}(1)-F_{t,c}(0)
+C_{t,c}(1)-C_{t,c}(0)+\lambda>0,
\]
the corresponding critical contraction is
\[
\rho^*_{t,c}
=
\sqrt{
1-
\frac{
F_{t,c}(1)-F_{t,c}(0)
+C_{t,c}(1)-C_{t,c}(0)+\lambda
}{A_{t,c}}
}.
\]
When the expression lies in $[0,1]$, $k=1$ is preferred only below this
threshold. If the noise-floor or cost term changes with $t$, the threshold
also changes with $t$. No monotone phase transition follows without an
additional assumption on these terms.

## Proof sketch

On the simultaneous certificate event, every estimated action value differs
from its true value by at most its stated radius. Evaluating the estimated
objective at the estimated minimizer and at the true minimizer gives two error
terms, producing the factor two. The DP quantization contributes
$\varepsilon_{\rm quant}$. Subtracting the true dynamic objective from the
constant-action objective gives the phase-gap inequality. The transition
condition follows by comparing the two adjacent proxy values and adding one
switch penalty.

The proposition supports stage-aware allocation as a finite-confidence
optimization result. It does not establish a test-set improvement, a global
continuous optimum, or a universal monotonic decrease of $\rho$.
