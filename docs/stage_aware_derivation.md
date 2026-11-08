# Stage-aware risk-envelope allocation: derivation package

## Target

Formalize a long-horizon extension of PV-MFAPS that can change inner-loop
precision across training stages without assuming that a larger `k` is always
beneficial. The derivation must separate exact identities, finite-action
optimization, and empirical proxies.

## Status

**COHERENT AFTER REFRAMING.** The original rule “force every class to have
`k >= 1` after the first 20% of training” is not generally justified. It is
replaced by a confidence-gated finite-action sequence optimization with an
optional switching penalty.

## Invariant object

The invariant object is the stage-conditional penalized feedback risk of a
shared class action policy. Let `s = 0, ..., S-1` index stages, `p` index
partitions, `c` index classes, and `a_c=(k_c, alpha_c)` be a legal action from
the finite set `A`. Define

\[
U_{c,p,s}(a_c)=B_{c,p,s}(k_c)+S_{c,p,s}(k_c)+E_{c,p,s}(a_c),
\]

where `B` is the readout-bias proxy, `S` is a stability term, and `E` is an
empirical-Bernstein confidence term. These are not interchangeable: `B` is a
solver proxy, whereas `S` and `E` require explicit assumptions and confidence
calibration.

## Assumptions and notation

1. The inner loss is `L`-Lipschitz in the readout parameters, has gradients
   bounded by `G`, and is `beta`-smooth. The step sizes satisfy `eta_j >= 0`.
2. Class `c` has `n_{c,p,s} >= 2` calibration examples in the data used to
   estimate its risk. The loss is bounded by `M`.
3. The action set `A` and every stage frontier `F_s` are finite. An action is
   legal only if `k=0` implies `alpha=0`.
4. Costs are measured and quantized to integer units. A stage action is
   feasible only when its quantized cost fits the stage budget.
5. For a strict high-probability claim, calibration data are independent of
   the data used to choose the final action, or are handled with cross-fitting.
   Reusing the same FIT data for selection and evaluation gives a diagnostic
   proxy, not the theorem below.

## Exact stability identity

For a `k`-step gradient inner solver, let `d_j` be the parameter difference
between two neighboring calibration sets after step `j`. Smoothness and
bounded influence give the recursion

\[
d_{j+1}\le (1+\beta\eta_j)d_j+\frac{2LG\eta_j}{n_{c,p,s}},
\qquad d_0=0.
\]

Unrolling it gives the uniform-stability bound

\[
S_{c,p,s}(k)\le
\frac{2LG}{n_{c,p,s}}
\sum_{j=0}^{k-1}\eta_j
\prod_{\ell=j+1}^{k-1}(1+\beta\eta_\ell).
\tag{1}
\]

For constant step size `eta` and `beta>0`, (1) becomes

\[
S_{c,p,s}(k)\le
\frac{2LG}{n_{c,p,s}\beta}
\left((1+\beta\eta)^k-1\right).
\tag{2}
\]

Equation (2) explains why an exponential stability proxy may appear. It does
not justify inserting an arbitrary `gamma^k` term: `gamma` must be tied to
`1+beta*eta` and the prefactor must be retained.

## Confidence envelope

For a fixed finite action and a fresh calibration sample, an empirical
Bernstein radius can be written as

\[
E_{c,p,s}(a) =
\sqrt{\frac{2\widehat V_{c,p,s}(a)\,\ell}{n_{c,p,s}}}
 + \frac{7M\ell}{3(n_{c,p,s}-1)},
\quad
\ell=\log\frac{2C P S |A|}{\delta}.
\tag{3}
\]

With probability at least `1-delta`, (3) holds simultaneously over the
finite class/partition/stage/action index set. This simultaneous statement
requires the stated finite-set and fresh-calibration assumptions. If the same
FIT examples are repeatedly used to select actions across stages, (3) is not
automatically valid; use a held-out calibration split, cross-fitting, or a
sequential complexity bound.

## Partition-robust stage objective

For a shared policy `A=(a_1,...,a_C)`, define the partition risk vector

\[
R_{p,s}(A)=\frac1C\sum_{c=1}^C U_{c,p,s}(a_c).
\]

The KL-robust aggregate is

\[
\mathcal D_{\epsilon_s}(R_s)=
\sup_{w\in\Delta_P:\,D_{\rm KL}(w\|u)\le\epsilon_s}
\sum_{p=1}^P w_pR_{p,s}.
\tag{4}
\]

Its exact dual form is

\[
\mathcal D_{\epsilon_s}(R_s)=
\inf_{\tau>0}\left[
\tau\epsilon_s+\tau\log\sum_{p=1}^P u_p
\exp\left(\frac{R_{p,s}}{\tau}\right)
\right].
\tag{5}
\]

At `epsilon=0`, (4) is the nominal weighted mean; as the ambiguity set
expands it approaches the worst-partition risk. The observed epsilon sweep
can therefore be interpreted as a sensitivity study of (4), not as evidence
that the partition shift is large.

## Sequence optimization and exact discrete solver

Let `F_s(B_s)` be the feasible Pareto frontier generated from the finite
action grid at stage `s`. For a previous action `A'`, let `d(A,A')` be a
nonnegative switching distance (for example, the number of changed class
actions). The stage sequence problem is

\[
\min_{A_0,\ldots,A_{S-1}}
\sum_{s=0}^{S-1}\omega_s\mathcal D_{\epsilon_s}(R_s(A_s))
 +\mu\sum_s\omega_s\bar R_s(A_s)
 +\lambda_{\rm sw}\sum_{s=1}^{S-1}d(A_s,A_{s-1})
\tag{6}
\]

subject to

\[
\sum_s L_s\,\mathrm{cost}_s(A_s)\le B_{\rm total},
\qquad A_s\in F_s(B_s).
\tag{7}
\]

After quantizing cost, the exact dynamic program is

\[
V_s(b,A)=J_s(A)+
\min_{A'\in F_{s-1}:\,b-\widehat c_s(A)\ge0}
V_{s-1}(b-\widehat c_s(A),A'),
\tag{8}
\]

where `J_s` is the stage part of (6). Backtracking (8) is exact for the
quantized finite surrogate. Its complexity is
`O(S * |F|^2 * B_quantized)` in the dense form; sparse frontier pruning can
reduce the practical cost. This exactness does not imply optimality for the
continuous training problem.

## Why a hard `k >= 1` floor is unsafe

Suppose two legal actions differ only by `k=0` and `k=1`. If

\[
U_{c,p,s}(1)+\lambda_{\rm cost}\Delta C
\ge U_{c,p,s}(0)
\]

for a partition that is active in (4), then forcing `k=1` cannot improve the
penalized robust objective for that class. Since (1) is nondecreasing in `k`
while the bias proxy may be nearly flat when `rho` is close to one, this
inequality is plausible in the long-horizon regime. A stage floor is therefore
an assumption, not a theorem.

## Certified stage activation

Let `r_{c,p,s}(a)` be a simultaneous confidence radius and let `h_s>0` be a
minimum improvement margin. Activate a positive-`k` action only if

\[
\max_p\{\widehat U_{c,p,s}(a)+r_{c,p,s}(a)\}
 +\lambda_{\rm cost}\Delta C
 +\lambda_{\rm sw}\Delta d
 <
\min_p\{\widehat U_{c,p,s}(0)-r_{c,p,s}(0)\}-h_s.
\tag{9}
\]

On the event that all confidence intervals are valid, (9) implies that the
true penalized risk of the activated action is lower than the zero-action
alternative by at least `h_s`. If (9) fails, retaining `k=0` is the safe
decision. A two-threshold hysteresis rule can prevent oscillation when the
estimated gap is close to zero.

## Annealing and its finite-frontier guarantee

Entropy regularization is useful for exploration but is not itself a
generalization theorem. For a finite frontier `F_s`, define

\[
\pi_s^\tau(A)=
\frac{\exp[-\widehat J_s(A)/\tau_s]}
{\sum_{A'\in F_s}\exp[-\widehat J_s(A')/\tau_s]}.
\tag{10}
\]

Then

\[
\mathbb E_{\pi_s^\tau}[\widehat J_s(A)]-\min_{A\in F_s}\widehat J_s(A)
\le \tau_s\log|F_s|.
\tag{11}
\]

Moreover, for any `g>0`,

\[
\Pr_{A\sim\pi_s^\tau}
\left[\widehat J_s(A)\ge J_s^*+g\right]
\le |F_s|e^{-g/\tau_s}.
\tag{12}
\]

Thus a decreasing temperature has an explicit finite-frontier optimization
interpretation. It does not, by itself, prove a held-set gain.

## Approximation error bound for the sequence solver

Assume the stage objective estimates satisfy
`|J_hat_s(A)-J_s(A)| <= xi_s` for every frontier action. Let `A_hat` be the
sequence returned by exact DP on `J_hat`, with the same budget and switching
constraints as the true optimum `A*`. Then

\[
J(A_{\rm hat})-J(A^*)\le 2\sum_s\omega_s\xi_s.
\tag{13}
\]

If a Gibbs action is sampled at each stage, add at most
`sum_s omega_s tau_s log |F_s|` in expectation. Equation (13) is a conditional
approximation result; the remaining work is to obtain valid `xi_s` from
independent calibration or cross-fitting.

## Required experiments

The stage claim is not established by a formula or by PlantSPD's 8-step
result. A valid test must include, under the same measured budget:

1. 8-step and 50-step horizons with fixed seeds and a fresh calibration split.
2. Fixed-`k`, hard `k>=1` floor, entropy-only annealing, and certified gate (9).
3. Stage action histograms, confidence margins, switching counts, real wall/FLOPs,
   FIT and held macro/tail metrics.
4. A no-switching and a no-stability-term ablation.
5. An independent long-tailed benchmark or a registered failure boundary.

Until these runs complete, the sequence theorem is a property of the finite
surrogate and the stage-gate assumptions; it is not a measured generalization
or SOTA claim.
