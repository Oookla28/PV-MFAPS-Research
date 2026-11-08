# Stage-TRR-10 algorithm registration

## Scope

This registration specifies the next independent validation of the stage-aware
readout refinement policy. It is written after the exploratory 30-epoch pilot
and must not be used to relabel that pilot as confirmatory.

## Algorithm

For epochs $t=0,\\ldots,T-1$, let $N=10$.

* If $t<N$, update the readout by one detached inner gradient step and update
  the encoder with the refined readout feedback.
* If $t\\ge N$, optimize the standard logit-adjusted cross-entropy jointly over
  encoder and readout.
* Use the same cosine schedule, batch size, optimizer base learning rate,
  class prior, and data order for the LA comparator.
* The readout inner step uses learning rate $\\eta_h=0.05$ and the outer
  optimizer uses $\\eta=0.1$.

The training objective is the stage-dependent surrogate
\\[
L_t(\\psi,W)=
\\begin{cases}
\\ell_{\\rm LA}(\\psi,W^{(1)}(\\psi)), & 0\\le t<N,\\\\
\\ell_{\\rm LA}(\\psi,W), & N\\le t<T,
\\end{cases}
\\]
where $W^{(1)}(\\psi)=W-\\eta_h\\nabla_W\\ell_{\\rm LA}(\\psi,W)$ is formed on
the current FIT minibatch and detached before the encoder update.

## Required evidence

The next run must be registered before reading test data. It requires:

1. three seeds for Stage-TRR-10 and three matched LA controls;
2. one FIT freeze manifest per arm;
3. exactly one post-freeze evaluator pass over the selected evaluation split;
4. paired seed deltas, mean, standard deviation, and wall-clock;
5. no selection based on the evaluation split.

The previous 30-epoch prefix10 result is exploratory because its horizon was
selected after direct test reads. It remains evidence for a mechanism signal,
not a confirmatory generalization claim.

## Conditional stage-switching bound

With a simultaneous FIT certificate
$|\\widehat U_{t,c}(k)-U_{t,c}(k)|\\le\\epsilon_{t,c}$, the finite-action
stage optimizer satisfies
\\[
J(\\widehat k_{1:T})-J(k^*_{1:T})
\\le 2\\sum_{t,c}\\omega_c\\epsilon_{t,c}
+\\epsilon_{\\rm quant}.
\\]
This is a conditional surrogate-risk guarantee. It does not imply test-set
improvement or universal superiority.
