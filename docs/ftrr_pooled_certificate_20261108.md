# Pooled paired certificate for FTRR

The class-wise empirical-Bernstein gate becomes extremely conservative when a
tail class contributes only a few calibration samples. FTRR therefore adds a
stage-level eligibility certificate over all FIT calibration samples.

For a candidate refinement depth (k), let

\[
D_i(k)=\operatorname{clip}\bigl(\ell_i(W^{(0)})-\ell_i(W^{(k)}),[-M,M]\bigr)
\]

for every calibration sample (i=1,\ldots,N_t). The pooled certificate is

\[
\widehat G_t(k)=\frac1{N_t}\sum_iD_i(k),\qquad
\underline G_t(k)=\widehat G_t(k)-r_t(k),
\]

where (r_t(k)) is the empirical-Bernstein radius with a union allocation over
stages and candidate depths. A stage may activate any class-level action using
depth (k) only if

\[
\underline G_t(k)\geq c_t(k)+s_t(k)+h_t.
\]

The pooled certificate has a stage-level interpretation. It certifies the
average bounded paired gain over the FIT calibration population. It does not
certify each tail class separately. The class allocation remains a finite
surrogate DP and must report its own proxy risk and uncertainty.

The raw-loss smoke with `M=1` remains the default conservative configuration.
An exploratory clipped-proxy variant with fixed `M=0.1` is registered
separately. It must be evaluated under a fresh manifest and cannot be merged
with the raw-loss certificate. The smaller bound changes the certified object
to a clipped paired gain, so any resulting action or accuracy claim is scoped
to that proxy.

The certificate implementation clips at the declared bound on entry and records
the number and fraction of clipped samples. This prevents floating-point
boundary failures while keeping the bounded object explicit.

This separation fixes a concrete failure mode without weakening the confidence
level. With (N_t) pooled samples, the linear empirical-Bernstein term scales
with (1/(N_t-1)), whereas a tail-class certificate scales with its much
smaller (n_{c,t}). The offline smoke records both radii and keeps
`held_reads=0`.
