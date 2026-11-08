# PV-MFAPS: finite-action discrete Pareto allocation

This document describes the finite-action allocator used by the PlantSPD
training entry point. The solver, brute-force exactness check, scale bench,
and measured PlantSPD runs are exposed through the source and configuration
files listed below.

## 1. Problem

Finite action set `A = {(k, alpha)}` with `k in {0,1,2,4}` and
`alpha in {0.25}`, per class `c`. Costs are measured/quantised; the envelope
`U_c(a)` is a risk upper bound. Solve

```text
min_{a_c, r}  r + (mu/C) sum_c U_c(a_c)
s.t.  U_c(a_c) <= r            for c in tail set T
      (1/C) sum_c U_c(a_c) <= U_bar + eps
      sum_c q_c cost_c(a_c) <= B
      k_c = 0 implies alpha_c = 0
```

## 2. Envelope components

- **Bias** `b_c(k) = A_c * rho_c^(2k)` — proxy only.
- **Stability** (uniform stability of a `k`-step gradient inner solver under
  bounded loss `M`, `L`-Lipschitz loss, `beta`-smoothness, gradient bound `G`):

  ```text
  Stab_c(k) <= (2 L G / n_c) * sum_{j=0}^{k-1} eta_j * prod_{l=j+1}^{k-1} (1 + beta eta_l)
  ```

- **Empirical Bernstein** for the finite class-action set:

  ```text
  EB_c <= sqrt(2 V_c log(2 C |A| / delta) / n_c) + 7 M log(2 C |A| / delta) / (3 (n_c - 1))
  ```

Registered constants used in the benchmark: `L=1, G=1, beta=1, eta=0.1,
delta=0.05, M=1, budget_scale=1000`.

Stability values computed by `pv_mfaps_dp_solver.stability_bound` (EXECUTED):

| n_c | k=0 | k=1 | k=2 | k=4 |
|---|---|---|---|---|
| 5 | 0 | 0.0400 | 0.0840 | 0.1856 |
| 10 | 0 | 0.0200 | 0.0420 | 0.0928 |
| 25 | 0 | 0.0080 | 0.0168 | 0.0371 |
| 100 | 0 | 0.0020 | 0.0042 | 0.0093 |
| 344 | 0 | 0.00058 | 0.00122 | 0.00270 |

`xi` is retained only as a sensitivity parameter (`xi=1.01`); it is not a
theorem constant. `A_c`, `rho_c`, `sigma_c` remain proxies.

## 3. Exactness

`pv_mfaps_dp.py` enumerates tail caps, prunes actions that violate the cap,
and runs a cost-indexed DP over conservatively `ceil`-quantised costs, then
backtracks `(k, alpha)` and emits a certificate (quantised budget, quantised
cost, actual cost, slack, macro risk, tail max risk, frontier size).

Brute-force comparison (`C=4`, 4 actions, 6 trials): **all match**
(`brute_force_all_match=true`).

The claim is **exactness for the quantised finite-action surrogate**, not
equivalence to the continuous KKT problem.

## 4. Scale

| instance | status | wall | final frontier |
|---|---|---|---|
| C=69, exact DP | OPTIMAL | 0.17 s | 201 |
| C=200, exact DP | OPTIMAL | 0.73 s | 201 |
| C=1000, sparse tail-only DP (150 active) | OPTIMAL | 0.52 s | 197 |
| C=69, greedy marginal gain | GREEDY | 0.012 s | — |

Greedy vs exact on C=69: relative objective gap `1.9e-16` on this instance
(i.e. greedy happened to be exact here), but greedy has **no exactness
guarantee**; it is reported only as a large-class fallback.

Action histogram at C=69: `k=0: 50`, `k=1: 0`, `k=2: 1`, `k=4: 18`.

## 5. Reproduction entry

The finite-action solver is implemented in `src/model/pv_mfaps_dp.py` and
`src/model/pv_mfaps_dp_solver.py`. The exactness contract is run with:

```bash
python tests/test_pv_mfaps_dp.py
```

PlantSPD training uses the shared runner and a policy under
`configs/policies/`. The complete FIT, freeze, evaluation and result-summary
sequence is documented in the root README and in
`docs/protocols.md`.
