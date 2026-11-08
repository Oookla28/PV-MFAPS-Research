"""Deterministic numerical checks for the PR-PV-MFAPS mathematical contract."""
import math
from itertools import product


def phi(v, ubar=(0.5, 0.5), eps=0.1):
    # Direct two-state KL constrained maximisation by a dense simplex scan.
    best = -float("inf")
    for i in range(100001):
        u0 = i / 100000.0
        u1 = 1.0 - u0
        kl = 0.0
        for u, q in zip((u0, u1), ubar):
            if u > 0:
                kl += u * math.log(u / q)
        if kl <= eps + 1e-9:
            best = max(best, u0 * v[0] + u1 * v[1])
    return best


def exhaustive_actions(risk, cost, budget, ubar=(0.5, 0.5), eps=0.1):
    # Tiny exact finite-action reference: risk[c][p][a], cost[c][p][a].
    cnum, anum = len(risk), len(risk[0][0])
    feasible = []
    for acts in product(range(anum), repeat=cnum):
        rv = [sum(risk[c][p][acts[c]] for c in range(cnum)) for p in range(2)]
        cv = [sum(cost[c][p][acts[c]] for c in range(cnum)) for p in range(2)]
        if phi(cv, ubar, eps) <= budget + 1e-9:
            feasible.append((phi(rv, ubar, eps), acts, rv, cv))
    return min(feasible)


def main():
    # Dual-independent invariants used by the audit: monotonicity and infinity Lipschitz.
    v = (0.2, 0.7)
    w = (0.4, 0.8)
    pv, pw = phi(v), phi(w)
    assert pw >= pv - 1e-9  # coordinate-wise comparison is not asserted here
    assert abs(pw - pv) <= max(abs(v[0]-w[0]), abs(v[1]-w[1])) + 2e-3

    risk = [
        [[0.40, 0.15], [0.20, 0.45]],
        [[0.35, 0.10], [0.30, 0.30]],
    ]
    cost = [
        [[0.00, 0.60], [0.00, 0.60]],
        [[0.00, 0.50], [0.00, 0.50]],
    ]
    best = exhaustive_actions(risk, cost, budget=0.60)
    assert best[1] == (0, 1)
    print({"status": "PASS", "phi_v": pv, "phi_w": pw, "best": best})


if __name__ == "__main__":
    main()
