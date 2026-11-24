#!/usr/bin/env python3
"""Z3 verification of the EduFair deployment guardrails (manuscript Table 6).

A candidate fairness-mitigation method m is authorised for deployment only if

    Deploy(m) = (ieod_method   <  ieod_baseline)   # G1 fairness improvement
              AND (recall_method >= 0.70)          # G2 recall guard
              AND (collapse_rate <= 0.10)          # G3 collapse guard
              AND (subgroup_min  >= 50)            # G4 subgroup support
              AND (seed_std      <= 0.05)          # G5 seed stability

What this proves
----------------
For each guard G_i, the query  Deploy(m) AND NOT(G_i)  is UNSAT, i.e. no
assignment of metric values can be authorised while violating that guard.
The combined query  Deploy(m) AND (NOT G1 OR ... OR NOT G5)  is also UNSAT.
The weakened model (recall guard removed) is SAT: it admits the UCI
Thresholding configuration (G3 >= 12, no G1/G2) with IEOD = 0.253 and
recall = 0.223, showing the recall guard is necessary.

What this does NOT prove
------------------------
Z3 checks the internal logical consistency of the encoded deployment rules.
It does not verify trained classifiers, datasets, or the empirical fairness
estimates themselves, nor whether the chosen thresholds are appropriate.

Usage:
    pip install z3-solver
    python z3/verify_guardrails.py
"""

from __future__ import annotations

import sys
from time import perf_counter

from z3 import And, Not, Or, Q, Real, Solver, is_rational_value, sat, unsat

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
ieod_method = Real("ieod_method")
ieod_baseline = Real("ieod_baseline")
recall_method = Real("recall_method")
collapse_rate = Real("collapse_rate")
subgroup_min = Real("subgroup_min")
seed_std = Real("seed_std")

# Thresholds (as exact rationals)
RECALL_MIN = Q(70, 100)
COLLAPSE_MAX = Q(10, 100)
SUBGROUP_MIN_N = 50
SEED_STD_MAX = Q(5, 100)

# Physical domain of the metrics
domain = And(
    ieod_method >= 0, ieod_method <= 1,
    ieod_baseline >= 0, ieod_baseline <= 1,
    recall_method >= 0, recall_method <= 1,
    collapse_rate >= 0, collapse_rate <= 1,
    subgroup_min >= 0,
    seed_std >= 0,
)

# ---------------------------------------------------------------------------
# Guards and deployment rule
# ---------------------------------------------------------------------------
GUARDS = {
    "G1": ("Fairness improvement", "ieod_method < ieod_baseline", ieod_method < ieod_baseline),
    "G2": ("Recall guard", "recall_method >= 0.70", recall_method >= RECALL_MIN),
    "G3": ("Collapse guard", "collapse_rate <= 0.10", collapse_rate <= COLLAPSE_MAX),
    "G4": ("Subgroup support", "subgroup_min >= 50", subgroup_min >= SUBGROUP_MIN_N),
    "G5": ("Seed stability", "seed_std <= 0.05", seed_std <= SEED_STD_MAX),
}


def deploy(exclude: tuple[str, ...] = ()):
    """Deploy(m) as the conjunction of all guards not listed in `exclude`."""
    return And(*[g[2] for k, g in GUARDS.items() if k not in exclude])


def check(*constraints):
    s = Solver()
    s.add(domain, *constraints)
    result = s.check()
    return result, (s.model() if result == sat else None)


def combined_query():
    return check(deploy(), Or(*[Not(g[2]) for g in GUARDS.values()]))


def weakened_model():
    """Recall guard (G2) removed; UCI Thresholding values (G3 >= 12, no G1/G2)."""
    return check(
        deploy(exclude=("G2",)),
        Not(GUARDS["G2"][2]),
        ieod_method == Q(253, 1000),
        ieod_baseline == Q(412, 1000),
        recall_method == Q(223, 1000),
    )


def fmt(v) -> str:
    if is_rational_value(v):
        return f"{v.numerator_as_long() / v.denominator_as_long():g}"
    return str(v)


def main() -> int:
    rows = []  # (property, query, expected, observed)

    for i, (key, (name, expr, _)) in enumerate(GUARDS.items(), start=1):
        result, _ = check(deploy(), Not(GUARDS[key][2]))
        rows.append((f"P{i}: {name}", f"Deploy(m) AND NOT({expr})", unsat, result))

    combined, _ = combined_query()
    rows.append(("P6: Combined (P1-P5)", "Deploy(m) AND OR_i NOT(G_i)", unsat, combined))

    weak, cex = weakened_model()
    rows.append(("Weakened model (no recall guard)",
                 "Deploy'(m) AND recall < 0.70, IEOD=0.253", sat, weak))

    # ---- Table 6 -----------------------------------------------------------
    w = (34, 48, 8, 8, 4)
    header = ("Property", "Query", "Expected", "Result", "Pass")
    line = "+" + "+".join("-" * (n + 2) for n in w) + "+"
    print("Table 6. Z3 verification of deployment guardrails")
    print(line)
    print("| " + " | ".join(h.ljust(n) for h, n in zip(header, w)) + " |")
    print(line)
    all_ok = True
    for prop, query, expected, observed in rows:
        ok = observed == expected
        all_ok &= ok
        cells = (prop, query, str(expected).upper(), str(observed).upper(), "yes" if ok else "NO")
        print("| " + " | ".join(c.ljust(n) for c, n in zip(cells, w)) + " |")
    print(line)

    if cex is not None:
        print("\nWeakened-model counterexample (admitted without the recall guard):")
        for var in (ieod_method, ieod_baseline, recall_method, collapse_rate, subgroup_min, seed_std):
            print(f"  {str(var):<14} = {fmt(cex.eval(var, model_completion=True))}")

    # ---- Solver timing -----------------------------------------------------
    iterations = 10_000
    start = perf_counter()
    for _ in range(iterations):
        combined_query()
    mean_ms = (perf_counter() - start) * 1000 / iterations
    print(f"\nMean solver time, combined query (P6), over {iterations:,} iterations: "
          f"{mean_ms:.3f} ms (machine-dependent)")

    print("\nAll results match expectations." if all_ok else "\nMISMATCH against expected results.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
