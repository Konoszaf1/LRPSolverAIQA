"""Metamorphic robustness tests for the LLM solver.

Each test applies one dataset perturbation, solves both the original and the
perturbed instance with the LLM solver, then asserts a *metamorphic relation*
that must hold between the two outputs.

Metamorphic testing does not require knowing the correct answer — only the
*relationship* that should exist between the original and perturbed outputs.

Requires ``ANTHROPIC_API_KEY`` in the environment.

Run with::

    PYTHONUTF8=1 pytest qa_suite/metamorphic_tests/test_metamorphic.py -v -s --tb=long -m "llm and metamorphic"
"""

from __future__ import annotations

import pytest

from ai_agent.solver import LLMSolver
from qa_suite.common.fixtures import load_instance
from qa_suite.metamorphic_tests.perturbations import (
    double_all_demands,
    increase_vehicle_capacity,
    remove_customers,
    zero_all_fixed_costs,
)

pytestmark = [pytest.mark.llm, pytest.mark.metamorphic]

_INSTANCE = "Srivastava86"
_TOLERANCE = 0.15  # 15 % — LLMs are stochastic


@pytest.fixture(scope="module")
def solver() -> LLMSolver:
    """Lazily instantiate the LLM solver at test time, not at import time.

    This prevents ``anthropic.Anthropic()`` from being created during test
    collection, which would crash when ``ANTHROPIC_API_KEY`` is not set —
    even for test runs that deselect LLM tests via ``-m 'not llm'``.
    """
    return LLMSolver()


# ---------------------------------------------------------------------------
# Test 1 — more capacity per vehicle should not increase total cost
# ---------------------------------------------------------------------------

def test_capacity_increase_decreases_cost(solver: LLMSolver) -> None:
    """Increasing vehicle capacity should keep (or reduce) total routing cost.

    Metamorphic relation:
        perturbed_cost <= original_cost * (1 + tolerance)

    Reasoning: more capacity per vehicle → fewer vehicles needed per route →
    lower or equal total routing cost.
    """
    ds_orig = load_instance(_INSTANCE)
    ds_pert = increase_vehicle_capacity(ds_orig, 1.5)

    print(f"\n[capacity_increase] Solving original instance...")
    sol_orig, meta_orig = solver.solve(ds_orig)
    print(f"  Original : cost={sol_orig.total_cost:.2f}, routes={len(sol_orig.routes)}, "
          f"depots={sol_orig.open_depots} ({meta_orig['elapsed_seconds']:.1f}s)")

    print(f"[capacity_increase] Solving perturbed instance (vc x1.5)...")
    sol_pert, meta_pert = solver.solve(ds_pert)
    print(f"  Perturbed: cost={sol_pert.total_cost:.2f}, routes={len(sol_pert.routes)}, "
          f"depots={sol_pert.open_depots} ({meta_pert['elapsed_seconds']:.1f}s)")

    threshold = sol_orig.total_cost * (1 + _TOLERANCE)
    print(f"  Relation : {sol_pert.total_cost:.2f} <= {threshold:.2f} "
          f"(original {sol_orig.total_cost:.2f} + {_TOLERANCE*100:.0f}%)")

    assert sol_pert.total_cost <= threshold, (
        f"METAMORPHIC VIOLATION — higher vehicle capacity raised cost: "
        f"original={sol_orig.total_cost:.2f}, perturbed={sol_pert.total_cost:.2f}, "
        f"threshold={threshold:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — doubled demands should require at least as many routes
# ---------------------------------------------------------------------------

def test_doubled_demand_increases_routes(solver: LLMSolver) -> None:
    """Doubling all demands must not reduce the number of routes.

    Metamorphic relation:
        len(perturbed_routes) >= len(original_routes)

    Reasoning: each vehicle fills twice as fast, so the solver needs at
    least as many routes as in the original instance.
    """
    ds_orig = load_instance(_INSTANCE)
    ds_pert = double_all_demands(ds_orig)

    print(f"\n[doubled_demand] Solving original instance...")
    sol_orig, meta_orig = solver.solve(ds_orig)
    print(f"  Original : cost={sol_orig.total_cost:.2f}, routes={len(sol_orig.routes)} "
          f"({meta_orig['elapsed_seconds']:.1f}s)")

    print(f"[doubled_demand] Solving perturbed instance (demands x2)...")
    sol_pert, meta_pert = solver.solve(ds_pert)
    print(f"  Perturbed: cost={sol_pert.total_cost:.2f}, routes={len(sol_pert.routes)} "
          f"({meta_pert['elapsed_seconds']:.1f}s)")

    print(f"  Relation : {len(sol_pert.routes)} >= {len(sol_orig.routes)}")

    assert len(sol_pert.routes) >= len(sol_orig.routes), (
        f"METAMORPHIC VIOLATION — doubled demands produced fewer routes: "
        f"original={len(sol_orig.routes)}, perturbed={len(sol_pert.routes)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — zero fixed costs should open at least as many depots
# ---------------------------------------------------------------------------

def test_zero_fixed_cost_opens_all_depots(solver: LLMSolver) -> None:
    """Zero depot fixed costs must not reduce the number of open depots.

    Metamorphic relation:
        len(perturbed_open_depots) >= len(original_open_depots)

    Reasoning: no financial penalty for opening depots → the solver should
    open at least as many (or all) to reduce routing distances.
    """
    ds_orig = load_instance(_INSTANCE)
    ds_pert = zero_all_fixed_costs(ds_orig)

    print(f"\n[zero_fixed_cost] Solving original instance...")
    sol_orig, meta_orig = solver.solve(ds_orig)
    print(f"  Original : cost={sol_orig.total_cost:.2f}, open_depots={sol_orig.open_depots} "
          f"({meta_orig['elapsed_seconds']:.1f}s)")

    print(f"[zero_fixed_cost] Solving perturbed instance (fixed_costs=0)...")
    sol_pert, meta_pert = solver.solve(ds_pert)
    print(f"  Perturbed: cost={sol_pert.total_cost:.2f}, open_depots={sol_pert.open_depots} "
          f"({meta_pert['elapsed_seconds']:.1f}s)")

    print(f"  Relation : {len(sol_pert.open_depots)} >= {len(sol_orig.open_depots)}")

    assert len(sol_pert.open_depots) >= len(sol_orig.open_depots), (
        f"METAMORPHIC VIOLATION — zero fixed costs opened fewer depots: "
        f"original={sol_orig.open_depots}, perturbed={sol_pert.open_depots}"
    )


# ---------------------------------------------------------------------------
# Test 4 — fewer customers should not increase total cost
# ---------------------------------------------------------------------------

def test_fewer_customers_decreases_cost(solver: LLMSolver) -> None:
    """Removing half the customers should keep (or reduce) total routing cost.

    Metamorphic relation:
        perturbed_cost <= original_cost * (1 + tolerance)

    Reasoning: fewer customers to serve → lower total distance / cost.
    """
    ds_orig = load_instance(_INSTANCE)
    ds_pert = remove_customers(ds_orig, keep_ratio=0.5)

    print(f"\n[fewer_customers] Solving original instance ({len(ds_orig['customers'])} customers)...")
    sol_orig, meta_orig = solver.solve(ds_orig)
    print(f"  Original : cost={sol_orig.total_cost:.2f}, routes={len(sol_orig.routes)} "
          f"({meta_orig['elapsed_seconds']:.1f}s)")

    print(f"[fewer_customers] Solving perturbed instance ({len(ds_pert['customers'])} customers)...")
    sol_pert, meta_pert = solver.solve(ds_pert)
    print(f"  Perturbed: cost={sol_pert.total_cost:.2f}, routes={len(sol_pert.routes)} "
          f"({meta_pert['elapsed_seconds']:.1f}s)")

    threshold = sol_orig.total_cost * (1 + _TOLERANCE)
    print(f"  Relation : {sol_pert.total_cost:.2f} <= {threshold:.2f} "
          f"(original {sol_orig.total_cost:.2f} + {_TOLERANCE*100:.0f}%)")

    assert sol_pert.total_cost <= threshold, (
        f"METAMORPHIC VIOLATION — fewer customers raised cost: "
        f"original={sol_orig.total_cost:.2f}, perturbed={sol_pert.total_cost:.2f}, "
        f"threshold={threshold:.2f}"
    )
