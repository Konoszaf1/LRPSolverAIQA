"""Standalone deterministic QA runner — no pytest or DeepEval required.

Executes the Cuckoo Search solver on the Ch69 benchmark instance, then runs
all four deterministic validators and prints a formatted pass/fail report.

Usage::

    python -m qa_suite.run_deterministic_qa

The script exits with code 0 when every check passes, and code 1 when any
check fails.
"""

from __future__ import annotations

import sys

from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.builder import build_solution, depot_combinations
from lrp.config import CuckooConfig
from lrp.io.data_loader import load_customers, load_depots
from qa_suite.common.adapters import cuckoo_solution_to_schema
from qa_suite.common.fixtures import DATA_DIR, INSTANCES
from qa_suite.deterministic_checks.validators import (
    ValidationResult,
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_vehicle_capacity,
)

# Instance to benchmark
INSTANCE_NAME = "Ch69"
NUM_SOLUTIONS = 3
NUM_ITERATIONS = 10


def _run_solver() -> tuple[list[dict], dict, dict, float]:
    """Run Cuckoo Search on Ch69 and return routes + problem data.

    Returns:
        Tuple of (routes_as_dicts, customers_fixture_dict,
                  depots_fixture_dict, vehicle_capacity).
    """
    cli_file, dep_file, vc = INSTANCES[INSTANCE_NAME]
    customers_lrp = load_customers(DATA_DIR / cli_file)
    depots_lrp = load_depots(DATA_DIR / dep_file)

    combos = depot_combinations(len(depots_lrp), NUM_SOLUTIONS)
    solutions = [
        build_solution(customers_lrp, depots_lrp, combo, vc)
        for combo in combos
    ]

    config = CuckooConfig(num_solutions=NUM_SOLUTIONS, num_iterations=NUM_ITERATIONS)
    best = CuckooSearch(config).optimize(solutions)
    schema_sol = cuckoo_solution_to_schema(best)

    # Convert schema routes to plain dicts for the validators
    routes = [r.model_dump() for r in schema_sol.routes]

    # Build fixture-format customer/depot dicts (float-keyed matching the
    # fixture loader format)
    customers_fix: dict[int, dict] = {
        c.customer_number: {"x": c.x_cord, "y": c.y_cord, "demand": c.demand}
        for c in best.customers
    }
    depots_fix: dict[int, dict] = {
        d.depot_number: {
            "x": d.x_cord,
            "y": d.y_cord,
            "capacity": d.original_capacity,
            "fixed_cost": d.fixed_cost,
            "variable_cost": d.variable_cost,
        }
        for d in best.depots
    }

    return routes, customers_fix, depots_fix, float(vc)


def _format_line(label: str, result: ValidationResult) -> str:
    """Format a single validator result line for the report."""
    status = "PASS" if result.passed else "FAIL"
    detail = "All checks OK." if result.passed else result.violations[0]
    return f"  [{status}] {label:<22} {result.score:.2f}  — {detail}"


def main() -> None:
    """Run the QA report and exit with appropriate status code."""
    sep = "=" * 64

    print(sep)
    print(f"DETERMINISTIC QA REPORT — {INSTANCE_NAME} (Cuckoo Search Baseline)")
    print(sep)

    print("  Running solver...", end=" ", flush=True)
    routes, customers, depots, vc = _run_solver()
    n_routes = len(routes)
    n_customers = len(customers)
    print(f"done ({n_routes} routes, {n_customers} customers)")
    print()

    results: list[tuple[str, ValidationResult]] = [
        ("Vehicle Capacity", validate_vehicle_capacity(routes, customers, vc)),
        ("Customer Coverage", validate_customer_coverage(routes, customers)),
        ("Depot Capacity",   validate_depot_capacity(routes, customers, depots)),
        ("Route Distances",  validate_route_distances(routes, customers, depots)),
    ]

    for label, result in results:
        print(_format_line(label, result))

    n_passed = sum(1 for _, r in results if r.passed)
    n_total = len(results)

    print(sep)
    print(f"RESULT: {n_passed}/{n_total} checks passed")
    print(sep)

    if n_passed < n_total:
        print()
        print("Failures:")
        for label, result in results:
            if not result.passed:
                for v in result.violations:
                    print(f"  [{label}] {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
