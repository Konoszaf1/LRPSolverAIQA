"""Side-by-side comparison of Cuckoo Search vs LLM solver on an LRP instance.

Usage::

    python -m qa_suite.run_comparison [INSTANCE_NAME]

Default instance: Srivastava86.

Both solvers are run on the selected instance.  The Cuckoo Search solver
uses the instance-specific vehicle capacity from the ``INSTANCES`` registry.

Results are saved to ``results/{instance_name}_comparison.json``.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

from ai_agent.solver import LLMSolver

# ---------------------------------------------------------------------------
# Internal solver imports (only used for Cuckoo Search — may raise on float
# demand instances, which is caught and handled as SKIPPED)
# ---------------------------------------------------------------------------
from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.algorithms.nearest_neighbor import assign_depots, build_vehicle_routes
from lrp.config import CuckooConfig
from lrp.io.data_loader import load_customers as lrp_load_customers
from lrp.io.data_loader import load_depots as lrp_load_depots
from lrp.models.solution import Solution
from qa_suite.common.adapters import cuckoo_solution_to_schema
from qa_suite.common.faithfulness import manual_faithfulness_check
from qa_suite.common.fixtures import DATA_DIR, INSTANCES, load_instance
from qa_suite.deterministic_checks.validators import (
    ValidationResult,
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_vehicle_capacity,
)

NUM_SOLUTIONS = 3
NUM_ITERATIONS = 10
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Cuckoo Search runner
# ---------------------------------------------------------------------------

def _run_cuckoo(instance_name: str) -> tuple[list[dict], dict, dict, float, float, float] | None:
    """Run Cuckoo Search and return (routes, customers, depots, vc, elapsed).

    Returns ``None`` if the instance is incompatible (float demand parsing).
    """
    cli_file, dep_file, vc = INSTANCES[instance_name]
    try:
        customers_lrp = lrp_load_customers(DATA_DIR / cli_file)
        depots_lrp = lrp_load_depots(DATA_DIR / dep_file)
    except (ValueError, TypeError) as exc:
        print(f"  Cuckoo Search skipped: {exc}")
        return None

    all_ids = tuple(range(1, len(depots_lrp) + 1))
    combos = list(combinations(all_ids, len(depots_lrp)))[:NUM_SOLUTIONS]

    t0 = time.time()
    solutions = []
    for combo in combos:
        sol = Solution(customers_lrp, depots_lrp)
        sol.vehicle_capacity = vc
        sol.depots = [d for d in sol.depots if d.depot_number in combo]
        sol.build_distances()
        assign_depots(sol.customers)
        for depot in sol.depots:
            build_vehicle_routes(depot, vc)
        sol.calculate_total_distance()
        solutions.append(sol)

    config = CuckooConfig(num_solutions=NUM_SOLUTIONS, num_iterations=NUM_ITERATIONS)
    best = CuckooSearch(config).optimize(solutions)
    elapsed = time.time() - t0

    schema_sol = cuckoo_solution_to_schema(best)
    routes = [r.model_dump() for r in schema_sol.routes]

    customers_fix: dict[int, dict] = {
        c.customer_number: {"x": c.x_cord, "y": c.y_cord, "demand": c.demand}
        for c in best.customers
    }
    depots_fix: dict[int, dict] = {
        d.depot_number: {
            "x": d.x_cord, "y": d.y_cord,
            "capacity": d.original_capacity,
            "fixed_cost": d.fixed_cost,
            "variable_cost": d.variable_cost,
        }
        for d in best.depots
    }
    return routes, customers_fix, depots_fix, float(vc), elapsed, schema_sol.total_cost


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 64
_SKIP = "SKIPPED"


def _val_line(label: str, result: ValidationResult | None, width: int = 22) -> str:
    if result is None:
        return f"  {label:<{width}}  {_SKIP}"
    status = "PASS" if result.passed else "FAIL"
    return f"  {label:<{width}}  {status} ({result.score:.2f})"


def _cell(value: str | None, width: int = 16) -> str:
    return (value or _SKIP).ljust(width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(instance_name: str = "Srivastava86") -> None:
    """Run the comparison and print + save a report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_instance(instance_name)
    customers = dataset["customers"]
    depots = dataset["depots"]
    vc = dataset["vehicle_capacity"]

    print(_SEP)
    print(f"SIDE-BY-SIDE COMPARISON — {instance_name}")
    print(_SEP)

    # -----------------------------------------------------------------------
    # Cuckoo Search
    # -----------------------------------------------------------------------
    print("\n  Running Cuckoo Search...", end=" ", flush=True)
    cuckoo_result = _run_cuckoo(instance_name)
    if cuckoo_result is None:
        print("SKIPPED (float demand incompatibility)")
        cuckoo_routes: list[dict] | None = None
        cuckoo_customers: dict | None = None
        cuckoo_depots: dict | None = None
        cuckoo_elapsed: float | None = None
        cuckoo_cost: float | None = None
        cuckoo_n_routes: int | None = None
        v_cap_c = v_cov_c = v_dep_c = v_dist_c = None
    else:
        (
            cuckoo_routes, cuckoo_customers, cuckoo_depots,
            _, cuckoo_elapsed, cuckoo_cost,
        ) = cuckoo_result
        cuckoo_n_routes = len(cuckoo_routes)
        v_cap_c = validate_vehicle_capacity(cuckoo_routes, cuckoo_customers, vc)
        v_cov_c = validate_customer_coverage(cuckoo_routes, cuckoo_customers)
        v_dep_c = validate_depot_capacity(cuckoo_routes, cuckoo_customers, cuckoo_depots)
        v_dist_c = validate_route_distances(cuckoo_routes, cuckoo_customers, cuckoo_depots)
        print(f"done ({cuckoo_n_routes} routes, {cuckoo_elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # LLM Solver
    # -----------------------------------------------------------------------
    print("  Running LLM solver...", end=" ", flush=True)
    solver = LLMSolver()
    try:
        llm_solution, llm_meta = solver.solve(dataset)
        llm_routes = [r.model_dump() for r in llm_solution.routes]
        llm_elapsed = llm_meta["elapsed_seconds"]
        llm_cost = llm_solution.total_cost
        llm_n_routes = len(llm_routes)
        v_cap_l = validate_vehicle_capacity(llm_routes, customers, vc)
        v_cov_l = validate_customer_coverage(llm_routes, customers)
        v_dep_l = validate_depot_capacity(llm_routes, customers, depots)
        v_dist_l = validate_route_distances(llm_routes, customers, depots)
        faith_result = manual_faithfulness_check(dataset, llm_solution)
        llm_error: str | None = None
        print(f"done ({llm_n_routes} routes, {llm_elapsed:.1f}s)")
    except Exception as exc:
        llm_error = str(exc)
        llm_solution = None
        llm_routes = []
        llm_elapsed = llm_cost = llm_n_routes = None
        v_cap_l = v_cov_l = v_dep_l = v_dist_l = None
        faith_result = None
        print(f"ERROR: {exc}")

    # -----------------------------------------------------------------------
    # Side-by-side report
    # -----------------------------------------------------------------------
    w = 20  # label column width
    print()
    print(_SEP)
    print(f"{'':>{w}}  {'Cuckoo Search':<16}  {'LLM Solver':<16}")
    print(f"{'':>{w}}  {'-------------':<16}  {'----------':<16}")

    def row(label: str, c_val: str | None, l_val: str | None) -> None:
        print(f"  {label:<{w}}  {_cell(c_val):<16}  {_cell(l_val):<16}")

    def vrow(label: str, r_c: ValidationResult | None, r_l: ValidationResult | None) -> None:
        c_str = None if r_c is None else f"{'PASS' if r_c.passed else 'FAIL'} ({r_c.score:.2f})"
        l_str = None if r_l is None else f"{'PASS' if r_l.passed else 'FAIL'} ({r_l.score:.2f})"
        row(label, c_str, l_str)

    row("Total Cost",
        None if cuckoo_cost is None else f"{cuckoo_cost:.2f}",
        None if llm_cost is None else f"{llm_cost:.2f}")
    row("Routes",
        None if cuckoo_n_routes is None else str(cuckoo_n_routes),
        None if llm_n_routes is None else str(llm_n_routes))
    row("Time (s)",
        None if cuckoo_elapsed is None else f"{cuckoo_elapsed:.1f}",
        None if llm_elapsed is None else f"{llm_elapsed:.1f}")

    print()
    vrow("Vehicle Capacity",  v_cap_c,  v_cap_l)
    vrow("Customer Coverage", v_cov_c,  v_cov_l)
    vrow("Depot Capacity",    v_dep_c,  v_dep_l)
    vrow("Route Distances",   v_dist_c, v_dist_l)

    # Faithfulness row (LLM only)
    if faith_result is not None:
        fs = faith_result["score"]
        phantoms = faith_result["phantom_customers"] + faith_result["phantom_depots"]
        faith_str = f"{'PASS' if fs >= 1.0 else 'FAIL'} ({fs:.2f})"
        faith_detail = "No phantom IDs" if not phantoms else f"Phantoms: {phantoms}"
        row("Faithfulness", None, faith_str)
        if phantoms:
            print(f"  {'':>{w}}  {'':16}  {faith_detail}")
    else:
        row("Faithfulness", None, None)

    # LLM reasoning
    if llm_solution is not None and llm_solution.reasoning:
        reasoning = llm_solution.reasoning
        short = (reasoning[:120] + "...") if len(reasoning) > 120 else reasoning
        print()
        print("  LLM Reasoning:")
        print(f'    "{short}"')

    # Violations
    all_violations: list[str] = []
    for label, result in [
        ("Vehicle Capacity (LLM)", v_cap_l),
        ("Customer Coverage (LLM)", v_cov_l),
        ("Depot Capacity (LLM)", v_dep_l),
        ("Route Distances (LLM)", v_dist_l),
    ]:
        if result and not result.passed:
            for v in result.violations:
                all_violations.append(f"  [{label}] {v}")

    if all_violations:
        print()
        print("  Violations:")
        for v in all_violations:
            print(v)

    if llm_error:
        print()
        print(f"  LLM Error: {llm_error[:200]}")

    print(_SEP)

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    def _vr(r: ValidationResult | None) -> dict | None:
        if r is None:
            return None
        return {"passed": r.passed, "score": r.score, "violations": r.violations}

    output = {
        "instance": instance_name,
        "cuckoo_search": None if cuckoo_routes is None else {
            "total_cost": cuckoo_cost,
            "n_routes": cuckoo_n_routes,
            "elapsed_seconds": cuckoo_elapsed,
            "vehicle_capacity": _vr(v_cap_c),
            "customer_coverage": _vr(v_cov_c),
            "depot_capacity": _vr(v_dep_c),
            "route_distances": _vr(v_dist_c),
        },
        "llm_solver": None if llm_error else {
            "model": llm_meta.get("model"),
            "total_cost": llm_cost,
            "n_routes": llm_n_routes,
            "elapsed_seconds": llm_elapsed,
            "open_depots": llm_solution.open_depots if llm_solution else [],
            "vehicle_capacity": _vr(v_cap_l),
            "customer_coverage": _vr(v_cov_l),
            "depot_capacity": _vr(v_dep_l),
            "route_distances": _vr(v_dist_l),
            "faithfulness": faith_result,
            "reasoning": llm_solution.reasoning if llm_solution else None,
        },
        "llm_error": llm_error,
    }

    out_path = RESULTS_DIR / f"{instance_name}_comparison.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Srivastava86"
    main(name)
