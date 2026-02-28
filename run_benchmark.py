"""Master benchmark script — runs solvers on one or more LRP instances.

For each instance the script:
  1. Loads the dataset via ``load_instance()``.
  2. Runs Cuckoo Search (skips with a note on float-demand instances).
  3. Runs the LLM solver with the chosen strategy (or all three tiers).
  4. Validates all outputs with deterministic checks + faithfulness.
  5. Saves structured results to ``results/{instance_name}.json``.
  6. Prints a one-line summary.

Usage::

    python run_benchmark.py                          # Srivastava86 only (default)
    python run_benchmark.py Srivastava86 Gaskell67   # named instances
    python run_benchmark.py --all                    # every registered instance
    python run_benchmark.py --strategy all           # run all 3 LLM tiers
    python run_benchmark.py --strategy self_healing  # self-healing tier only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# QA-suite imports
# ---------------------------------------------------------------------------
from ai_agent.solver import LLMSolver, SolveStrategy

# ---------------------------------------------------------------------------
# Internal solver imports (Cuckoo Search)
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
from qa_suite.deterministic_checks.optimality_gap import compute_optimality_gap
from qa_suite.deterministic_checks.reasoning_audit import audit_reasoning
from qa_suite.deterministic_checks.validators import (
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_total_cost,
    validate_vehicle_capacity,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_CS_ITERATIONS = 10
_CS_SOLUTIONS = 3


# ---------------------------------------------------------------------------
# Cuckoo Search runner
# ---------------------------------------------------------------------------

def _run_cuckoo(instance_name: str, dataset: dict) -> dict:
    """Run Cuckoo Search; return result dict or a skip-reason dict."""
    cli_file, dep_file, vc = INSTANCES[instance_name]
    t0 = time.time()
    try:
        customers_lrp = lrp_load_customers(DATA_DIR / cli_file)
        depots_lrp = lrp_load_depots(DATA_DIR / dep_file)
    except (ValueError, TypeError) as exc:
        return {"available": False, "skip_reason": str(exc)}

    all_ids = tuple(range(1, len(depots_lrp) + 1))
    combos = list(combinations(all_ids, len(depots_lrp)))[:_CS_SOLUTIONS]

    solutions = []
    try:
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
    except (ValueError, TypeError) as exc:
        return {"available": False, "skip_reason": str(exc)}

    config = CuckooConfig(num_solutions=_CS_SOLUTIONS, num_iterations=_CS_ITERATIONS)
    best = CuckooSearch(config).optimize(solutions)
    elapsed = time.time() - t0

    schema_sol = cuckoo_solution_to_schema(best)
    routes = [r.model_dump() for r in schema_sol.routes]

    customers_d = dataset["customers"]
    depots_d = dataset["depots"]
    vc = dataset["vehicle_capacity"]

    v_cap = validate_vehicle_capacity(routes, customers_d, vc)
    v_cov = validate_customer_coverage(routes, customers_d)
    v_dep = validate_depot_capacity(routes, customers_d, depots_d)
    v_dist = validate_route_distances(routes, customers_d, depots_d)
    v_cost = validate_total_cost(
        routes, depots_d, schema_sol.open_depots, schema_sol.total_cost,
    )

    return {
        "available": True,
        "skip_reason": None,
        "total_cost": round(schema_sol.total_cost, 4),
        "open_depots": schema_sol.open_depots,
        "n_routes": len(routes),
        "time_seconds": round(elapsed, 2),
        "vehicle_capacity_valid": v_cap.passed,
        "vehicle_capacity_score": round(v_cap.score, 4),
        "customer_coverage_valid": v_cov.passed,
        "customer_coverage_score": round(v_cov.score, 4),
        "depot_capacity_valid": v_dep.passed,
        "depot_capacity_score": round(v_dep.score, 4),
        "route_distances_valid": v_dist.passed,
        "route_distances_score": round(v_dist.score, 4),
        "total_cost_valid": v_cost.passed,
        "total_cost_score": round(v_cost.score, 4),
        "_routes_raw": routes,  # kept for gap computation, stripped before save
    }


# ---------------------------------------------------------------------------
# LLM solver runner
# ---------------------------------------------------------------------------

def _run_llm(solver: LLMSolver, dataset: dict) -> dict:
    """Run LLM solver; return result dict or an error dict."""
    try:
        solution, meta = solver.solve(dataset)
    except Exception as exc:
        return {"available": False, "error": str(exc)[:400]}

    routes = [r.model_dump() for r in solution.routes]
    customers_d = dataset["customers"]
    depots_d = dataset["depots"]
    vc = dataset["vehicle_capacity"]

    v_cap = validate_vehicle_capacity(routes, customers_d, vc)
    v_cov = validate_customer_coverage(routes, customers_d)
    v_dep = validate_depot_capacity(routes, customers_d, depots_d)
    v_dist = validate_route_distances(routes, customers_d, depots_d)
    v_cost = validate_total_cost(
        routes, depots_d, solution.open_depots, solution.total_cost,
    )
    faith = manual_faithfulness_check(dataset, solution)

    # Reasoning-solution consistency audit (zero API cost)
    audit = audit_reasoning(solution, customers_d, depots_d)

    reasoning = solution.reasoning or ""
    excerpt = (reasoning[:120] + "...") if len(reasoning) > 120 else reasoning

    return {
        "available": True,
        "error": None,
        "model": meta.get("model"),
        "strategy": meta.get("strategy", "unknown"),
        "total_cost": solution.total_cost,
        "n_routes": len(routes),
        "time_seconds": meta["elapsed_seconds"],
        "heal_attempts": meta.get("heal_attempts"),
        "heal_exhausted": meta.get("heal_exhausted"),
        "vehicle_capacity_valid": v_cap.passed,
        "vehicle_capacity_score": round(v_cap.score, 4),
        "customer_coverage_valid": v_cov.passed,
        "customer_coverage_score": round(v_cov.score, 4),
        "depot_capacity_valid": v_dep.passed,
        "depot_capacity_score": round(v_dep.score, 4),
        "route_distances_valid": v_dist.passed,
        "route_distances_score": round(v_dist.score, 4),
        "total_cost_valid": v_cost.passed,
        "total_cost_score": round(v_cost.score, 4),
        "route_distance_violations": v_dist.violations[:10],
        "faithfulness_score": faith["score"],
        "phantom_customers": faith["phantom_customers"],
        "phantom_depots": faith["phantom_depots"],
        "open_depots": solution.open_depots,
        "reasoning_audit": audit.as_dict(),
        "reasoning_excerpt": excerpt,
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "_solution": solution,  # kept for gap computation, stripped before save
    }


# ---------------------------------------------------------------------------
# Summary line helpers
# ---------------------------------------------------------------------------

def _cs_summary(cs: dict) -> str:
    if not cs["available"]:
        return "CS: SKIP"
    n_pass = sum([
        cs["vehicle_capacity_valid"],
        cs["customer_coverage_valid"],
        cs["depot_capacity_valid"],
        cs["route_distances_valid"],
        cs.get("total_cost_valid", True),
    ])
    return f"CS: {n_pass}/5 PASS (cost={cs['total_cost']:.2f}, {cs['time_seconds']:.1f}s)"


def _llm_summary(llm: dict) -> str:
    if not llm["available"]:
        return "LLM: ERROR"
    n_pass = sum([
        llm["vehicle_capacity_valid"],
        llm["customer_coverage_valid"],
        llm["depot_capacity_valid"],
        llm["route_distances_valid"],
        llm.get("total_cost_valid", False),
        (llm.get("faithfulness_score") or 0) >= 1.0,
    ])
    strat = llm.get("strategy", "?")
    n_total = 6
    status = "VALID \u2713" if n_pass == n_total else "INVALID \u2717"

    # Severity: worst soft-score violation
    sev_parts = []
    for key in ["vehicle_capacity_score", "customer_coverage_score",
                "depot_capacity_score", "route_distances_score", "total_cost_score"]:
        s = llm.get(key, 1.0)
        sev_parts.append(1.0 - s)
    max_sev = max(sev_parts) if sev_parts else 0.0

    # Reasoning fidelity
    audit = llm.get("reasoning_audit", {})
    if audit.get("total_claims", 0) > 0:
        reasoning_str = f"{audit.get('consistency_score', 1.0):.0%} consistent"
    else:
        reasoning_str = "no claims"

    return (
        f"{strat}: {status} | Checks: {n_pass}/{n_total} | "
        f"Severity: {max_sev:.2f} | Reasoning: {reasoning_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_instance(
    instance_name: str,
    solvers: list[tuple[str, LLMSolver]],
) -> dict:
    """Run CS + LLM solver(s) on one instance and save the result JSON."""
    print(f"\n{'='*60}")
    print(f"  {instance_name}")
    print(f"{'='*60}")

    dataset = load_instance(instance_name)
    n_customers = len(dataset["customers"])
    n_depots = len(dataset["depots"])
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("  Cuckoo Search...", end=" ", flush=True)
    cs = _run_cuckoo(instance_name, dataset)
    if cs["available"]:
        print(f"done ({cs['n_routes']} routes, {cs['time_seconds']:.1f}s)")
    else:
        print(f"SKIPPED ({cs.get('skip_reason', '')[:60]})")

    llm_results: dict[str, dict] = {}
    for strat_name, solver in solvers:
        print(f"  LLM [{strat_name}]...", end=" ", flush=True)
        llm = _run_llm(solver, dataset)
        if llm["available"]:
            print(f"done ({llm['n_routes']} routes, {llm['time_seconds']:.1f}s)")
        else:
            print(f"ERROR: {llm.get('error', '')[:80]}")
        llm_results[strat_name] = llm

    # Compute optimality gaps (zero API cost, requires both CS + LLM)
    cs_schema = None
    if cs["available"]:
        from qa_suite.common.schemas import LRPSolution, Route
        cs_routes_raw = cs.get("_routes_raw")
        if cs_routes_raw:
            cs_schema = LRPSolution(
                routes=[Route(**r) for r in cs_routes_raw],
                open_depots=cs.get("open_depots", []),
                total_cost=cs["total_cost"],
            )

    for strat_name, llm in llm_results.items():
        llm_sol = llm.pop("_solution", None)
        if cs_schema and llm_sol:
            gap = compute_optimality_gap(llm_sol, cs_schema)
            llm["optimality_gap"] = gap.as_dict()

    # Strip internal keys before serialisation
    cs.pop("_routes_raw", None)

    result = {
        "instance": instance_name,
        "n_customers": n_customers,
        "n_depots": n_depots,
        "timestamp": timestamp,
        "cuckoo_search": cs,
        "llm_solvers": llm_results,
        # Backward compat: also store the first LLM result as "llm_solver".
        "llm_solver": next(iter(llm_results.values())) if llm_results else {},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{instance_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"  [{instance_name}] {_cs_summary(cs)}")
    for strat_name, llm in llm_results.items():
        print(f"  [{instance_name}] {_llm_summary(llm)}")
    print(f"  Saved → {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIQA benchmark — run both solvers on LRP instances."
    )
    parser.add_argument(
        "instances",
        nargs="*",
        help="Instance names to benchmark (default: Srivastava86).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every registered instance.",
    )
    parser.add_argument(
        "--strategy",
        choices=["naive", "cot", "self_healing", "all"],
        default="naive",
        help="LLM solving strategy. 'all' runs all three tiers. Default: naive.",
    )
    parser.add_argument(
        "--track-history",
        action="store_true",
        help="Append results to history.jsonl and check for regressions.",
    )
    parser.add_argument(
        "--compute-ci",
        action="store_true",
        help="Compute bootstrap confidence intervals after the run.",
    )
    args = parser.parse_args()

    if args.all:
        names = list(INSTANCES.keys())
    elif args.instances:
        names = args.instances
    else:
        names = ["Srivastava86"]

    # Build solver list based on strategy choice.
    if args.strategy == "all":
        solvers = [
            (s.value, LLMSolver(strategy=s))
            for s in SolveStrategy
        ]
    else:
        strat = SolveStrategy(args.strategy)
        solvers = [(strat.value, LLMSolver(strategy=strat))]

    all_results: list[dict] = []
    for name in names:
        if name not in INSTANCES:
            print(f"[WARN] Unknown instance {name!r} — skipping.", file=sys.stderr)
            continue
        result = run_instance(name, solvers)
        all_results.append(result)

    print(f"\nAll done. Results in {RESULTS_DIR}/")

    # ── Regression history tracking ──
    if args.track_history and all_results:
        from qa_suite.regression.regression_gate import (
            append_to_history,
            detect_regressions,
            load_history,
            print_report,
        )

        print("\n--- Regression Gate ---")
        append_to_history(all_results)
        history = load_history()
        regressions = detect_regressions(history)
        print_report(history, regressions)

    # ── Bootstrap CI ──
    if args.compute_ci:
        from qa_suite.probabilistic.bootstrap_ci import (
            bootstrap_validity_ci,
            plot_bootstrap_ci,
            print_ci_table,
        )

        print("\n--- Bootstrap Confidence Intervals ---")
        ci_results = bootstrap_validity_ci(RESULTS_DIR)
        if ci_results:
            print_ci_table(ci_results)
            out = plot_bootstrap_ci(ci_results)
            print(f"Chart saved to {out}")


if __name__ == "__main__":
    main()
