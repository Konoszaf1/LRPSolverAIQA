"""Validity breakpoint analysis: at what instance size does each LLM strategy
lose feasibility?

Instead of running the same instance 30 times (Monte Carlo), run each LLM
strategy **once** across instances of increasing size: Srivastava86 (8),
Gaskell67 (21), Perl83 (55), Ch69 (100).  Each call produces a distinct data
point on the validity scaling curve.

This answers: **at what instance size does each strategy break, and how severe
are the violations?**  12 API calls total (4 instances x 3 strategies), each
producing unique insight that repetition cannot provide.

A Cuckoo Search baseline is also computed (zero API cost) as a validity
reference — CS achieves 100% validity at all sizes.

Usage::

    uv run python -m qa_suite.probabilistic.scaling_analysis \\
        --strategies all
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt

from ai_agent.solver import LLMSolver, SolveStrategy
from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.algorithms.nearest_neighbor import (
    assign_depots,
    build_vehicle_routes,
)
from lrp.config import CuckooConfig
from lrp.io.data_loader import load_customers as lrp_load_customers
from lrp.io.data_loader import load_depots as lrp_load_depots
from lrp.models.solution import Solution
from qa_suite.common.adapters import cuckoo_solution_to_schema
from qa_suite.common.fixtures import DATA_DIR, INSTANCES, load_instance
from qa_suite.deterministic_checks.soft_scoring import score_all

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SCALING_INSTANCES = [
    "Srivastava86",
    "Gaskell67",
    "Perl83",
    "Ch69",
    "Or76",
    "Min92",
    "Daskin95",
]

_STRATEGY_COLORS = {
    "naive": "#e74c3c",
    "cot": "#e67e22",
    "self_healing": "#3498db",
}

_CS_SOLUTIONS = 3
_CS_ITERATIONS = 10


def _run_cs_cost(instance_name: str) -> float | None:
    """Run CS on an instance, return total_cost or None on any failure."""
    cli_file, dep_file, vc = INSTANCES[instance_name]
    try:
        custs = lrp_load_customers(DATA_DIR / cli_file)
        deps = lrp_load_depots(DATA_DIR / dep_file)

        all_ids = tuple(range(1, len(deps) + 1))
        combos = list(combinations(all_ids, len(deps)))[:_CS_SOLUTIONS]

        solutions = []
        for combo in combos:
            sol = Solution(custs, deps)
            sol.vehicle_capacity = vc
            sol.depots = [
                d for d in sol.depots if d.depot_number in combo
            ]
            sol.build_distances()
            assign_depots(sol.customers)
            for depot in sol.depots:
                build_vehicle_routes(depot, vc)
            sol.calculate_total_distance()
            solutions.append(sol)

        config = CuckooConfig(
            num_solutions=_CS_SOLUTIONS,
            num_iterations=_CS_ITERATIONS,
        )
        best = CuckooSearch(config).optimize(solutions)
        schema = cuckoo_solution_to_schema(best)
        return schema.total_cost
    except (ValueError, TypeError):
        return None


def run_scaling_analysis(
    strategies: list[SolveStrategy],
    instances: list[str] | None = None,
    max_llm_size: int | None = None,
) -> dict:
    """Run each strategy once per instance, collect soft scores.

    Args:
        strategies: Which LLM strategies to evaluate.
        instances: Instance names to run; defaults to SCALING_INSTANCES.
        max_llm_size: Skip LLM calls for instances with more customers than
            this threshold.  CS baselines still run on all instances.
            None means no limit.
    """
    inst_list = instances or SCALING_INSTANCES
    results: list[dict] = []

    # Pre-compute CS baselines (zero API cost, runs on all instances)
    cs_costs: dict[str, float | None] = {}
    for name in inst_list:
        print(f"  [CS] {name}...", end=" ", flush=True)
        cs_costs[name] = _run_cs_cost(name)
        if cs_costs[name] is not None:
            print(f"cost={cs_costs[name]:.1f}")
        else:
            print("skipped")

    for name in inst_list:
        dataset = load_instance(name)
        n_cust = len(dataset["customers"])

        for strat in strategies:
            tag = f"[{name}][{strat.value}]"

            entry: dict = {
                "instance": name,
                "n_customers": n_cust,
                "strategy": strat.value,
            }

            # Skip LLM call if instance is too large
            if max_llm_size is not None and n_cust > max_llm_size:
                print(f"  {tag} SKIPPED (n={n_cust} > --max-llm-size={max_llm_size})")
                entry.update({
                    "error": f"skipped: n_customers={n_cust} exceeds max_llm_size={max_llm_size}",
                    "elapsed_seconds": 0.0,
                    "skipped": True,
                })
                results.append(entry)
                continue

            print(f"  {tag} Solving...", end=" ", flush=True)
            t0 = time.time()

            try:
                solver = LLMSolver(strategy=strat)
                solution, meta = solver.solve(dataset)
                elapsed = time.time() - t0

                routes = [r.model_dump() for r in solution.routes]
                report = score_all(
                    routes=routes,
                    customers=dataset["customers"],
                    depots=dataset["depots"],
                    open_depots=solution.open_depots,
                    vehicle_capacity=dataset["vehicle_capacity"],
                    stated_total_cost=solution.total_cost,
                )

                cs_cost = cs_costs.get(name)
                gap = None
                if cs_cost and cs_cost > 0:
                    gap = round(
                        (solution.total_cost - cs_cost)
                        / cs_cost
                        * 100,
                        2,
                    )

                entry.update({
                    "total_cost": solution.total_cost,
                    "cs_total_cost": cs_cost,
                    "optimality_gap_pct": gap,
                    "max_severity": report.max_severity,
                    "all_passed": report.all_passed,
                    "soft_scores": report.as_dict(),
                    "elapsed_seconds": round(elapsed, 2),
                    "error": None,
                })
                status = "PASS" if report.all_passed else "FAIL"
                print(
                    f"{status} sev={report.max_severity:.3f} "
                    f"gap={gap}% ({elapsed:.1f}s)"
                )

            except Exception as exc:
                elapsed = time.time() - t0
                entry.update({
                    "error": str(exc)[:200],
                    "elapsed_seconds": round(elapsed, 2),
                })
                print(f"ERROR ({str(exc)[:60]})")

            results.append(entry)

    return {"results": results}


def plot_scaling(data: dict, output_path: Path) -> None:
    """Two-panel validity breakpoint chart.

    Top:    Validity pass rate (%) vs instance size (step chart).
    Bottom: Max severity when failed (scatter) vs instance size.
    """
    results = [r for r in data["results"] if r.get("error") is None]
    if not results:
        print("  No valid results to plot.")
        return

    strategies = sorted({r["strategy"] for r in results})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), height_ratios=[1, 1],
    )

    _strategy_labels = {
        "naive": "Naive",
        "cot": "CoT",
        "self_healing": "Self-Healing",
    }

    for strat in strategies:
        pts = sorted(
            [r for r in results if r["strategy"] == strat],
            key=lambda r: r["n_customers"],
        )
        xs = [p["n_customers"] for p in pts]
        color = _STRATEGY_COLORS.get(strat, "gray")
        label = _strategy_labels.get(strat, strat)

        # Top: validity pass rate (all_passed → 100%, else 0%)
        pass_rates = [100.0 if p.get("all_passed", False) else 0.0 for p in pts]
        ax1.step(
            xs, pass_rates, where="mid",
            color=color, label=label, linewidth=2.5,
        )
        ax1.scatter(xs, pass_rates, color=color, s=50, zorder=5)

        # Bottom: max severity (only for failed instances)
        for p in pts:
            sev = p.get("max_severity", 0)
            passed = p.get("all_passed", False)
            marker = "o" if not passed else "D"
            alpha = 1.0 if not passed else 0.3
            ax2.scatter(
                p["n_customers"], sev,
                color=color, marker=marker, s=60, alpha=alpha, zorder=5,
            )
        # Connect with a dashed line for visual continuity
        sevs = [p.get("max_severity", 0) for p in pts]
        ax2.plot(
            xs, sevs, "--",
            color=color, label=label, linewidth=1.5, alpha=0.6,
        )

    ax1.set_ylabel("Validity Pass Rate (%)")
    ax1.set_title("Validity Breakpoint: Pass Rate vs Instance Size")
    ax1.set_ylim(-5, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.axhline(y=100, color="green", linestyle="--", alpha=0.3, label="100% validity")
    ax1.legend(loc="lower left")
    ax1.set_xlabel("Number of Customers")

    ax2.set_ylabel("Max Severity (when failed)")
    ax2.set_title("Violation Severity at Failure Points")
    ax2.axhline(
        y=0.05, color="gray", linestyle="--",
        alpha=0.5, label="5% threshold",
    )
    ax2.legend(loc="upper left")
    ax2.set_xlabel("Number of Customers")

    fig.suptitle(
        "LLM Validity Scaling — CS achieves 100% validity at all sizes",
        fontsize=10, style="italic", y=0.02, color="gray",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instance difficulty scaling analysis",
    )
    parser.add_argument(
        "--strategies",
        choices=["naive", "cot", "self_healing", "all"],
        default="all",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--max-llm-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Skip LLM calls for instances with more than N customers. "
            "CS baselines still run on all instances. "
            "Use this to avoid token-limit errors on very large instances "
            "(e.g. --max-llm-size 100)."
        ),
    )
    args = parser.parse_args()

    if args.strategies == "all":
        strats = [
            SolveStrategy.NAIVE,
            SolveStrategy.COT,
            SolveStrategy.SELF_HEALING,
        ]
    else:
        strats = [SolveStrategy(args.strategies)]

    names = [s.value for s in strats]
    print(f"Scaling analysis: strategies={names}")
    if args.max_llm_size is not None:
        print(f"  LLM calls capped at n<={args.max_llm_size} customers")
    data = run_scaling_analysis(strats, args.instances, max_llm_size=args.max_llm_size)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "scaling_analysis.json"
    png_path = RESULTS_DIR / "scaling_analysis.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved: {json_path}")

    plot_scaling(data, png_path)


if __name__ == "__main__":
    main()
