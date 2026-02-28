"""Cross-model comparison — run the LLM pipeline on multiple Claude models.

Runs each model × strategy × instance combination with cost safeguards:
a hard per-model dollar cap prevents runaway spend.

Usage::

    uv run python run_cross_model.py
    uv run python run_cross_model.py --max-cost-usd 1.00 --instances Srivastava86
    uv run python run_cross_model.py --strategies naive self_healing \\
        --models claude-haiku-4-5-20251001 claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_agent.solver import LLMSolver, SolveStrategy
from qa_suite.common.cost_guard import CostGuard
from qa_suite.common.faithfulness import manual_faithfulness_check
from qa_suite.common.fixtures import INSTANCES, load_instance
from qa_suite.deterministic_checks.validators import (
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_total_cost,
    validate_vehicle_capacity,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"]
DEFAULT_INSTANCES = ["Srivastava86", "Gaskell67", "Perl83"]
DEFAULT_STRATEGIES = ["naive", "self_healing"]

_MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-6": "Opus",
}


def _run_one(
    model: str,
    strategy: str,
    instance_name: str,
    guard: CostGuard,
) -> dict:
    """Run a single model × strategy × instance combination."""
    dataset = load_instance(instance_name)

    if not guard.can_afford():
        return {"available": False, "error": "budget_exhausted"}

    solver = LLMSolver(
        model=model,
        strategy=SolveStrategy(strategy),
    )

    try:
        solution, meta = solver.solve(dataset)
    except Exception as exc:
        return {"available": False, "error": str(exc)[:400]}

    # Record spend
    guard.record(
        meta.get("input_tokens", 0),
        meta.get("output_tokens", 0),
    )

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

    return {
        "available": True,
        "error": None,
        "model": model,
        "strategy": strategy,
        "total_cost": solution.total_cost,
        "n_routes": len(routes),
        "time_seconds": round(meta["elapsed_seconds"], 2),
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
        "faithfulness_score": faith["score"],
        "phantom_customers": faith["phantom_customers"],
        "phantom_depots": faith["phantom_depots"],
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "heal_attempts": meta.get("heal_attempts"),
        "heal_exhausted": meta.get("heal_exhausted"),
    }


def run_cross_model(
    models: list[str] | None = None,
    instances: list[str] | None = None,
    strategies: list[str] | None = None,
    max_cost_usd: float = 2.00,
) -> dict:
    """Run the full model × strategy × instance grid with cost guards.

    Returns the result dict (also saved to JSON).
    """
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    models = models or MODELS
    instances = instances or DEFAULT_INSTANCES
    strategies = strategies or DEFAULT_STRATEGIES
    console = Console()

    guards: dict[str, CostGuard] = {
        m: CostGuard(m, max_cost_usd=max_cost_usd) for m in models
    }
    results: dict[str, dict[str, dict[str, dict]]] = {m: {} for m in models}

    def _make_spend_table() -> Table:
        table = Table(title="Cross-Model Spend Tracker", show_lines=True)
        table.add_column("Model", style="bold")
        table.add_column("Calls", justify="right")
        table.add_column("Cost (USD)", justify="right")
        table.add_column("Budget", justify="right")
        table.add_column("Status", justify="center")
        for m in models:
            g = guards[m]
            status = "[red]EXHAUSTED[/red]" if g.budget_exhausted else "[green]OK[/green]"
            table.add_row(
                _MODEL_SHORT.get(m, m),
                str(g.calls),
                f"${g.total_cost_usd:.4f}",
                f"${max_cost_usd:.2f}",
                status,
            )
        return table

    with Live(_make_spend_table(), console=console, refresh_per_second=2) as live:
        for model in models:
            guard = guards[model]
            results[model] = {}
            for strat in strategies:
                results[model][strat] = {}
                for inst in instances:
                    if guard.budget_exhausted:
                        results[model][strat][inst] = {
                            "available": False,
                            "error": "budget_exhausted",
                        }
                    else:
                        console.print(
                            f"  [{_MODEL_SHORT.get(model, model)}] {strat} / {inst}...",
                            end=" ",
                        )
                        r = _run_one(model, strat, inst, guard)
                        if r["available"]:
                            console.print("done")
                        else:
                            console.print(f"[red]{r.get('error', 'ERROR')[:60]}[/red]")
                        results[model][strat][inst] = r
                    live.update(_make_spend_table())

    # Build model_comparison summary
    model_comparison = {}
    for m in models:
        g = guards[m]
        completed = sum(
            1
            for s in strategies
            for i in instances
            if results[m].get(s, {}).get(i, {}).get("available", False)
        )
        model_comparison[m] = {
            "total_cost_usd": round(g.total_cost_usd, 4),
            "instances_completed": completed,
            "budget_exhausted": g.budget_exhausted,
        }

    output = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instances": instances,
        "strategies": strategies,
        "models": models,
        "max_cost_usd": max_cost_usd,
        "results": results,
        "model_comparison": model_comparison,
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"cross_model_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    console.print(f"\nResults saved to {out_path}")
    console.print(_make_spend_table())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model comparison — run LLM pipeline on multiple Claude models."
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help=f"Instance names (default: {DEFAULT_INSTANCES}).",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help=f"Strategies to run (default: {DEFAULT_STRATEGIES}).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Model IDs (default: {MODELS}).",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=2.00,
        help="Per-model budget cap in USD (default: 2.00).",
    )
    args = parser.parse_args()

    # Validate instances
    instances = args.instances or DEFAULT_INSTANCES
    for name in instances:
        if name not in INSTANCES:
            print(f"[WARN] Unknown instance {name!r} — skipping.", file=sys.stderr)
    instances = [n for n in instances if n in INSTANCES]

    run_cross_model(
        models=args.models,
        instances=instances,
        strategies=args.strategies,
        max_cost_usd=args.max_cost_usd,
    )


if __name__ == "__main__":
    main()
