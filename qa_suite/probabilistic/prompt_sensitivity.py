"""Prompt format sensitivity analysis.

Tests whether the LLM's solution quality is sensitive to cosmetic changes in
how the same data is presented.  Five formatting variants of the same instance
are sent through the same strategy — if results diverge, the model is fragile
to prompt layout rather than reasoning about the problem.

5 API calls total (1 per variant), each producing a distinct data point.

Usage::

    uv run python -m qa_suite.probabilistic.prompt_sensitivity \\
        --instance Srivastava86 --strategy cot
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ai_agent.prompt_templates import COT_SYSTEM_PROMPT, NAIVE_SYSTEM_PROMPT
from ai_agent.solver import LLMSolver, SolveStrategy
from qa_suite.common.fixtures import load_instance
from qa_suite.deterministic_checks.soft_scoring import score_all

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


# ---------------------------------------------------------------------------
# Format variant builders
# ---------------------------------------------------------------------------

def _format_default(dataset: dict) -> str:
    """Standard column order: ID, X, Y, Demand (ascending ID)."""
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers = dataset.get("customers", {})
    depots = dataset.get("depots", {})

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")
    lines.append(f"Customers ({len(customers)}):")
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Demand':>8}")
    for nid, c in sorted(customers.items()):
        lines.append(
            f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}"
            f"  {c['demand']:>8.1f}"
        )

    lines.append(f"Depots ({len(depots)}):")
    lines.append(
        f"  {'ID':>4}  {'X':>8}  {'Y':>8}"
        f"  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}"
    )
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}"
            f"  {d['fixed_cost']:>10.2f}"
            f"  {d['variable_cost']:>8.3f}"
        )
    return "\n".join(lines)


def _format_demand_descending(dataset: dict) -> str:
    """Customers sorted by demand (highest first)."""
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers = dataset.get("customers", {})
    depots = dataset.get("depots", {})

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")
    lines.append(f"Customers ({len(customers)}) [sorted by demand, descending]:")
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Demand':>8}")
    for nid, c in sorted(
        customers.items(), key=lambda kv: kv[1]["demand"], reverse=True,
    ):
        lines.append(
            f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}"
            f"  {c['demand']:>8.1f}"
        )

    lines.append(f"Depots ({len(depots)}):")
    lines.append(
        f"  {'ID':>4}  {'X':>8}  {'Y':>8}"
        f"  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}"
    )
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}"
            f"  {d['fixed_cost']:>10.2f}"
            f"  {d['variable_cost']:>8.3f}"
        )
    return "\n".join(lines)


def _format_distance_from_centroid(dataset: dict) -> str:
    """Customers sorted by distance from the customer centroid."""
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers = dataset.get("customers", {})
    depots = dataset.get("depots", {})

    cx = sum(c["x"] for c in customers.values()) / max(len(customers), 1)
    cy = sum(c["y"] for c in customers.values()) / max(len(customers), 1)

    def dist_to_centroid(kv: tuple[int, dict]) -> float:
        c = kv[1]
        return math.dist((c["x"], c["y"]), (cx, cy))

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")
    lines.append(
        f"Customers ({len(customers)}) "
        f"[sorted by distance from centroid ({cx:.1f}, {cy:.1f})]:"
    )
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Demand':>8}")
    for nid, c in sorted(customers.items(), key=dist_to_centroid):
        lines.append(
            f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}"
            f"  {c['demand']:>8.1f}"
        )

    lines.append(f"Depots ({len(depots)}):")
    lines.append(
        f"  {'ID':>4}  {'X':>8}  {'Y':>8}"
        f"  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}"
    )
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}"
            f"  {d['fixed_cost']:>10.2f}"
            f"  {d['variable_cost']:>8.3f}"
        )
    return "\n".join(lines)


def _format_depots_first(dataset: dict) -> str:
    """Depot section printed before customer section."""
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers = dataset.get("customers", {})
    depots = dataset.get("depots", {})

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")

    # Depots first
    lines.append(f"Depots ({len(depots)}):")
    lines.append(
        f"  {'ID':>4}  {'X':>8}  {'Y':>8}"
        f"  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}"
    )
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}"
            f"  {d['fixed_cost']:>10.2f}"
            f"  {d['variable_cost']:>8.3f}"
        )

    # Then customers
    lines.append(f"Customers ({len(customers)}):")
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Demand':>8}")
    for nid, c in sorted(customers.items()):
        lines.append(
            f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}"
            f"  {c['demand']:>8.1f}"
        )
    return "\n".join(lines)


def _format_swapped_xy_labels(dataset: dict) -> str:
    """Column headers say Y before X (data values unchanged)."""
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers = dataset.get("customers", {})
    depots = dataset.get("depots", {})

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")
    lines.append(f"Customers ({len(customers)}):")
    # Swapped labels: Y before X in header, but data columns stay the same
    lines.append(f"  {'ID':>4}  {'Y':>8}  {'X':>8}  {'Demand':>8}")
    for nid, c in sorted(customers.items()):
        lines.append(
            f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}"
            f"  {c['demand']:>8.1f}"
        )

    lines.append(f"Depots ({len(depots)}):")
    lines.append(
        f"  {'ID':>4}  {'Y':>8}  {'X':>8}"
        f"  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}"
    )
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}"
            f"  {d['fixed_cost']:>10.2f}"
            f"  {d['variable_cost']:>8.3f}"
        )
    return "\n".join(lines)


VARIANTS: dict[str, callable] = {
    "default_order": _format_default,
    "demand_descending": _format_demand_descending,
    "distance_from_centroid": _format_distance_from_centroid,
    "depots_first": _format_depots_first,
    "swapped_xy_labels": _format_swapped_xy_labels,
}

_VALIDATOR_NAMES = [
    "vehicle_capacity",
    "customer_coverage",
    "depot_capacity",
    "route_distances",
    "total_cost",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _wrap_user_prompt(
    dataset_text: str, n_customers: int, strategy: SolveStrategy,
) -> str:
    """Wrap formatted data text with the appropriate user prompt template."""
    if strategy == SolveStrategy.NAIVE:
        return (
            "Solve the following Location-Routing Problem instance. "
            "Return only the JSON solution — no preamble, no markdown fences."
            f"\n\n=== PROBLEM DATA ===\n{dataset_text}\n"
            "===================\n\nOutput the JSON solution now.\n"
        )
    # CoT / self_healing
    if n_customers > 30:
        return (
            "Solve the following Location-Routing Problem. Show your work in "
            'the "reasoning" field — depot selection rationale and per-route '
            "capacity tallies. For distances, compute them accurately but you "
            "do NOT need to show every leg-by-leg calculation — just state "
            "each route's total stated_distance.\n\n"
            "IMPORTANT: Keep your reasoning CONCISE. Focus on correctness, "
            "not verbosity.\n\n"
            f"=== PROBLEM DATA ===\n{dataset_text}\n"
            "===================\n\n"
            "Follow the 6-step strategy from your instructions. "
            "Output the JSON now.\n"
        )
    return (
        "Solve the following Location-Routing Problem. You MUST show your "
        'work in the "reasoning" field — depot selection rationale, per-route '
        "capacity tallies, and leg-by-leg distance calculations.\n\n"
        f"=== PROBLEM DATA ===\n{dataset_text}\n"
        "===================\n\n"
        "Follow the 6-step strategy from your instructions. "
        "Output the JSON now.\n"
    )


def run_sensitivity_analysis(
    instance_name: str = "Srivastava86",
    strategy: SolveStrategy = SolveStrategy.COT,
) -> dict:
    """Run each format variant through the solver. 5 API calls."""
    dataset = load_instance(instance_name)
    n_customers = len(dataset["customers"])

    system_prompt = (
        NAIVE_SYSTEM_PROMPT
        if strategy == SolveStrategy.NAIVE
        else COT_SYSTEM_PROMPT
    )
    solver = LLMSolver(strategy=strategy)

    results: list[dict] = []

    for variant_name, format_fn in VARIANTS.items():
        tag = f"[{variant_name}]"
        print(f"  {tag} Formatting + calling API...", end=" ", flush=True)

        dataset_text = format_fn(dataset)
        user_prompt = _wrap_user_prompt(dataset_text, n_customers, strategy)

        entry: dict = {
            "variant": variant_name,
            "instance": instance_name,
            "strategy": strategy.value,
        }

        t0 = time.time()
        try:
            response = solver._call_api(
                user_prompt, system_prompt=system_prompt,
            )
            solution, raw_text = solver._parse_response(response)
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

            entry.update({
                "total_cost": solution.total_cost,
                "n_routes": len(solution.routes),
                "open_depots": solution.open_depots,
                "max_severity": report.max_severity,
                "all_passed": report.all_passed,
                "soft_scores": report.as_dict(),
                "elapsed_seconds": round(elapsed, 2),
                "error": None,
            })
            status = "PASS" if report.all_passed else "FAIL"
            print(
                f"{status} sev={report.max_severity:.3f} "
                f"cost={solution.total_cost:.1f} ({elapsed:.1f}s)"
            )

        except Exception as exc:
            elapsed = time.time() - t0
            entry.update({
                "error": str(exc)[:200],
                "elapsed_seconds": round(elapsed, 2),
            })
            print(f"ERROR ({str(exc)[:60]})")

        results.append(entry)

    return {"instance": instance_name, "strategy": strategy.value, "results": results}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_sensitivity(data: dict, output_path: Path) -> None:
    """Heatmap: validators (rows) x format variants (columns), green/red."""
    results = [r for r in data["results"] if r.get("error") is None]
    if not results:
        print("  No valid results to plot.")
        return

    variants = [r["variant"] for r in results]
    validators = _VALIDATOR_NAMES

    # Build matrix: 1.0 = passed, severity value if failed
    matrix = []
    for vname in validators:
        row = []
        for r in results:
            scores = r.get("soft_scores", {})
            score_entry = scores.get(vname, {})
            passed = score_entry.get("passed", True)
            severity = score_entry.get("severity", 0.0)
            row.append(0.0 if passed else max(severity, 0.01))
        matrix.append(row)

    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pass_fail", ["#2ecc71", "#f1c40f", "#e74c3c"],
    )
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(
        [v.replace("_", "\n") for v in variants],
        fontsize=8, ha="center",
    )
    ax.set_yticks(range(len(validators)))
    ax.set_yticklabels(
        [v.replace("_", " ").title() for v in validators], fontsize=9,
    )

    # Annotate cells
    for i in range(len(validators)):
        for j in range(len(variants)):
            val = arr[i, j]
            label = "PASS" if val == 0.0 else f"{val:.2f}"
            color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, label, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_title(
        f"Prompt Sensitivity: {data['instance']} ({data['strategy']})",
    )
    fig.colorbar(im, ax=ax, label="Severity (0 = pass)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt format sensitivity analysis",
    )
    parser.add_argument("--instance", default="Srivastava86")
    parser.add_argument(
        "--strategy",
        choices=["naive", "cot", "self_healing"],
        default="cot",
    )
    args = parser.parse_args()

    strategy = SolveStrategy(args.strategy)
    print(
        f"Prompt sensitivity: {args.instance}, {args.strategy}"
    )
    data = run_sensitivity_analysis(args.instance, strategy)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.instance}_{args.strategy}"
    json_path = RESULTS_DIR / f"prompt_sensitivity_{tag}.json"
    png_path = RESULTS_DIR / f"prompt_sensitivity_{tag}.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved: {json_path}")

    plot_sensitivity(data, png_path)


if __name__ == "__main__":
    main()
