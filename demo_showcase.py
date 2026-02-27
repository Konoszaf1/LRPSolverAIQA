"""AIQA Showcase: Multi-Tier LLM Benchmarking vs Cuckoo Search.

Runs FOUR solvers on the same LRP instance and prints a side-by-side
constraint-validation table:

* **Cuckoo Search** — classical metaheuristic, always satisfies constraints.
* **Naive LLM** — zero-shot baseline (Tier 1).
* **CoT + Heuristic LLM** — Chain-of-Thought with routing guidance (Tier 2).
* **Self-Healing LLM** — CoT + iterative validator-feedback repair loop (Tier 3).

The comparison demonstrates that prompt engineering and agentic QA loops
progressively reduce constraint violations in LLM-generated solutions.

Usage::

    # Random instances (1 small, 1 large) — different every run:
    uv run python demo_showcase.py --api-key sk-ant-...

    # Specific instances:
    uv run python demo_showcase.py --api-key sk-ant-... --instances Gaskell67 Min92

    # Via environment variable:
    $env:ANTHROPIC_API_KEY="sk-ant-..."; uv run python demo_showcase.py

Available instances: Srivastava86 (8), Gaskell67 (21), Perl83 (55),
                     Ch69 (100), Or76 (117), Min92 (134), Daskin95 (150)
"""

from __future__ import annotations

import argparse
import os
import random
import time
from itertools import combinations

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ai_agent.solver import LLMSolver, SolveStrategy
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
    validate_total_cost,
    validate_vehicle_capacity,
)

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LLM_MODEL = "claude-haiku-4-5-20251001"  # Fastest Claude — low latency, lower cost

# Instance pools. Small: LLM usually passes. Large: LLM usually fails.
_SMALL_POOL = ["Srivastava86", "Gaskell67"]
_LARGE_POOL = ["Ch69", "Or76", "Min92"]

# Keep CS fast — this is a demo, not a full benchmark.
_CS_SOLUTIONS = 2
_CS_ITERATIONS = 8

# Labels for the comparison table rows (matches validator order below).
_LABELS = [
    "Vehicle Capacity",
    "Customer Coverage",
    "Depot Capacity",
    "Route Distances",
    "Total Cost",
    "ID Grounding",   # faithfulness — LLM only
]

# The three LLM tiers in order.
_TIERS: list[tuple[str, SolveStrategy]] = [
    ("Naive", SolveStrategy.NAIVE),
    ("CoT", SolveStrategy.COT),
    ("Self-Heal", SolveStrategy.SELF_HEALING),
]


# ---------------------------------------------------------------------------
# Cuckoo Search runner
# ---------------------------------------------------------------------------

def _run_cuckoo(
    instance_name: str,
) -> tuple[list[dict], list[int], float, float]:
    """Run Cuckoo Search; return (routes, open_depots, total_cost, elapsed_s)."""
    cli_file, dep_file, vc = INSTANCES[instance_name]
    customers_lrp = lrp_load_customers(DATA_DIR / cli_file)
    depots_lrp = lrp_load_depots(DATA_DIR / dep_file)

    all_ids = tuple(range(1, len(depots_lrp) + 1))
    combos = list(combinations(all_ids, len(depots_lrp)))[:_CS_SOLUTIONS]

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

    config = CuckooConfig(num_solutions=_CS_SOLUTIONS, num_iterations=_CS_ITERATIONS)
    best = CuckooSearch(config).optimize(solutions)
    elapsed = time.time() - t0

    schema_sol = cuckoo_solution_to_schema(best)
    routes = [r.model_dump() for r in schema_sol.routes]
    return routes, schema_sol.open_depots, schema_sol.total_cost, elapsed


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_all(
    routes: list[dict],
    customers: dict,
    depots: dict,
    vc: float,
    open_depots: list[int],
    total_cost: float,
    faith_result: dict | None,
) -> list[ValidationResult]:
    """Run all 5 deterministic validators + faithfulness; return list of 6 results."""
    faith_pass = faith_result is not None and faith_result["score"] >= 1.0
    faith_violations: list[str] = []
    if faith_result and not faith_pass:
        pc = faith_result["phantom_customers"]
        pd = faith_result["phantom_depots"]
        if pc:
            faith_violations.append(f"Phantom customer IDs: {pc}")
        if pd:
            faith_violations.append(f"Phantom depot IDs: {pd}")

    return [
        validate_vehicle_capacity(routes, customers, vc),
        validate_customer_coverage(routes, customers),
        validate_depot_capacity(routes, customers, depots),
        validate_route_distances(routes, customers, depots),
        validate_total_cost(routes, depots, open_depots, total_cost),
        ValidationResult(
            passed=faith_pass,
            violations=faith_violations,
            score=faith_result["score"] if faith_result else 0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------

def _badge(passed: bool) -> Text:
    return Text(" ✓ ", style="bold green") if passed else Text(" ✗ ", style="bold red")


def _score_bar(score: float, width: int = 8) -> Text:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if score >= 1.0 else ("yellow" if score >= 0.5 else "red")
    t = Text(f"{bar} {score:.0%}")
    t.stylize(color)
    return t


# ---------------------------------------------------------------------------
# Data structures for solver results
# ---------------------------------------------------------------------------

class _SolverResult:
    """Holds the output + validation of a single solver run."""

    def __init__(self, name: str, n_checks: int = 6) -> None:
        self.name = name
        self.n_checks = n_checks
        self.routes: list[dict] | None = None
        self.open_depots: list[int] = []
        self.cost: float = 0.0
        self.elapsed: float = 0.0
        self.results: list[ValidationResult] | None = None
        self.error: str | None = None
        self.meta: dict = {}
        self.pass_count: int = 0

    def count_passes(self) -> int:
        if self.results is None:
            return 0
        self.pass_count = sum(1 for r in self.results if r.passed)
        return self.pass_count


# ---------------------------------------------------------------------------
# Per-instance comparison
# ---------------------------------------------------------------------------

def run_comparison(solvers: dict[str, LLMSolver], instance_name: str) -> dict:
    """Run CS + all three LLM tiers; print side-by-side validation table."""
    dataset = load_instance(instance_name)
    n_cust = len(dataset["customers"])
    n_dep = len(dataset["depots"])
    vc = dataset["vehicle_capacity"]
    customers = dataset["customers"]
    depots = dataset["depots"]

    console.print()
    console.print(Panel(
        f"[bold]{instance_name}[/]   "
        f"[dim]{n_cust} customers · {n_dep} depots · vehicle capacity {vc}[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    # ---- Cuckoo Search ----
    cs = _SolverResult("Cuckoo Search", n_checks=5)
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
        console=console, transient=True,
    ) as prog:
        prog.add_task("[cyan]Cuckoo Search solving...", total=None)
        try:
            cs.routes, cs.open_depots, cs.cost, cs.elapsed = _run_cuckoo(instance_name)
            cs.results = _validate_all(
                cs.routes, customers, depots, vc, cs.open_depots, cs.cost, None
            )
        except Exception as exc:
            cs.error = str(exc)[:120]
    cs.count_passes()

    # ---- LLM tiers (sequential to avoid rate-limiting) ----
    tier_results: list[_SolverResult] = []
    for tier_label, strategy in _TIERS:
        sr = _SolverResult(tier_label)
        solver = solvers[strategy.value]
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
            console=console, transient=True,
        ) as prog:
            prog.add_task(f"[cyan]{tier_label} LLM ({solver.model}) solving...", total=None)
            try:
                solution, meta = solver.solve(dataset)
                sr.routes = [r.model_dump() for r in solution.routes]
                sr.cost = solution.total_cost
                sr.elapsed = meta["elapsed_seconds"]
                sr.open_depots = solution.open_depots
                sr.meta = meta
                faith = manual_faithfulness_check(dataset, solution)
                sr.results = _validate_all(
                    sr.routes, customers, depots, vc, sr.open_depots, sr.cost, faith
                )
            except Exception as exc:
                sr.error = str(exc)[:300]
        sr.count_passes()
        tier_results.append(sr)

    # ---- Side-by-side table ----
    tbl = Table(
        title=f"[bold cyan]Constraint Validation — {instance_name}[/]",
        expand=True,
        show_lines=True,
        header_style="bold",
    )
    tbl.add_column("Check", style="bold", width=18)
    tbl.add_column("CS", justify="center", width=12)
    for tier_label, _ in _TIERS:
        tbl.add_column(tier_label, justify="center", width=14)

    for i, label in enumerate(_LABELS):
        is_faith = i == 5
        row: list[Text | str] = [label]

        # --- CS cell ---
        if is_faith:
            row.append(Text("deterministic", style="dim"))
        elif cs.error:
            row.append(Text("ERR", style="bold red"))
        else:
            r = cs.results[i]  # type: ignore[index]
            row.append(_score_bar(r.score) if r.passed else Text(f"✗ {r.score:.0%}", style="red"))

        # --- LLM tier cells ---
        for sr in tier_results:
            if sr.error:
                row.append(Text("ERR", style="bold red"))
            else:
                r = sr.results[i]  # type: ignore[index]
                if r.passed:
                    row.append(Text(f"✓ {r.score:.0%}", style="bold green"))
                else:
                    row.append(Text(f"✗ {r.score:.0%}", style="bold red"))

        tbl.add_row(*row)

    # Summary row
    tbl.add_section()
    summary_row: list[Text | str] = ["[bold]Result[/]"]

    # CS summary
    if cs.error:
        summary_row.append(Text(f"ERR\n{cs.error[:60]}", style="bold red"))
    else:
        cs_color = "green" if cs.pass_count == cs.n_checks else "yellow"
        summary_row.append(Text(
            f"{cs.pass_count}/{cs.n_checks}\n{cs.cost:.0f}  {cs.elapsed:.1f}s",
            style=f"bold {cs_color}",
        ))

    # LLM tier summaries
    for sr in tier_results:
        if sr.error:
            # Show a short error excerpt so failures are debuggable
            summary_row.append(Text(f"ERR\n{sr.error[:80]}", style="bold red"))
        else:
            color = "green" if sr.pass_count == sr.n_checks else "red"
            heal_info = ""
            if sr.meta.get("heal_attempts") is not None and sr.meta["heal_attempts"] > 0:
                heal_info = f"\n{sr.meta['heal_attempts']} heal"
            tok = f"{sr.meta.get('input_tokens', '?')}/{sr.meta.get('output_tokens', '?')}tok"
            line = (
                f"{sr.pass_count}/{sr.n_checks}\n"
                f"{sr.cost:.0f}  {sr.elapsed:.1f}s{heal_info}\n{tok}"
            )
            summary_row.append(Text(line, style=f"bold {color}"))

    tbl.add_row(*summary_row)
    console.print(tbl)

    # Collect all violations from all tiers for the narrative
    all_violations: dict[str, list[str]] = {}
    for sr in tier_results:
        if sr.results:
            viols = [v for r in sr.results if not r.passed for v in r.violations]
            if viols:
                all_violations[sr.name] = viols

    return {
        "instance": instance_name,
        "n_customers": n_cust,
        "cs_pass": cs.pass_count,
        "cs_total": cs.n_checks,
        "cs_cost": cs.cost,
        "tiers": {
            sr.name: {
                "pass": sr.pass_count,
                "total": sr.n_checks,
                "cost": sr.cost,
                "elapsed": sr.elapsed,
                "error": sr.error,
                "heal_attempts": sr.meta.get("heal_attempts"),
            }
            for sr in tier_results
        },
        "violations": all_violations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIQA Showcase: Multi-Tier LLM Benchmarking"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        help=(
            "Instance names to run (default: 1 random small + 1 random large). "
            "Choices: Srivastava86, Gaskell67, Perl83, Ch69, Or76, Min92, Daskin95"
        ),
    )
    args = parser.parse_args()

    if args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key

    # Select instances
    if args.instances:
        instances = args.instances
    else:
        small = random.choice(_SMALL_POOL)
        large = random.choice(_LARGE_POOL)
        instances = [small, large]

    # Hero banner
    console.print(Panel(
        "[bold cyan]"
        "    _    ___ ___    _      ____  _\n"
        r"   / \  |_ _/ _ \  / \    / ___|| |__   _____      _____  __ _ ___  ___"
        "\n"
        r"  / _ \  | | | | |/ _ \   \___ \| '_ \ / _ \ \ /\ / / __|/ _` / __|/ _ \ "
        "\n"
        r" / ___ \ | | |_| / ___ \   ___) | | | | (_) \ V  V / (__| (_| \__ \  __/"
        "\n"
        r"/_/   \_\___\__\_\_/   \_\ |____/|_| |_|\___/ \_/\_/ \___|\__,_|___/\___|"
        "[/]",
        title="[bold]Multi-Tier LLM Benchmarking — AIQA Validation Demo[/]",
        subtitle="[dim]Classical optimization vs 3 LLM strategies · Real-time QA[/]",
        border_style="cyan",
    ))

    console.print(f"\n  [dim]Instances :[/] [bold]{', '.join(instances)}[/]")
    console.print(f"  [dim]LLM model :[/] [bold]{_LLM_MODEL}[/]  (fast Haiku)")
    console.print(
        "  [dim]LLM tiers :[/] Naive (zero-shot) → CoT (heuristic) → Self-Healing (validator loop)"
    )
    console.print(
        f"  [dim]CS config :[/] {_CS_SOLUTIONS} solutions × {_CS_ITERATIONS} iterations  "
        f"(demo speed)\n"
    )

    # Create one solver per strategy (all use the same model).
    solvers = {
        strat.value: LLMSolver(model=_LLM_MODEL, strategy=strat)
        for _, strat in _TIERS
    }

    results = []
    for name in instances:
        result = run_comparison(solvers, name)
        results.append(result)

    # ---- Narrative ----
    console.print()
    console.rule("[bold yellow]Why This Matters")

    # Find the instance where tier progression is most visible.
    best_example = None
    for r in results:
        tiers = r.get("tiers", {})
        naive_p = tiers.get("Naive", {}).get("pass", 99)
        heal_p = tiers.get("Self-Heal", {}).get("pass", 0)
        if heal_p > naive_p:
            best_example = r
            break

    if best_example:
        tiers = best_example["tiers"]
        cs_p = f"{best_example['cs_pass']}/{best_example['cs_total']}"
        n_p = f"{tiers['Naive']['pass']}/{tiers['Naive']['total']}"
        c_p = f"{tiers['CoT']['pass']}/{tiers['CoT']['total']}"
        h_p = f"{tiers['Self-Heal']['pass']}/{tiers['Self-Heal']['total']}"
        inst = best_example["instance"]
        n_c = best_example["n_customers"]
        body = (
            f"On [bold]{inst}[/] ({n_c} customers):\n\n"
            f"  [green bold]Cuckoo Search[/]   → {cs_p} satisfied\n"
            f"  [red bold]Naive LLM[/]       → {n_p} — zero-shot\n"
            f"  [yellow bold]CoT LLM[/]         → {c_p} — heuristic\n"
            f"  [green bold]Self-Healing[/]    → {h_p} — feedback\n\n"
            "[bold]Key insight:[/] Each tier progressively "
            "reduces violations.\n"
            "The Self-Healing agent uses our AIQA validators "
            "as a feedback signal,\nproving that "
            "[bold]QA-driven agentic workflows[/] compensate "
            "for LLM weaknesses.\n\n"
            "[dim]Stack: 5 validators · faithfulness · "
            "metamorphic · DeepEval · Phoenix[/]"
        )
    elif any(r.get("violations") for r in results):
        # LLM failed somewhere — show generic narrative.
        failed = next(r for r in results if r.get("violations"))
        viols = list(failed["violations"].values())
        sample = viols[0][:3] if viols else []
        body = (
            f"On [bold]{failed['instance']}[/], the LLM produced plausible JSON "
            f"that [bold red]silently violates hard constraints[/].\n\n"
            "[bold]AIQA caught what manual review would miss:[/]\n"
        )
        for v in sample:
            body += f"  [dim red]✗[/] [dim]{v[:110]}[/]\n"
        body += (
            "\nAutomated QA at every layer — schema validation, deterministic checks, "
            "faithfulness, metamorphic tests —\nis the only way to deploy LLMs safely "
            "for constrained mathematical optimization.\n\n"
            "[dim]Stack: 5 validators · faithfulness · metamorphic tests · "
            "DeepEval metrics · Arize Phoenix[/]"
        )
    else:
        body = (
            "[green]All LLM tiers passed all checks on this run.[/]\n\n"
            "LLM failure rates are stochastic. Re-run or force larger instances:\n"
            "  [dim]uv run python demo_showcase.py --instances Or76 Min92[/]"
        )

    console.print(Panel(body, border_style="yellow", padding=(1, 2)))


if __name__ == "__main__":
    main()
