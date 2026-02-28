"""Optimality gap analysis: compare LLM solution quality against CS baseline.

Computes structural differences between an LLM-generated solution and the
Cuckoo Search ground truth: cost gap percentage, depot selection overlap
(Jaccard index), and route count ratio.  Zero API calls — works entirely
on existing ``LRPSolution`` objects or benchmark JSON files.

Usage::

    uv run python -m qa_suite.deterministic_checks.optimality_gap \\
        results/Srivastava86.json --strategy cot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qa_suite.common.schemas import LRPSolution


@dataclass
class OptimalityGapResult:
    """Structural comparison between an LLM solution and a CS baseline."""

    llm_total_cost: float
    cs_total_cost: float
    gap_percent: float
    depot_overlap: float  # Jaccard index of open depot sets
    llm_n_routes: int
    cs_n_routes: int
    route_count_ratio: float
    llm_open_depots: list[int]
    cs_open_depots: list[int]
    depot_only_llm: list[int]
    depot_only_cs: list[int]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "llm_total_cost": round(self.llm_total_cost, 2),
            "cs_total_cost": round(self.cs_total_cost, 2),
            "gap_percent": round(self.gap_percent, 2),
            "depot_overlap": round(self.depot_overlap, 4),
            "llm_n_routes": self.llm_n_routes,
            "cs_n_routes": self.cs_n_routes,
            "route_count_ratio": round(self.route_count_ratio, 2),
            "llm_open_depots": self.llm_open_depots,
            "cs_open_depots": self.cs_open_depots,
            "depot_only_llm": self.depot_only_llm,
            "depot_only_cs": self.depot_only_cs,
        }


def compute_optimality_gap(
    llm_solution: LRPSolution,
    cs_solution: LRPSolution,
) -> OptimalityGapResult:
    """Compare two solutions and compute structural gap metrics."""
    llm_set = set(llm_solution.open_depots)
    cs_set = set(cs_solution.open_depots)
    union = llm_set | cs_set
    overlap = len(llm_set & cs_set) / len(union) if union else 1.0

    cs_cost = cs_solution.total_cost
    gap = (
        (llm_solution.total_cost - cs_cost) / cs_cost * 100
        if cs_cost > 0
        else 0.0
    )

    cs_routes = len(cs_solution.routes)
    ratio = (
        len(llm_solution.routes) / cs_routes
        if cs_routes > 0
        else 1.0
    )

    return OptimalityGapResult(
        llm_total_cost=llm_solution.total_cost,
        cs_total_cost=cs_solution.total_cost,
        gap_percent=gap,
        depot_overlap=overlap,
        llm_n_routes=len(llm_solution.routes),
        cs_n_routes=cs_routes,
        route_count_ratio=ratio,
        llm_open_depots=sorted(llm_solution.open_depots),
        cs_open_depots=sorted(cs_solution.open_depots),
        depot_only_llm=sorted(llm_set - cs_set),
        depot_only_cs=sorted(cs_set - llm_set),
    )


def load_from_benchmark_json(
    json_path: str | Path,
    strategy: str = "cot",
) -> OptimalityGapResult | None:
    """Load a run_benchmark.py JSON file and compute the gap.

    Returns None if either solver was unavailable.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cs = data.get("cuckoo_search", {})
    if not cs.get("available"):
        return None

    # Support multi-tier and legacy formats
    tiers = data.get("llm_solvers", {})
    llm = tiers.get(strategy)
    if llm is None:
        llm = data.get("llm_solver", {})
    if not llm or not llm.get("available"):
        return None

    cs_cost = cs["total_cost"]
    llm_cost = llm["total_cost"]

    cs_depots = cs.get("open_depots", [])
    llm_depots = llm.get("open_depots", [])

    cs_routes = cs.get("n_routes", 0)
    llm_routes = llm.get("n_routes", 0)

    llm_set = set(llm_depots)
    cs_set = set(cs_depots)
    union = llm_set | cs_set
    overlap = len(llm_set & cs_set) / len(union) if union else 1.0

    gap = (
        (llm_cost - cs_cost) / cs_cost * 100
        if cs_cost > 0
        else 0.0
    )
    ratio = llm_routes / cs_routes if cs_routes > 0 else 1.0

    return OptimalityGapResult(
        llm_total_cost=llm_cost,
        cs_total_cost=cs_cost,
        gap_percent=gap,
        depot_overlap=overlap,
        llm_n_routes=llm_routes,
        cs_n_routes=cs_routes,
        route_count_ratio=ratio,
        llm_open_depots=sorted(llm_depots),
        cs_open_depots=sorted(cs_depots),
        depot_only_llm=sorted(llm_set - cs_set),
        depot_only_cs=sorted(cs_set - llm_set),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimality gap analysis (LLM vs CS)"
    )
    parser.add_argument(
        "input_json",
        help="Path to benchmark JSON from run_benchmark.py",
    )
    parser.add_argument(
        "--strategy",
        choices=["naive", "cot", "self_healing"],
        default="cot",
    )
    args = parser.parse_args()

    result = load_from_benchmark_json(args.input_json, args.strategy)
    if result is None:
        print("  Could not compute gap (solver data missing).")
        return

    print(f"  CS cost       : {result.cs_total_cost:.2f}")
    print(f"  LLM cost      : {result.llm_total_cost:.2f}")
    print(f"  Gap           : {result.gap_percent:+.1f}%")
    print(f"  Depot overlap : {result.depot_overlap:.0%} (Jaccard)")
    print(f"  Routes        : {result.llm_n_routes} LLM vs "
          f"{result.cs_n_routes} CS "
          f"({result.route_count_ratio:.1f}x)")
    if result.depot_only_llm:
        print(f"  LLM-only depots: {result.depot_only_llm}")
    if result.depot_only_cs:
        print(f"  CS-only depots : {result.depot_only_cs}")


if __name__ == "__main__":
    main()
