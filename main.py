"""Entry point for the Location-Routing Problem solver.

Reads benchmark data, constructs an initial solution population via a
nearest-neighbour heuristic, then optimises with Cuckoo Search.

Usage::

    uv run python main.py
    uv run python main.py --customers DATALRP/DATALRP/Ch69Cli100x10 \\
                          --depots   DATALRP/DATALRP/Ch69Dep100x10 \\
                          --solutions 20 --iterations 100 --no-plot
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.builder import build_solution, depot_combinations
from lrp.config import VEHICLE_CAPACITY, CuckooConfig
from lrp.io.data_loader import load_customers, load_depots
from lrp.models.solution import Solution
from lrp.visualization import plot_routes

_DEFAULT_CUSTOMERS = Path("DATALRP/DATALRP/Ch69Cli100x10")
_DEFAULT_DEPOTS = Path("DATALRP/DATALRP/Ch69Dep100x10")


def print_solution(solution: Solution) -> None:
    """Print a human-readable summary of a solution to stdout.

    Args:
        solution: The solution to display.
    """
    separator = "*" * 50
    print(separator)
    print(f"Total distance / cost: {solution.total_distance:.4f}")
    active = [d for d in solution.depots if d.opened]
    print(f"Active depots: {[d.depot_number for d in active]}")
    for depot in solution.depots:
        if not depot.vehicles:
            continue
        print(f"\nDepot {depot.depot_number}:")
        for vehicle in depot.vehicles:
            print(f"  Vehicle {vehicle.vehicle_number}: {vehicle.vector}")
            print(f"    Distance: {vehicle.vehicle_distance:.4f}")
    print(separator)


def main() -> None:
    """Parse CLI arguments and run the LRP solver."""
    parser = argparse.ArgumentParser(
        description="LRP solver — Cuckoo Search metaheuristic"
    )
    parser.add_argument(
        "--customers",
        type=Path,
        default=_DEFAULT_CUSTOMERS,
        help="Path to customer data file.",
    )
    parser.add_argument(
        "--depots",
        type=Path,
        default=_DEFAULT_DEPOTS,
        help="Path to depot data file.",
    )
    parser.add_argument(
        "--solutions",
        type=int,
        default=10,
        help="Number of initial solutions in the population (default: 10).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Cuckoo Search iterations per solution (default: 100).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering the route visualisation.",
    )
    parser.add_argument(
        "--vehicle-capacity",
        type=int,
        default=VEHICLE_CAPACITY,
        help=f"Maximum load per vehicle (default: {VEHICLE_CAPACITY}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    print(f"Loading customers from: {args.customers}")
    print(f"Loading depots from:    {args.depots}")
    customers = load_customers(args.customers)
    depots = load_depots(args.depots)
    print(f"  {len(customers)} customers, {len(depots)} depots loaded.")

    print(f"\nBuilding {args.solutions} initial solutions...")
    combos = depot_combinations(len(depots), args.solutions)
    solutions = [
        build_solution(customers, depots, combo, args.vehicle_capacity)
        for combo in combos
    ]

    config = CuckooConfig(num_iterations=args.iterations)
    searcher = CuckooSearch(config)

    print(
        f"Running Cuckoo Search "
        f"({args.iterations} iterations, population={len(solutions)})..."
    )
    best = searcher.optimize(solutions)

    violations = best.validate_feasibility()
    if violations:
        print("\nWARNING — Solution has feasibility violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("  Feasibility check passed.")

    print("\n--- Best Solution Found ---")
    print_solution(best)

    if args.seed is not None:
        print(f"(Seed used: {args.seed})")

    if not args.no_plot:
        plot_routes(best.depots, best.customers)


if __name__ == "__main__":
    main()
