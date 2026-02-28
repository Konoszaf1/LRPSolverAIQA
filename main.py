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
from itertools import combinations
from pathlib import Path

from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.algorithms.nearest_neighbor import assign_depots, build_vehicle_routes
from lrp.config import VEHICLE_CAPACITY, CuckooConfig
from lrp.io.data_loader import load_customers, load_depots
from lrp.models.node import CustomerNode, DepotNode
from lrp.models.solution import Solution
from lrp.visualization import plot_routes

_DEFAULT_CUSTOMERS = Path("DATALRP/DATALRP/Ch69Cli100x10")
_DEFAULT_DEPOTS = Path("DATALRP/DATALRP/Ch69Dep100x10")


def _depot_combinations(
    num_depots: int, target_count: int
) -> list[tuple[int, ...]]:
    """Return diverse depot-subset combinations for initial population seeding.

    Starts with the full set and progressively removes one depot at a time
    until enough combinations have been generated.

    Args:
        num_depots: Total number of available depots.
        target_count: Desired number of combinations.

    Returns:
        List of tuples, each containing the 1-based depot IDs to keep open.
    """
    all_ids = range(1, num_depots + 1)
    result: list[tuple[int, ...]] = []
    remove_count = 0
    while len(result) < target_count:
        subset_size = num_depots - remove_count
        if subset_size < 1:
            break
        result.extend(combinations(all_ids, subset_size))
        remove_count += 1
    return result[:target_count]


def _build_solution(
    customers: list[CustomerNode],
    depots: list[DepotNode],
    active_depot_ids: tuple[int, ...],
    vehicle_capacity: int,
) -> Solution:
    """Construct and evaluate one solution for a given depot subset.

    Args:
        customers: Full list of customer nodes.
        depots: Full list of depot nodes.
        active_depot_ids: 1-based depot IDs to keep in this solution.
        vehicle_capacity: Maximum load each vehicle can carry.

    Returns:
        An initialised Solution with distances computed, customers assigned,
        and vehicle routes built.
    """
    solution = Solution(customers, depots)
    solution.vehicle_capacity = vehicle_capacity
    solution.depots = [
        d for d in solution.depots if d.depot_number in active_depot_ids
    ]
    solution.build_distances()
    assign_depots(solution.customers)
    for depot in solution.depots:
        build_vehicle_routes(depot, vehicle_capacity)
    solution.calculate_total_distance()
    return solution


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
    combos = _depot_combinations(len(depots), args.solutions)
    solutions = [
        _build_solution(customers, depots, combo, args.vehicle_capacity)
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
