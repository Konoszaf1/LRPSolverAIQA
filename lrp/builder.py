"""Solution construction helpers for the LRP solver.

Provides reusable factory functions for building initial solution populations.
Used by ``main.py``, the QA runner, and test suites to avoid duplicating the
solution-building sequence.
"""

from __future__ import annotations

from itertools import combinations

from lrp.algorithms.nearest_neighbor import assign_depots, build_vehicle_routes
from lrp.models.node import CustomerNode, DepotNode
from lrp.models.solution import Solution


def build_solution(
    customers: list[CustomerNode],
    depots: list[DepotNode],
    active_depot_ids: tuple[int, ...],
    vehicle_capacity: float,
) -> Solution:
    """Construct and evaluate one solution for a given depot subset.

    Creates a deep-copied Solution, filters to the specified depots,
    builds distances, assigns customers via nearest-neighbour, constructs
    vehicle routes, and computes the total objective.

    Args:
        customers: Full list of customer nodes (will be deep-copied).
        depots: Full list of depot nodes (will be deep-copied).
        active_depot_ids: 1-based depot IDs to keep in this solution.
        vehicle_capacity: Maximum load each vehicle can carry.

    Returns:
        An initialised Solution ready for optimisation.
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


def depot_combinations(
    num_depots: int, target_count: int,
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
