"""Solution container for the Location-Routing Problem."""

from __future__ import annotations

import copy

from lrp.config import VEHICLE_CAPACITY
from lrp.models.distance import Distance
from lrp.models.node import CustomerNode, DepotNode


class Solution:
    """A complete candidate solution to the LRP.

    Holds independent deep copies of customer and depot nodes so that each
    member of the solution population can be modified without affecting others.

    Attributes:
        customers: Customer nodes in this solution.
        depots: Depot nodes available in this solution.
        total_distance: Combined objective value (opened depot fixed costs
            plus the sum of all vehicle route distances).
    """

    def __init__(
        self,
        customers: list[CustomerNode],
        depots: list[DepotNode],
    ) -> None:
        """Initialise a solution from deep copies of the provided nodes.

        Args:
            customers: Customer nodes to copy into this solution.
            depots: Depot nodes to copy into this solution.
        """
        self.customers: list[CustomerNode] = copy.deepcopy(customers)
        self.depots: list[DepotNode] = copy.deepcopy(depots)
        self.total_distance: float = 0.0
        self.vehicle_capacity: float = float(VEHICLE_CAPACITY)

    def remove_depot(self, depot_number: int) -> None:
        """Remove a depot from this solution by its identifier.

        Args:
            depot_number: The depot to remove.
        """
        self.depots = [d for d in self.depots if d.depot_number != depot_number]

    def build_distances(self) -> None:
        """Compute and store all pairwise distances between nodes.

        Populates ``depot_distances`` and ``customer_distances`` on every
        customer in this solution. Must be called before any routing or
        assignment heuristic is applied.
        """
        for customer in self.customers:
            customer.depot_distances = [
                Distance(customer, depot) for depot in self.depots
            ]
        for customer in self.customers:
            customer.customer_distances = [
                Distance(customer, other) for other in self.customers
            ]

    def calculate_total_distance(self) -> float:
        """Compute the total objective value and store it in ``total_distance``.

        The objective is the sum of fixed costs for all *opened* depots plus
        the distances of all vehicle routes. Resets ``total_distance`` to zero
        before accumulating.

        Returns:
            The computed total distance.
        """
        self.total_distance = 0.0
        for depot in self.depots:
            if depot.opened:
                self.total_distance += depot.fixed_cost
            for vehicle in depot.vehicles:
                self.total_distance += vehicle.vehicle_distance
        return self.total_distance

    def validate_feasibility(self) -> list[str]:
        """Check solution feasibility and return a list of violation messages.

        Checks:

        - Every customer in ``self.customers`` is served by exactly one
          vehicle route across all depots.
        - No vehicle route's total demand exceeds the vehicle capacity
          (read from ``self.vehicle_capacity`` when set, otherwise the
          global ``VEHICLE_CAPACITY`` constant).
        - No depot's total routed demand exceeds its ``original_capacity``
          (when available).

        Returns:
            Empty list when the solution is feasible; otherwise a list of
            human-readable violation strings, one per detected problem.
        """
        vc = self.vehicle_capacity
        violations: list[str] = []
        served: dict[int, int] = {}  # customer_number -> serve count

        for depot in self.depots:
            depot_load = 0
            for vehicle in depot.vehicles:
                vehicle_load = sum(c.demand for c in vehicle.customers)
                if vehicle_load > vc:
                    violations.append(
                        f"Vehicle {vehicle.vehicle_number} at depot "
                        f"{depot.depot_number} overloaded: "
                        f"{vehicle_load} > {vc}"
                    )
                for c in vehicle.customers:
                    served[c.customer_number] = served.get(c.customer_number, 0) + 1
                depot_load += vehicle_load

            if hasattr(depot, "original_capacity"):
                if depot_load > depot.original_capacity:
                    violations.append(
                        f"Depot {depot.depot_number} overloaded: "
                        f"{depot_load} > {depot.original_capacity}"
                    )

        for customer in self.customers:
            count = served.get(customer.customer_number, 0)
            if count == 0:
                violations.append(
                    f"Customer {customer.customer_number} is not served by any vehicle."
                )
            elif count > 1:
                violations.append(
                    f"Customer {customer.customer_number} is served "
                    f"{count} times (expected 1)."
                )

        return violations
