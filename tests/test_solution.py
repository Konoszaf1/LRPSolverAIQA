"""Unit tests for lrp/models/solution.py."""

from __future__ import annotations

import pytest

from lrp.algorithms.nearest_neighbor import assign_depots, build_vehicle_routes
from lrp.models.node import CustomerNode, DepotNode
from lrp.models.solution import Solution


def _make_simple_instance() -> tuple[list[CustomerNode], list[DepotNode]]:
    """Create a minimal 2-customer, 1-depot instance."""
    customers = [CustomerNode(1, 0, 0, 50), CustomerNode(2, 1, 0, 30)]
    depots = [DepotNode(1, 0, 1, 200, 100.0, 1.0)]
    return customers, depots


class TestSolution:
    def test_deepcopies_nodes(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        # Modifying the solution's nodes shouldn't affect originals
        sol.customers[0].demand = 999
        assert customers[0].demand == 50

    def test_build_distances_populates_lists(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        sol.build_distances()

        for c in sol.customers:
            assert len(c.depot_distances) == 1  # 1 depot
            assert len(c.customer_distances) == 2  # self + other

    def test_calculate_total_distance_includes_fixed_cost(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        sol.build_distances()
        assign_depots(sol.customers)
        for d in sol.depots:
            build_vehicle_routes(d, 160)

        total = sol.calculate_total_distance()
        # Must include the depot's fixed cost (100.0) plus route distances
        assert total >= 100.0
        assert sol.total_distance == total

    def test_calculate_total_distance_resets_between_calls(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        sol.build_distances()
        assign_depots(sol.customers)
        for d in sol.depots:
            build_vehicle_routes(d, 160)

        first = sol.calculate_total_distance()
        second = sol.calculate_total_distance()
        assert first == pytest.approx(second)

    def test_remove_depot(self) -> None:
        depots = [
            DepotNode(1, 0, 0, 100, 50.0, 1.0),
            DepotNode(2, 5, 5, 100, 50.0, 1.0),
        ]
        sol = Solution([], depots)
        assert len(sol.depots) == 2
        sol.remove_depot(1)
        assert len(sol.depots) == 1
        assert sol.depots[0].depot_number == 2


class TestValidateFeasibility:
    def test_feasible_solution_returns_empty(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        sol.build_distances()
        assign_depots(sol.customers)
        for d in sol.depots:
            build_vehicle_routes(d, 160)
        sol.calculate_total_distance()

        violations = sol.validate_feasibility()
        assert violations == []

    def test_unserved_customer_detected(self) -> None:
        customers, depots = _make_simple_instance()
        sol = Solution(customers, depots)
        # No routes built → all customers unserved
        violations = sol.validate_feasibility()
        assert len(violations) == 2  # both customers unserved
        assert all("not served" in v for v in violations)
