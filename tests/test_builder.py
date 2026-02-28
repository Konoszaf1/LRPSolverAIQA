"""Unit tests for lrp/builder.py."""

from __future__ import annotations

import pytest

from lrp.builder import build_solution, depot_combinations
from lrp.models.node import CustomerNode, DepotNode


class TestDepotCombinations:
    def test_returns_full_set_first(self) -> None:
        combos = depot_combinations(3, 5)
        assert combos[0] == (1, 2, 3)

    def test_respects_target_count(self) -> None:
        combos = depot_combinations(10, 3)
        assert len(combos) == 3

    def test_returns_fewer_if_exhausted(self) -> None:
        # 2 depots: (1,2), (1,), (2,) = 3 total, requesting 10
        combos = depot_combinations(2, 10)
        assert len(combos) == 3

    def test_single_depot(self) -> None:
        combos = depot_combinations(1, 5)
        assert combos == [(1,)]


class TestBuildSolution:
    def test_produces_feasible_solution(self) -> None:
        customers = [
            CustomerNode(1, 0, 0, 30),
            CustomerNode(2, 5, 0, 40),
            CustomerNode(3, 10, 0, 20),
        ]
        depots = [DepotNode(1, 2, 2, 200, 100.0, 1.0)]

        sol = build_solution(customers, depots, (1,), vehicle_capacity=160)
        assert sol.total_distance > 0
        violations = sol.validate_feasibility()
        assert violations == []

    def test_does_not_mutate_originals(self) -> None:
        customers = [CustomerNode(1, 0, 0, 30)]
        depots = [DepotNode(1, 2, 2, 200, 100.0, 1.0)]

        build_solution(customers, depots, (1,), vehicle_capacity=160)
        # Original nodes must be untouched
        assert customers[0].assigned_depot is None
        assert depots[0].opened is False

    def test_filters_depots_by_active_ids(self) -> None:
        customers = [CustomerNode(1, 0, 0, 30)]
        depots = [
            DepotNode(1, 0, 0, 200, 100.0, 1.0),
            DepotNode(2, 5, 5, 200, 200.0, 1.0),
        ]

        sol = build_solution(customers, depots, (1,), vehicle_capacity=160)
        assert len(sol.depots) == 1
        assert sol.depots[0].depot_number == 1

    def test_insufficient_capacity_raises(self) -> None:
        customers = [CustomerNode(1, 0, 0, 300)]  # demand > depot capacity
        depots = [DepotNode(1, 0, 0, 100, 50.0, 1.0)]

        with pytest.raises(ValueError, match="cannot be assigned"):
            build_solution(customers, depots, (1,), vehicle_capacity=160)
