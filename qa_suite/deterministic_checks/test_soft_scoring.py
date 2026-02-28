"""Unit tests for the soft-scoring layer — no API key needed."""

from __future__ import annotations

import math

from qa_suite.deterministic_checks.soft_scoring import (
    soft_customer_coverage,
    soft_depot_capacity,
    soft_route_distances,
    soft_total_cost,
    soft_vehicle_capacity,
)

# ---------------------------------------------------------------------------
# Vehicle capacity
# ---------------------------------------------------------------------------

class TestSoftVehicleCapacity:
    def test_exact_capacity(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2]}]
        customers = {1: {"x": 0, "y": 0, "demand": 80}, 2: {"x": 1, "y": 1, "demand": 80}}
        result = soft_vehicle_capacity(routes, customers, 160.0)
        assert result.passed is True
        assert result.severity == 0.0

    def test_exceeded_15_pct(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2]}]
        customers = {1: {"x": 0, "y": 0, "demand": 92}, 2: {"x": 1, "y": 1, "demand": 92}}
        # load = 184, capacity = 160, excess = 24, frac = 24/160 = 0.15
        result = soft_vehicle_capacity(routes, customers, 160.0)
        assert result.passed is False
        assert result.severity == 0.15

    def test_empty_routes(self) -> None:
        result = soft_vehicle_capacity([], {}, 100.0)
        assert result.passed is True
        assert result.severity == 0.0

    def test_multiple_routes_worst_reported(self) -> None:
        routes = [
            {"depot_id": 1, "customer_ids": [1]},
            {"depot_id": 1, "customer_ids": [2]},
        ]
        customers = {
            1: {"x": 0, "y": 0, "demand": 100},  # within 160
            2: {"x": 1, "y": 1, "demand": 200},  # 40 over → 40/160 = 0.25
        }
        result = soft_vehicle_capacity(routes, customers, 160.0)
        assert result.passed is False
        assert result.severity == 0.25
        assert result.score == 0.5  # 1 of 2 valid


# ---------------------------------------------------------------------------
# Customer coverage
# ---------------------------------------------------------------------------

class TestSoftCustomerCoverage:
    def test_all_covered(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2, 3]}]
        customers: dict[int, dict] = {1: {}, 2: {}, 3: {}}
        result = soft_customer_coverage(routes, customers)
        assert result.passed is True
        assert result.severity == 0.0

    def test_two_of_ten_missing(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": list(range(1, 9))}]
        customers: dict[int, dict] = {i: {} for i in range(1, 11)}
        result = soft_customer_coverage(routes, customers)
        assert result.passed is False
        assert result.severity == 0.2  # 2 missing / 10 total
        assert result.detail["n_missing"] == 2

    def test_phantom_customer(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2, 99]}]
        customers: dict[int, dict] = {1: {}, 2: {}}
        result = soft_customer_coverage(routes, customers)
        assert result.passed is False
        assert result.detail["n_phantom"] == 1


# ---------------------------------------------------------------------------
# Depot capacity
# ---------------------------------------------------------------------------

class TestSoftDepotCapacity:
    def test_within_capacity(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2]}]
        customers = {1: {"demand": 50}, 2: {"demand": 50}}
        depots = {1: {"capacity": 200}}
        result = soft_depot_capacity(routes, customers, depots)
        assert result.passed is True
        assert result.severity == 0.0

    def test_exceeded_20_pct(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1, 2]}]
        customers = {1: {"demand": 60}, 2: {"demand": 60}}
        depots = {1: {"capacity": 100}}
        # load = 120, cap = 100, excess = 20, frac = 20/100 = 0.2
        result = soft_depot_capacity(routes, customers, depots)
        assert result.passed is False
        assert result.severity == 0.2


# ---------------------------------------------------------------------------
# Route distances
# ---------------------------------------------------------------------------

class TestSoftRouteDistances:
    def test_exact_distance(self) -> None:
        # depot at (0,0), customer at (3,4), round trip = 5 + 5 = 10
        actual_dist = 2 * math.dist((0, 0), (3, 4))
        routes = [{"depot_id": 1, "customer_ids": [1], "stated_distance": actual_dist}]
        customers = {1: {"x": 3, "y": 4}}
        depots = {1: {"x": 0, "y": 0}}
        result = soft_route_distances(routes, customers, depots)
        assert result.passed is True
        assert result.severity == 0.0

    def test_25_pct_off(self) -> None:
        actual_dist = 2 * math.dist((0, 0), (3, 4))  # 10.0
        stated = actual_dist * 1.25  # 12.5 → 25% off
        routes = [{"depot_id": 1, "customer_ids": [1], "stated_distance": stated}]
        customers = {1: {"x": 3, "y": 4}}
        depots = {1: {"x": 0, "y": 0}}
        result = soft_route_distances(routes, customers, depots)
        assert result.passed is False
        assert abs(result.severity - 0.25) < 0.01

    def test_no_stated_distance(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1]}]
        customers = {1: {"x": 3, "y": 4}}
        depots = {1: {"x": 0, "y": 0}}
        result = soft_route_distances(routes, customers, depots)
        assert result.passed is True
        assert result.severity == 0.0


# ---------------------------------------------------------------------------
# Total cost
# ---------------------------------------------------------------------------

class TestSoftTotalCost:
    def test_exact_cost(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1], "stated_distance": 50.0}]
        depots = {1: {"fixed_cost": 100.0}}
        result = soft_total_cost(routes, depots, [1], 150.0)
        assert result.passed is True
        assert result.severity == 0.0

    def test_30_pct_off(self) -> None:
        routes = [{"depot_id": 1, "customer_ids": [1], "stated_distance": 50.0}]
        depots = {1: {"fixed_cost": 100.0}}
        # recomputed = 150, stated = 195 → |195-150|/150 = 0.3
        result = soft_total_cost(routes, depots, [1], 195.0)
        assert result.passed is False
        assert result.severity == 0.3
