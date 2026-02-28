"""Unit tests for lrp/algorithms/two_opt.py."""

from __future__ import annotations

from lrp.algorithms.two_opt import two_opt_route
from lrp.models.distance import Distance
from lrp.models.node import CustomerNode, DepotNode
from lrp.models.vehicle_route import VehicleRoute


def _make_route_with_distances(
    depot: DepotNode,
    customers: list[CustomerNode],
) -> VehicleRoute:
    """Build a VehicleRoute with all required distance lookups populated."""
    # Build all depot distances for each customer
    for c in customers:
        c.depot_distances = [Distance(c, depot)]
    # Build all customer-customer distances
    for c in customers:
        c.customer_distances = [Distance(c, other) for other in customers]

    vehicle = VehicleRoute(0, depot, vehicle_capacity=9999)
    vehicle.customers = list(customers)
    vehicle.create_vector()
    vehicle.calculate_vehicle_distance()
    return vehicle


def test_two_opt_does_not_change_short_route() -> None:
    """A 2-customer route cannot be improved by 2-opt (needs >=3)."""
    depot = DepotNode(1, 0, 0, 1000, 100.0, 1.0)
    c1 = CustomerNode(1, 1, 0, 10)
    c2 = CustomerNode(2, 2, 0, 10)
    vehicle = _make_route_with_distances(depot, [c1, c2])

    improved = two_opt_route(vehicle)
    assert improved is False


def test_two_opt_improves_suboptimal_route() -> None:
    """A route with crossing edges should be improved by 2-opt."""
    depot = DepotNode(1, 0, 0, 1000, 100.0, 1.0)
    # Create a crossing pattern: depot(0,0) → c1(1,5) → c2(5,0) → c3(1,0) → c4(5,5)
    # The edges c1→c2 and c3→c4 cross, so 2-opt should uncross them.
    c1 = CustomerNode(1, 1, 5, 10)
    c2 = CustomerNode(2, 5, 0, 10)
    c3 = CustomerNode(3, 1, 0, 10)
    c4 = CustomerNode(4, 5, 5, 10)

    vehicle = _make_route_with_distances(depot, [c1, c2, c3, c4])
    original_dist = vehicle.vehicle_distance

    improved = two_opt_route(vehicle)
    assert improved is True
    assert vehicle.vehicle_distance < original_dist


def test_two_opt_updates_vector() -> None:
    """After 2-opt, the vector attribute must reflect the new customer order."""
    depot = DepotNode(1, 0, 0, 1000, 100.0, 1.0)
    c1 = CustomerNode(1, 1, 0, 10)
    c2 = CustomerNode(2, 3, 0, 10)
    c3 = CustomerNode(3, 2, 0, 10)

    vehicle = _make_route_with_distances(depot, [c1, c2, c3])
    two_opt_route(vehicle)

    assert vehicle.vector == [c.customer_number for c in vehicle.customers]
