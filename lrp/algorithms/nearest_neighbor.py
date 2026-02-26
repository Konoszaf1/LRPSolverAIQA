"""Nearest-neighbour heuristic for depot assignment and vehicle routing."""

from lrp.config import VEHICLE_CAPACITY
from lrp.models.node import CustomerNode, DepotNode
from lrp.models.vehicle_route import VehicleRoute


def _assign_customer_to_depot(
    customer: CustomerNode,
    depot: DepotNode,
    total_cost: float,
) -> float:
    """Assign a customer to a depot and return the updated fixed cost.

    Opens the depot if it was previously closed, adding its fixed cost to
    the running total. Modifies both ``customer`` and ``depot`` in place.

    Args:
        customer: The customer to assign.
        depot: The target depot.
        total_cost: Current accumulated fixed-cost total.

    Returns:
        Updated total fixed cost after this assignment.
    """
    if not depot.opened:
        depot.opened = True
        total_cost += depot.fixed_cost
    depot.assigned_customers.append(customer)
    customer.assigned_depot = depot.depot_number
    depot.capacity -= customer.demand
    return total_cost


def assign_depots(customers: list[CustomerNode]) -> float:
    """Assign each customer to a depot using the nearest-neighbour heuristic.

    Iterates customers and assigns each to the nearest depot (by pre-computed
    ``depot_distances``) that has sufficient remaining capacity.  Unopened
    depots are opened on demand; their fixed costs are accumulated and
    returned.

    ``customer.depot_distances`` must be populated before calling this
    function (see ``Solution.build_distances``).

    Args:
        customers: Customer nodes to assign (mutated in place).  Depot nodes
            are reached through each customer's ``depot_distances`` list.

    Returns:
        Total fixed cost of all depots opened during assignment.
    """
    total_cost = 0.0
    for customer in customers:
        sorted_depots: list[DepotNode] = [
            d.node_2 for d in sorted(customer.depot_distances)
        ]
        assigned = False
        for depot in sorted_depots:
            if depot.capacity >= customer.demand:
                total_cost = _assign_customer_to_depot(customer, depot, total_cost)
                assigned = True
                break
        if not assigned:
            raise ValueError(
                f"Customer {customer.customer_number} (demand={customer.demand}) "
                f"cannot be assigned — no depot has sufficient capacity."
            )
    return total_cost


def build_vehicle_routes(
    depot: DepotNode, vehicle_capacity: int = VEHICLE_CAPACITY
) -> None:
    """Build greedy nearest-neighbour vehicle routes for one depot.

    Repeatedly selects the unassigned customer nearest to the depot as the
    first stop, then chains the nearest unvisited neighbour until the vehicle
    is full, then opens a new vehicle. Modifies ``depot`` in place; empties
    ``depot.assigned_customers`` as customers are loaded onto vehicles.

    ``customer.customer_distances`` and ``customer.depot_distances`` must be
    populated before calling this function.

    Args:
        depot: The depot to route (mutated in place).
        vehicle_capacity: Maximum load each vehicle can carry.  Defaults to
            the global ``VEHICLE_CAPACITY`` constant.
    """
    vehicle_number = 0
    while depot.assigned_customers:
        vehicle = VehicleRoute(vehicle_number, depot, vehicle_capacity)
        depot.vehicles.append(vehicle)

        first = depot.get_closest_customer()
        if first.demand > vehicle.capacity:
            raise ValueError(
                f"Customer {first.customer_number} (demand={first.demand}) "
                f"exceeds vehicle capacity ({vehicle.capacity})."
            )
        vehicle.customers.append(first)
        vehicle.capacity -= first.demand
        depot.assigned_customers.remove(first)

        current = first
        while depot.assigned_customers and vehicle.capacity > 0:
            next_customer = current.get_closest_depot_customer(
                depot.assigned_customers
            )
            if vehicle.capacity >= next_customer.demand:
                vehicle.customers.append(next_customer)
                vehicle.capacity -= next_customer.demand
                depot.assigned_customers.remove(next_customer)
                current = next_customer
            else:
                break

        vehicle_number += 1

    for vehicle in depot.vehicles:
        vehicle.create_vector()
        vehicle.calculate_vehicle_distance()
