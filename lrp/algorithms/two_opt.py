"""2-opt local search for improving individual vehicle routes.

Applies the classic 2-opt edge-swap improvement heuristic to a single
VehicleRoute.  A 2-opt move reverses a contiguous sub-sequence of customers,
replacing two edges with two shorter alternatives whenever the reversal
reduces total route distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lrp.models.vehicle_route import VehicleRoute


def two_opt_route(vehicle: VehicleRoute) -> bool:
    """Apply 2-opt improvement to a single vehicle route.

    Iterates over all pairs of non-adjacent edges in the route.  When
    reversing the sub-sequence between edge endpoints yields a shorter total
    distance, the reversal is applied immediately and the scan restarts.
    Continues until a full pass finds no improving move.

    Uses the distance lookups already stored on each CustomerNode, so no
    additional data is required.

    Args:
        vehicle: The vehicle route to improve (mutated in place).  Both
            ``vehicle.customers`` and ``vehicle.vehicle_distance`` are
            updated on exit.

    Returns:
        ``True`` if at least one improving swap was applied, ``False`` if
        the route was already locally optimal.
    """
    customers = vehicle.customers
    if len(customers) < 3:
        return False

    depot_id = vehicle.vehicle_depot_id
    improved = False
    changed = True

    while changed:
        changed = False
        n = len(customers)

        for i in range(n - 1):
            for j in range(i + 2, n):
                # --- Cost of the two edges being considered for removal ---
                # Edge A: predecessor-of-i  →  i
                if i == 0:
                    cost_a = customers[i].get_depot_distance(depot_id).dist
                else:
                    cost_a = customers[i - 1].get_customer_distance(
                        customers[i].customer_number
                    ).dist

                # Edge B: j  →  successor-of-j
                if j == n - 1:
                    cost_b = customers[j].get_depot_distance(depot_id).dist
                else:
                    cost_b = customers[j].get_customer_distance(
                        customers[j + 1].customer_number
                    ).dist

                # --- Cost of the two replacement edges ---
                # New edge A': predecessor-of-i  →  j
                if i == 0:
                    new_cost_a = customers[j].get_depot_distance(depot_id).dist
                else:
                    new_cost_a = customers[i - 1].get_customer_distance(
                        customers[j].customer_number
                    ).dist

                # New edge B': i  →  successor-of-j
                if j == n - 1:
                    new_cost_b = customers[i].get_depot_distance(depot_id).dist
                else:
                    new_cost_b = customers[i].get_customer_distance(
                        customers[j + 1].customer_number
                    ).dist

                if (new_cost_a + new_cost_b) < (cost_a + cost_b):
                    # Reverse the sub-sequence customers[i..j]
                    customers[i : j + 1] = customers[i : j + 1][::-1]
                    changed = True
                    improved = True

    vehicle.calculate_vehicle_distance()
    vehicle.create_vector()
    return improved
