"""Vehicle route model for the Location-Routing Problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lrp.config import VEHICLE_CAPACITY

if TYPE_CHECKING:
    from lrp.models.node import CustomerNode, DepotNode


class VehicleRoute:
    """A single vehicle route departing from and returning to a depot.

    Attributes:
        vehicle_number: Sequential vehicle identifier within its depot.
        customers: Ordered list of customers visited on this route.
        vehicle_depot_id: Identifier of the originating depot.
        capacity: Remaining load capacity of the vehicle.
        vector: Customer numbers in visit order.
        vehicle_distance: Total distance travelled on this route.
    """

    def __init__(
        self, number: int, depot: DepotNode, vehicle_capacity: int = VEHICLE_CAPACITY
    ) -> None:
        """Initialise a vehicle route.

        Args:
            number: Sequential vehicle number within its depot.
            depot: The depot this vehicle departs from and returns to.
            vehicle_capacity: Maximum load this vehicle can carry.  Defaults
                to the global ``VEHICLE_CAPACITY`` constant.
        """
        self.vehicle_number = number
        self.customers: list[CustomerNode] = []
        self.vehicle_depot_id: int = depot.depot_number
        self.capacity: int = vehicle_capacity
        self.vector: list[int] = []
        self.vehicle_distance: float = 0.0

    def create_vector(self) -> None:
        """Populate ``vector`` from current customers."""
        self.vector = [c.customer_number for c in self.customers]

    def calculate_vehicle_distance(self) -> None:
        """Compute and store the total route distance.

        Traversal order: depot -> c[0] -> c[1] -> ... -> c[n-1] -> depot.
        Resets ``vehicle_distance`` to zero before accumulating.
        """
        self.vehicle_distance = 0.0
        if not self.customers:
            return
        depot_id = self.vehicle_depot_id
        # Leg: depot -> first customer
        self.vehicle_distance += self.customers[0].get_depot_distance(depot_id).dist
        # Legs: customer[i] -> customer[i+1]
        for i in range(len(self.customers) - 1):
            self.vehicle_distance += (
                self.customers[i]
                .get_customer_distance(self.customers[i + 1].customer_number)
                .dist
            )
        # Leg: last customer -> depot
        self.vehicle_distance += self.customers[-1].get_depot_distance(depot_id).dist

    def __repr__(self) -> str:
        ids = [c.customer_number for c in self.customers]
        return f"VehicleRoute(#{self.vehicle_number}, customers={ids})"
