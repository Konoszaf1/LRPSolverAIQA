"""Node data models for the Location-Routing Problem.

Provides a base ``Node`` class and concrete ``CustomerNode`` / ``DepotNode``
subclasses that represent the two kinds of locations in the LRP network.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lrp.models.distance import Distance


class Node:
    """Base class for all network nodes.

    Attributes:
        x_cord: X-coordinate of the node.
        y_cord: Y-coordinate of the node.
    """

    def __init__(self, x_cord: int, y_cord: int) -> None:
        """Initialise node coordinates.

        Args:
            x_cord: X-coordinate.
            y_cord: Y-coordinate.
        """
        self.x_cord = x_cord
        self.y_cord = y_cord


class CustomerNode(Node):
    """Represents a customer location in the LRP network.

    Attributes:
        customer_number: Unique customer identifier.
        demand: Quantity of goods the customer requires.
        depot_distances: Distances to all depot nodes in the current solution.
        customer_distances: Distances to all other customer nodes.
        assigned_depot: Depot number this customer has been assigned to,
            or ``None`` if not yet assigned.
    """

    def __init__(
        self,
        customer_number: int,
        x_cord: int,
        y_cord: int,
        demand: int,
    ) -> None:
        """Initialise a customer node.

        Args:
            customer_number: Unique identifier for this customer.
            x_cord: X-coordinate.
            y_cord: Y-coordinate.
            demand: Units of goods required by this customer.
        """
        super().__init__(x_cord, y_cord)
        self.customer_number = customer_number
        self.demand = demand
        self.depot_distances: list[Distance] = []
        self.customer_distances: list[Distance] = []
        self.assigned_depot: int | None = None

    def get_closest_depot_customer(
        self, other_customers: list[CustomerNode]
    ) -> CustomerNode:
        """Return the nearest customer from a candidate pool.

        Uses ``customer_number`` for matching so that shared distance objects
        (which may reference original nodes) still resolve correctly after a
        deep copy.

        Args:
            other_customers: Pool of candidate customers to search within.

        Returns:
            The CustomerNode in ``other_customers`` with the smallest
            distance to this node.

        Raises:
            ValueError: If no matching customer is found.
        """
        other_numbers = {c.customer_number for c in other_customers}
        relevant = [
            d for d in self.customer_distances
            if cast("CustomerNode", d.node_2).customer_number in other_numbers
        ]
        best = min(relevant)
        for customer in other_customers:
            if customer.customer_number == cast("CustomerNode", best.node_2).customer_number:
                return customer
        raise ValueError("No matching customer found in pool.")

    def get_depot_distance(self, depot_number: int) -> Distance:
        """Return the Distance object to the specified depot.

        Args:
            depot_number: Identifier of the target depot.

        Returns:
            The Distance between this customer and the depot.

        Raises:
            ValueError: If no distance to the depot is found.
        """
        for distance in self.depot_distances:
            if cast("DepotNode", distance.node_2).depot_number == depot_number:
                return distance
        raise ValueError(
            f"No distance found to depot {depot_number} "
            f"from customer {self.customer_number}."
        )

    def get_customer_distance(self, customer_number: int) -> Distance:
        """Return the Distance object to another customer.

        Args:
            customer_number: Identifier of the target customer.

        Returns:
            The Distance between this customer and the target.

        Raises:
            ValueError: If no distance to the target customer is found.
        """
        for distance in self.customer_distances:
            if cast("CustomerNode", distance.node_2).customer_number == customer_number:
                return distance
        raise ValueError(
            f"No distance found to customer {customer_number} "
            f"from customer {self.customer_number}."
        )

    def __deepcopy__(self, memo: dict) -> CustomerNode:
        """Deep-copy this node, sharing distance lists across copies.

        Distance objects are immutable after construction and reference the
        entire node graph; sharing them avoids quadratic memory growth when
        copying large solution populations.

        Args:
            memo: Standard deepcopy memo dictionary for cycle detection.

        Returns:
            A new CustomerNode sharing the original distance lists.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("depot_distances", "customer_distances"):
                setattr(result, k, v)  # Shared — distances are immutable
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __repr__(self) -> str:
        return (
            f"CustomerNode(#{self.customer_number}, x={self.x_cord},"
            f" y={self.y_cord}, demand={self.demand})"
        )


class DepotNode(Node):
    """Represents a depot location in the LRP network.

    Attributes:
        depot_number: Unique depot identifier.
        capacity: Remaining capacity available for customer assignment.
        original_capacity: Full capacity at construction time; used to reset
            the depot between random solution generations.
        fixed_cost: One-time cost incurred when this depot is opened.
        variable_cost: Per-unit transportation cost for this depot.
        opened: Whether this depot is currently active in the solution.
        assigned_customers: Customers currently allocated to this depot.
        vehicles: Vehicle routes operating out of this depot.
    """

    def __init__(
        self,
        depot_number: int,
        x_cord: int,
        y_cord: int,
        capacity: float,
        fixed_cost: float,
        variable_cost: float,
    ) -> None:
        """Initialise a depot node.

        Args:
            depot_number: Unique identifier for this depot.
            x_cord: X-coordinate.
            y_cord: Y-coordinate.
            capacity: Maximum units this depot can supply.
            fixed_cost: Cost to open this depot.
            variable_cost: Cost per unit transported from this depot.
        """
        super().__init__(x_cord, y_cord)
        self.depot_number = depot_number
        self.capacity = capacity
        self.original_capacity = capacity
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.opened: bool = False
        self.assigned_customers: list[CustomerNode] = []
        self.vehicles: list = []  # list[VehicleRoute] — avoids circular import

    def get_closest_customer(self) -> CustomerNode:
        """Return the assigned customer nearest to this depot.

        Returns:
            The CustomerNode with the smallest distance to this depot.

        Raises:
            ValueError: If no customers are currently assigned.
        """
        if not self.assigned_customers:
            raise ValueError(
                f"Depot {self.depot_number} has no assigned customers."
            )
        return min(
            self.assigned_customers,
            key=lambda c: c.get_depot_distance(self.depot_number).dist,
        )

    def __repr__(self) -> str:
        return (
            f"DepotNode(#{self.depot_number}, x={self.x_cord}, y={self.y_cord},"
            f" capacity={self.capacity}, fixed_cost={self.fixed_cost},"
            f" opened={self.opened})"
        )
