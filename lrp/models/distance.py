"""Euclidean distance between two LRP network nodes."""

from __future__ import annotations

import math

from lrp.models.node import CustomerNode, DepotNode, Node


class Distance:
    """Euclidean distance between two nodes in the LRP network.

    Attributes:
        node_1: The source node.
        node_2: The destination node.
        dist: Euclidean distance, or ``math.inf`` when both nodes are the
            same object.
        is_customer_depot: True when one node is a depot and the other a
            customer.
        is_customer_customer: True when both nodes are customers.
    """

    def __init__(self, node_1: Node, node_2: Node) -> None:
        """Compute and store the distance between two nodes.

        Args:
            node_1: Source node.
            node_2: Destination node.
        """
        self.node_1 = node_1
        self.node_2 = node_2
        self.dist: float = (
            math.inf
            if node_1 is node_2
            else math.dist(
                (node_1.x_cord, node_1.y_cord),
                (node_2.x_cord, node_2.y_cord),
            )
        )
        self.is_customer_depot: bool = isinstance(node_1, DepotNode) or isinstance(
            node_2, DepotNode
        )
        self.is_customer_customer: bool = isinstance(
            node_1, CustomerNode
        ) and isinstance(node_2, CustomerNode)

    def __lt__(self, other: Distance) -> bool:
        return self.dist < other.dist

    def __gt__(self, other: Distance) -> bool:
        return self.dist > other.dist

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distance):
            return NotImplemented
        return (
            self.dist == other.dist
            and {id(self.node_1), id(self.node_2)}
            == {id(other.node_1), id(other.node_2)}
        )

    def __repr__(self) -> str:
        return (
            f"Distance({self.node_1!r} <-> {self.node_2!r}: {self.dist:.4f})"
        )
