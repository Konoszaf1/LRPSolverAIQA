"""Unit tests for lrp/models/node.py."""

from __future__ import annotations

import copy

import pytest

from lrp.models.distance import Distance
from lrp.models.node import CustomerNode, DepotNode

# ---------------------------------------------------------------------------
# CustomerNode
# ---------------------------------------------------------------------------

class TestCustomerNode:
    def test_basic_construction(self) -> None:
        c = CustomerNode(1, 10, 20, 50)
        assert c.customer_number == 1
        assert c.x_cord == 10
        assert c.y_cord == 20
        assert c.demand == 50
        assert c.assigned_depot is None
        assert c.depot_distances == []
        assert c.customer_distances == []

    def test_deepcopy_shares_distance_lists(self) -> None:
        """The __deepcopy__ optimisation must share distance lists."""
        c1 = CustomerNode(1, 0, 0, 10)
        c2 = CustomerNode(2, 3, 4, 20)
        c1.customer_distances = [Distance(c1, c2)]
        c1.depot_distances = []

        c1_copy = copy.deepcopy(c1)

        # Distance lists are the same objects (shared, not copied)
        assert c1_copy.customer_distances is c1.customer_distances
        assert c1_copy.depot_distances is c1.depot_distances
        # But scalar attributes are independent copies
        c1_copy.demand = 999
        assert c1.demand == 10

    def test_get_closest_depot_customer(self) -> None:
        c1 = CustomerNode(1, 0, 0, 10)
        c2 = CustomerNode(2, 1, 0, 10)  # closer
        c3 = CustomerNode(3, 10, 0, 10)  # farther
        c1.customer_distances = [Distance(c1, c2), Distance(c1, c3)]

        result = c1.get_closest_depot_customer([c2, c3])
        assert result is c2

    def test_get_closest_depot_customer_empty_raises(self) -> None:
        c1 = CustomerNode(1, 0, 0, 10)
        c1.customer_distances = []
        with pytest.raises(ValueError):
            c1.get_closest_depot_customer([])

    def test_get_depot_distance(self) -> None:
        c = CustomerNode(1, 0, 0, 10)
        d = DepotNode(5, 3, 4, 1000, 100.0, 1.0)
        c.depot_distances = [Distance(c, d)]

        dist = c.get_depot_distance(5)
        assert dist.dist == pytest.approx(5.0)

    def test_get_depot_distance_missing_raises(self) -> None:
        c = CustomerNode(1, 0, 0, 10)
        c.depot_distances = []
        with pytest.raises(ValueError, match="No distance found to depot"):
            c.get_depot_distance(99)

    def test_get_customer_distance(self) -> None:
        c1 = CustomerNode(1, 0, 0, 10)
        c2 = CustomerNode(2, 3, 4, 20)
        c1.customer_distances = [Distance(c1, c2)]

        dist = c1.get_customer_distance(2)
        assert dist.dist == pytest.approx(5.0)

    def test_get_customer_distance_missing_raises(self) -> None:
        c = CustomerNode(1, 0, 0, 10)
        c.customer_distances = []
        with pytest.raises(ValueError, match="No distance found to customer"):
            c.get_customer_distance(99)

    def test_repr(self) -> None:
        c = CustomerNode(3, 7, 8, 42)
        assert "CustomerNode" in repr(c)
        assert "#3" in repr(c)


# ---------------------------------------------------------------------------
# DepotNode
# ---------------------------------------------------------------------------

class TestDepotNode:
    def test_basic_construction(self) -> None:
        d = DepotNode(1, 5, 5, 1000, 500.0, 1.5)
        assert d.depot_number == 1
        assert d.capacity == 1000
        assert d.original_capacity == 1000
        assert d.fixed_cost == 500.0
        assert d.variable_cost == 1.5
        assert d.opened is False
        assert d.assigned_customers == []
        assert d.vehicles == []

    def test_get_closest_customer(self) -> None:
        d = DepotNode(1, 0, 0, 1000, 100.0, 1.0)
        c_near = CustomerNode(1, 1, 0, 10)
        c_far = CustomerNode(2, 10, 0, 20)
        # Populate distances on the customers (so get_depot_distance works)
        c_near.depot_distances = [Distance(c_near, d)]
        c_far.depot_distances = [Distance(c_far, d)]
        d.assigned_customers = [c_far, c_near]

        result = d.get_closest_customer()
        assert result is c_near

    def test_get_closest_customer_empty_raises(self) -> None:
        d = DepotNode(1, 0, 0, 1000, 100.0, 1.0)
        with pytest.raises(ValueError, match="no assigned customers"):
            d.get_closest_customer()

    def test_repr(self) -> None:
        d = DepotNode(2, 3, 4, 500, 200.0, 0.5)
        assert "DepotNode" in repr(d)
        assert "#2" in repr(d)
