"""Unit tests for lrp/models/distance.py."""

from __future__ import annotations

import math

import pytest

from lrp.models.distance import Distance
from lrp.models.node import CustomerNode, DepotNode


def test_euclidean_distance_3_4_5_triangle() -> None:
    c = CustomerNode(1, 0, 0, 10)
    d = DepotNode(1, 3, 4, 100, 50.0, 1.0)
    dist = Distance(c, d)
    assert dist.dist == pytest.approx(5.0)


def test_same_node_is_infinity() -> None:
    c = CustomerNode(1, 5, 5, 10)
    dist = Distance(c, c)
    assert dist.dist == math.inf


def test_distance_comparison_lt_gt() -> None:
    c = CustomerNode(1, 0, 0, 10)
    d1 = DepotNode(1, 1, 0, 100, 50.0, 1.0)
    d2 = DepotNode(2, 10, 0, 100, 50.0, 1.0)
    dist1 = Distance(c, d1)
    dist2 = Distance(c, d2)
    assert dist1 < dist2
    assert dist2 > dist1


def test_distance_equality() -> None:
    c = CustomerNode(1, 0, 0, 10)
    d = DepotNode(1, 3, 4, 100, 50.0, 1.0)
    dist_a = Distance(c, d)
    dist_b = Distance(c, d)
    assert dist_a == dist_b


def test_distance_equality_different_nodes() -> None:
    c1 = CustomerNode(1, 0, 0, 10)
    c2 = CustomerNode(2, 0, 0, 10)
    d = DepotNode(1, 3, 4, 100, 50.0, 1.0)
    dist_a = Distance(c1, d)
    dist_b = Distance(c2, d)
    # Same distance value but different node objects
    assert dist_a != dist_b


def test_customer_depot_flag() -> None:
    c = CustomerNode(1, 0, 0, 10)
    d = DepotNode(1, 3, 4, 100, 50.0, 1.0)
    dist = Distance(c, d)
    assert dist.is_customer_depot is True
    assert dist.is_customer_customer is False


def test_customer_customer_flag() -> None:
    c1 = CustomerNode(1, 0, 0, 10)
    c2 = CustomerNode(2, 3, 4, 20)
    dist = Distance(c1, c2)
    assert dist.is_customer_depot is False
    assert dist.is_customer_customer is True


def test_repr_contains_distance_value() -> None:
    c = CustomerNode(1, 0, 0, 10)
    d = DepotNode(1, 3, 4, 100, 50.0, 1.0)
    dist = Distance(c, d)
    assert "5.0000" in repr(dist)
