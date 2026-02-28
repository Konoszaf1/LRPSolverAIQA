"""Unit tests for lrp/io/data_loader.py."""

from __future__ import annotations

import pytest

from lrp.io.data_loader import load_customers, load_depots


@pytest.fixture()
def customer_file(tmp_path):
    """Create a temporary customer data file."""
    p = tmp_path / "customers.txt"
    p.write_text("1 10 20 50\n2 30 40 80\n")
    return p


@pytest.fixture()
def depot_file(tmp_path):
    """Create a temporary depot data file."""
    p = tmp_path / "depots.txt"
    p.write_text("1 5 5 1000 500.0 1.5\n")
    return p


def test_load_customers(customer_file) -> None:
    customers = load_customers(customer_file)
    assert len(customers) == 2
    assert customers[0].customer_number == 1
    assert customers[0].x_cord == 10
    assert customers[0].y_cord == 20
    assert customers[0].demand == 50
    assert customers[1].customer_number == 2


def test_load_customers_skips_blank_lines(tmp_path) -> None:
    p = tmp_path / "customers.txt"
    p.write_text("1 10 20 50\n\n\n2 30 40 80\n")
    customers = load_customers(p)
    assert len(customers) == 2


def test_load_customers_handles_float_demands(tmp_path) -> None:
    """Some benchmarks (Srivastava86) have float-formatted demands like 112.0."""
    p = tmp_path / "customers.txt"
    p.write_text("1 10.0 20.0 112.0\n")
    customers = load_customers(p)
    assert customers[0].demand == 112


def test_load_customers_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_customers("/nonexistent/path/to/file")


def test_load_depots(depot_file) -> None:
    depots = load_depots(depot_file)
    assert len(depots) == 1
    assert depots[0].depot_number == 1
    assert depots[0].capacity == 1000
    assert depots[0].fixed_cost == 500.0
    assert depots[0].variable_cost == 1.5


def test_load_depots_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_depots("/nonexistent/path/to/file")
