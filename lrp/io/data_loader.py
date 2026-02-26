"""I/O utilities for reading LRP benchmark dataset files.

Benchmark files use whitespace-delimited columns with no header row.
Customer files: node_id  x  y  demand
Depot files:    node_id  x  y  capacity  fixed_cost  variable_cost
"""

from pathlib import Path

from lrp.models.node import CustomerNode, DepotNode


def _read_rows(path: str | Path) -> list[list[str]]:
    """Read a whitespace-delimited file and return non-empty rows of tokens.

    Args:
        path: Filesystem path to the input file.

    Returns:
        A list of token lists, one entry per non-blank line.
    """
    with open(path) as fh:
        return [line.split() for line in fh if line.strip()]


def load_customers(path: str | Path) -> list[CustomerNode]:
    """Load customer nodes from a benchmark data file.

    Expected column order: node_id  x  y  demand.

    Args:
        path: Path to the customer data file.

    Returns:
        Ordered list of CustomerNode objects.
    """
    return [
        CustomerNode(
            customer_number=int(row[0]),
            x_cord=int(row[1]),
            y_cord=int(row[2]),
            demand=int(row[3]),
        )
        for row in _read_rows(path)
    ]


def load_depots(path: str | Path) -> list[DepotNode]:
    """Load depot nodes from a benchmark data file.

    Expected column order: node_id  x  y  capacity  fixed_cost  variable_cost.

    Args:
        path: Path to the depot data file.

    Returns:
        Ordered list of DepotNode objects.
    """
    return [
        DepotNode(
            depot_number=int(row[0]),
            x_cord=int(row[1]),
            y_cord=int(row[2]),
            capacity=float(row[3]),
            fixed_cost=float(row[4]),
            variable_cost=float(row[5]),
        )
        for row in _read_rows(path)
    ]
