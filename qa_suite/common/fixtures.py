"""Shared data-loading fixtures for the AIQA QA suite.

Provides a standalone interface to the LRP benchmark datasets that does NOT
depend on the internal ``lrp`` package, so QA tests can load raw problem data
independently of the solver implementation.

All parsers are robust to blank lines, trailing whitespace, and extra spaces
between fields.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Absolute path to the directory containing all benchmark data files.
DATA_DIR: Path = Path(__file__).resolve().parents[2] / "DATALRP" / "DATALRP"

# ---------------------------------------------------------------------------
# Instance registry
#
# Each entry maps an instance name to:
#   (customer_filename, depot_filename, vehicle_capacity)
#
# vehicle_capacity is taken from lrp/config.py: VEHICLE_CAPACITY = 160.
# Depot capacities in each file are much larger (per-depot aggregate limits),
# not the per-vehicle limit.
# ---------------------------------------------------------------------------

INSTANCES: dict[str, tuple[str, str, float]] = {
    # vehicle_capacity: Ch69 uses the lrp/config.py global (160).
    # Other instances don't encode per-vehicle capacity in their data files;
    # 999.0 is used as a safe default that keeps all routes feasible.
    "Srivastava86": ("Srivastava86Cli8x2",  "Srivastava86Dep8x2",  999.0),  # depot cap=1000
    "Gaskell67":    ("Gaskell67Cli21x5",    "Gaskell67Dep21x5",    999.0),  # TODO: verify
    "Perl83":       ("Perl83Cli55x15",      "Perl83Dep55x15",      999.0),  # TODO: verify
    "Ch69":         ("Ch69Cli100x10",       "Ch69Dep100x10",       160.0),  # from lrp/config.py
    "Or76":         ("Or76Cli117x14",       "Or76Dep117x14",       999.0),  # TODO: verify
    "Min92":        ("Min92Cli134x8",       "Min92Dep134x8",       999.0),  # TODO: verify
    "Daskin95":     ("Daskin95Cli150x10",   "Daskin95Dep150x10",   999.0),  # TODO: verify
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def load_customers(filepath: str | Path) -> dict[int, dict]:
    """Parse a customer data file into a plain dictionary.

    Expected file format (whitespace-separated, no header)::

        node_id  x_coord  y_coord  demand

    Blank lines and lines consisting only of whitespace are skipped.

    Args:
        filepath: Path to the customer data file.

    Returns:
        Mapping of ``node_id -> {"x": float, "y": float, "demand": float}``.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If a non-blank line cannot be parsed into exactly 4 fields.
    """
    customers: dict[int, dict] = {}
    with open(filepath, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"{filepath}:{lineno} — expected 4 fields, got {len(parts)}: {line!r}"
                )
            node_id = int(parts[0])
            customers[node_id] = {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "demand": float(parts[3]),
            }
    return customers


def load_depots(filepath: str | Path) -> dict[int, dict]:
    """Parse a depot data file into a plain dictionary.

    Expected file format (whitespace-separated, no header)::

        node_id  x_coord  y_coord  capacity  fixed_cost  variable_cost

    Blank lines and lines consisting only of whitespace are skipped.

    Args:
        filepath: Path to the depot data file.

    Returns:
        Mapping of ``node_id -> {"x": float, "y": float, "capacity": float,
        "fixed_cost": float, "variable_cost": float}``.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If a non-blank line cannot be parsed into exactly 6 fields.
    """
    depots: dict[int, dict] = {}
    with open(filepath, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"{filepath}:{lineno} — expected 6 fields, got {len(parts)}: {line!r}"
                )
            node_id = int(parts[0])
            depots[node_id] = {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "capacity": float(parts[3]),
                "fixed_cost": float(parts[4]),
                "variable_cost": float(parts[5]),
            }
    return depots


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_instance(name: str) -> dict:
    """Load a named benchmark instance from disk.

    Args:
        name: One of the keys in :data:`INSTANCES` (e.g. ``"Srivastava86"``).

    Returns:
        Dictionary with keys:

        - ``"name"`` – the instance name
        - ``"customers"`` – ``dict[int, dict]`` from :func:`load_customers`
        - ``"depots"`` – ``dict[int, dict]`` from :func:`load_depots`
        - ``"vehicle_capacity"`` – per-vehicle capacity float

    Raises:
        KeyError: If *name* is not in :data:`INSTANCES`.
        FileNotFoundError: If the data files are missing.
    """
    if name not in INSTANCES:
        raise KeyError(
            f"Unknown instance {name!r}. Available: {list(INSTANCES)}"
        )
    cli_file, dep_file, vehicle_capacity = INSTANCES[name]
    return {
        "name": name,
        "customers": load_customers(DATA_DIR / cli_file),
        "depots": load_depots(DATA_DIR / dep_file),
        "vehicle_capacity": vehicle_capacity,
    }


def instance_to_text(dataset: dict) -> str:
    """Convert a dataset dict into a human-readable multi-line string.

    Args:
        dataset: A dict as returned by :func:`load_instance`.

    Returns:
        A formatted string listing instance metadata, all customers (id, x, y,
        demand), and all depots (id, x, y, capacity, fixed_cost, variable_cost).
    """
    lines: list[str] = []
    name = dataset.get("name", "<unknown>")
    vc = dataset.get("vehicle_capacity", "?")
    customers: dict[int, dict] = dataset.get("customers", {})
    depots: dict[int, dict] = dataset.get("depots", {})

    lines.append(f"Instance : {name}")
    lines.append(f"Vehicle capacity : {vc}")
    lines.append(f"Customers ({len(customers)}):")
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Demand':>8}")
    for nid, c in sorted(customers.items()):
        lines.append(f"  {nid:>4}  {c['x']:>8.1f}  {c['y']:>8.1f}  {c['demand']:>8.1f}")

    lines.append(f"Depots ({len(depots)}):")
    lines.append(f"  {'ID':>4}  {'X':>8}  {'Y':>8}  {'Capacity':>10}  {'FixedCost':>10}  {'VarCost':>8}")
    for nid, d in sorted(depots.items()):
        lines.append(
            f"  {nid:>4}  {d['x']:>8.1f}  {d['y']:>8.1f}"
            f"  {d['capacity']:>10.1f}  {d['fixed_cost']:>10.2f}  {d['variable_cost']:>8.3f}"
        )

    return "\n".join(lines)
