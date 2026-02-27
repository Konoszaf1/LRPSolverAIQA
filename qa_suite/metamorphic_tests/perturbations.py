"""Dataset perturbation functions for metamorphic testing of the LRP solver.

Each function accepts a dataset dict (same format as ``load_instance()`` output)
and returns a NEW, independent deep copy with one targeted modification applied.
The original dataset is never mutated.
"""

from __future__ import annotations

import copy
import random


def increase_vehicle_capacity(dataset: dict, factor: float = 1.5) -> dict:
    """Multiply ``vehicle_capacity`` by *factor*.

    More capacity per vehicle means fewer vehicles are needed per route,
    so total routing cost should not increase.

    Args:
        dataset: Dataset dict from ``load_instance()``.
        factor:  Multiplicative factor (default 1.5).

    Returns:
        Deep-copied dataset with ``vehicle_capacity`` scaled by *factor*.
    """
    result = copy.deepcopy(dataset)
    result["vehicle_capacity"] = dataset["vehicle_capacity"] * factor
    return result


def double_all_demands(dataset: dict) -> dict:
    """Double every customer's demand.

    Each vehicle fills twice as fast, so the solver needs at least as many
    routes as in the original instance (and likely more).

    Args:
        dataset: Dataset dict from ``load_instance()``.

    Returns:
        Deep-copied dataset with each customer demand multiplied by 2.
    """
    result = copy.deepcopy(dataset)
    for cid in result["customers"]:
        result["customers"][cid]["demand"] *= 2.0
    return result


def zero_all_fixed_costs(dataset: dict) -> dict:
    """Set every depot's ``fixed_cost`` to 0.

    With no financial penalty for opening a depot, the solver should open
    at least as many (and possibly all) depots to minimise routing distances.

    Args:
        dataset: Dataset dict from ``load_instance()``.

    Returns:
        Deep-copied dataset with every depot's ``fixed_cost`` set to 0.0.
    """
    result = copy.deepcopy(dataset)
    for did in result["depots"]:
        result["depots"][did]["fixed_cost"] = 0.0
    return result


def remove_customers(dataset: dict, keep_ratio: float = 0.5) -> dict:
    """Randomly remove customers, retaining only *keep_ratio* of them.

    Uses a fixed random seed (42) so results are reproducible across runs.
    Keeps exactly ``floor(n * keep_ratio)`` customers.

    Args:
        dataset:    Dataset dict from ``load_instance()``.
        keep_ratio: Fraction of customers to retain (default 0.5).

    Returns:
        Deep-copied dataset with a random subset of customers retained.
    """
    result = copy.deepcopy(dataset)
    all_ids = sorted(result["customers"].keys())
    n_keep = max(1, int(len(all_ids) * keep_ratio))
    rng = random.Random(42)
    kept_ids = set(rng.sample(all_ids, n_keep))
    result["customers"] = {cid: v for cid, v in result["customers"].items() if cid in kept_ids}
    return result


def add_nearby_customer(dataset: dict) -> dict:
    """Add one new customer near the centroid of existing customers.

    The new customer's demand equals the average demand of existing customers,
    rounded to one decimal place.  Its ID is ``max(existing_ids) + 1``.
    The coordinates are the exact centroid (no random offset, ensuring
    determinism).

    Args:
        dataset: Dataset dict from ``load_instance()``.

    Returns:
        Deep-copied dataset with one additional customer appended.
    """
    result = copy.deepcopy(dataset)
    customers = result["customers"]

    xs = [c["x"] for c in customers.values()]
    ys = [c["y"] for c in customers.values()]
    demands = [c["demand"] for c in customers.values()]

    centroid_x = round(sum(xs) / len(xs), 1)
    centroid_y = round(sum(ys) / len(ys), 1)
    avg_demand = round(sum(demands) / len(demands), 1)

    new_id = max(customers.keys()) + 1
    customers[new_id] = {"x": centroid_x, "y": centroid_y, "demand": avg_demand}
    return result
