"""Deterministic validators for LRP solution correctness.

Each validator inspects a list of route dicts (as produced by
``qa_suite.common.adapters`` or an AI agent) against the raw problem data
from ``qa_suite.common.fixtures``.  All validators return a ``ValidationResult``
and make no LLM calls — they are fast, deterministic, and suitable for use
as DeepEval BaseMetric implementations or standalone QA scripts.

Route dict shape expected by every validator::

    {
        "depot_id": int,
        "customer_ids": list[int],
        "stated_distance": float | None,   # optional
    }

Customer dict shape (values of the dict from ``fixtures.load_customers``)::

    {"x": float, "y": float, "demand": float}

Depot dict shape (values of the dict from ``fixtures.load_depots``)::

    {"x": float, "y": float, "capacity": float,
     "fixed_cost": float, "variable_cost": float}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a single validation pass.

    Attributes:
        passed: True when no violations were detected.
        violations: Human-readable description of each detected violation.
        score: Fraction of items that passed, in [0.0, 1.0].
    """

    passed: bool
    violations: list[str] = field(default_factory=list)
    score: float = 1.0


# ---------------------------------------------------------------------------
# Validator 1 — Vehicle capacity
# ---------------------------------------------------------------------------

def validate_vehicle_capacity(
    routes: list[dict],
    customers: dict[int, dict],
    vehicle_capacity: float,
) -> ValidationResult:
    """Check that no route's total demand exceeds the vehicle capacity.

    Args:
        routes: List of route dicts with ``"customer_ids"`` key.
        customers: Mapping of customer ID → attributes (must include ``"demand"``).
        vehicle_capacity: Maximum load a single vehicle can carry.

    Returns:
        A :class:`ValidationResult` where ``score`` = valid routes / total routes.
        If there are no routes, ``passed`` is True and ``score`` is 1.0.
    """
    if not routes:
        return ValidationResult(passed=True, violations=[], score=1.0)

    violations: list[str] = []
    valid = 0

    for i, route in enumerate(routes):
        depot_id = route.get("depot_id", "?")
        cids = route.get("customer_ids", [])

        # Flag phantom customer IDs — their demand is unknown so the load
        # calculation would silently undercount.
        unknown = [cid for cid in cids if cid not in customers]
        if unknown:
            violations.append(
                f"Route {i} (depot {depot_id}): unknown customer IDs {unknown} "
                f"— demand cannot be verified"
            )
            continue

        load = sum(customers[cid]["demand"] for cid in cids)
        if load > vehicle_capacity:
            violations.append(
                f"Route {i} (depot {depot_id}, customers {cids}): "
                f"demand {load:.1f} exceeds vehicle capacity {vehicle_capacity:.1f}"
            )
        else:
            valid += 1

    total = len(routes)
    score = valid / total if total > 0 else 1.0
    return ValidationResult(passed=len(violations) == 0, violations=violations, score=score)


# ---------------------------------------------------------------------------
# Validator 2 — Customer coverage
# ---------------------------------------------------------------------------

def validate_customer_coverage(
    routes: list[dict],
    customers: dict[int, dict],
) -> ValidationResult:
    """Check that every customer is served by exactly one route.

    Detects three kinds of violations:
    - *Missing*: customer exists in the problem but appears in no route.
    - *Duplicate*: customer appears in more than one route.
    - *Phantom*: route contains a customer ID not in the problem.

    Args:
        routes: List of route dicts with ``"customer_ids"`` key.
        customers: Mapping of customer ID → attributes.

    Returns:
        A :class:`ValidationResult` where ``score`` = correctly served customers
        / total customers in the problem.
    """
    violations: list[str] = []
    serve_count: dict[int, int] = {}

    for route in routes:
        for cid in route.get("customer_ids", []):
            serve_count[cid] = serve_count.get(cid, 0) + 1

    all_ids = set(customers.keys())
    routed_ids = set(serve_count.keys())

    # Phantom customers (in routes but not in the problem)
    phantom = routed_ids - all_ids
    for cid in sorted(phantom):
        violations.append(f"Phantom customer {cid}: appears in a route but is not in the problem.")

    # Missing customers (in the problem but served 0 times)
    missing = all_ids - routed_ids
    for cid in sorted(missing):
        violations.append(f"Customer {cid}: not served by any route.")

    # Duplicate customers (served more than once)
    duplicates = {cid: cnt for cid, cnt in serve_count.items() if cnt > 1 and cid in all_ids}
    for cid, cnt in sorted(duplicates.items()):
        violations.append(f"Customer {cid}: served {cnt} times (expected exactly 1).")

    # Score: fraction of real customers served exactly once
    total = len(all_ids)
    correctly_served = sum(
        1 for cid in all_ids if serve_count.get(cid, 0) == 1
    )
    score = correctly_served / total if total > 0 else 1.0
    return ValidationResult(passed=len(violations) == 0, violations=violations, score=score)


# ---------------------------------------------------------------------------
# Validator 3 — Depot capacity
# ---------------------------------------------------------------------------

def validate_depot_capacity(
    routes: list[dict],
    customers: dict[int, dict],
    depots: dict[int, dict],
) -> ValidationResult:
    """Check that total demand assigned to each depot does not exceed its capacity.

    Args:
        routes: List of route dicts with ``"depot_id"`` and ``"customer_ids"`` keys.
        customers: Mapping of customer ID → attributes (must include ``"demand"``).
        depots: Mapping of depot ID → attributes (must include ``"capacity"``).

    Returns:
        A :class:`ValidationResult` where ``score`` = valid depots / used depots.
        If no depots are used, ``passed`` is True and ``score`` is 1.0.
    """
    # Accumulate demand per depot
    depot_load: dict[int, float] = {}
    for route in routes:
        did = route.get("depot_id")
        if did is None:
            continue
        load = sum(customers[cid]["demand"] for cid in route.get("customer_ids", []) if cid in customers)
        depot_load[did] = depot_load.get(did, 0.0) + load

    if not depot_load:
        return ValidationResult(passed=True, violations=[], score=1.0)

    violations: list[str] = []
    valid = 0

    for did, load in sorted(depot_load.items()):
        if did not in depots:
            violations.append(
                f"Depot {did}: referenced in routes but not found in problem data."
            )
            continue
        cap = depots[did]["capacity"]
        if load > cap:
            violations.append(
                f"Depot {did}: total assigned demand {load:.1f} "
                f"exceeds depot capacity {cap:.1f}"
            )
        else:
            valid += 1

    total = len(depot_load)
    score = valid / total if total > 0 else 1.0
    return ValidationResult(passed=len(violations) == 0, violations=violations, score=score)


# ---------------------------------------------------------------------------
# Validator 4 — Route distances
# ---------------------------------------------------------------------------

def validate_route_distances(
    routes: list[dict],
    customers: dict[int, dict],
    depots: dict[int, dict],
    tolerance: float = 0.1,
) -> ValidationResult:
    """Check that stated route distances match recomputed Euclidean distances.

    Only routes where ``"stated_distance"`` is not None are checked.
    The recomputed distance follows the path:
    depot → customer₁ → customer₂ → … → customerₙ → depot.

    Args:
        routes: List of route dicts.
        customers: Mapping of customer ID → ``{"x": float, "y": float, ...}``.
        depots: Mapping of depot ID → ``{"x": float, "y": float, ...}``.
        tolerance: Relative tolerance; a route fails when
            ``|actual − stated| > tolerance × actual``.

    Returns:
        A :class:`ValidationResult` where ``score`` = accurate routes /
        routes with a stated distance.  If no routes have a stated distance,
        ``passed`` is True and ``score`` is 1.0.
    """
    checkable = [r for r in routes if r.get("stated_distance") is not None]
    if not checkable:
        return ValidationResult(passed=True, violations=[], score=1.0)

    violations: list[str] = []
    accurate = 0

    for route in checkable:
        did = route.get("depot_id")
        cids = route.get("customer_ids", [])
        stated = float(route["stated_distance"])

        if did not in depots:
            violations.append(f"Route depot {did} not found in depots dict.")
            continue

        dep_pos = (depots[did]["x"], depots[did]["y"])

        # Build ordered position list: depot, c1, c2, ..., cn, depot
        positions: list[tuple[float, float]] = [dep_pos]
        skip = False
        for cid in cids:
            if cid not in customers:
                violations.append(f"Customer {cid} in route not found in customers dict.")
                skip = True
                break
            positions.append((customers[cid]["x"], customers[cid]["y"]))
        if skip:
            continue
        positions.append(dep_pos)

        # Sum consecutive Euclidean legs
        actual = sum(
            math.dist(positions[i], positions[i + 1])
            for i in range(len(positions) - 1)
        )

        # When actual == 0 (all nodes co-located), stated must also be ~0.
        # Use absolute tolerance of 1e-6 as a fallback for degenerate routes.
        abs_tol = tolerance * actual if actual > 0 else 1e-6
        if abs(actual - stated) > abs_tol:
            violations.append(
                f"Route (depot {did}, customers {cids}): "
                f"stated distance {stated:.4f} differs from "
                f"recomputed {actual:.4f} "
                f"(|Δ| = {abs(actual - stated):.4f}, "
                f"tolerance = {abs_tol:.4f})"
            )
        else:
            accurate += 1

    total = len(checkable)
    score = accurate / total if total > 0 else 1.0
    return ValidationResult(passed=len(violations) == 0, violations=violations, score=score)


# ---------------------------------------------------------------------------
# Validator 5 — Total cost recomputation
# ---------------------------------------------------------------------------

def validate_total_cost(
    routes: list[dict],
    depots: dict[int, dict],
    open_depots: list[int],
    stated_total_cost: float,
    tolerance: float = 0.1,
) -> ValidationResult:
    """Check that the stated total cost matches recomputed fixed costs + route distances.

    Recomputes ``total_cost = Σ(fixed_cost for open depots) + Σ(stated_distance for routes)``.

    Args:
        routes: List of route dicts with optional ``"stated_distance"`` key.
        depots: Mapping of depot ID → attributes (must include ``"fixed_cost"``).
        open_depots: List of depot IDs that are open.
        stated_total_cost: The solver's reported total cost.
        tolerance: Relative tolerance for the comparison.

    Returns:
        A :class:`ValidationResult` with score 1.0 if the cost matches, else 0.0.
    """
    violations: list[str] = []

    # Sum depot fixed costs
    fixed_cost = 0.0
    for did in open_depots:
        if did in depots:
            fixed_cost += depots[did]["fixed_cost"]
        else:
            violations.append(f"Open depot {did} not found in problem data — cannot verify fixed cost.")

    # Sum route distances
    routing_cost = 0.0
    missing_distances = 0
    for i, route in enumerate(routes):
        sd = route.get("stated_distance")
        if sd is not None:
            routing_cost += float(sd)
        else:
            missing_distances += 1

    if missing_distances:
        violations.append(
            f"{missing_distances} route(s) have no stated_distance — "
            f"routing cost may be underestimated."
        )

    recomputed = fixed_cost + routing_cost
    abs_tol = tolerance * recomputed if recomputed > 0 else 1e-6

    if abs(recomputed - stated_total_cost) > abs_tol:
        violations.append(
            f"Stated total cost {stated_total_cost:.4f} differs from "
            f"recomputed {recomputed:.4f} "
            f"(fixed={fixed_cost:.4f} + routing={routing_cost:.4f}, "
            f"|Δ| = {abs(recomputed - stated_total_cost):.4f}, "
            f"tolerance = {abs_tol:.4f})"
        )

    score = 1.0 if len(violations) == 0 else 0.0
    return ValidationResult(passed=len(violations) == 0, violations=violations, score=score)
