"""Soft-scoring layer: continuous severity measurement for constraint violations.

Binary pass/fail validators answer *whether* a constraint was broken.  This module
answers *by how much* — turning discrete test outcomes into a continuous signal
suitable for statistical analysis across many stochastic LLM runs.

Each function returns a ``SoftScore`` dataclass:

- ``passed``: bool — same semantics as ``ValidationResult.passed``.
- ``score``: float 0.0–1.0 — fraction of items (routes / depots) that satisfy the
  constraint.
- ``severity``: float ≥ 0.0 — **magnitude** of the worst (or aggregate) violation.
  0.0 = no violation; higher = worse.

  ============= ===============================================
  Validator     Severity interpretation
  ============= ===============================================
  Vehicle cap.  max fractional overshoot (0.15 → 15 % overload)
  Coverage      error fraction (missing + dupl. + phantom) / N
  Depot cap.    max fractional overshoot across depots
  Distances     mean |stated − actual| / actual across routes
  Total cost    |stated − recomputed| / recomputed
  ============= ===============================================

- ``detail``: dict — per-item breakdown for debugging and drill-down.

The ``score_all()`` convenience function runs all five soft scorers at once and
returns both the individual ``SoftScore`` results and a single aggregate
``max_severity`` value — the worst violation magnitude across all five checks.
This aggregate is the primary metric used by the Monte Carlo profiler to track
how far a stochastic LLM output deviates from feasibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SoftScore:
    """Extended validation result with violation severity magnitude."""

    passed: bool
    score: float  # 0.0-1.0, same semantics as ValidationResult.score
    severity: float  # 0.0 = perfect; unbounded positive = worse
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Vehicle capacity
# ---------------------------------------------------------------------------

def soft_vehicle_capacity(
    routes: list[dict],
    customers: dict[int, dict],
    vehicle_capacity: float,
) -> SoftScore:
    """Measure how much vehicle capacity is exceeded, as a fraction.

    ``severity`` = max over all routes of ``(excess_demand / vehicle_capacity)``.
    E.g. 0.15 means the worst route exceeds capacity by 15 %.
    """
    if not routes:
        return SoftScore(passed=True, score=1.0, severity=0.0)

    max_excess_frac = 0.0
    violations = 0
    per_route: list[dict[str, Any]] = []

    for i, route in enumerate(routes):
        cids = route.get("customer_ids", [])
        load = sum(customers.get(cid, {}).get("demand", 0) for cid in cids)
        excess = max(0.0, load - vehicle_capacity)
        frac = excess / vehicle_capacity if vehicle_capacity > 0 else 0.0
        if excess > 0:
            violations += 1
        max_excess_frac = max(max_excess_frac, frac)
        per_route.append({"route_idx": i, "load": load, "excess_frac": round(frac, 4)})

    valid = len(routes) - violations
    score = valid / len(routes)
    return SoftScore(
        passed=violations == 0,
        score=score,
        severity=round(max_excess_frac, 4),
        detail={"per_route": per_route},
    )


# ---------------------------------------------------------------------------
# 2. Customer coverage
# ---------------------------------------------------------------------------

def soft_customer_coverage(
    routes: list[dict],
    customers: dict[int, dict],
) -> SoftScore:
    """Measure coverage gap: fraction of customers missing, duplicated, or phantom.

    ``severity`` = ``(n_missing + n_duplicated + n_phantom) / n_total_customers``.
    """
    serve_count: dict[int, int] = {}
    for route in routes:
        for cid in route.get("customer_ids", []):
            serve_count[cid] = serve_count.get(cid, 0) + 1

    all_ids = set(customers.keys())
    routed_ids = set(serve_count.keys())

    n_phantom = len(routed_ids - all_ids)
    n_missing = len(all_ids - routed_ids)
    n_duplicated = sum(1 for cid, cnt in serve_count.items() if cnt > 1 and cid in all_ids)

    total = len(all_ids)
    correctly_served = sum(1 for cid in all_ids if serve_count.get(cid, 0) == 1)
    score = correctly_served / total if total > 0 else 1.0

    error_count = n_missing + n_duplicated + n_phantom
    severity = error_count / total if total > 0 else 0.0

    return SoftScore(
        passed=error_count == 0,
        score=score,
        severity=round(severity, 4),
        detail={"n_missing": n_missing, "n_duplicated": n_duplicated, "n_phantom": n_phantom},
    )


# ---------------------------------------------------------------------------
# 3. Depot capacity
# ---------------------------------------------------------------------------

def soft_depot_capacity(
    routes: list[dict],
    customers: dict[int, dict],
    depots: dict[int, dict],
) -> SoftScore:
    """Measure how much depot capacity is exceeded, as a fraction.

    ``severity`` = max over all depots of ``(excess_load / depot_capacity)``.
    """
    depot_load: dict[int, float] = {}
    for route in routes:
        did = route.get("depot_id")
        if did is None:
            continue
        load = sum(
            customers[cid]["demand"]
            for cid in route.get("customer_ids", [])
            if cid in customers
        )
        depot_load[did] = depot_load.get(did, 0.0) + load

    if not depot_load:
        return SoftScore(passed=True, score=1.0, severity=0.0)

    max_excess_frac = 0.0
    violations = 0
    per_depot: list[dict[str, Any]] = []

    for did, load in sorted(depot_load.items()):
        if did not in depots:
            violations += 1
            per_depot.append({"depot_id": did, "error": "not_in_problem"})
            continue
        cap = depots[did]["capacity"]
        excess = max(0.0, load - cap)
        frac = excess / cap if cap > 0 else 0.0
        if excess > 0:
            violations += 1
        max_excess_frac = max(max_excess_frac, frac)
        per_depot.append({
            "depot_id": did, "load": load,
            "capacity": cap, "excess_frac": round(frac, 4),
        })

    total = len(depot_load)
    valid = total - violations
    score = valid / total if total > 0 else 1.0

    return SoftScore(
        passed=violations == 0,
        score=score,
        severity=round(max_excess_frac, 4),
        detail={"per_depot": per_depot},
    )


# ---------------------------------------------------------------------------
# 4. Route distances
# ---------------------------------------------------------------------------

def soft_route_distances(
    routes: list[dict],
    customers: dict[int, dict],
    depots: dict[int, dict],
) -> SoftScore:
    """Measure distance error magnitude.

    ``severity`` = mean over routes of ``|stated - actual| / actual``.
    """
    checkable = [r for r in routes if r.get("stated_distance") is not None]
    if not checkable:
        return SoftScore(passed=True, score=1.0, severity=0.0)

    rel_errors: list[float] = []
    per_route: list[dict[str, Any]] = []
    violations = 0

    for route in checkable:
        did = route.get("depot_id")
        cids = route.get("customer_ids", [])
        stated = float(route["stated_distance"])

        if did not in depots:
            violations += 1
            per_route.append({"depot_id": did, "error": "depot_not_found"})
            continue

        dep_pos = (depots[did]["x"], depots[did]["y"])
        positions: list[tuple[float, float]] = [dep_pos]
        skip = False
        for cid in cids:
            if cid not in customers:
                skip = True
                break
            positions.append((customers[cid]["x"], customers[cid]["y"]))
        if skip:
            violations += 1
            per_route.append({"depot_id": did, "error": "customer_not_found"})
            continue
        positions.append(dep_pos)

        actual = sum(
            math.dist(positions[i], positions[i + 1])
            for i in range(len(positions) - 1)
        )

        rel_err = abs(actual - stated) / actual if actual > 0 else (0.0 if stated == 0 else 1.0)
        if rel_err > 0.1:
            violations += 1
        rel_errors.append(rel_err)
        per_route.append({
            "depot_id": did,
            "stated": round(stated, 4),
            "actual": round(actual, 4),
            "rel_error": round(rel_err, 4),
        })

    total = len(checkable)
    accurate = total - violations
    score = accurate / total if total > 0 else 1.0
    mean_err = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0

    return SoftScore(
        passed=violations == 0,
        score=score,
        severity=round(mean_err, 4),
        detail={"per_route": per_route},
    )


# ---------------------------------------------------------------------------
# 5. Total cost
# ---------------------------------------------------------------------------

def soft_total_cost(
    routes: list[dict],
    depots: dict[int, dict],
    open_depots: list[int],
    stated_total_cost: float,
) -> SoftScore:
    """Measure total cost discrepancy as a fraction.

    ``severity`` = ``|stated - recomputed| / recomputed``.
    """
    fixed_cost = sum(depots[did]["fixed_cost"] for did in open_depots if did in depots)
    routing_cost = sum(
        float(r["stated_distance"])
        for r in routes
        if r.get("stated_distance") is not None
    )
    recomputed = fixed_cost + routing_cost

    if recomputed > 0:
        rel_err = abs(recomputed - stated_total_cost) / recomputed
    else:
        rel_err = 0.0 if stated_total_cost == 0 else 1.0

    passed = rel_err <= 0.1
    score = 1.0 if passed else 0.0

    return SoftScore(
        passed=passed,
        score=score,
        severity=round(rel_err, 4),
        detail={
            "stated": stated_total_cost,
            "recomputed": round(recomputed, 4),
            "fixed_cost": round(fixed_cost, 4),
            "routing_cost": round(routing_cost, 4),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------

@dataclass
class SoftScoreReport:
    """Aggregated results from all five soft scorers."""

    vehicle_capacity: SoftScore
    customer_coverage: SoftScore
    depot_capacity: SoftScore
    route_distances: SoftScore
    total_cost: SoftScore

    @property
    def max_severity(self) -> float:
        """Worst violation magnitude across all five checks."""
        return max(
            self.vehicle_capacity.severity,
            self.customer_coverage.severity,
            self.depot_capacity.severity,
            self.route_distances.severity,
            self.total_cost.severity,
        )

    @property
    def all_passed(self) -> bool:
        """True only when every check passes."""
        return all([
            self.vehicle_capacity.passed,
            self.customer_coverage.passed,
            self.depot_capacity.passed,
            self.route_distances.passed,
            self.total_cost.passed,
        ])

    def as_dict(self) -> dict[str, Any]:
        """Flat dict keyed by validator name — ready for JSON serialisation."""
        return {
            name: {"passed": s.passed, "score": s.score, "severity": s.severity}
            for name, s in [
                ("vehicle_capacity", self.vehicle_capacity),
                ("customer_coverage", self.customer_coverage),
                ("depot_capacity", self.depot_capacity),
                ("route_distances", self.route_distances),
                ("total_cost", self.total_cost),
            ]
        }


def score_all(
    routes: list[dict],
    customers: dict[int, dict],
    depots: dict[int, dict],
    open_depots: list[int],
    vehicle_capacity: float,
    stated_total_cost: float,
) -> SoftScoreReport:
    """Run all five soft scorers and return an aggregated report.

    This is the single entry point used by the scaling analysis and
    prompt sensitivity modules to obtain continuous severity measurements
    for every constraint dimension in one call.
    """
    return SoftScoreReport(
        vehicle_capacity=soft_vehicle_capacity(routes, customers, vehicle_capacity),
        customer_coverage=soft_customer_coverage(routes, customers),
        depot_capacity=soft_depot_capacity(routes, customers, depots),
        route_distances=soft_route_distances(routes, customers, depots),
        total_cost=soft_total_cost(routes, depots, open_depots, stated_total_cost),
    )
