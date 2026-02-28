"""DeepEval BaseMetric wrappers for LRP deterministic validators.

Each class wraps one of the four validators from
``qa_suite.deterministic_checks.validators`` as a DeepEval metric so that
standard DeepEval tooling (``evaluate``, ``assert_test``, CI reporting) can
drive the QA pipeline without LLM calls.

Expected ``LLMTestCase`` layout
--------------------------------
- ``actual_output`` — JSON string of the solution produced by the solver or
  agent, with keys ``"routes"``, ``"open_depots"``, ``"total_cost"``.
  Each route has ``"depot_id"``, ``"customer_ids"``, and optionally
  ``"stated_distance"``.
- ``context[0]`` — JSON string of the problem dataset, with keys
  ``"customers"``, ``"depots"``, and ``"vehicle_capacity"``.

JSON key-type note
------------------
When a ``dict[int, ...]`` is serialised via ``json.dumps`` and then
deserialised via ``json.loads``, all keys become strings.  The helper
``_int_keys`` converts them back to ``int`` so the validators receive the
expected types.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from qa_suite.deterministic_checks.validators import (
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_total_cost,
    validate_vehicle_capacity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_keys(d: dict) -> dict:
    """Convert string keys back to int after a JSON round-trip.

    Args:
        d: Dictionary whose keys should be integers but may have been
           serialised as strings by ``json.dumps``.

    Returns:
        A new dictionary with all keys converted to ``int``.
    """
    return {int(k): v for k, v in d.items()}


def _parse_inputs(test_case: LLMTestCase) -> tuple[list[dict], dict, dict, float]:
    """Extract and deserialise routes, customers, depots, and vehicle_capacity.

    Args:
        test_case: A DeepEval ``LLMTestCase`` with ``actual_output`` and
            ``context[0]`` populated as JSON strings.

    Returns:
        Tuple of (routes, customers, depots, vehicle_capacity).
    """
    solution: dict[str, Any] = json.loads(test_case.actual_output)
    dataset: dict[str, Any] = json.loads(test_case.context[0])

    routes: list[dict] = solution.get("routes", [])
    customers: dict[int, dict] = _int_keys(dataset.get("customers", {}))
    depots: dict[int, dict] = _int_keys(dataset.get("depots", {}))
    vehicle_capacity: float = float(dataset.get("vehicle_capacity", 160.0))

    return routes, customers, depots, vehicle_capacity


# ---------------------------------------------------------------------------
# Metric classes
# ---------------------------------------------------------------------------

class VehicleCapacityMetric(BaseMetric):
    """Checks that no vehicle route's total demand exceeds the vehicle capacity.

    Threshold is 1.0 — every route must be within capacity for the metric to
    pass.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.async_mode: bool = False

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Vehicle Capacity"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        routes, customers, _, vehicle_capacity = _parse_inputs(test_case)
        result = validate_vehicle_capacity(routes, customers, vehicle_capacity)
        self.score = result.score
        if result.passed:
            self.reason = (
                f"All {len(routes)} routes within vehicle capacity {vehicle_capacity:.1f}."
            )
        else:
            self.reason = "; ".join(result.violations[:5])
            if len(result.violations) > 5:
                self.reason += f" (and {len(result.violations) - 5} more)"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class CustomerCoverageMetric(BaseMetric):
    """Checks that every customer is served by exactly one route.

    Threshold is 1.0 — complete, duplicate-free coverage is required.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.async_mode: bool = False

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Customer Coverage"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        routes, customers, _, _ = _parse_inputs(test_case)
        result = validate_customer_coverage(routes, customers)
        self.score = result.score
        if result.passed:
            self.reason = f"All {len(customers)} customers served exactly once."
        else:
            self.reason = "; ".join(result.violations[:5])
            if len(result.violations) > 5:
                self.reason += f" (and {len(result.violations) - 5} more)"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class DepotCapacityMetric(BaseMetric):
    """Checks that total demand assigned to each depot does not exceed its capacity.

    Threshold is 1.0 — all used depots must be within their capacity.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.async_mode: bool = False

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Depot Capacity"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        routes, customers, depots, _ = _parse_inputs(test_case)
        result = validate_depot_capacity(routes, customers, depots)
        self.score = result.score
        if result.passed:
            self.reason = "All depot capacities respected."
        else:
            self.reason = "; ".join(result.violations[:5])
            if len(result.violations) > 5:
                self.reason += f" (and {len(result.violations) - 5} more)"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class RouteDistanceMetric(BaseMetric):
    """Checks that stated route distances match recomputed Euclidean distances.

    Threshold is 1.0 — every route with a stated distance must be within
    the relative tolerance (default 10 %) of the recomputed distance.
    """

    def __init__(self, threshold: float = 1.0, tolerance: float = 0.1) -> None:
        self.threshold = threshold
        self._tolerance = tolerance
        self.score: float = 0.0
        self.reason: str = ""
        self.async_mode: bool = False

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Route Distance"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        routes, customers, depots, _ = _parse_inputs(test_case)
        result = validate_route_distances(routes, customers, depots, self._tolerance)
        self.score = result.score
        if result.passed:
            self.reason = f"All route distances within {self._tolerance * 100:.0f}% tolerance."
        else:
            self.reason = "; ".join(result.violations[:5])
            if len(result.violations) > 5:
                self.reason += f" (and {len(result.violations) - 5} more)"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class TotalCostMetric(BaseMetric):
    """Checks that stated total cost matches recomputed fixed costs + route distances.

    Threshold is 1.0 — the cost must be within the relative tolerance.
    """

    def __init__(self, threshold: float = 1.0, tolerance: float = 0.1) -> None:
        self.threshold = threshold
        self._tolerance = tolerance
        self.score: float = 0.0
        self.reason: str = ""
        self.async_mode: bool = False

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Total Cost"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        routes, _, depots, _ = _parse_inputs(test_case)
        solution: dict[str, Any] = json.loads(test_case.actual_output)
        open_depots: list[int] = solution.get("open_depots", [])
        stated_total_cost: float = float(solution.get("total_cost", 0.0))

        result = validate_total_cost(
            routes, depots, open_depots, stated_total_cost, self._tolerance
        )
        self.score = result.score
        if result.passed:
            self.reason = f"Total cost {stated_total_cost:.2f} matches recomputed value."
        else:
            self.reason = "; ".join(result.violations[:5])
            if len(result.violations) > 5:
                self.reason += f" (and {len(result.violations) - 5} more)"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score >= self.threshold
