"""Pytest test suite: Cuckoo Search solver baseline via DeepEval metrics.

Runs the real LRP solver on benchmark instances and validates the output
with the three deterministic DeepEval metrics.  The Cuckoo Search solver
always produces constraint-feasible solutions, so every metric is expected
to score 1.0.

Instance notes
--------------
- **Ch69** (100 customers, 10 depots) — the standard large benchmark instance.
- **Srivastava86** (8 customers, 2 depots) — small instance with float-formatted
  demands (e.g. ``112.0``); the data loader handles these via ``int(float(...))``.

Solver invocation pattern (from ``main.py``)
---------------------------------------------
  sol = Solution(customers, depots)
  sol.vehicle_capacity = VEHICLE_CAPACITY
  sol.depots = [d for d in sol.depots if d.depot_number in active_ids]
  sol.build_distances()
  assign_depots(sol.customers)
  for depot in sol.depots:
      build_vehicle_routes(depot, VEHICLE_CAPACITY)
  sol.calculate_total_distance()
"""

from __future__ import annotations

import json
from itertools import combinations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from lrp.algorithms.cuckoo_search import CuckooSearch
from lrp.algorithms.nearest_neighbor import assign_depots, build_vehicle_routes
from lrp.config import CuckooConfig
from lrp.io.data_loader import load_customers, load_depots
from lrp.models.solution import Solution
from qa_suite.common.adapters import cuckoo_solution_to_schema, schema_to_json
from qa_suite.common.fixtures import DATA_DIR, INSTANCES, load_instance
from qa_suite.deepeval_tests.metrics import (
    CustomerCoverageMetric,
    DepotCapacityMetric,
    RouteDistanceMetric,
    TotalCostMetric,
    VehicleCapacityMetric,
)

# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------

def _run_cuckoo(instance_name: str, num_solutions: int = 3, num_iterations: int = 10):
    """Run the Cuckoo Search solver on a named benchmark instance.

    Uses the real invocation pattern from ``main.py``.  Returns the best
    solution converted to an ``LRPSolution`` schema object.

    Args:
        instance_name: Key in the ``INSTANCES`` registry (e.g. ``"Ch69"``).
        num_solutions: Population size for initial solutions.
        num_iterations: Number of Cuckoo Search iterations.

    Returns:
        Tuple of (schema_solution, fixture_dataset_dict).
    """
    cli_file, dep_file, vc = INSTANCES[instance_name]
    customers = load_customers(DATA_DIR / cli_file)
    depots = load_depots(DATA_DIR / dep_file)

    # Build initial solution population (same pattern as main.py)
    all_ids = tuple(range(1, len(depots) + 1))
    combos = list(combinations(all_ids, len(depots)))[:num_solutions]

    solutions = []
    for combo in combos:
        sol = Solution(customers, depots)
        sol.vehicle_capacity = vc
        sol.depots = [d for d in sol.depots if d.depot_number in combo]
        sol.build_distances()
        assign_depots(sol.customers)
        for depot in sol.depots:
            build_vehicle_routes(depot, vc)
        sol.calculate_total_distance()
        solutions.append(sol)

    config = CuckooConfig(num_solutions=num_solutions, num_iterations=num_iterations)
    best = CuckooSearch(config).optimize(solutions)
    schema_sol = cuckoo_solution_to_schema(best)

    # Also load fixture dataset for the context (so depots/customers are in
    # the qa_suite format rather than the lrp internal format)
    fixture_ds = load_instance(instance_name)

    return schema_sol, fixture_ds


# ---------------------------------------------------------------------------
# Test parametrisation — add more instances here as the data_loader is fixed
# ---------------------------------------------------------------------------

SOLVER_INSTANCES = [
    "Ch69",
    "Srivastava86",
    # "Gaskell67",   # integer demands — safe to add later
    # "Or76",        # integer demands — safe to add later
]


@pytest.mark.parametrize("instance_name", SOLVER_INSTANCES)
def test_cuckoo_search_deterministic(instance_name: str) -> None:
    """Validate Cuckoo Search output for *instance_name* with all three metrics.

    The solver is constraint-aware, so all metrics are expected to achieve
    score=1.0.
    """
    schema_sol, fixture_ds = _run_cuckoo(instance_name)

    solution_json = schema_to_json(schema_sol)

    # Build the dataset context JSON (use fixture_ds which has int-keyed dicts;
    # json.dumps converts int keys to strings automatically — the metrics
    # handle the round-trip with _int_keys)
    dataset_context = {
        "customers": fixture_ds["customers"],
        "depots": fixture_ds["depots"],
        "vehicle_capacity": fixture_ds["vehicle_capacity"],
    }
    context_json = json.dumps(dataset_context)

    tc = LLMTestCase(
        input=f"Solve the LRP instance: {instance_name}",
        actual_output=solution_json,
        context=[context_json],
    )

    metrics = [
        VehicleCapacityMetric(threshold=1.0),
        CustomerCoverageMetric(threshold=1.0),
        DepotCapacityMetric(threshold=1.0),
        RouteDistanceMetric(threshold=1.0),
        TotalCostMetric(threshold=1.0),
    ]

    assert_test(test_case=tc, metrics=metrics, run_async=False)
