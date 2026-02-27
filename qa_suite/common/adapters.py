"""Adapters that convert between the internal LRP solver types and QA schemas.

This module bridges the gap between ``lrp.models.solution.Solution`` objects
produced by the Cuckoo Search optimiser and the Pydantic schemas defined in
``qa_suite.common.schemas``, enabling QA tests to work with a stable,
serialisable representation of solver output.

Real attribute names (verified against source):
- ``Solution.depots``         — list[DepotNode]
- ``Solution.total_distance`` — float  (objective value)
- ``DepotNode.opened``        — bool   (True when depot is active)
- ``DepotNode.depot_number``  — int    (depot identifier)
- ``DepotNode.vehicles``      — list[VehicleRoute]
- ``VehicleRoute.customers``  — list[CustomerNode]
- ``VehicleRoute.vehicle_distance`` — float
- ``CustomerNode.customer_number``  — int
"""

from __future__ import annotations

import json

from lrp.models.solution import Solution

from qa_suite.common.schemas import LRPSolution, Route


def cuckoo_solution_to_schema(solution: Solution) -> LRPSolution:
    """Convert a Cuckoo Search ``Solution`` to an ``LRPSolution`` schema object.

    Iterates over all depots in the solution. For each *opened* depot, every
    vehicle route is converted to a :class:`~qa_suite.common.schemas.Route`.

    Args:
        solution: A ``Solution`` instance returned by
            ``lrp.algorithms.cuckoo_search.CuckooSearch.optimize``.

    Returns:
        An ``LRPSolution`` populated with all active routes, the list of open
        depot identifiers, and the total objective cost.

    Raises:
        ValueError: If no routes are found (empty or all-closed solution).
    """
    routes: list[Route] = []
    open_depots: list[int] = []

    for depot in solution.depots:
        if not depot.opened:
            continue
        open_depots.append(depot.depot_number)
        for vehicle in depot.vehicles:
            customer_ids = [c.customer_number for c in vehicle.customers]
            if not customer_ids:
                continue  # skip empty vehicles (shouldn't happen in valid solutions)
            routes.append(
                Route(
                    depot_id=depot.depot_number,
                    customer_ids=customer_ids,
                    stated_distance=round(vehicle.vehicle_distance, 6),
                )
            )

    return LRPSolution(
        routes=routes,
        open_depots=open_depots,
        total_cost=round(solution.total_distance, 6),
    )


def schema_to_json(solution: LRPSolution) -> str:
    """Serialise an ``LRPSolution`` to a pretty-printed JSON string.

    Args:
        solution: The schema object to serialise.

    Returns:
        A human-readable JSON string with 2-space indentation.
    """
    return json.dumps(solution.model_dump(), indent=2)
