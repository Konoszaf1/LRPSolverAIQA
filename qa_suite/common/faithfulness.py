"""Faithfulness check for LLM-generated LRP solutions.

Verifies that every ID referenced in the solution actually exists in the
problem dataset — i.e., the solution is "grounded" in the input data.
"""

from __future__ import annotations

from typing import Any

from qa_suite.common.schemas import LRPSolution


def manual_faithfulness_check(dataset: dict, solution: LRPSolution) -> dict[str, Any]:
    """Check that every ID in the solution exists in the dataset.

    Validates:
    - Every ``customer_id`` in every route exists in ``dataset["customers"]``.
    - Every ``depot_id`` in every route exists in ``dataset["depots"]``.
    - Every ``depot_id`` in ``open_depots`` exists in ``dataset["depots"]``.

    Args:
        dataset: Dict from ``load_instance`` with ``"customers"`` and
            ``"depots"`` keys (``dict[int, dict]``).
        solution: A validated ``LRPSolution`` from the LLM solver.

    Returns:
        Dict with keys:
        - ``"score"`` (float 0–1): fraction of referenced IDs that exist.
        - ``"phantom_customers"`` (list[int]): customer IDs not in dataset.
        - ``"phantom_depots"`` (list[int]): depot IDs not in dataset.
    """
    known_customers: set[int] = set(dataset["customers"].keys())
    known_depots: set[int] = set(dataset["depots"].keys())

    phantom_customers: list[int] = []
    phantom_depots: list[int] = []
    total_refs = 0
    valid_refs = 0

    for route in solution.routes:
        # Depot ID in route
        total_refs += 1
        if route.depot_id in known_depots:
            valid_refs += 1
        else:
            if route.depot_id not in phantom_depots:
                phantom_depots.append(route.depot_id)

        # Customer IDs in route
        for cid in route.customer_ids:
            total_refs += 1
            if cid in known_customers:
                valid_refs += 1
            else:
                if cid not in phantom_customers:
                    phantom_customers.append(cid)

    # open_depots list
    for did in solution.open_depots:
        total_refs += 1
        if did in known_depots:
            valid_refs += 1
        else:
            if did not in phantom_depots:
                phantom_depots.append(did)

    score = valid_refs / total_refs if total_refs > 0 else 1.0
    return {
        "score": round(score, 4),
        "phantom_customers": sorted(phantom_customers),
        "phantom_depots": sorted(phantom_depots),
    }
