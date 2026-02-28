"""Pytest tests: Multi-tier LLM solver output validated through DeepEval metrics.

Runs the LLM solver on benchmark instances using all three strategies
(naive, cot, self_healing) and validates the returned ``LRPSolution``
with the five deterministic DeepEval metrics.

Both PASS and FAIL outcomes are valuable:
- PASS confirms the LLM satisfied the hard constraints on this run.
- FAIL proves the QA pipeline catches real constraint violations.

Requires ``ANTHROPIC_API_KEY`` to be set in the environment.

Run with::

    PYTHONUTF8=1 pytest qa_suite/deepeval_tests/test_llm_solver.py -v -s --tb=long -m llm
"""

from __future__ import annotations

import json

import pytest
from deepeval import assert_test  # type: ignore[attr-defined]
from deepeval.test_case import LLMTestCase

from ai_agent.solver import LLMSolver, SolveStrategy
from qa_suite.common.adapters import schema_to_json
from qa_suite.common.fixtures import load_instance
from qa_suite.deepeval_tests.metrics import (
    CustomerCoverageMetric,
    DepotCapacityMetric,
    RouteDistanceMetric,
    TotalCostMetric,
    VehicleCapacityMetric,
)

# Mark every test in this module as requiring API access
pytestmark = pytest.mark.llm


# ---------------------------------------------------------------------------
# Fixtures — one solver per strategy, cached per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def naive_solver() -> LLMSolver:
    return LLMSolver(strategy=SolveStrategy.NAIVE)


@pytest.fixture(scope="module")
def cot_solver() -> LLMSolver:
    return LLMSolver(strategy=SolveStrategy.COT)


@pytest.fixture(scope="module")
def self_healing_solver() -> LLMSolver:
    return LLMSolver(strategy=SolveStrategy.SELF_HEALING)


@pytest.fixture(scope="module")
def solver_map(
    naive_solver: LLMSolver,
    cot_solver: LLMSolver,
    self_healing_solver: LLMSolver,
) -> dict[str, LLMSolver]:
    return {
        "naive": naive_solver,
        "cot": cot_solver,
        "self_healing": self_healing_solver,
    }


# ---------------------------------------------------------------------------
# Instances × strategies to test
# ---------------------------------------------------------------------------

LLM_INSTANCES = [
    "Srivastava86",
    # "Gaskell67",   # 21 customers — add when Srivastava86 passes reliably
]

LLM_STRATEGIES = [
    SolveStrategy.NAIVE,
    SolveStrategy.COT,
    SolveStrategy.SELF_HEALING,
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", LLM_STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize("instance_name", LLM_INSTANCES)
def test_llm_solver_deterministic(
    instance_name: str,
    strategy: SolveStrategy,
    solver_map: dict[str, LLMSolver],
) -> None:
    """Run the LLM solver and validate output with all five DeepEval metrics.

    Prints scores and violations before asserting so that failures are
    immediately informative in the pytest output.
    """
    solver = solver_map[strategy.value]
    dataset = load_instance(instance_name)

    tag = f"[{instance_name}][{strategy.value}]"
    print(f"\n{tag} Calling LLM solver...", flush=True)
    solution, meta = solver.solve(dataset)

    print(f"{tag} Model  : {meta['model']}")
    print(f"{tag} Time   : {meta['elapsed_seconds']:.1f}s")
    in_tok = meta["input_tokens"]
    out_tok = meta["output_tokens"]
    print(f"{tag} Tokens : {in_tok} in / {out_tok} out")
    if meta.get("heal_attempts") is not None:
        heals = meta["heal_attempts"]
        exh = meta.get("heal_exhausted")
        print(f"{tag} Heals  : {heals} (exhausted={exh})")
    print(f"{tag} Depots : {solution.open_depots}")
    print(f"{tag} Routes : {len(solution.routes)}")
    print(f"{tag} Cost   : {solution.total_cost:.2f}")

    solution_json = schema_to_json(solution)
    dataset_context = {
        "customers": dataset["customers"],
        "depots": dataset["depots"],
        "vehicle_capacity": dataset["vehicle_capacity"],
    }

    tc = LLMTestCase(
        input=f"Solve the LRP instance: {instance_name} (strategy={strategy.value})",
        actual_output=solution_json,
        context=[json.dumps(dataset_context)],
    )

    metrics = [
        VehicleCapacityMetric(threshold=1.0),
        CustomerCoverageMetric(threshold=1.0),
        DepotCapacityMetric(threshold=1.0),
        RouteDistanceMetric(threshold=1.0),
        TotalCostMetric(threshold=1.0),
    ]

    # Run metrics synchronously and print results before asserting
    for m in metrics:
        m.measure(tc)
        status = "PASS" if m.is_successful() else "FAIL"
        print(
            f"{tag} [{status}] {m.__name__}: "
            f"score={m.score:.2f} — {m.reason}"
        )
        if not m.is_successful():
            print("  Violations captured in reason above.")

    # Now assert — test will fail here if any metric did not pass
    assert_test(test_case=tc, metrics=metrics, run_async=False)
