"""RAGAS faithfulness evaluation for LLM-generated LRP solutions.

Conceptual mapping for RAGAS:
- user_input        : "Solve the <name> LRP instance"
- response          : the LLM solution JSON
- retrieved_contexts: [instance_to_text(dataset)]  — the raw problem data
                       the LLM was given as context

Two modes
---------
1. **Manual faithfulness check** (no API key beyond Anthropic needed):
   Verifies that every ID in the solution actually exists in the dataset.
   Run with just ``ANTHROPIC_API_KEY`` set.

2. **Full RAGAS check** (needs an evaluator LLM — OpenAI or Anthropic via LiteLLM):
   Uses ``ragas.metrics.Faithfulness`` to score whether the solution's claims
   are grounded in the retrieved context.

   Option A — OpenAI evaluator (default RAGAS setup)::

       export OPENAI_API_KEY="sk-..."
       pytest qa_suite/ragas_tests/test_faithfulness.py -v -s -m llm

   Option B — Anthropic via LiteLLM::

       pip install litellm
       # Then in the test, replace llm_factory with:
       # from ragas.llms import LangchainLLMWrapper
       # from langchain_anthropic import ChatAnthropic
       # evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))
"""

from __future__ import annotations

import pytest

from ai_agent.solver import LLMSolver
from qa_suite.common.faithfulness import manual_faithfulness_check  # noqa: F401
from qa_suite.common.fixtures import load_instance, instance_to_text
from qa_suite.common.schemas import LRPSolution


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def srivastava_solution() -> tuple[dict, LRPSolution]:
    """Run the LLM solver once for Srivastava86 and cache the result."""
    dataset = load_instance("Srivastava86")
    solver = LLMSolver()
    solution, meta = solver.solve(dataset)
    print(f"\n[fixture] LLM solved Srivastava86 in {meta['elapsed_seconds']:.1f}s")
    return dataset, solution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_manual_faithfulness(srivastava_solution: tuple[dict, LRPSolution]) -> None:
    """Manual faithfulness: all IDs in the LLM solution must exist in the dataset."""
    dataset, solution = srivastava_solution
    result = manual_faithfulness_check(dataset, solution)

    print(f"\nManual faithfulness score : {result['score']:.2f}")
    print(f"Phantom customers         : {result['phantom_customers']}")
    print(f"Phantom depots            : {result['phantom_depots']}")

    assert result["score"] >= 1.0, (
        f"Faithfulness score {result['score']:.2f} < 1.0. "
        f"Phantom customers: {result['phantom_customers']}, "
        f"Phantom depots: {result['phantom_depots']}"
    )


@pytest.mark.llm
def test_ragas_faithfulness(srivastava_solution: tuple[dict, LRPSolution]) -> None:
    """Full RAGAS faithfulness score (requires OPENAI_API_KEY or LiteLLM setup).

    Skips automatically if ``ragas`` or ``openai`` are not installed, or if
    ``OPENAI_API_KEY`` is not set.
    """
    # Lazy imports — skip gracefully if not installed
    try:
        import os
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness
    except ImportError as exc:
        pytest.skip(f"ragas not installed: {exc}")

    openai_key = __import__("os").environ.get("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set — skipping RAGAS faithfulness test.")

    try:
        from ragas.llms import llm_factory
        from openai import AsyncOpenAI
        evaluator_llm = llm_factory("gpt-4o-mini", openai_client=AsyncOpenAI())
    except Exception as exc:
        pytest.skip(f"Could not initialise RAGAS evaluator LLM: {exc}")

    dataset, solution = srivastava_solution
    name = dataset.get("name", "Unknown")
    n_customers = len(dataset["customers"])
    n_depots = len(dataset["depots"])

    sample = SingleTurnSample(
        user_input=f"Solve the {name} LRP instance with {n_customers} customers and {n_depots} depots.",
        response=solution.model_dump_json(indent=2),
        retrieved_contexts=[instance_to_text(dataset)],
    )

    scorer = Faithfulness(llm=evaluator_llm)
    score = scorer.single_turn_score(sample)

    print(f"\nRAGAS faithfulness score: {score:.2f}")
    assert score >= 0.3, f"Faithfulness score {score:.2f} below threshold 0.3"
