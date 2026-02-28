"""LLM-based LRP solver using the Anthropic API.

Supports three solving strategies:

* **naive** — zero-shot prompt, no heuristic guidance (baseline).
* **cot** — Chain-of-Thought with nearest-neighbour heuristic hints.
* **self_healing** — CoT initial attempt + iterative repair loop driven by
  deterministic validator feedback (max 3 retries).

All API errors and JSON parse failures are caught and re-raised with the raw
response attached so failures are debuggable.
"""

from __future__ import annotations

import re
import sys
import time
from enum import StrEnum
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_agent.prompt_templates import (
    COT_SYSTEM_PROMPT,
    NAIVE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_cot_user_prompt,
    build_naive_user_prompt,
    build_repair_user_prompt,
)
from qa_suite.common.faithfulness import manual_faithfulness_check
from qa_suite.common.fixtures import instance_to_text
from qa_suite.common.schemas import LRPSolution
from qa_suite.deterministic_checks.validators import (
    validate_customer_coverage,
    validate_depot_capacity,
    validate_route_distances,
    validate_total_cost,
    validate_vehicle_capacity,
)


class SolveStrategy(StrEnum):
    """LLM solving strategy for the multi-tier evaluation suite."""

    NAIVE = "naive"
    COT = "cot"
    SELF_HEALING = "self_healing"


def _extract_json(text: str) -> str:
    """Extract the outermost JSON object from LLM response text.

    Handles three common output patterns:

    1. Pure JSON response.
    2. JSON wrapped in triple-backtick markdown fences (with or without 'json' tag).
    3. Prose reasoning followed (or preceded) by JSON — extracts the first
       complete top-level object found by brace-balancing.

    Args:
        text: Raw text from the LLM response.

    Returns:
        The extracted JSON string, stripped of surrounding whitespace.

    Raises:
        ValueError: If no complete brace-balanced block can be found.
    """
    text = text.strip()

    # 1. Strip markdown fences if present
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        # A fenced block may itself contain prose before the JSON;
        # fall through to brace-balancing on the candidate.
        text = candidate

    # 2. Brace-balance scan — find the FIRST '{' and its matching '}'
    #    so we always capture the outermost JSON object.
    first_open = text.find("{")
    if first_open == -1:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape_next = False
    for i in range(first_open, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first_open:i + 1].strip()

    # 3. Truncated response: return from first '{' onward and let Pydantic
    #    report the parse error with the partial JSON visible.
    return text[first_open:].strip()


class LLMSolver:
    """Calls a Claude model to solve an LRP instance.

    Supports three strategies (see :class:`SolveStrategy`):

    * **naive** — zero-shot, no heuristic hints (baseline).
    * **cot** — Chain-of-Thought with nearest-neighbour guidance.
    * **self_healing** — CoT + iterative repair driven by validator feedback.

    Reads ``ANTHROPIC_API_KEY`` from the environment automatically via the
    ``anthropic`` SDK.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 32_768,
        strategy: SolveStrategy = SolveStrategy.NAIVE,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.temperature = temperature
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, dataset: dict, tracer=None) -> tuple[LRPSolution, dict]:
        """Invoke the LLM to solve the given LRP instance.

        Routes to the correct internal method based on ``self.strategy``.

        Args:
            dataset: A dict as returned by ``qa_suite.common.fixtures.load_instance``,
                with keys ``"name"``, ``"customers"``, ``"depots"``,
                ``"vehicle_capacity"``.
            tracer:  Optional OpenTelemetry tracer.  When provided, each
                logical step is wrapped in a span so the call chain is
                visible in Arize Phoenix (or any OTLP-compatible backend).

        Returns:
            A tuple ``(solution, metadata)`` where ``metadata`` includes
            ``elapsed_seconds``, ``input_tokens``, ``output_tokens``,
            ``model``, ``raw_response``, ``strategy``.  For self-healing,
            also ``heal_attempts`` (int) and ``heal_exhausted`` (bool).

        Raises:
            ValueError: If the LLM response cannot be parsed/validated.
            anthropic.APIError: On network or API-level failures.
        """
        if tracer is not None:
            return self._solve_traced(dataset, tracer)

        if self.strategy == SolveStrategy.SELF_HEALING:
            return self._solve_with_healing(dataset)
        return self._solve_single(dataset, self.strategy)

    # ------------------------------------------------------------------
    # Strategy-aware single-shot solve
    # ------------------------------------------------------------------

    def _solve_single(
        self,
        dataset: dict,
        strategy: SolveStrategy | None = None,
    ) -> tuple[LRPSolution, dict]:
        """One-shot solve using the specified strategy's prompts."""
        strat = strategy or self.strategy
        system_prompt, user_prompt = self._build_prompts(strat, dataset)

        t0 = time.time()
        response = self._call_api(user_prompt, system_prompt=system_prompt)
        elapsed = time.time() - t0

        solution, raw_text = self._parse_response(response)

        metadata = {
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": response.model,
            "raw_response": raw_text,
            "strategy": strat.value,
        }
        return solution, metadata

    # ------------------------------------------------------------------
    # Self-healing loop (Tier 3)
    # ------------------------------------------------------------------

    def _solve_with_healing(
        self,
        dataset: dict,
        max_retries: int = 3,
    ) -> tuple[LRPSolution, dict]:
        """CoT initial attempt + up to *max_retries* repair iterations.

        Each iteration: validate → collect violations → send repair prompt.
        """
        # Initial attempt uses CoT prompts.
        solution, meta = self._solve_single(dataset, SolveStrategy.COT)
        meta["strategy"] = SolveStrategy.SELF_HEALING.value

        dataset_text = instance_to_text(dataset)
        n_customers = len(dataset["customers"])
        vc = dataset["vehicle_capacity"]
        total_input_tokens = meta["input_tokens"]
        total_output_tokens = meta["output_tokens"]

        for attempt in range(max_retries):
            violations = self._run_validators(solution, dataset)
            if not violations:
                meta["heal_attempts"] = attempt
                meta["heal_exhausted"] = False
                meta["input_tokens"] = total_input_tokens
                meta["output_tokens"] = total_output_tokens
                return solution, meta

            print(
                f"  [heal] Attempt {attempt + 1}/{max_retries}: "
                f"{len(violations)} violation(s), sending repair prompt...",
                file=sys.stderr,
            )

            repair_prompt = build_repair_user_prompt(
                violations, dataset_text, n_customers, vc,
            )

            try:
                t0 = time.time()
                response = self._call_api(repair_prompt, system_prompt=COT_SYSTEM_PROMPT)
                elapsed = time.time() - t0
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

                solution, raw_text = self._parse_response(response)
                meta["elapsed_seconds"] = round(
                    meta["elapsed_seconds"] + elapsed, 2
                )
                meta["raw_response"] = raw_text
            except KeyboardInterrupt:
                print(
                    "\n  [heal] Interrupted by user — returning last valid solution.",
                    file=sys.stderr,
                )
                meta["heal_attempts"] = attempt
                meta["heal_exhausted"] = False
                meta["input_tokens"] = total_input_tokens
                meta["output_tokens"] = total_output_tokens
                return solution, meta
            except Exception as exc:
                # Parse/validation failure counts as a spent attempt.
                print(
                    f"  [heal] Repair attempt {attempt + 1} failed "
                    f"({type(exc).__name__}): {exc!s:.100}",
                    file=sys.stderr,
                )
                continue

        # Exhausted all retries — return last valid solution.
        meta["heal_attempts"] = max_retries
        meta["heal_exhausted"] = True
        meta["input_tokens"] = total_input_tokens
        meta["output_tokens"] = total_output_tokens
        return solution, meta

    # ------------------------------------------------------------------
    # Deterministic validators (Task 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_validators(solution: LRPSolution, dataset: dict) -> list[str]:
        """Run all 5 deterministic validators + faithfulness; return violations.

        Returns an empty list when everything passes.
        """
        routes = [r.model_dump() for r in solution.routes]
        customers = dataset["customers"]
        depots = dataset["depots"]
        vc = dataset["vehicle_capacity"]

        checks: list[tuple[str, list[str]]] = []

        v = validate_vehicle_capacity(routes, customers, vc)
        if not v.passed:
            checks.append(("Vehicle Capacity", v.violations))

        v = validate_customer_coverage(routes, customers)
        if not v.passed:
            checks.append(("Customer Coverage", v.violations))

        v = validate_depot_capacity(routes, customers, depots)
        if not v.passed:
            checks.append(("Depot Capacity", v.violations))

        v = validate_route_distances(routes, customers, depots)
        if not v.passed:
            checks.append(("Route Distances", v.violations))

        v = validate_total_cost(
            routes, depots, solution.open_depots, solution.total_cost,
        )
        if not v.passed:
            checks.append(("Total Cost", v.violations))

        faith = manual_faithfulness_check(dataset, solution)
        if faith["score"] < 1.0:
            faith_v: list[str] = []
            if faith["phantom_customers"]:
                faith_v.append(f"Phantom customer IDs: {faith['phantom_customers']}")
            if faith["phantom_depots"]:
                faith_v.append(f"Phantom depot IDs: {faith['phantom_depots']}")
            checks.append(("ID Grounding", faith_v))

        # Flatten with validator-name prefixes.
        flat: list[str] = []
        for name, viols in checks:
            for msg in viols:
                flat.append(f"[{name}] {msg}")
        return flat

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompts(
        strategy: SolveStrategy, dataset: dict,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for the given strategy."""
        dataset_text = instance_to_text(dataset)
        n_customers = len(dataset.get("customers", {}))
        if strategy == SolveStrategy.NAIVE:
            return NAIVE_SYSTEM_PROMPT, build_naive_user_prompt(dataset_text)
        # COT and SELF_HEALING initial call both use CoT prompts.
        return COT_SYSTEM_PROMPT, build_cot_user_prompt(dataset_text, n_customers)

    # ------------------------------------------------------------------
    # API call (with tenacity retry for transient errors)
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        )),
        before_sleep=lambda rs: print(  # type: ignore[union-attr]
            f"  [retry] Attempt {rs.attempt_number} failed "
            f"({rs.outcome.exception().__class__.__name__}), "  # type: ignore[union-attr]
            f"retrying in {rs.next_action.sleep:.1f}s...",  # type: ignore[union-attr]
            file=sys.stderr,
        ),
        reraise=True,
    )
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        *,
        temperature: float | None = None,
    ) -> anthropic.types.Message:
        """Send a single API request with automatic retry on transient errors.

        Uses streaming mode, which the Anthropic API requires when max_tokens
        is large enough that the response may take an extended time.
        ``stream.get_final_message()`` returns the same ``Message`` type as
        ``messages.create()``, so all downstream parsing is unchanged.
        """
        temp = temperature if temperature is not None else self.temperature
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if temp is not None:
            kwargs["temperature"] = temp
        with self._client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: anthropic.types.Message,
    ) -> tuple[LRPSolution, str]:
        """Extract JSON from API response, validate, and return (solution, raw_text).

        Raises ValueError on truncation or parse failure.
        """
        if response.stop_reason == "max_tokens":
            raise ValueError(
                f"LLM response truncated (hit {self.max_tokens} token limit). "
                f"Increase max_tokens or simplify the instance."
            )

        raw_text: str = response.content[0].text  # type: ignore[union-attr]
        cleaned = _extract_json(raw_text)

        try:
            solution = LRPSolution.model_validate_json(cleaned)
        except Exception as exc:
            raise ValueError(
                f"LLM response could not be parsed as LRPSolution.\n"
                f"Parse error: {exc}\n"
                f"--- Raw response ---\n{raw_text}\n"
                f"--- Cleaned text ---\n{cleaned}\n"
            ) from exc

        return solution, raw_text

    # ------------------------------------------------------------------
    # Traced path (preserves Phoenix/OTEL spans)
    # ------------------------------------------------------------------

    def _solve_traced(self, dataset: dict, tracer) -> tuple[LRPSolution, dict]:
        """Solve with OpenTelemetry tracing spans wrapping the standard pipeline.

        Delegates to :meth:`_solve_single` or :meth:`_solve_with_healing` so
        that all logic (token accounting, elapsed time, KeyboardInterrupt
        handling) is shared rather than reimplemented.
        """
        with tracer.start_as_current_span("solve_lrp") as root_span:
            root_span.set_attribute("instance", dataset.get("name", "unknown"))
            root_span.set_attribute("strategy", self.strategy.value)
            root_span.set_attribute("n_customers", len(dataset["customers"]))
            root_span.set_attribute("n_depots", len(dataset["depots"]))

            if self.strategy == SolveStrategy.SELF_HEALING:
                solution, metadata = self._solve_with_healing(dataset)
            else:
                solution, metadata = self._solve_single(dataset, self.strategy)

            root_span.set_attribute("elapsed_seconds", metadata["elapsed_seconds"])
            root_span.set_attribute("input_tokens", metadata["input_tokens"])
            root_span.set_attribute("output_tokens", metadata["output_tokens"])
            root_span.set_attribute("model", metadata["model"])

            return solution, metadata
