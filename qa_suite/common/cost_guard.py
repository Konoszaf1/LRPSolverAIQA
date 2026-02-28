"""Cost tracking and budget enforcement for LLM API calls.

Provides a ``CostGuard`` that accumulates spend from token counts and
enforces a hard per-model dollar cap.  Also exposes the Claude pricing
constants used by the tier selector and cross-model comparison modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Claude pricing per 1M tokens (approximate, April 2025)
COST_PER_1M_INPUT: dict[str, float] = {
    "claude-haiku-4-5-20251001": 0.80,
    "claude-sonnet-4-6": 3.00,
    "claude-opus-4-6": 15.00,
}
COST_PER_1M_OUTPUT: dict[str, float] = {
    "claude-haiku-4-5-20251001": 4.00,
    "claude-sonnet-4-6": 15.00,
    "claude-opus-4-6": 75.00,
}

# Self-healing does 1 CoT + up to 3 repairs; avg ~1.5 repairs observed.
SELF_HEALING_TOKEN_MULTIPLIER: float = 2.5


def token_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute dollar cost for a single API call."""
    c_in = COST_PER_1M_INPUT.get(model, 3.00)
    c_out = COST_PER_1M_OUTPUT.get(model, 15.00)
    return input_tokens * c_in / 1_000_000 + output_tokens * c_out / 1_000_000


@dataclass
class CostGuard:
    """Track cumulative LLM spend and enforce a hard dollar cap.

    Usage::

        guard = CostGuard("claude-haiku-4-5-20251001", max_cost_usd=2.00)
        if guard.can_afford():
            # ... make API call ...
            guard.record(meta["input_tokens"], meta["output_tokens"])
        if guard.budget_exhausted:
            print("Budget reached")
    """

    model: str
    max_cost_usd: float = 2.00
    _total_input: int = field(default=0, init=False, repr=False)
    _total_output: int = field(default=0, init=False, repr=False)
    _total_cost: float = field(default=0.0, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def budget_exhausted(self) -> bool:
        return self._total_cost >= self.max_cost_usd

    @property
    def calls(self) -> int:
        return self._calls

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate cost from one API call."""
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cost += token_cost_usd(self.model, input_tokens, output_tokens)
        self._calls += 1

    def can_afford(
        self,
        estimated_input: int = 2000,
        estimated_output: int = 1000,
    ) -> bool:
        """Return True if the next call is expected to stay within budget."""
        projected = self._total_cost + token_cost_usd(
            self.model, estimated_input, estimated_output,
        )
        return projected <= self.max_cost_usd
