"""Cost-aware tier selection — recommend the cheapest passing LLM strategy.

Given an instance size and optional token budget, loads observed validity
data from ``results/scaling_analysis.json`` (or falls back to hardcoded
priors) and recommends the most cost-efficient tier expected to pass all
validators.

CLI::

    uv run python -m qa_suite.probabilistic.tier_selector --n-customers 55
    uv run python -m qa_suite.probabilistic.tier_selector --n-customers 55 --budget 5000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from qa_suite.common.cost_guard import (
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
    SELF_HEALING_TOKEN_MULTIPLIER,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_PASS_RATE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TierRecommendation:
    """Recommendation output from :func:`recommend_tier`."""

    recommended_strategy: str
    expected_pass_rate: float
    estimated_tokens: int  # total (input + output)
    estimated_cost_usd: float
    rationale: str


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_scaling_data() -> list[dict] | None:
    """Load ``results/scaling_analysis.json`` if it exists."""
    path = RESULTS_DIR / "scaling_analysis.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results: list[dict] = data.get("results", [])
        return results
    except (json.JSONDecodeError, OSError):
        return None


def _load_benchmark_token_data() -> dict[str, dict[str, float]]:
    """Scan ``results/*.json`` for average token usage per strategy.

    Returns ``{strategy: {"avg_input": float, "avg_output": float}}``.
    """
    from collections import defaultdict

    totals: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for path in RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        tiers = data.get("llm_solvers", {})
        if not tiers:
            llm = data.get("llm_solver", {})
            if llm and llm.get("input_tokens"):
                strat = llm.get("strategy", "naive")
                tiers = {strat: llm}
        for strat, llm in tiers.items():
            i_tok = llm.get("input_tokens")
            o_tok = llm.get("output_tokens")
            if i_tok and o_tok:
                totals[strat].append((i_tok, o_tok))

    result: dict[str, dict[str, float]] = {}
    for strat, pairs in totals.items():
        avg_in = sum(p[0] for p in pairs) / len(pairs)
        avg_out = sum(p[1] for p in pairs) / len(pairs)
        result[strat] = {"avg_input": avg_in, "avg_output": avg_out}
    return result


def _hardcoded_priors() -> list[dict]:
    """Fallback validity priors when no scaling data is available.

    Based on observed patterns: naive fails at n>8, CoT at n>55,
    self_healing extends to n~100.
    """
    return [
        {"instance": "small",  "n_customers": 8,   "strategy": "naive",        "all_passed": True},
        {"instance": "small",  "n_customers": 8,   "strategy": "cot",          "all_passed": True},
        {"instance": "small",  "n_customers": 8,   "strategy": "self_healing", "all_passed": True},
        {"instance": "medium", "n_customers": 21,  "strategy": "naive",        "all_passed": False},
        {"instance": "medium", "n_customers": 21,  "strategy": "cot",          "all_passed": True},
        {"instance": "medium", "n_customers": 21,  "strategy": "self_healing", "all_passed": True},
        {"instance": "large",  "n_customers": 55,  "strategy": "naive",        "all_passed": False},
        {"instance": "large",  "n_customers": 55,  "strategy": "cot",          "all_passed": True},
        {"instance": "large",  "n_customers": 55,  "strategy": "self_healing", "all_passed": True},
        {"instance": "xlarge", "n_customers": 100, "strategy": "naive",        "all_passed": False},
        {"instance": "xlarge", "n_customers": 100, "strategy": "cot",          "all_passed": False},
        {"instance": "xlarge", "n_customers": 100, "strategy": "self_healing", "all_passed": False},
    ]


# ---------------------------------------------------------------------------
# Default token estimates when no benchmark data available
# ---------------------------------------------------------------------------

_DEFAULT_TOKENS: dict[str, dict[str, float]] = {
    "naive":        {"avg_input": 1500, "avg_output": 800},
    "cot":          {"avg_input": 2500, "avg_output": 1500},
    "self_healing": {"avg_input": 2500, "avg_output": 1500},
}

# Rough ordering by cost (cheapest first for same model)
_STRATEGY_COST_ORDER = ["naive", "cot", "self_healing"]


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def recommend_tier(
    n_customers: int,
    token_budget: int | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> TierRecommendation:
    """Recommend the cheapest LLM tier expected to pass all validators.

    Algorithm:
      1. Load observed validity from scaling data or hardcoded priors.
      2. For each tier, compute pass rate on instances of size <= n_customers.
      3. Estimate token usage and dollar cost.
      4. Filter by token budget if set.
      5. Pick cheapest tier with pass_rate >= 0.9; else highest pass rate.
    """
    scaling = _load_scaling_data() or _hardcoded_priors()
    token_data = _load_benchmark_token_data() or _DEFAULT_TOKENS

    # Per-strategy pass rates at sizes <= n_customers
    strategy_rates: dict[str, float] = {}
    for strat in _STRATEGY_COST_ORDER:
        relevant = [
            r for r in scaling
            if r.get("strategy") == strat
            and r.get("n_customers", 0) <= n_customers
            and not r.get("skipped", False)
            and not r.get("error")
        ]
        if relevant:
            passed = sum(1 for r in relevant if r.get("all_passed", False))
            strategy_rates[strat] = passed / len(relevant)
        else:
            strategy_rates[strat] = 0.0

    # Token + cost estimates
    c_in = COST_PER_1M_INPUT.get(model, 3.00)
    c_out = COST_PER_1M_OUTPUT.get(model, 15.00)

    candidates: list[tuple[str, float, int, float]] = []  # (strat, rate, tokens, cost)
    for strat in _STRATEGY_COST_ORDER:
        fallback = _DEFAULT_TOKENS.get(strat, {"avg_input": 2000, "avg_output": 1000})
        t = token_data.get(strat, fallback)
        avg_in = t["avg_input"]
        avg_out = t["avg_output"]

        # Self-healing multiplier
        if strat == "self_healing":
            avg_in *= SELF_HEALING_TOKEN_MULTIPLIER
            avg_out *= SELF_HEALING_TOKEN_MULTIPLIER

        total_tokens = int(avg_in + avg_out)
        cost = avg_in * c_in / 1_000_000 + avg_out * c_out / 1_000_000

        # Filter by token budget
        if token_budget is not None and total_tokens > token_budget:
            continue

        rate = strategy_rates.get(strat, 0.0)
        candidates.append((strat, rate, total_tokens, cost))

    if not candidates:
        return TierRecommendation(
            recommended_strategy="naive",
            expected_pass_rate=0.0,
            estimated_tokens=0,
            estimated_cost_usd=0.0,
            rationale="No tier fits within the token budget.",
        )

    # Pick cheapest with rate >= threshold
    passing = [(s, r, t, c) for s, r, t, c in candidates if r >= _PASS_RATE_THRESHOLD]
    if passing:
        best = min(passing, key=lambda x: x[3])  # cheapest
        rationale = (
            f"{best[0]} meets {_PASS_RATE_THRESHOLD:.0%} pass-rate threshold "
            f"at n<={n_customers} and is the cheapest qualifying tier."
        )
    else:
        best = max(candidates, key=lambda x: x[1])  # highest rate
        rationale = (
            f"No tier meets {_PASS_RATE_THRESHOLD:.0%} threshold. "
            f"{best[0]} has the highest observed pass rate ({best[1]:.0%})."
        )

    return TierRecommendation(
        recommended_strategy=best[0],
        expected_pass_rate=best[1],
        estimated_tokens=best[2],
        estimated_cost_usd=round(best[3], 6),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-aware tier selection — recommend the cheapest passing LLM strategy."
    )
    parser.add_argument(
        "--n-customers", type=int, required=True,
        help="Instance size (number of customers).",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Maximum token budget (input + output). Optional.",
    )
    parser.add_argument(
        "--model", type=str, default="claude-haiku-4-5-20251001",
        help="Model ID for cost estimation.",
    )
    args = parser.parse_args()

    rec = recommend_tier(args.n_customers, args.budget, args.model)

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_row("Recommended tier", rec.recommended_strategy)
    table.add_row("Expected pass rate", f"{rec.expected_pass_rate:.0%}")
    table.add_row("Estimated tokens", f"{rec.estimated_tokens:,}")
    table.add_row("Estimated cost", f"${rec.estimated_cost_usd:.4f}")
    table.add_row("Model", args.model)
    table.add_row("Rationale", rec.rationale)

    console.print(Panel(
        table,
        title=f"Tier Recommendation for n_customers={args.n_customers}",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
