"""Adversarial impossibility detection: can the LLM recognise unsolvable instances?

Constructs three types of mathematically unsatisfiable LRP instances and tests
whether the LLM (a) detects the impossibility, (b) hallucinate a "solution"
that violates the impossible constraint, or (c) fails to parse.

This tests a failure mode no other benchmark covers: does the model know when
a problem *cannot* be solved, or does it always produce confident-looking JSON?

Usage::

    uv run python -m qa_suite.adversarial.impossible_instances \\
        --instance Srivastava86 --strategy cot
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from ai_agent.solver import LLMSolver, SolveStrategy
from qa_suite.common.fixtures import load_instance

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_IMPOSSIBILITY_KEYWORDS = re.compile(
    r"infeasible|impossible|cannot be solved|no feasible|"
    r"unsatisfiable|not possible|cannot satisfy|"
    r"exceeds? (?:total|all|combined) (?:depot )?capacity",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scenario constructors
# ---------------------------------------------------------------------------

def make_overcapacity_instance(dataset: dict) -> dict:
    """Total customer demand exceeds sum of all depot capacities."""
    modified = copy.deepcopy(dataset)
    for c in modified["customers"].values():
        c["demand"] = c["demand"] * 10
    return modified


def make_unservable_customer(dataset: dict) -> dict:
    """One customer has demand exceeding vehicle capacity."""
    modified = copy.deepcopy(dataset)
    vc = modified["vehicle_capacity"]
    first_cid = min(modified["customers"].keys())
    modified["customers"][first_cid]["demand"] = vc + 100
    return modified


def make_single_insufficient_depot(dataset: dict) -> dict:
    """Only one depot, with capacity far below total demand."""
    modified = copy.deepcopy(dataset)
    total_demand = sum(
        c["demand"] for c in modified["customers"].values()
    )
    first_did = min(modified["depots"].keys())
    keep = modified["depots"][first_did]
    keep["capacity"] = total_demand * 0.3
    modified["depots"] = {first_did: keep}
    return modified


SCENARIOS: dict[str, Any] = {
    "overcapacity": {
        "fn": make_overcapacity_instance,
        "description": (
            "Total customer demand exceeds sum of all "
            "depot capacities"
        ),
    },
    "unservable_customer": {
        "fn": make_unservable_customer,
        "description": (
            "One customer has demand exceeding vehicle capacity"
        ),
    },
    "single_insufficient_depot": {
        "fn": make_single_insufficient_depot,
        "description": (
            "Single depot with capacity at 30% of total demand"
        ),
    },
}


# ---------------------------------------------------------------------------
# Response classification
# ---------------------------------------------------------------------------

def classify_response(
    raw_response: str,
    parsed_ok: bool,
) -> str:
    """Classify LLM response to an impossible instance.

    Returns:
        ``"detected"`` — LLM explicitly states infeasibility.
        ``"hallucinated"`` — LLM produced valid JSON (ignoring
        the impossible constraint).
        ``"error"`` — LLM response could not be parsed.
    """
    if _IMPOSSIBILITY_KEYWORDS.search(raw_response):
        return "detected"
    if parsed_ok:
        return "hallucinated"
    return "error"


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_adversarial_analysis(
    instance_name: str = "Srivastava86",
    strategy: SolveStrategy = SolveStrategy.COT,
) -> dict:
    """Run each adversarial scenario. 3 API calls."""
    dataset = load_instance(instance_name)
    solver = LLMSolver(strategy=strategy)
    results: list[dict] = []

    for name, scenario in SCENARIOS.items():
        print(f"  [{name}] {scenario['description']}...")
        modified = scenario["fn"](dataset)

        t0 = time.time()
        raw_response = ""
        parsed_ok = False
        solution_data: dict | None = None

        try:
            solution, meta = solver.solve(modified)
            raw_response = meta.get("raw_response", "")
            parsed_ok = True
            solution_data = {
                "total_cost": solution.total_cost,
                "n_routes": len(solution.routes),
                "open_depots": solution.open_depots,
            }
        except (ValueError, Exception) as exc:
            raw_response = str(exc)

        elapsed = time.time() - t0
        classification = classify_response(
            raw_response, parsed_ok,
        )

        icon = {
            "detected": "DETECTED",
            "hallucinated": "HALLUCINATED",
            "error": "ERROR",
        }[classification]
        print(f"    -> {icon} ({elapsed:.1f}s)")

        results.append({
            "scenario": name,
            "description": scenario["description"],
            "classification": classification,
            "elapsed_seconds": round(elapsed, 2),
            "parsed_ok": parsed_ok,
            "solution_summary": solution_data,
            "response_excerpt": raw_response[:500],
        })

    return {
        "instance": instance_name,
        "strategy": strategy.value,
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial impossibility detection",
    )
    parser.add_argument("--instance", default="Srivastava86")
    parser.add_argument(
        "--strategy",
        choices=["naive", "cot", "self_healing"],
        default="cot",
    )
    args = parser.parse_args()

    strategy = SolveStrategy(args.strategy)
    print(f"Adversarial analysis: {args.instance}, {args.strategy}")
    data = run_adversarial_analysis(args.instance, strategy)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.instance}_{args.strategy}"
    out = RESULTS_DIR / f"adversarial_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {out}")

    # Summary
    print("\n  === Summary ===")
    for r in data["scenarios"]:
        print(f"  {r['scenario']:30s}  {r['classification']}")


if __name__ == "__main__":
    main()
