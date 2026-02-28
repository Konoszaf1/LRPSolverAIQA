"""Prompt templates for the multi-tier LLM-based LRP solver.

Three tiers of prompt engineering:

* **Naive (zero-shot):** Minimal instructions, no heuristic guidance.
* **CoT + Heuristic:** Step-by-step strategy with capacity tracking.
* **Self-Healing Repair:** Feedback prompt injecting validator violations.

Exports backward-compatible ``SYSTEM_PROMPT`` alias (= ``COT_SYSTEM_PROMPT``)
and ``build_user_prompt`` alias (= ``build_cot_user_prompt``) so existing
callers continue to work.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1 — Naive (zero-shot) prompts
# ---------------------------------------------------------------------------

NAIVE_SYSTEM_PROMPT: str = """\
You are an optimization solver. Given customer and depot data for a
Location-Routing Problem (LRP), output a JSON solution.

## Output format
Output a SINGLE JSON object with this schema:

{
  "routes": [
    {
      "depot_id": <int>,
      "customer_ids": [<int>, ...],
      "stated_distance": <float>
    }
  ],
  "open_depots": [<int>, ...],
  "total_cost": <float>,
  "reasoning": "<string>",
  "confidence_score": <float 0.0-1.0>
}

## Constraints
- Each route's total customer demand must not exceed vehicle_capacity.
- Each depot's total assigned demand must not exceed that depot's capacity.
- Every customer must appear in exactly one route. No skips, no duplicates.
- Only use customer IDs and depot IDs that appear in the problem data.
- stated_distance = Euclidean tour length: depot → c1 → c2 → ... → cn → depot.
  Each leg = sqrt((x2-x1)² + (y2-y1)²). Round to 4 decimal places.
- total_cost = sum(fixed_cost of open depots) + sum(stated_distance of all routes).
- confidence_score: your subjective confidence (0.0 = no idea, 1.0 = certain) that
  this solution satisfies all constraints.

Output ONLY the JSON object. No markdown fences, no explanation outside the JSON.
"""


def build_naive_user_prompt(dataset_text: str) -> str:
    """Wrap a formatted dataset string with minimal solve instructions."""
    return f"""\
Solve the following Location-Routing Problem instance. Return only the JSON
solution — no preamble, no markdown fences.

=== PROBLEM DATA ===
{dataset_text}
===================

Output the JSON solution now.
"""


# ---------------------------------------------------------------------------
# Tier 2 — Chain-of-Thought + Heuristic prompts
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT: str = """\
You are an expert combinatorial optimisation solver specialising in the
Location-Routing Problem (LRP).

## Solving Strategy (follow these steps IN ORDER)

### Step 1: Depot Selection
- Calculate each depot's "reach score": count customers closer to it than to
  any other depot. Prefer depots that serve many nearby customers relative to
  their fixed cost.
- Open the minimum set of depots that can cover all customer demand without
  breaching any depot's capacity.

### Step 2: Customer Assignment (Nearest-Neighbour Clustering)
- Assign each customer to the nearest open depot.
- After assignment, verify that no depot's total assigned demand exceeds its
  capacity. If it does, reassign the furthest-out customer to the next nearest
  depot with remaining capacity.

### Step 3: Route Construction
- For each depot, build routes using a nearest-neighbour insertion:
  1. Start at the depot. Pick the nearest unrouted customer.
  2. Add it to the current route if doing so does not exceed vehicle_capacity.
  3. If adding the next customer would exceed capacity, close this route
     (return to depot) and start a new route from the depot.
  4. Repeat until all customers assigned to this depot are routed.

### Step 4: Capacity Verification (MANDATORY — do this before outputting JSON)
For EACH route, write out explicitly in your reasoning:
  Route R (depot D): customers [c1, c2, ...] → demands [d1, d2, ...] → total = X / vehicle_capacity
If total > vehicle_capacity for ANY route, you MUST split that route.

### Step 5: Distance Calculation
For each route, compute:
  stated_distance = dist(depot, c1) + dist(c1, c2) + ... + dist(cn, depot)
  where dist(a, b) = sqrt((ax - bx)² + (ay - by)²)
Show the leg-by-leg calculation in your reasoning.

### Step 6: Cost Calculation
total_cost = Σ(fixed_cost for each open depot) + Σ(stated_distance for all routes)
Show this sum explicitly in your reasoning.

## Output Format
First, write your step-by-step reasoning inside a "reasoning" field.
Then output a SINGLE JSON object with this schema:

{
  "routes": [
    {
      "depot_id": <int>,
      "customer_ids": [<int>, ...],
      "stated_distance": <float>
    }
  ],
  "open_depots": [<int>, ...],
  "total_cost": <float>,
  "reasoning": "<your step-by-step work from above>",
  "confidence_score": <float 0.0-1.0>
}

Also include a "confidence_score" (0.0-1.0) representing how confident you are that
every constraint is satisfied.

CRITICAL RULES:
- Every customer must appear in EXACTLY one route. Count them.
- Route demand must NEVER exceed vehicle_capacity.
- Depot aggregate demand must NEVER exceed depot capacity.
- Only use IDs from the problem data. Do NOT invent new ones.
- Output ONLY the JSON object. No text outside it.
"""


def build_cot_user_prompt(dataset_text: str, n_customers: int = 0) -> str:
    """Wrap a dataset string with CoT solve instructions.

    For large instances (>30 customers), omits the verbose leg-by-leg
    distance requirement to keep reasoning within token limits.
    """
    if n_customers > 30:
        return f"""\
Solve the following Location-Routing Problem. Show your work in the
"reasoning" field — depot selection rationale and per-route capacity tallies.
For distances, compute them accurately but you do NOT need to show every
leg-by-leg calculation — just state each route's total stated_distance.

IMPORTANT: Keep your reasoning CONCISE. Focus on correctness, not verbosity.

=== PROBLEM DATA ===
{dataset_text}
===================

Follow the 6-step strategy from your instructions. Output the JSON now.
"""
    return f"""\
Solve the following Location-Routing Problem. You MUST show your work in the
"reasoning" field — depot selection rationale, per-route capacity tallies, and
leg-by-leg distance calculations.

=== PROBLEM DATA ===
{dataset_text}
===================

Follow the 6-step strategy from your instructions. Output the JSON now.
"""


# ---------------------------------------------------------------------------
# Tier 3 — Self-Healing repair prompt
# ---------------------------------------------------------------------------

def build_repair_user_prompt(
    violations: list[str],
    dataset_text: str,
    n_customers: int,
    vehicle_capacity: float,
) -> str:
    """Build the repair prompt injecting validator failures back to the LLM.

    Args:
        violations: Flat list of violation strings from ``_run_validators``,
            each prefixed with the validator name (e.g. ``"[Vehicle Capacity] ..."``).
        dataset_text: Full problem data text for context.
        n_customers: Expected total customer count for the coverage sanity check.
        vehicle_capacity: Per-vehicle capacity limit.
    """
    numbered = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(violations))
    return f"""\
Your previous solution FAILED the following automated quality checks:

=== VALIDATION ERRORS ===
{numbered}
=========================

The original problem data is repeated below for reference:

=== PROBLEM DATA ===
{dataset_text}
===================

Instructions:
1. Read each violation carefully.
2. In your reasoning, explain what went wrong and how you will fix each one.
3. Output a COMPLETE corrected JSON solution (not a partial patch).
4. Re-verify:
   - Count all customers to ensure full coverage (expect {n_customers} total).
   - Re-tally demand per route vs vehicle_capacity ({vehicle_capacity}).
   - Recalculate stated_distance for any modified routes.
   - Recalculate total_cost = sum(depot fixed costs) + sum(route distances).
   - Include your updated confidence_score (0.0-1.0).

Output ONLY the corrected JSON object.
"""


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

#: Legacy alias — points to the CoT system prompt (used by existing callers).
SYSTEM_PROMPT: str = COT_SYSTEM_PROMPT

#: Legacy alias for the user-prompt builder.
build_user_prompt = build_cot_user_prompt
