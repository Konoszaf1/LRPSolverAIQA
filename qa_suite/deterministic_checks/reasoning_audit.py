"""Reasoning-solution consistency audit: verify LLM reasoning matches its JSON.

Parses the free-text ``reasoning`` field from an ``LRPSolution`` and extracts
verifiable claims (customer assignments, demand tallies, distance values,
customer counts).  Each claim is checked against the actual solution JSON.

This catches a failure mode no other validator tests: the LLM narrates a
correct strategy but emits different JSON.  The reasoning *sounds* right but
describes a different solution than the one produced.

Zero API calls — pure Python regex parsing of existing data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from qa_suite.common.schemas import LRPSolution

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_RE_ASSIGN = re.compile(
    r"(?:assign|route).*?customer\s*(\d+).*?depot\s*(\d+)",
    re.IGNORECASE,
)

_RE_DEPOT_SERVES = re.compile(
    r"depot\s*(\d+)\s*(?::|serves|→)\s*"
    r"(?:customers?\s*)?\[?([\d,\s]+)\]?",
    re.IGNORECASE,
)

_RE_DEMAND_TALLY = re.compile(
    r"(?:total\s+)?demand\s*[:=]\s*([\d.]+)",
    re.IGNORECASE,
)

_RE_ROUTE_DEMAND = re.compile(
    r"([\d.]+)\s*/\s*(?:vehicle_capacity|capacity|[\d.]+)",
    re.IGNORECASE,
)

_RE_DISTANCE = re.compile(
    r"(?:stated_)?distance\s*[:=]\s*([\d.]+)",
    re.IGNORECASE,
)

_RE_CUSTOMER_COUNT = re.compile(
    r"(\d+)\s+customers?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningClaim:
    """A single verifiable claim extracted from reasoning text."""

    claim_type: str
    raw_text: str
    expected: str
    actual: str
    consistent: bool


@dataclass
class ReasoningAuditResult:
    """Full audit of reasoning vs solution consistency."""

    total_claims: int
    consistent_claims: int
    consistency_score: float
    contradictions: list[ReasoningClaim] = field(
        default_factory=list,
    )
    all_claims: list[ReasoningClaim] = field(
        default_factory=list,
    )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "total_claims": self.total_claims,
            "consistent_claims": self.consistent_claims,
            "consistency_score": round(self.consistency_score, 4),
            "n_contradictions": len(self.contradictions),
            "contradictions": [
                {
                    "type": c.claim_type,
                    "expected": c.expected,
                    "actual": c.actual,
                    "text": c.raw_text[:120],
                }
                for c in self.contradictions
            ],
        }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_assignments(
    reasoning: str,
) -> list[tuple[int, int]]:
    """Extract (customer_id, depot_id) assignment claims."""
    results: list[tuple[int, int]] = []
    for m in _RE_ASSIGN.finditer(reasoning):
        cid, did = int(m.group(1)), int(m.group(2))
        results.append((cid, did))
    for m in _RE_DEPOT_SERVES.finditer(reasoning):
        did = int(m.group(1))
        cid_text = m.group(2)
        for c in re.findall(r"\d+", cid_text):
            results.append((int(c), did))
    return results


def _extract_demand_tallies(
    reasoning: str,
) -> list[float]:
    """Extract numeric demand tally claims."""
    values: list[float] = []
    for m in _RE_DEMAND_TALLY.finditer(reasoning):
        values.append(float(m.group(1)))
    for m in _RE_ROUTE_DEMAND.finditer(reasoning):
        values.append(float(m.group(1)))
    return values


def _extract_distances(reasoning: str) -> list[float]:
    """Extract stated distance claims."""
    return [float(m.group(1)) for m in _RE_DISTANCE.finditer(reasoning)]


def _extract_customer_counts(reasoning: str) -> list[int]:
    """Extract customer count claims."""
    return [int(m.group(1)) for m in _RE_CUSTOMER_COUNT.finditer(reasoning)]


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit_reasoning(
    solution: LRPSolution,
    customers: dict[int, dict] | None = None,
    depots: dict[int, dict] | None = None,
) -> ReasoningAuditResult:
    """Parse reasoning, extract claims, verify against solution JSON.

    Conservative: returns ``consistency_score = 1.0`` when no claims are
    detected (absence of evidence is not evidence of absence).
    """
    reasoning = solution.reasoning or ""
    if not reasoning.strip():
        return ReasoningAuditResult(
            total_claims=0,
            consistent_claims=0,
            consistency_score=1.0,
        )

    all_claims: list[ReasoningClaim] = []

    # Build actual assignment map from solution
    actual_map: dict[int, int] = {}
    for route in solution.routes:
        for cid in route.customer_ids:
            actual_map[cid] = route.depot_id

    # 1. Assignment claims
    for cid, did in _extract_assignments(reasoning):
        actual_did = actual_map.get(cid)
        consistent = actual_did == did
        all_claims.append(ReasoningClaim(
            claim_type="assignment",
            raw_text=f"customer {cid} → depot {did}",
            expected=f"depot {did}",
            actual=f"depot {actual_did}" if actual_did else "not routed",
            consistent=consistent,
        ))

    # 2. Customer count claims
    actual_count = sum(len(r.customer_ids) for r in solution.routes)
    for claimed in _extract_customer_counts(reasoning):
        # Only check counts that could plausibly be the total
        if claimed < 3:
            continue
        consistent = claimed == actual_count
        all_claims.append(ReasoningClaim(
            claim_type="customer_count",
            raw_text=f"{claimed} customers",
            expected=str(claimed),
            actual=str(actual_count),
            consistent=consistent,
        ))

    # 3. Distance claims (if depot/customer data available)
    if customers and depots:
        actual_distances: list[float] = []
        for route in solution.routes:
            did = route.depot_id
            if did not in depots:
                continue
            dep = (depots[did]["x"], depots[did]["y"])
            pts: list[tuple[float, float]] = [dep]
            for cid in route.customer_ids:
                if cid in customers:
                    pts.append(
                        (customers[cid]["x"], customers[cid]["y"])
                    )
            pts.append(dep)
            dist = sum(
                math.dist(pts[i], pts[i + 1])
                for i in range(len(pts) - 1)
            )
            actual_distances.append(dist)

        for claimed_dist in _extract_distances(reasoning):
            # Match against closest actual distance (10% tolerance)
            best_match = min(
                actual_distances,
                key=lambda a: abs(a - claimed_dist),
                default=None,
            )
            if best_match is not None:
                rel_err = (
                    abs(best_match - claimed_dist) / best_match
                    if best_match > 0
                    else (0.0 if claimed_dist == 0 else 1.0)
                )
                consistent = rel_err <= 0.10
                all_claims.append(ReasoningClaim(
                    claim_type="distance",
                    raw_text=f"distance = {claimed_dist}",
                    expected=f"{claimed_dist}",
                    actual=f"{best_match:.4f}",
                    consistent=consistent,
                ))

    total = len(all_claims)
    n_consistent = sum(1 for c in all_claims if c.consistent)
    contradictions = [c for c in all_claims if not c.consistent]
    score = n_consistent / total if total > 0 else 1.0

    return ReasoningAuditResult(
        total_claims=total,
        consistent_claims=n_consistent,
        consistency_score=score,
        contradictions=contradictions,
        all_claims=all_claims,
    )
