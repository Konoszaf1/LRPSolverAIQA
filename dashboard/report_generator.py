"""Benchmark report generator — reads ``results/*.json`` and writes Markdown.

Reads every JSON file in the ``results/`` directory that was produced by
``run_benchmark.py`` and generates a comprehensive Markdown report at
``results/BENCHMARK_REPORT.md``.

Supports both single-tier (``llm_solver``) and multi-tier (``llm_solvers``)
result formats.

Usage::

    python -m dashboard.report_generator
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPORT_PATH = RESULTS_DIR / "BENCHMARK_REPORT.md"

# Ordered list of LLM strategies for column display.
_STRATEGY_ORDER = ["naive", "cot", "self_healing"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    """Load all valid benchmark JSON files from the results directory."""
    results: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "BENCHMARK_REPORT.md":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "n_customers" not in data:
            continue
        results.append(data)
    return results


def _pass_icon(passed: bool | None) -> str:
    if passed is None:
        return "—"
    return "PASS" if passed else "FAIL"


def _score_str(score: float | None) -> str:
    return "—" if score is None else f"{score:.2f}"


def _get_llm_tiers(r: dict) -> dict[str, dict]:
    """Extract LLM tier results, supporting both old and new formats."""
    # New multi-tier format.
    if "llm_solvers" in r and r["llm_solvers"]:
        return r["llm_solvers"]
    # Legacy single-tier format.
    llm = r.get("llm_solver", {})
    if llm:
        strat = llm.get("strategy", "naive")
        return {strat: llm}
    return {}


def _cs_checks_count(cs: dict) -> int:
    return sum([
        cs.get("vehicle_capacity_valid", False),
        cs.get("customer_coverage_valid", False),
        cs.get("depot_capacity_valid", False),
        cs.get("route_distances_valid", False),
        cs.get("total_cost_valid", True),
    ])


def _llm_checks_count(llm: dict) -> int:
    return sum([
        llm.get("vehicle_capacity_valid", False),
        llm.get("customer_coverage_valid", False),
        llm.get("depot_capacity_valid", False),
        llm.get("route_distances_valid", False),
        llm.get("total_cost_valid", False),
        (llm.get("faithfulness_score") or 0) >= 1.0,
    ])


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _summary_table(results: list[dict]) -> str:
    """Build the main summary table."""
    # Determine which strategies are present across all results.
    all_strats: set[str] = set()
    for r in results:
        all_strats.update(_get_llm_tiers(r).keys())
    strat_cols = [s for s in _STRATEGY_ORDER if s in all_strats]
    if not strat_cols:
        strat_cols = sorted(all_strats)

    lines = [
        "## Results Summary",
        "",
    ]

    # Header row.
    header = "| Instance | Cust | CS Cost | CS Checks |"
    sep = "|----------|------|---------|-----------|"
    for s in strat_cols:
        header += f" {s.title()} Checks | {s.title()} Cost | {s.title()} Time |"
        sep += "-------------|------------|------------|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        cs = r.get("cuckoo_search", {})
        tiers = _get_llm_tiers(r)

        cs_cost = f"{cs['total_cost']:.2f}" if cs.get("available") else "—"
        cs_checks = f"{_cs_checks_count(cs)}/5" if cs.get("available") else "SKIP"

        row = f"| {r['instance']} | {r['n_customers']} | {cs_cost} | {cs_checks} |"

        for s in strat_cols:
            llm = tiers.get(s, {})
            if not llm or not llm.get("available"):
                row += " — | — | — |"
                continue
            n_pass = _llm_checks_count(llm)
            icon = "OK" if n_pass == 6 else "FAIL"
            cost = f"{llm['total_cost']:.2f}"
            t = f"{llm['time_seconds']:.1f}s"
            heal = ""
            if llm.get("heal_attempts"):
                heal = f" ({llm['heal_attempts']}h)"
            row += f" {n_pass}/6 {icon} | {cost} | {t}{heal} |"

        lines.append(row)

    return "\n".join(lines)


def _instance_detail(r: dict) -> str:
    """Generate per-instance detail section."""
    lines: list[str] = []
    cs = r.get("cuckoo_search", {})
    tiers = _get_llm_tiers(r)
    name = r["instance"]
    n_c = r["n_customers"]
    n_d = r["n_depots"]

    lines.append(f"### {name} ({n_c} customers, {n_d} depots)")
    lines.append("")

    # --- Cuckoo Search ---
    if cs.get("available"):
        lines.append("**Cuckoo Search Results:**")
        lines.append(f"- Total Cost: {cs['total_cost']:.2f}")
        lines.append(f"- Routes: {cs['n_routes']}")
        lines.append(f"- Time: {cs['time_seconds']:.1f}s")
        lines.append("")
        lines.append("| Check | Score | Status |")
        lines.append("|-------|-------|--------|")
        for key, label in [
            ("vehicle_capacity", "Vehicle Capacity"),
            ("customer_coverage", "Customer Coverage"),
            ("depot_capacity", "Depot Capacity"),
            ("route_distances", "Route Distances"),
            ("total_cost", "Total Cost"),
        ]:
            score = _score_str(cs.get(f"{key}_score"))
            status = _pass_icon(cs.get(f"{key}_valid"))
            lines.append(f"| {label} | {score} | {status} |")
        lines.append("")
    else:
        reason = cs.get("skip_reason", "")
        lines.append(f"**Cuckoo Search:** SKIPPED — {reason}")
        lines.append("")

    # --- LLM tiers ---
    for strat in _STRATEGY_ORDER:
        llm = tiers.get(strat)
        if not llm:
            continue

        lines.append(f"**LLM Solver [{strat.title()}]:**")
        if not llm.get("available"):
            lines.append(f"- ERROR: {llm.get('error', '')}")
            lines.append("")
            continue

        lines.append(f"- Model: {llm.get('model', '—')}")
        lines.append(f"- Strategy: {llm.get('strategy', strat)}")
        lines.append(f"- Total Cost: {llm['total_cost']:.2f}")
        lines.append(f"- Routes: {llm['n_routes']}")
        lines.append(f"- Time: {llm['time_seconds']:.1f}s")
        if llm.get("heal_attempts") is not None:
            lines.append(
                f"- Heal Attempts: {llm['heal_attempts']} "
                f"(exhausted={llm.get('heal_exhausted', False)})"
            )
        lines.append("")

        lines.append("| Check | Score | Status |")
        lines.append("|-------|-------|--------|")
        for key, label in [
            ("vehicle_capacity", "Vehicle Capacity"),
            ("customer_coverage", "Customer Coverage"),
            ("depot_capacity", "Depot Capacity"),
            ("route_distances", "Route Distances"),
            ("total_cost", "Total Cost"),
        ]:
            score = _score_str(llm.get(f"{key}_score"))
            status = _pass_icon(llm.get(f"{key}_valid"))
            lines.append(f"| {label} | {score} | {status} |")

        faith_score = llm.get("faithfulness_score")
        if faith_score is not None:
            faith_icon = "PASS" if faith_score >= 1.0 else "FAIL"
            lines.append(f"| Faithfulness | {faith_score:.2f} | {faith_icon} |")
        lines.append("")

        # Violations
        violations = llm.get("route_distance_violations", [])
        phantoms_c = llm.get("phantom_customers", [])
        phantoms_d = llm.get("phantom_depots", [])
        if violations or phantoms_c or phantoms_d:
            lines.append("**Violations:**")
            for v in violations[:5]:
                lines.append(f"- {v}")
            if phantoms_c:
                lines.append(f"- Phantom customer IDs: {phantoms_c}")
            if phantoms_d:
                lines.append(f"- Phantom depot IDs: {phantoms_d}")
            lines.append("")

    return "\n".join(lines)


def _key_findings(results: list[dict]) -> str:
    """Generate key findings section with tier-by-tier analysis."""
    lines = ["## Key Findings", ""]

    # Per-tier stats.
    tier_stats: dict[str, dict] = {}
    for r in results:
        tiers = _get_llm_tiers(r)
        for strat, llm in tiers.items():
            if not llm.get("available"):
                continue
            if strat not in tier_stats:
                tier_stats[strat] = {
                    "pass_count": 0, "total": 0, "times": [], "heals": [],
                }
            stats = tier_stats[strat]
            stats["total"] += 1
            n_pass = _llm_checks_count(llm)
            if n_pass == 6:
                stats["pass_count"] += 1
            stats["times"].append(llm["time_seconds"])
            if llm.get("heal_attempts") is not None:
                stats["heals"].append(llm["heal_attempts"])

    if tier_stats:
        lines.append("### Tier Comparison")
        lines.append("")
        lines.append("| Strategy | Instances | All-Pass Rate | Avg Time |")
        lines.append("|----------|-----------|---------------|----------|")
        for strat in _STRATEGY_ORDER:
            s = tier_stats.get(strat)
            if not s:
                continue
            rate = f"{s['pass_count']}/{s['total']}"
            avg_t = f"{sum(s['times']) / len(s['times']):.1f}s" if s["times"] else "—"
            lines.append(f"| {strat.title()} | {s['total']} | {rate} | {avg_t} |")
        lines.append("")

    # Self-healing convergence.
    heals = tier_stats.get("self_healing", {}).get("heals", [])
    if heals:
        avg_heals = sum(heals) / len(heals)
        lines.append(
            f"**Self-Healing Convergence:** "
            f"avg {avg_heals:.1f} repair iterations per instance"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report() -> None:
    results = _load_results()
    if not results:
        print(f"No benchmark results found in {RESULTS_DIR}/")
        print("Run: python run_benchmark.py <instance_name>")
        return

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = [
        "# AIQA Benchmark Report",
        "",
        f"> Generated: {now}",
        "> Framework: Multi-Tier LLM Benchmarking & Validation (AIQA)",
        "",
        _summary_table(results),
        "",
        "---",
        "",
        "## Per-Instance Details",
        "",
    ]

    for r in results:
        sections.append(_instance_detail(r))
        sections.append("---")
        sections.append("")

    sections.append(_key_findings(results))

    report = "\n".join(sections)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")
    print(f"Instances processed: {[r['instance'] for r in results]}")


if __name__ == "__main__":
    generate_report()
