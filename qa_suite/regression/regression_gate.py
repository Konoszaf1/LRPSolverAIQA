"""Regression gate — validity tracking over time.

Appends benchmark results to a JSONL history file and detects regressions
(a tier that was passing now fails) between consecutive runs.

CLI::

    uv run python -m qa_suite.regression.regression_gate --report
    # Prints Rich table, exits with code 1 if any regression detected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
HISTORY_DIR = RESULTS_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "history.jsonl"

_VALIDATORS = [
    "vehicle_capacity",
    "customer_coverage",
    "depot_capacity",
    "route_distances",
    "total_cost",
]
_TIERS = ["naive", "cot", "self_healing"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegressionEvent:
    """A validator that regressed between two consecutive runs."""

    tier: str
    validator: str
    previous_pass_rate: float
    current_pass_rate: float
    delta: float  # current - previous (negative = regression)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# History I/O
# ---------------------------------------------------------------------------

def load_history(history_path: Path | None = None) -> list[dict]:
    """Load all entries from ``history.jsonl``."""
    path = history_path or HISTORY_FILE
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _aggregate_results(results: list[dict]) -> dict[str, dict]:
    """Aggregate per-instance results into per-tier pass rates.

    Returns ``{tier: {pass_rate, mean_severity, per_validator: {v: rate}}}``
    """
    from collections import defaultdict

    # tier -> validator -> list[bool]
    counts: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        tiers = r.get("llm_solvers", {})
        if not tiers:
            llm = r.get("llm_solver", {})
            if llm:
                strat = llm.get("strategy", "naive")
                tiers = {strat: llm}

        for tier_name, llm in tiers.items():
            if not llm or not llm.get("available", llm.get("vehicle_capacity_valid") is not None):
                continue
            for v in _VALIDATORS:
                passed = llm.get(f"{v}_valid", False)
                counts[tier_name][v].append(bool(passed))

    aggregated: dict[str, dict] = {}
    for tier_name in _TIERS:
        if tier_name not in counts:
            continue
        per_v: dict[str, float] = {}
        all_rates: list[float] = []
        for v in _VALIDATORS:
            bools = counts[tier_name].get(v, [])
            rate = sum(bools) / len(bools) if bools else 0.0
            per_v[v] = round(rate, 4)
            all_rates.append(rate)
        aggregated[tier_name] = {
            "pass_rate": round(sum(all_rates) / len(all_rates), 4) if all_rates else 0.0,
            "mean_severity": 0.0,  # placeholder — could be enriched later
            "per_validator": per_v,
        }
    return aggregated


def append_to_history(
    results: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    history_path: Path | None = None,
) -> dict:
    """Aggregate ``results`` and append one JSONL line to history.

    Idempotent for the same git commit + instance set: if an entry with
    matching ``git_commit`` and ``instances`` already exists, it is replaced.
    """
    path = history_path or HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    instances = sorted({r["instance"] for r in results if "instance" in r})
    git_commit = _git_short_sha()

    entry = {
        "run_id": str(uuid.uuid4()),
        "git_commit": git_commit,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instances": instances,
        "model": model,
        "tiers": _aggregate_results(results),
    }

    # Idempotency: remove prior entry with same commit + instances.
    existing = load_history(path)
    filtered = [
        e for e in existing
        if not (e.get("git_commit") == git_commit and sorted(e.get("instances", [])) == instances)
    ]
    filtered.append(entry)

    # Rewrite (still append-only semantics from the caller's perspective).
    with open(path, "w", encoding="utf-8") as fh:
        for e in filtered:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    return entry


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def detect_regressions(history: list[dict]) -> list[RegressionEvent]:
    """Compare the last two history entries and flag regressions.

    A regression = a (tier, validator) pair whose pass rate decreased.
    Requires at least 2 entries.
    """
    if len(history) < 2:
        return []

    prev = history[-2]
    curr = history[-1]

    events: list[RegressionEvent] = []
    for tier in _TIERS:
        prev_tier = prev.get("tiers", {}).get(tier, {})
        curr_tier = curr.get("tiers", {}).get(tier, {})
        if not prev_tier or not curr_tier:
            continue
        prev_pv = prev_tier.get("per_validator", {})
        curr_pv = curr_tier.get("per_validator", {})
        for v in _VALIDATORS:
            p = prev_pv.get(v, 0.0)
            c = curr_pv.get(v, 0.0)
            if c < p:
                events.append(RegressionEvent(
                    tier=tier,
                    validator=v,
                    previous_pass_rate=p,
                    current_pass_rate=c,
                    delta=round(c - p, 4),
                ))
    return events


# ---------------------------------------------------------------------------
# Rich report
# ---------------------------------------------------------------------------

def print_report(
    history: list[dict],
    regressions: list[RegressionEvent],
) -> None:
    """Print a Rich table showing pass rate changes between the last two runs."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if len(history) < 2:
        console.print("[yellow]Need at least 2 history entries for comparison.[/yellow]")
        return

    prev = history[-2]
    curr = history[-1]

    table = Table(
        title="Regression Gate — Validity Pass Rate Changes",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tier", style="bold")
    for v in _VALIDATORS:
        table.add_column(v.replace("_", " ").title(), justify="center")

    reg_set = {(r.tier, r.validator) for r in regressions}

    for tier in _TIERS:
        prev_pv = prev.get("tiers", {}).get(tier, {}).get("per_validator", {})
        curr_pv = curr.get("tiers", {}).get(tier, {}).get("per_validator", {})
        if not prev_pv and not curr_pv:
            continue
        cells: list[str] = []
        for v in _VALIDATORS:
            p = prev_pv.get(v, 0.0)
            c = curr_pv.get(v, 0.0)
            if (tier, v) in reg_set:
                arrow = f"[red]{c:.0%} ↓[/red]"
            elif c > p:
                arrow = f"[green]{c:.0%} ↑[/green]"
            else:
                arrow = f"{c:.0%} ="
            cells.append(arrow)
        table.add_row(tier, *cells)

    console.print(table)

    if regressions:
        console.print(f"\n[bold red]REGRESSIONS DETECTED: {len(regressions)}[/bold red]")
        for r in regressions:
            console.print(
                f"  {r.tier} / {r.validator}: "
                f"{r.previous_pass_rate:.0%} → {r.current_pass_rate:.0%} ({r.delta:+.0%})"
            )
    else:
        console.print("\n[green]No regressions detected.[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression gate — detect validity regressions across benchmark runs."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print Rich table and exit with code 1 if regressions found.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help=f"Path to history directory (default: {HISTORY_DIR})",
    )
    args = parser.parse_args()

    history_file = (args.history_dir / "history.jsonl") if args.history_dir else HISTORY_FILE
    history = load_history(history_file)

    if not history:
        print(f"No history found at {history_file}")
        print("Run benchmarks with --track-history to build history.")
        sys.exit(0)

    regressions = detect_regressions(history)

    if args.report:
        print_report(history, regressions)
        # Plot trend if we have data
        if len(history) >= 2:
            from qa_suite.regression.trend_plot import plot_trend
            out = plot_trend(history, out_path=history_file.parent / "trend.png")
            print(f"Trend chart saved to {out}")

    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
