"""Bootstrap confidence intervals for tier-level validity rates.

Uses instance-distribution bootstrap (zero extra API calls) to estimate
uncertainty in pass rates.  Each benchmark instance is treated as an
independent observation from the distribution of LRP problems.

**Statistical note:** This estimates "how reliable is our estimate of
tier validity rate across the instance distribution" — the right question
for deployment decisions ("should I deploy CoT for instances of this
size?").  This is distinct from run-to-run variance of a single LLM
invocation on a specific instance (which would require Monte Carlo).

CLI::

    uv run python -m qa_suite.probabilistic.bootstrap_ci --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

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
class BootstrapResult:
    """Confidence interval for one (strategy, validator) combination."""

    strategy: str
    validator: str
    n_instances: int
    observed_rate: float
    ci_lower: float  # bootstrap percentile
    ci_upper: float  # bootstrap percentile
    ci_width: float  # ci_upper - ci_lower
    n_bootstrap: int
    wilson_lower: float
    wilson_upper: float


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion k/n.

    Closed-form confidence interval that works well even for small n
    and extreme proportions (near 0 or 1).

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    z : float
        Z-score for desired confidence (1.96 for 95%).

    Returns
    -------
    (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(results_dir: Path | str) -> list[dict]:
    """Load all valid benchmark JSON files from the results directory."""
    results_dir = Path(results_dir)
    results: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "n_customers" not in data:
            continue
        results.append(data)
    return results


def _extract_pass_vectors(
    results: list[dict],
) -> dict[tuple[str, str], list[bool]]:
    """Extract binary pass/fail vectors for each (strategy, validator).

    Returns ``{(strategy, validator): [True, False, True, ...]}``, one
    entry per instance.
    """
    vectors: dict[tuple[str, str], list[bool]] = {}

    for r in results:
        tiers = r.get("llm_solvers", {})
        if not tiers:
            llm = r.get("llm_solver", {})
            if llm:
                strat = llm.get("strategy", "naive")
                tiers = {strat: llm}

        for strat in _TIERS:
            llm = tiers.get(strat)
            if not llm or not llm.get("available", llm.get("vehicle_capacity_valid") is not None):
                continue
            for v in _VALIDATORS:
                key = (strat, v)
                passed = bool(llm.get(f"{v}_valid", False))
                vectors.setdefault(key, []).append(passed)

    return vectors


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_validity_ci(
    results_dir: Path | str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> list[BootstrapResult]:
    """Compute bootstrap CIs for each (strategy, validator) pair.

    Loads all ``results/*.json``, extracts binary pass/fail vectors,
    and bootstraps over instances (not over LLM runs) to estimate
    uncertainty in tier-level pass rates.

    Parameters
    ----------
    results_dir:
        Directory containing benchmark JSON files.
    n_bootstrap:
        Number of bootstrap resamples.
    confidence:
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    List of ``BootstrapResult`` objects.
    """
    results_data = _load_results(results_dir)
    if not results_data:
        return []

    vectors = _extract_pass_vectors(results_data)
    alpha = 1.0 - confidence
    z = 1.96  # for 95% Wilson CI (standard)

    output: list[BootstrapResult] = []

    for strat in _TIERS:
        for v in _VALIDATORS:
            key = (strat, v)
            passes = vectors.get(key, [])
            n = len(passes)

            if n == 0:
                continue

            k = sum(passes)
            observed = k / n

            # Wilson CI (always computed)
            w_lo, w_hi = wilson_ci(k, n, z)

            # Bootstrap CI (skip if too few instances)
            if n < 4:
                b_lo, b_hi = w_lo, w_hi
                actual_n_bootstrap = 0
            else:
                rng = random.Random(42)  # reproducible
                boot_rates: list[float] = []
                for _ in range(n_bootstrap):
                    sample = rng.choices(passes, k=n)
                    boot_rates.append(sum(sample) / n)
                boot_rates.sort()
                lo_idx = int(n_bootstrap * alpha / 2)
                hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
                b_lo = boot_rates[max(0, lo_idx)]
                b_hi = boot_rates[min(hi_idx, len(boot_rates) - 1)]
                actual_n_bootstrap = n_bootstrap

            output.append(BootstrapResult(
                strategy=strat,
                validator=v,
                n_instances=n,
                observed_rate=round(observed, 4),
                ci_lower=round(b_lo, 4),
                ci_upper=round(b_hi, 4),
                ci_width=round(b_hi - b_lo, 4),
                n_bootstrap=actual_n_bootstrap,
                wilson_lower=round(w_lo, 4),
                wilson_upper=round(w_hi, 4),
            ))

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bootstrap_ci(
    results: list[BootstrapResult],
    out_path: Path | None = None,
) -> Path:
    """Error bar chart: strategies on x-axis, validators as grouped bars.

    Error bars show 95% bootstrap CI.  CS reference line at 100%.
    Saves ``results/bootstrap_ci.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = RESULTS_DIR / "bootstrap_ci.png"

    val_colours = ["#1565C0", "#2E7D32", "#00838F", "#E65100", "#B71C1C"]
    val_labels = [
        "Veh. Capacity", "Cust. Coverage", "Depot Capacity",
        "Route Distances", "Total Cost",
    ]
    strat_labels = {"naive": "Naive", "cot": "CoT", "self_healing": "Self-Healing"}

    # Index results
    idx: dict[tuple[str, str], BootstrapResult] = {
        (r.strategy, r.validator): r for r in results
    }

    present_strats = [s for s in _TIERS if any((s, v) in idx for v in _VALIDATORS)]
    if not present_strats:
        # Nothing to plot
        return out_path

    n_v = len(_VALIDATORS)
    bar_w = 0.12
    group_gap = 0.20
    group_width = n_v * bar_w + group_gap
    x_centers = [i * group_width for i in range(len(present_strats))]
    offsets = [(j - (n_v - 1) / 2) * bar_w for j in range(n_v)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F7F9FC")

    for j, (v, v_label, v_color) in enumerate(
        zip(_VALIDATORS, val_labels, val_colours),
    ):
        for i, strat in enumerate(present_strats):
            r = idx.get((strat, v))
            if not r:
                continue
            rate = r.observed_rate * 100
            err_lo = (r.observed_rate - r.ci_lower) * 100
            err_hi = (r.ci_upper - r.observed_rate) * 100
            xpos = x_centers[i] + offsets[j]
            ax.bar(
                xpos, rate, width=bar_w * 0.9,
                color=v_color, alpha=0.85, zorder=3,
                yerr=[[err_lo], [err_hi]],
                capsize=3, error_kw=dict(lw=1.5, capthick=1.2, color="#37474F"),
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [strat_labels.get(s, s) for s in present_strats],
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, 120)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_title(
        "Bootstrap 95% Confidence Intervals — Validity Pass Rates by Tier\n"
        "Instance-distribution bootstrap (zero extra API calls)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.axhline(y=100, color="#2E7D32", lw=1.5, linestyle=":", alpha=0.6)
    ax.text(x_centers[-1] + group_width / 2, 101, "CS reference (100%)",
            fontsize=8, color="#2E7D32", va="bottom")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=c, label=label, alpha=0.85)
        for label, c in zip(val_labels, val_colours)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#CFD8DC", ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Rich table
# ---------------------------------------------------------------------------

def print_ci_table(results: list[BootstrapResult]) -> None:
    """Print a Rich table: strategy x validator with observed rate and CIs."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="Bootstrap & Wilson Confidence Intervals — Validity Pass Rates",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Validator")
    table.add_column("n", justify="right")
    table.add_column("Observed", justify="right")
    table.add_column("Bootstrap 95% CI", justify="center")
    table.add_column("Wilson 95% CI", justify="center")

    for r in results:
        obs_str = f"{r.observed_rate:.0%}"
        if r.n_bootstrap > 0:
            boot_str = f"[{r.ci_lower:.0%}, {r.ci_upper:.0%}]"
        else:
            boot_str = "[dim]n<4, Wilson used[/dim]"
        wilson_str = f"[{r.wilson_lower:.0%}, {r.wilson_upper:.0%}]"

        # Colour code the observed rate
        if r.observed_rate >= 0.9:
            obs_str = f"[green]{obs_str}[/green]"
        elif r.observed_rate >= 0.5:
            obs_str = f"[yellow]{obs_str}[/yellow]"
        else:
            obs_str = f"[red]{obs_str}[/red]"

        table.add_row(
            r.strategy, r.validator.replace("_", " ").title(),
            str(r.n_instances), obs_str, boot_str, wilson_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for LLM tier validity rates."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Path to results directory (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default: 0.95).",
    )
    args = parser.parse_args()

    results = bootstrap_validity_ci(
        args.results_dir,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
    )

    if not results:
        print(f"No benchmark results found in {args.results_dir}/")
        print("Run benchmarks first: uv run python run_benchmark.py --all --strategy all")
        sys.exit(0)

    print_ci_table(results)

    out = plot_bootstrap_ci(results)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    main()
