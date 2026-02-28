"""Cross-model comparison bar chart.

Grouped bar chart: x-axis = model, groups = validator checks, y = pass rate.
One chart per strategy.  Saves ``results/cross_model_comparison.png``.

Usage::

    from dashboard.cross_model_plot import plot_cross_model
    plot_cross_model(Path("results/cross_model_20260228T100000.json"))
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_VALIDATORS = [
    ("vehicle_capacity_valid", "Veh. Capacity"),
    ("customer_coverage_valid", "Cust. Coverage"),
    ("depot_capacity_valid", "Depot Capacity"),
    ("route_distances_valid", "Route Distances"),
    ("total_cost_valid", "Total Cost"),
]
_VALIDATOR_COLOURS = [
    "#1565C0", "#2E7D32", "#00838F", "#E65100", "#B71C1C",
]

_MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-6": "Opus",
}


def plot_cross_model(
    results_path: Path,
    out_path: Path | None = None,
) -> Path:
    """Create grouped bar charts from cross-model results.

    Parameters
    ----------
    results_path:
        Path to ``cross_model_*.json``.
    out_path:
        Where to save the PNG. Defaults to ``results/cross_model_comparison.png``.
    """
    if out_path is None:
        out_path = RESULTS_DIR / "cross_model_comparison.png"

    data = json.loads(results_path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    strategies = data.get("strategies", [])
    instances = data.get("instances", [])
    results = data.get("results", {})

    n_strats = len(strategies)
    fig, axes = plt.subplots(1, max(n_strats, 1), figsize=(7 * n_strats, 6), squeeze=False)
    fig.patch.set_facecolor("#FAFAFA")

    for s_idx, strat in enumerate(strategies):
        ax = axes[0][s_idx]
        ax.set_facecolor("#F7F9FC")

        n_validators = len(_VALIDATORS)
        bar_w = 0.12
        group_gap = 0.20
        group_width = n_validators * bar_w + group_gap
        x_centers = [i * group_width for i in range(len(models))]
        offsets = [(j - (n_validators - 1) / 2) * bar_w for j in range(n_validators)]

        for j, ((v_key, v_label), v_color) in enumerate(zip(_VALIDATORS, _VALIDATOR_COLOURS)):
            for i, model in enumerate(models):
                # Compute pass rate across instances for this model/strategy/validator
                strat_results = results.get(model, {}).get(strat, {})
                n_pass = 0
                n_total = 0
                for inst in instances:
                    r = strat_results.get(inst, {})
                    if r.get("available", False):
                        n_total += 1
                        if r.get(v_key, False):
                            n_pass += 1

                rate = (n_pass / n_total * 100) if n_total > 0 else 0
                xpos = x_centers[i] + offsets[j]
                ax.bar(xpos, rate, width=bar_w * 0.9, color=v_color, alpha=0.85, zorder=3)
                if rate > 0:
                    ax.text(xpos, rate + 1.5, f"{rate:.0f}%",
                            ha="center", va="bottom", fontsize=6,
                            color=v_color, fontweight="bold", rotation=90)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [_MODEL_SHORT.get(m, m) for m in models],
            fontsize=10, fontweight="bold",
        )
        ax.set_ylim(0, 120)
        ax.set_ylabel("Pass Rate (%)", fontsize=10)
        ax.set_title(f"Strategy: {strat}", fontsize=11, fontweight="bold")
        ax.axhline(y=100, color="#2E7D32", lw=1.0, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=c, label=label, alpha=0.85)
        for (_, label), c in zip(_VALIDATORS, _VALIDATOR_COLOURS)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=len(_VALIDATORS),
        fontsize=8, framealpha=0.95, edgecolor="#CFD8DC",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "Cross-Model Validity Comparison",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
