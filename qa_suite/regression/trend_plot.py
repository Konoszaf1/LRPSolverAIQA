"""Trend plot — validity pass rate over time, one line per tier.

Saves ``results/history/trend.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TIER_COLORS = {"naive": "#e74c3c", "cot": "#e67e22", "self_healing": "#3498db"}
_TIER_LABELS = {"naive": "Naive", "cot": "CoT", "self_healing": "Self-Healing"}
_TIERS = ["naive", "cot", "self_healing"]


def plot_trend(
    history: list[dict],
    out_path: Path | None = None,
) -> Path:
    """Line chart of per-tier pass rate (%) over run index.

    Parameters
    ----------
    history:
        List of history entries (as loaded from ``history.jsonl``).
    out_path:
        Where to save the PNG.  Defaults to ``results/history/trend.png``.

    Returns
    -------
    Path to the saved image.
    """
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "results" / "history" / "trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F7F9FC")

    xs = list(range(len(history)))
    x_labels = [
        e.get("timestamp", "")[:10] or f"Run {i}"
        for i, e in enumerate(history)
    ]

    for tier in _TIERS:
        rates = []
        for entry in history:
            tier_data = entry.get("tiers", {}).get(tier, {})
            rates.append(tier_data.get("pass_rate", 0.0) * 100)

        if any(r > 0 for r in rates):
            ax.plot(
                xs, rates,
                marker="o", markersize=6,
                color=_TIER_COLORS.get(tier, "gray"),
                label=_TIER_LABELS.get(tier, tier),
                linewidth=2.0,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Validity Pass Rate (%)", fontsize=10)
    ax.set_xlabel("Benchmark Run", fontsize=10)
    ax.set_title("Validity Trend Over Time", fontsize=12, fontweight="bold")
    ax.axhline(y=100, color="#2E7D32", lw=1.5, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
