"""Root pytest configuration for the LRPSolver test suite.

Handles graceful degradation when optional runtime requirements are absent:

* ``ANTHROPIC_API_KEY`` — required for any test that calls the Claude API.
  Tests marked with ``llm``, ``probabilistic``, ``adversarial``, or
  ``cross_model`` are automatically **skipped** (not failed) when the key is
  missing, with a clear message explaining how to enable them.

* Expensive / intentionally-manual marks (``regression``, ``metamorphic``)
  are left to the developer to invoke explicitly.

Quick-start reference printed at the top of every pytest run (see the header
section below for live output).
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Marks that require an Anthropic API key to run
# ---------------------------------------------------------------------------

_API_KEY_MARKS: frozenset[str] = frozenset(
    {"llm", "probabilistic", "adversarial", "cross_model"}
)

# Marks that are intentionally opt-in (expensive / require specific setup)
# but are NOT silently skipped — the user must pass -m explicitly.
_OPT_IN_MARKS: frozenset[str] = frozenset({"metamorphic", "regression"})


# ---------------------------------------------------------------------------
# Session-level environment checks (run once at collection time)
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return bool(key)


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------

def pytest_report_header(config: pytest.Config) -> list[str]:
    """Emit a status banner at the top of every test run."""
    lines: list[str] = []
    sep = "─" * 60

    lines.append(sep)
    lines.append("LRPSolver Test Suite")
    lines.append(sep)

    if _has_api_key():
        lines.append("  ANTHROPIC_API_KEY : ✓  set  (LLM tests ENABLED)")
        lines.append("  Enabled marks     : llm · probabilistic · adversarial · cross_model")
    else:
        lines.append("  ANTHROPIC_API_KEY : ✗  not set  (LLM tests SKIPPED)")
        lines.append("")
        lines.append("  To run LLM tests, export the key and re-run:")
        lines.append("    set ANTHROPIC_API_KEY=sk-ant-...")
        lines.append("    uv run pytest -m llm -v -s")
        lines.append("")
        lines.append("  Skipping marks    : llm · probabilistic · adversarial · cross_model")

    lines.append("")
    lines.append("  Opt-in marks (always skipped unless passed with -m):")
    lines.append("    metamorphic  — perturbation robustness  (needs API key)")
    lines.append("    regression   — quality gate tracking    (needs API key)")
    lines.append(sep)

    return lines


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip API-key-requiring tests when ANTHROPIC_API_KEY is not set.

    This runs at collection time so the tests show as 's' (skipped) in the
    output rather than 'E' (error), with an explanation attached to each
    skipped test.
    """
    if _has_api_key():
        return  # Nothing to skip — all tests can run

    skip_reason = pytest.mark.skip(
        reason=(
            "ANTHROPIC_API_KEY is not set. "
            "Export the key and re-run with: "
            "uv run pytest -m llm -v -s"
        )
    )

    for item in items:
        item_marks = {m.name for m in item.iter_markers()}
        if item_marks & _API_KEY_MARKS:
            item.add_marker(skip_reason, append=False)


def pytest_terminal_summary(
    terminalreporter,  # type: ignore[type-arg]
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Append a footer to the terminal summary when tests were auto-skipped."""
    skipped = terminalreporter.stats.get("skipped", [])

    # Count how many were skipped due to missing API key (vs other reasons)
    api_skips = [
        r for r in skipped
        if "ANTHROPIC_API_KEY is not set" in getattr(r, "longrepr", ("", "", ""))[2]
    ]

    if not api_skips:
        return

    terminalreporter.write_sep(
        "-",
        f"{len(api_skips)} test(s) skipped — ANTHROPIC_API_KEY not set",
    )
    terminalreporter.write_line(
        "  Run LLM tests with:"
        "  set ANTHROPIC_API_KEY=sk-ant-...  &&  uv run pytest -m llm -v -s"
    )
