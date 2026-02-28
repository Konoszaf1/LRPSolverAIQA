"""Unit tests for exception handling in ai_agent/solver.py.

Tests the Claim 6 fix: KeyboardInterrupt must propagate cleanly out of
the self-healing loop, returning the last valid solution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_agent.solver import LLMSolver, SolveStrategy, _extract_json

# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_pure_json(self) -> None:
        result = _extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_markdown_fenced(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_prose_then_json(self) -> None:
        text = 'Here is the solution:\n{"routes": []}'
        assert _extract_json(text) == '{"routes": []}'

    def test_nested_braces(self) -> None:
        text = '{"a": {"b": 1}}'
        assert _extract_json(text) == '{"a": {"b": 1}}'

    def test_braces_inside_strings_ignored(self) -> None:
        text = '{"msg": "hello { world }"}'
        assert _extract_json(text) == '{"msg": "hello { world }"}'

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            _extract_json("no json here")


# ---------------------------------------------------------------------------
# Self-healing KeyboardInterrupt handling
# ---------------------------------------------------------------------------

MOCK_JSON = (
    '{"routes": [{"depot_id": 1, "customer_ids": [1], "stated_distance": 10.0}],'
    ' "open_depots": [1], "total_cost": 110.0,'
    ' "reasoning": "test", "confidence_score": 0.9}'
)


def _mock_response(text: str = MOCK_JSON) -> MagicMock:
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=100, output_tokens=200)
    response.model = "claude-sonnet-4-6"
    return response


DATASET = {
    "name": "test",
    "customers": {1: {"x": 0, "y": 0, "demand": 10}},
    "depots": {1: {"x": 5, "y": 5, "capacity": 100,
                   "fixed_cost": 100, "variable_cost": 1.0}},
    "vehicle_capacity": 160,
}


@patch.object(LLMSolver, "_call_api")
@patch.object(LLMSolver, "_run_validators")
def test_keyboard_interrupt_returns_last_valid(mock_validators, mock_api) -> None:
    """KeyboardInterrupt during a repair call must return the last valid solution."""
    # First call (initial CoT): return valid response
    mock_api.side_effect = [
        _mock_response(),  # initial solve
        KeyboardInterrupt(),  # first repair attempt interrupted
    ]
    # Validators report a violation so the healing loop kicks in
    mock_validators.return_value = ["[Vehicle Capacity] Route 1 overloaded"]

    solver = LLMSolver(strategy=SolveStrategy.SELF_HEALING)
    solution, meta = solver.solve(DATASET)

    # Should return the solution from the initial (successful) call
    assert solution.total_cost == 110.0
    assert meta["heal_exhausted"] is False


@patch.object(LLMSolver, "_call_api")
@patch.object(LLMSolver, "_run_validators")
def test_parse_failure_logs_exception_type(mock_validators, mock_api, capsys) -> None:
    """Parse failures in the healing loop must log the exception type."""
    mock_api.side_effect = [
        _mock_response(),  # initial solve
        _mock_response("not valid json {{{{"),  # broken repair response
        _mock_response("still broken {{{"),  # broken repair response
        _mock_response("nope {{{"),  # broken repair response
    ]
    mock_validators.return_value = ["[Coverage] Missing customer 5"]

    solver = LLMSolver(strategy=SolveStrategy.SELF_HEALING)
    solution, meta = solver.solve(DATASET)

    assert meta["heal_exhausted"] is True
    # Check that exception type is logged to stderr
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
