"""Tests for helper utilities in agents/multi_agent.py.

These functions are pure string/array operations with no LLM calls,
so they run entirely offline and deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.multi_agent import (
    _extract_code,
    _strip_thinking,
    _truncate_to_valid_function,
    _parse_hypotheses,
    _format_task_description,
    _format_training_examples,
    _format_error_info,
    _format_diff,
    _grid_to_str,
    _diff_summary,
)


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def test_removes_think_tags(self):
        text = "<think>internal reasoning</think>actual answer"
        assert _strip_thinking(text) == "actual answer"

    def test_no_tags_unchanged(self):
        text = "plain response"
        assert _strip_thinking(text) == "plain response"

    def test_multiline_think_stripped(self):
        text = "<think>\nline1\nline2\n</think>result"
        assert _strip_thinking(text) == "result"

    def test_multiple_think_blocks(self):
        text = "<think>a</think>X<think>b</think>Y"
        assert _strip_thinking(text) == "XY"

    def test_empty_string(self):
        assert _strip_thinking("") == ""

    def test_only_think_block(self):
        assert _strip_thinking("<think>hidden</think>") == ""


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_python_fenced_block(self):
        text = "Here is the code:\n```python\ndef transform(g):\n    return g\n```"
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_plain_fenced_block_with_def(self):
        text = "```\ndef transform(g):\n    return g\n```"
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_bare_def_transform(self):
        text = "def transform(input_grid):\n    return input_grid.copy()"
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_returns_last_fenced_block(self):
        text = (
            "```python\ndef transform(g):\n    return g * 0\n```\n"
            "Actually:\n"
            "```python\ndef transform(g):\n    return g.copy()\n```"
        )
        code = _extract_code(text)
        assert "copy" in code      # last block preferred

    def test_none_when_no_code(self):
        text = "I cannot find a solution."
        assert _extract_code(text) is None

    def test_strips_thinking_first(self):
        text = "<think>ignore this def fake(g): pass</think>\n```python\ndef transform(g):\n    return g\n```"
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_numpy_import_block(self):
        text = "import numpy as np\ndef transform(g):\n    return np.zeros_like(g)"
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_empty_response_returns_none(self):
        assert _extract_code("") is None
        assert _extract_code("   ") is None


# ---------------------------------------------------------------------------
# _truncate_to_valid_function
# ---------------------------------------------------------------------------

class TestTruncateToValidFunction:
    def test_stops_at_top_level_non_def(self):
        text = "def transform(g):\n    return g\n\nsome_random_var = 1"
        result = _truncate_to_valid_function(text)
        assert "some_random_var" not in result

    def test_preserves_nested_code(self):
        text = "def transform(g):\n    x = 1\n    return g"
        result = _truncate_to_valid_function(text)
        assert "x = 1" in result

    def test_returns_text_when_no_def(self):
        text = "no function here"
        assert _truncate_to_valid_function(text) == text


# ---------------------------------------------------------------------------
# _parse_hypotheses
# ---------------------------------------------------------------------------

class TestParseHypotheses:
    # Each hypothesis must be > _MIN_HYPOTHESIS_CHARS=80 characters to pass the filter
    NUMBERED = (
        "1. The output grid is produced by rotating the input 90 degrees clockwise "
        "and replacing every blue cell with a red cell throughout the entire grid.\n\n"
        "2. Each row in the input is individually reversed left-to-right, then all "
        "zero-valued cells are replaced with the most frequently occurring colour.\n\n"
        "3. Discrete foreground objects are found using connected-component analysis, "
        "sorted by pixel area from largest to smallest, then re-arranged top-to-bottom."
    )

    def test_parses_three_hypotheses(self):
        result = _parse_hypotheses(self.NUMBERED)
        assert len(result) == 3

    def test_each_hypothesis_starts_with_number(self):
        result = _parse_hypotheses(self.NUMBERED)
        for i, h in enumerate(result, 1):
            assert h.startswith(f"{i}.")

    def test_max_n_respected(self):
        result = _parse_hypotheses(self.NUMBERED, max_n=2)
        assert len(result) == 2

    def test_short_hypotheses_filtered(self):
        short = "1. short\n\n2. also short\n\n3. and this too"
        result = _parse_hypotheses(short)
        # All are too short (<80 chars) — fallback returns original stripped text
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_single_paragraph_returns_list(self):
        text = "The transformation rotates the grid and applies a colour mapping."
        result = _parse_hypotheses(text)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_strips_thinking_before_parse(self):
        text = "<think>ignore</think>1. First long hypothesis about rotating and flipping grids.\n\n2. Second long hypothesis about colour mapping rules in ARC puzzles."
        result = _parse_hypotheses(text)
        assert result[0].startswith("1.")


# ---------------------------------------------------------------------------
# _format_training_examples
# ---------------------------------------------------------------------------

class TestFormatTrainingExamples:
    def test_contains_example_header(self, identity_task):
        text = _format_training_examples(identity_task)
        assert "Training examples" in text

    def test_contains_example_numbers(self, identity_task):
        text = _format_training_examples(identity_task)
        assert "Example 1" in text
        assert "Example 2" in text

    def test_contains_grid_values(self, identity_task):
        text = _format_training_examples(identity_task)
        # The identity task has values 1,2,3,4
        assert "1" in text and "2" in text


# ---------------------------------------------------------------------------
# _format_task_description
# ---------------------------------------------------------------------------

class TestFormatTaskDescription:
    def test_contains_training_pairs(self, identity_task):
        text = _format_task_description(identity_task)
        assert "Training pair" in text

    def test_contains_test_input(self, identity_task):
        text = _format_task_description(identity_task)
        assert "Test input" in text

    def test_contains_shape_info(self, identity_task):
        text = _format_task_description(identity_task)
        assert "×" in text   # e.g. "2×2"

    def test_large_grid_truncates(self):
        big = np.ones((20, 20), dtype=np.int32)
        task = {
            "train": [
                {"input": big, "output": big},
                {"input": big, "output": big},
                {"input": big, "output": big},
            ],
            "test": [{"input": big}],
        }
        text = _format_task_description(task)
        # Large grids only show up to _LARGE_GRID_MAX_PAIRS = 2 pairs
        assert "Training pair 3" not in text


# ---------------------------------------------------------------------------
# _grid_to_str
# ---------------------------------------------------------------------------

class TestGridToStr:
    def test_format(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        s = _grid_to_str(grid)
        assert s == "[[1, 2], [3, 4]]"

    def test_single_cell(self):
        grid = np.array([[7]], dtype=np.int32)
        assert _grid_to_str(grid) == "[[7]]"


# ---------------------------------------------------------------------------
# _format_error_info
# ---------------------------------------------------------------------------

class TestFormatErrorInfo:
    def test_shows_correct_count(self):
        result = {
            "n_correct": 1, "n_total": 3,
            "pairs": [
                {"correct": True,  "error": None},
                {"correct": False, "error": "RuntimeError: boom"},
                {"correct": False, "error": None},
            ],
        }
        text = _format_error_info(result)
        assert "1/3" in text
        assert "RuntimeError" in text

    def test_all_correct_message(self):
        result = {
            "n_correct": 2, "n_total": 2,
            "pairs": [
                {"correct": True, "error": None},
                {"correct": True, "error": None},
            ],
        }
        text = _format_error_info(result)
        assert "2/2" in text


# ---------------------------------------------------------------------------
# _diff_summary
# ---------------------------------------------------------------------------

class TestDiffSummary:
    def test_no_diffs(self):
        g = np.array([[1, 2]], dtype=np.int32)
        assert "(no differences)" in _diff_summary(g, g)

    def test_shape_mismatch(self):
        expected  = np.array([[1, 2]], dtype=np.int32)
        predicted = np.array([[1]], dtype=np.int32)
        result = _diff_summary(expected, predicted)
        assert "Shape mismatch" in result

    def test_none_predicted(self):
        g = np.array([[1]], dtype=np.int32)
        assert "(no output produced)" in _diff_summary(g, None)

    def test_shows_diff_positions(self):
        expected  = np.array([[1, 2]], dtype=np.int32)
        predicted = np.array([[1, 9]], dtype=np.int32)
        result = _diff_summary(expected, predicted)
        assert "[0,1]" in result
        assert "9" in result
