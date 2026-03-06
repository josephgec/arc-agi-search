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
    _format_spatial_diff,
    _format_eval_diff,
    _grid_to_str,
    _grid_to_rle,
    _grid_to_sparse,
    _diff_summary,
    _diff_annotation,
    _structural_note,
    _output_shape_constraint,
    _cross_pair_notes,
    _format_grid_comparison,
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

    # --- deepseek-r1 / think-block scenarios ---

    def test_think_then_fenced_block(self):
        """Long think chain followed by a real code fence — primary path."""
        text = (
            "<think>Let me reason step by step.\n"
            "The grid rotates 90 degrees.\n"
            "I should use rotate().\n"
            "</think>\n"
            "Here is the implementation:\n"
            "```python\n"
            "def transform(input_grid):\n"
            "    return rotate(input_grid, 1)\n"
            "```"
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code
        assert "rotate" in code

    def test_code_inside_think_block_extracted_as_fallback(self):
        """Model emits code inside <think> and nothing outside — fallback path."""
        text = (
            "<think>\n"
            "```python\n"
            "def transform(input_grid):\n"
            "    return flip(input_grid, 1)\n"
            "```\n"
            "</think>\n"
            "The answer is shown above."
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code
        assert "flip" in code

    def test_bare_def_in_think_block_extracted_as_fallback(self):
        """Bare def inside <think>, no fence anywhere — deepest fallback."""
        text = (
            "<think>\n"
            "def transform(input_grid):\n"
            "    return input_grid[::-1]\n"
            "</think>\n"
            "Done."
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_no_code_anywhere_returns_none(self):
        """No code inside or outside think — must return None, not crash."""
        text = (
            "<think>I am thinking but have no code.</think>\n"
            "Sorry, I cannot solve this."
        )
        assert _extract_code(text) is None

    def test_fake_code_in_think_ignored_when_real_code_outside(self):
        """Fake def inside think must not shadow real code outside."""
        text = (
            "<think>ignore: def transform(g): return g*0</think>\n"
            "```python\n"
            "def transform(input_grid):\n"
            "    return recolor(input_grid, 1, 2)\n"
            "```"
        )
        code = _extract_code(text)
        assert code is not None
        assert "recolor" in code  # real code wins
        assert "g*0" not in code

    def test_backtick_py_fence_extracted(self):
        """```py ... ``` (without 'python') should be accepted as a code fence."""
        text = (
            "Here is my solution:\n"
            "```py\n"
            "def transform(input_grid):\n"
            "    return input_grid[::-1]\n"
            "```"
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_raw_text_fallback_finds_last_fence_in_whole_response(self):
        """Final fallback: scan entire raw text (think included) for last ```python block."""
        # Both blocks are inside a malformed think block so earlier passes may miss them;
        # the final raw-scan should find the LAST one.
        text = (
            "<think>\n"
            "```python\n"
            "def transform(input_grid):\n"
            "    return input_grid  # draft\n"
            "```\n"
            "Actually let me improve it:\n"
            "```python\n"
            "def transform(input_grid):\n"
            "    return flip(input_grid, 0)\n"
            "```\n"
            "</think>\n"
            "I could not produce a clean answer."
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_py_fence_in_think_raw_fallback(self):
        """```py fence inside a think block is found by the raw final fallback."""
        text = (
            "<think>\n"
            "```py\n"
            "def transform(input_grid):\n"
            "    return rotate(input_grid, 1)\n"
            "```\n"
            "</think>\n"
            "The answer is above."
        )
        code = _extract_code(text)
        assert code is not None
        assert "def transform" in code


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

    def test_large_grid_uses_visual_with_labels(self):
        big = np.ones((21, 21), dtype=np.int32)
        task = {
            "train": [
                {"input": big, "output": big},
                {"input": big, "output": big},
                {"input": big, "output": big},
            ],
            "test": [{"input": big}],
        }
        text = _format_task_description(task)
        # All pairs shown with visual format
        assert "Training pair 3" in text
        # Large grids (>10 in either dim) get row/col index labels
        assert " 0:" in text
        # Old compression markers gone
        assert "sparse" not in text.lower()


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


# ---------------------------------------------------------------------------
# _structural_note
# ---------------------------------------------------------------------------

class TestStructuralNote:
    def test_new_color_detected(self):
        inp = np.array([[3, 3], [3, 3]], dtype=np.int32)
        out = np.array([[3, 4], [4, 3]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "NEW color" in note
        assert "4" in note

    def test_no_new_color_no_note(self):
        inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        out = np.array([[4, 3], [2, 1]], dtype=np.int32)
        note = _structural_note(inp, out)
        # No new colors, no shape change, no structural pattern → None
        assert note is None or "NEW color" not in (note or "")

    def test_scale_up_detected(self):
        inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        # 2x scale: each cell repeated 2×2
        out = np.array([[1, 1, 2, 2], [1, 1, 2, 2],
                        [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "2×" in note or "scaled" in note.lower()

    def test_shape_change_noted(self):
        inp = np.array([[1, 2, 3]], dtype=np.int32)
        out = np.array([[1, 2]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "Shape changes" in note or "scaled" in note.lower()

    def test_anti_diagonal_detected_multiple_colors(self):
        # Multiple distinct colors across diagonals → pattern reported
        # k=0: (0,0)=1;  k=1: (0,1)=2, (1,0)=2;  k=2: (1,1)=3
        inp = np.array([[1, 2], [2, 3]], dtype=np.int32)
        out = np.array([[1, 2], [2, 3]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "anti-diagonal" in note

    def test_anti_diagonal_single_color_not_reported(self):
        # All non-zero cells share the same color → trivial, not reported
        inp = np.array([[0, 1], [1, 0]], dtype=np.int32)
        out = np.array([[0, 1], [1, 0]], dtype=np.int32)
        note = _structural_note(inp, out)
        # Should NOT contain anti-diagonal note (only one color in input)
        assert note is None or "anti-diagonal" not in (note or "")

    def test_row_cycle_extension_detected(self):
        # Rows cycle with period 2; output is extended version (1→2 color swap)
        inp = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1],
                        [0, 1, 0], [1, 0, 1]], dtype=np.int32)
        out = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0], [2, 0, 2],
                        [0, 2, 0], [2, 0, 2], [0, 2, 0], [2, 0, 2],
                        [0, 2, 0]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "cycle" in note.lower()

    def test_none_for_identical_simple_grids(self):
        g = np.array([[1, 0], [0, 1]], dtype=np.int32)
        # Same grid, same shape, no new colors
        note = _structural_note(g, g)
        # May be None or have no alarming content
        assert note is None or isinstance(note, str)

    def test_vertical_divider_detected(self):
        # Column 2 is all 5s → vertical divider
        inp = np.array([[1, 0, 5, 0, 1],
                        [0, 1, 5, 1, 0],
                        [1, 0, 5, 0, 1]], dtype=np.int32)
        out = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "DIVIDER" in note.upper()
        assert "column 2" in note

    def test_horizontal_divider_detected(self):
        # Row 1 is all 9s → horizontal divider
        inp = np.array([[1, 0, 1],
                        [9, 9, 9],
                        [0, 1, 0]], dtype=np.int32)
        out = np.array([[0, 0],
                        [0, 0],
                        [0, 0]], dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "DIVIDER" in note.upper()
        assert "row 1" in note

    def test_object_count_reported(self):
        # Two separate single-cell objects of color 3
        inp = np.array([[3, 0, 3],
                        [0, 0, 0],
                        [0, 0, 0]], dtype=np.int32)
        out = np.array([[0, 0, 0],
                        [0, 3, 0],
                        [0, 0, 0]], dtype=np.int32)
        note = _structural_note(inp, out)
        # Should mention 2 objects for the input
        assert note is not None
        assert "2" in note

    def test_anti_diagonal_cycle_detected(self):
        # 3-cycle: k=0→1, k=1→2, k=2→3
        inp = np.array([[1, 2, 3],
                        [2, 3, 0],
                        [3, 0, 0]], dtype=np.int32)
        out = np.zeros((3, 3), dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "CYCLE" in note.upper() or "cycle" in note.lower()


# ---------------------------------------------------------------------------
# _grid_to_rle
# ---------------------------------------------------------------------------

class TestGridToRle:
    def test_run_encoded_with_count(self):
        g = np.array([[1, 1, 1]], dtype=np.int32)
        result = _grid_to_rle(g)
        assert "x3" in result

    def test_singleton_has_no_count(self):
        g = np.array([[5]], dtype=np.int32)
        result = _grid_to_rle(g)
        assert "x" not in result

    def test_mixed_row(self):
        g = np.array([[0, 0, 1]], dtype=np.int32)
        result = _grid_to_rle(g)
        assert "x2" in result

    def test_empty_row_handled(self):
        # 0-column grid → each row is an empty list → the `if not row:` path
        g = np.zeros((2, 0), dtype=np.int32)
        result = _grid_to_rle(g)
        assert isinstance(result, str)

    def test_multi_row_grid(self):
        g = np.array([[1, 1], [2, 2]], dtype=np.int32)
        result = _grid_to_rle(g)
        assert "x2" in result


# ---------------------------------------------------------------------------
# _grid_to_sparse
# ---------------------------------------------------------------------------

class TestGridToSparse:
    def test_all_zeros_returns_empty_message(self):
        g = np.zeros((3, 3), dtype=np.int32)
        result = _grid_to_sparse(g)
        assert "empty" in result.lower()

    def test_nonzero_cells_listed(self):
        g = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int32)
        result = _grid_to_sparse(g)
        assert "(0,1)=1" in result
        assert "(1,2)=2" in result


# ---------------------------------------------------------------------------
# _diff_annotation
# ---------------------------------------------------------------------------

class TestDiffAnnotation:
    def test_no_changes(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = _diff_annotation(g, g)
        assert "no cells changed" in result

    def test_many_changes_truncated(self):
        # 5×5 all different → 25 changes > 20 → short message
        pred = np.zeros((5, 5), dtype=np.int32)
        out  = np.ones((5, 5), dtype=np.int32)
        result = _diff_annotation(pred, out)
        assert "25" in result

    def test_few_changes_listed(self):
        pred = np.array([[1, 2], [3, 4]], dtype=np.int32)
        out  = np.array([[1, 9], [3, 4]], dtype=np.int32)
        result = _diff_annotation(pred, out)
        assert "[0,1]" in result

    def test_shape_mismatch_returns_none(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1], [2]], dtype=np.int32)
        assert _diff_annotation(a, b) is None

    def test_more_than_8_changes_has_ellipsis(self):
        pred = np.zeros((3, 4), dtype=np.int32)   # 12 cells
        out  = np.ones((3, 4), dtype=np.int32)
        result = _diff_annotation(pred, out)
        assert "…" in result or "12" in result


# ---------------------------------------------------------------------------
# _format_task_description — size dispatch paths
# ---------------------------------------------------------------------------

def _make_task(train_shape, test_shape):
    t = np.zeros(train_shape, dtype=np.int32)
    s = np.zeros(test_shape,  dtype=np.int32)
    return {"train": [{"input": t, "output": t}], "test": [{"input": s}]}


class TestFormatTaskDescriptionSizes:
    def test_very_large_train_grid_shown(self):
        # All grids shown regardless of size — visual format with row/col labels
        task = _make_task((30, 30), (2, 2))
        desc = _format_task_description(task)
        assert "Training pair 1" in desc
        # 30×30 grid → row label pattern present (r_width=2 → " 0:")
        assert " 0:" in desc
        assert "omitted" not in desc.lower()

    def test_medium_train_grid_visual(self):
        # 8×8 grid (≤10 in both dims) → plain visual format, no [RLE]
        task = _make_task((8, 8), (2, 2))
        desc = _format_task_description(task)
        assert "Training pair 1" in desc
        assert "[RLE]" not in desc

    def test_very_large_test_input_shown(self):
        # All grids shown — 30×30 test input shown with row/col labels
        task = _make_task((2, 2), (30, 30))
        desc = _format_task_description(task)
        assert "Test input" in desc
        assert " 0:" in desc
        assert "omitted" not in desc.lower()

    def test_medium_test_input_visual(self):
        # 8×8 test input → plain visual format
        task = _make_task((2, 2), (8, 8))
        desc = _format_task_description(task)
        assert "Test input" in desc
        assert "[RLE]" not in desc


# ---------------------------------------------------------------------------
# _output_shape_constraint — variable output sizes
# ---------------------------------------------------------------------------

class TestOutputShapeConstraintVariable:
    def test_variable_sizes_shows_guidance(self):
        # Two training pairs with different output shapes → variable guidance
        inp1 = np.zeros((2, 2), dtype=np.int32)
        out1 = np.zeros((2, 2), dtype=np.int32)
        out2 = np.zeros((3, 3), dtype=np.int32)
        task = {"train": [
            {"input": inp1, "output": out1},
            {"input": inp1, "output": out2},
        ]}
        result = _output_shape_constraint(task)
        assert "varies" in result.lower() or "variable" in result.lower() or "dynamically" in result.lower()

    def test_variable_larger_output_noted(self):
        inp = np.zeros((2, 2), dtype=np.int32)
        out = np.zeros((4, 4), dtype=np.int32)
        task = {"train": [
            {"input": inp, "output": out},
            {"input": inp, "output": np.zeros((3, 3), dtype=np.int32)},
        ]}
        result = _output_shape_constraint(task)
        assert "larger" in result.lower() or "expanded" in result.lower()


# ---------------------------------------------------------------------------
# _format_eval_diff
# ---------------------------------------------------------------------------

class TestFormatEvalDiff:
    def test_error_pair_reported(self):
        eval_res = {"pairs": [
            {"error": "RuntimeError: boom", "predicted": None, "expected": None, "correct": False}
        ]}
        result = _format_eval_diff(eval_res)
        assert "runtime error" in result.lower()

    def test_no_prediction_pair(self):
        exp = np.array([[1, 2]], dtype=np.int32)
        eval_res = {"pairs": [
            {"error": None, "predicted": None, "expected": exp, "correct": False}
        ]}
        result = _format_eval_diff(eval_res)
        assert "no prediction" in result.lower()

    def test_shape_mismatch_pair(self):
        pred = np.array([[1, 2, 3]], dtype=np.int32)   # 1×3
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)  # 2×2
        eval_res = {"pairs": [
            {"error": None, "predicted": pred, "expected": exp, "correct": False}
        ]}
        result = _format_eval_diff(eval_res)
        assert "wrong shape" in result.lower() or "shape" in result.lower()

    def test_excess_wrongs_truncated(self):
        pred = np.zeros((5, 5), dtype=np.int32)
        exp  = np.ones((5, 5), dtype=np.int32)
        eval_res = {"pairs": [
            {"error": None, "predicted": pred, "expected": exp, "correct": False}
        ]}
        result = _format_eval_diff(eval_res, max_mismatches=3)
        assert "more wrong" in result.lower() or "…" in result

    def test_empty_pairs_returns_no_diff_message(self):
        result = _format_eval_diff({"pairs": []})
        assert result == "No diff available"


# ---------------------------------------------------------------------------
# _format_diff
# ---------------------------------------------------------------------------

class TestFormatDiff:
    def test_failing_pair_shows_diff_section(self):
        pred = np.array([[1, 2], [3, 0]], dtype=np.int32)
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)
        eval_result = {"pairs": [
            {"correct": False, "predicted": pred, "expected": exp, "error": None}
        ]}
        result = _format_diff(eval_result)
        assert "Pair 1" in result

    def test_all_correct_returns_short_message(self):
        g = np.array([[1]], dtype=np.int32)
        eval_result = {"pairs": [
            {"correct": True, "predicted": g, "expected": g, "error": None}
        ]}
        result = _format_diff(eval_result)
        assert "all pairs correct" in result.lower()

    def test_failing_pair_includes_full_grids_for_small_exp(self):
        # exp.size = 4 ≤ 200 → full grids appended to section
        pred = np.array([[1, 2], [3, 0]], dtype=np.int32)
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)
        eval_result = {"pairs": [
            {"correct": False, "predicted": pred, "expected": exp, "error": None}
        ]}
        result = _format_diff(eval_result)
        assert "Full expected" in result or "Full predicted" in result


# ---------------------------------------------------------------------------
# _format_spatial_diff
# ---------------------------------------------------------------------------

class TestFormatSpatialDiff:
    def test_failing_pair_described(self):
        pred = np.array([[1, 0]], dtype=np.int32)
        exp  = np.array([[1, 2]], dtype=np.int32)
        eval_result = {"pairs": [
            {"correct": False, "predicted": pred, "expected": exp, "error": None}
        ]}
        result = _format_spatial_diff(eval_result)
        assert "Pair 1" in result

    def test_none_expected_skipped(self):
        eval_result = {"pairs": [
            {"correct": False, "predicted": None, "expected": None}
        ]}
        result = _format_spatial_diff(eval_result)
        assert result == "(all pairs correct)"

    def test_correct_pair_skipped(self):
        g = np.array([[1]], dtype=np.int32)
        eval_result = {"pairs": [
            {"correct": True, "predicted": g, "expected": g}
        ]}
        result = _format_spatial_diff(eval_result)
        assert result == "(all pairs correct)"


# ---------------------------------------------------------------------------
# _structural_note — downscale branch (lines 215-216)
# ---------------------------------------------------------------------------

class TestStructuralNoteDownscale:
    def test_downscale_detected(self):
        # 4×4 input → 2×2 output is a 1/2 downscale
        inp = np.ones((4, 4), dtype=np.int32)
        out = np.ones((2, 2), dtype=np.int32)
        note = _structural_note(inp, out)
        assert note is not None
        assert "1/2" in note


class TestStructuralNoteSatelliteAnchor:
    """Tests for movement detection and satellite-anchor proximity hints."""

    def test_moved_cells_detected(self):
        """Color with same count but different positions triggers movement note."""
        inp = np.zeros((5, 5), dtype=np.int32)
        out = np.zeros((5, 5), dtype=np.int32)
        inp[0, 0] = 3  # satellite far from anchor
        out[2, 2] = 3  # moved to near anchor
        note = _structural_note(inp, out)
        assert note is not None
        assert "MOVE" in note
        assert "3" in note

    def test_anchor_stays_in_place(self):
        """A color that doesn't move is NOT reported as moving."""
        inp = np.zeros((5, 5), dtype=np.int32)
        out = np.zeros((5, 5), dtype=np.int32)
        inp[2, 2] = 1  # anchor stays
        out[2, 2] = 1
        inp[0, 0] = 3  # satellite moves
        out[2, 1] = 3
        note = _structural_note(inp, out)
        # Color 1 is anchor → should not appear in a "MOVE" note
        move_lines = [l for l in (note or "").splitlines() if "MOVE" in l]
        assert all("1" not in l or "Color 3" in l for l in move_lines)

    def test_satellite_adjacent_to_anchor_hint(self):
        """When moved cells are adjacent to anchor in output, strong hint fires."""
        inp = np.zeros((5, 5), dtype=np.int32)
        out = np.zeros((5, 5), dtype=np.int32)
        # anchor: color 2 at (2, 2)
        inp[2, 2] = 2; out[2, 2] = 2
        # satellite: color 3 far in input, adjacent to anchor in output
        inp[0, 0] = 3
        out[1, 2] = 3  # adjacent (above) anchor
        note = _structural_note(inp, out)
        assert note is not None
        assert "satellites compress" in note.lower() or "STRONG HINT" in note

    def test_partial_adjacency_softer_hint(self):
        """Only some moved cells adjacent → softer hint (not STRONG)."""
        inp = np.zeros((7, 7), dtype=np.int32)
        out = np.zeros((7, 7), dtype=np.int32)
        inp[2, 2] = 2; out[2, 2] = 2   # anchor
        inp[0, 0] = 3                   # far satellite 1
        inp[0, 6] = 3                   # far satellite 2
        out[1, 2] = 3                   # satellite 1 → adjacent
        out[0, 4] = 3                   # satellite 2 → NOT adjacent
        note = _structural_note(inp, out)
        assert note is not None
        # Should report partial adjacency but NOT the "STRONG HINT" message
        assert "STRONG HINT" not in (note or "")
        assert "adjacent" in note.lower()


# ---------------------------------------------------------------------------
# _format_training_examples — adaptive grid format (lines 342-349)
# ---------------------------------------------------------------------------

class TestFormatTrainingExamplesAdaptive:
    def _make_task(self, inp, out):
        return {"train": [{"input": inp, "output": out}],
                "test":  [{"input": inp}]}

    def test_large_grid_shown_with_labels(self):
        # All grids shown — 30×30 gets row/col labels (>10 in both dims)
        big = np.zeros((30, 30), dtype=np.int32)
        task = self._make_task(big, big)
        result = _format_training_examples(task)
        assert "Example 1" in result
        # Row label present for large grid (r_width=2 → " 0:")
        assert " 0:" in result
        assert "large grid" not in result.lower()

    def test_medium_grid_shown_visually(self):
        # 25×20 grid → shown with visual format and labels (>10 in both dims)
        med = np.zeros((25, 20), dtype=np.int32)  # 500 cells
        med[0, 0] = 3
        task = self._make_task(med, med)
        result = _format_training_examples(task)
        assert "Example 1" in result
        assert "[sparse]" not in result
        # Green (color 3) shown as "G"
        assert "G" in result

    def test_small_grid_visual_format(self):
        # 8×8 grid (≤10 in both dims) → plain visual format, no [RLE]
        rle_grid = np.zeros((8, 8), dtype=np.int32)  # 64 cells
        rle_grid[0, :] = 1
        task = self._make_task(rle_grid, rle_grid)
        result = _format_training_examples(task)
        assert "[RLE]" not in result
        # Blue (color 1) shown as "B"
        assert "B" in result

    def test_small_grid_dense(self):
        # ≤50 cells → visual format (no [RLE], no [sparse])
        small = np.array([[1, 2], [3, 4]], dtype=np.int32)
        task = self._make_task(small, small)
        result = _format_training_examples(task)
        assert "1" in result
        assert "[RLE]" not in result
        assert "[sparse]" not in result


# ---------------------------------------------------------------------------
# _format_task_description — large test input (line 384)
# ---------------------------------------------------------------------------

class TestFormatTaskDescLargeTest:
    def test_large_test_input_shown(self):
        # test input 30×30 → shown with visual format + row/col labels (not omitted)
        small  = np.array([[1, 2], [3, 4]], dtype=np.int32)
        big    = np.zeros((30, 30), dtype=np.int32)
        task   = {"train": [{"input": small, "output": small}],
                  "test":  [{"input": big}]}
        result = _format_task_description(task)
        assert "Test input" in result
        # Large grid labels present (r_width=2 → " 0:")
        assert " 0:" in result
        assert "too large" not in result.lower()


# ---------------------------------------------------------------------------
# MultiAgent.__init__ (lines 673-708)
# ---------------------------------------------------------------------------

class TestMultiAgentInit:
    def test_init_sets_attributes(self):
        from unittest.mock import patch, MagicMock
        from agents.multi_agent import MultiAgent

        with patch("agents.multi_agent.LLMClient") as MockClient:
            MockClient.return_value = MagicMock(model="mock-model")
            agent = MultiAgent(backend="ollama", model="test-model",
                               max_cycles=5, debug=False)

        assert agent.max_cycles == 5
        assert agent.debug is False
        assert agent.backend == "ollama"

    def test_init_role_models_default_to_model(self):
        from unittest.mock import patch, MagicMock
        from agents.multi_agent import MultiAgent

        with patch("agents.multi_agent.LLMClient") as MockClient:
            MockClient.return_value = MagicMock(model="default-model")
            agent = MultiAgent(model="default-model")

        # All role clients default to the base model
        assert agent.model == "default-model"


# ---------------------------------------------------------------------------
# Orchestrator.__init__ (orchestrator.py lines 53-71)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Code deduplication in MultiAgent.solve()
# ---------------------------------------------------------------------------

class TestCodeDeduplication:
    """Verify that duplicate code is detected and skipped without re-running sandbox."""

    def _make_task(self):
        """Minimal 1-pair task returning a fixed 1×1 grid."""
        g = np.array([[1]], dtype=np.int32)
        return {"train": [{"input": g, "output": g}], "test": [{"input": g}]}

    def test_duplicate_code_gets_new_approach_feedback(self):
        """When Coder returns the same code twice, solve() injects a diversity prompt."""
        from unittest.mock import patch, MagicMock
        from agents.multi_agent import MultiAgent

        code_once = "def transform(input_grid):\n    return input_grid.copy()\n"

        with patch("agents.multi_agent.LLMClient") as MockClient:
            MockClient.return_value = MagicMock(model="mock")
            agent = MultiAgent(max_cycles=6, use_decomposer=False, use_verifier=False)

        call_count = [0]
        def _fake_coder_generate(hyp, fb=None, training_context=None, temperature=None, **kw):
            call_count[0] += 1
            # Always return the same code regardless of feedback
            return f"```python\n{code_once}```"

        def _fake_hyp_generate(desc, feedback=None, **kw):
            return "1. The output is the input copied unchanged for all training examples."

        agent._hypothesizer.generate = _fake_hyp_generate
        agent._coder.generate = _fake_coder_generate

        with patch("agents.multi_agent.sandbox.evaluate_code") as mock_eval:
            mock_eval.return_value = {
                "all_correct": False,
                "n_correct": 0, "n_total": 1,
                "pairs": [{"correct": False, "predicted": None,
                           "expected": np.array([[1]]), "error": None}],
            }
            result = agent.solve(self._make_task())

        # sandbox.evaluate_code should NOT be called twice for the same code
        # (second call is a duplicate and must be skipped)
        assert mock_eval.call_count < call_count[0], (
            "sandbox should be skipped for duplicate code"
        )

    def test_different_codes_both_evaluated(self):
        """Two distinct code strings must each reach the sandbox."""
        from unittest.mock import patch, MagicMock
        from agents.multi_agent import MultiAgent
        from agents.roles import ROUTE_CODER

        codes = [
            "def transform(input_grid):\n    return input_grid.copy()\n",
            "def transform(input_grid):\n    return input_grid[::-1]\n",
        ]
        call_idx = [0]

        with patch("agents.multi_agent.LLMClient") as MockClient:
            MockClient.return_value = MagicMock(model="mock")
            agent = MultiAgent(max_cycles=8, use_decomposer=False, use_verifier=False)

        def _fake_coder(hyp, fb=None, training_context=None, temperature=None, **kw):
            code = codes[min(call_idx[0], len(codes) - 1)]
            call_idx[0] += 1
            return f"```python\n{code}```"

        agent._hypothesizer.generate = lambda *a, **kw: (
            "1. The output is a copy of the input grid without any modifications made."
        )
        agent._coder.generate = _fake_coder
        # Critic routes back to coder so the hypothesis stays alive for the 2nd attempt
        agent._critic.analyze = lambda *a, **kw: {
            "route": ROUTE_CODER, "feedback": "try a different approach"
        }

        with patch("agents.multi_agent.sandbox.evaluate_code") as mock_eval:
            mock_eval.return_value = {
                "all_correct": False,
                "n_correct": 0, "n_total": 1,
                "pairs": [{"correct": False, "predicted": None,
                           "expected": np.array([[1]]), "error": None}],
            }
            agent.solve(self._make_task())

        # Both distinct codes should reach the sandbox
        assert mock_eval.call_count >= 2


# ---------------------------------------------------------------------------
# _cross_pair_notes
# ---------------------------------------------------------------------------

class TestCrossPairNotes:
    def test_color_swap_detected(self):
        """Pairs where 1→5 and 5→1 → reports FIXED COLOR SWAP 1↔5."""
        inp1 = np.array([[1, 5]], dtype=np.int32)
        out1 = np.array([[5, 1]], dtype=np.int32)
        inp2 = np.array([[5, 1]], dtype=np.int32)
        out2 = np.array([[1, 5]], dtype=np.int32)
        result = _cross_pair_notes([(inp1, out1), (inp2, out2)])
        assert result is not None
        assert "1↔5" in result or "5↔1" in result
        assert "SWAP" in result.upper()

    def test_output_symmetry_4way(self):
        """All outputs are 4-way symmetric → reports 4-way symmetry hint."""
        # 2×2 grid that is symmetric both ways: [[1,1],[1,1]]
        inp = np.array([[0, 1]], dtype=np.int32)
        out = np.array([[1, 1], [1, 1]], dtype=np.int32)
        result = _cross_pair_notes([(inp, out), (inp, out)])
        assert result is not None
        assert "4-way" in result.lower() or "fourfold" in result.lower() or "horizontal + vertical" in result.lower()

    def test_scale_ratio_detected(self):
        """All pairs 3×3 → 6×6 → reports 2× scale."""
        inp = np.ones((3, 3), dtype=np.int32)
        out = np.ones((6, 6), dtype=np.int32)
        result = _cross_pair_notes([(inp, out), (inp, out)])
        assert result is not None
        assert "2" in result
        assert "scale" in result.lower() or "×" in result

    def test_block_selection_detected(self):
        """9×3 input, output matches one 3×3 block → reports block hint."""
        row = np.array([[1, 2, 3]], dtype=np.int32)
        block0 = np.tile(row, (3, 1))          # rows 0-2: [[1,2,3]]*3
        block1 = np.tile(row * 2, (3, 1))      # rows 3-5: [[2,4,6]]*3
        block2 = np.tile(row * 3, (3, 1))      # rows 6-8: [[3,6,9]]*3
        inp = np.vstack([block0, block1, block2])
        # Output is block1 for both pairs
        result = _cross_pair_notes([(inp, block1), (inp, block1)])
        assert result is not None
        assert "block" in result.lower()
        assert "3" in result  # K=3 blocks

    def test_returns_none_for_no_pattern(self):
        """Random dissimilar pairs → no cross-pair notes."""
        inp1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        out1 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        inp2 = np.array([[8, 7], [6, 5]], dtype=np.int32)
        out2 = np.array([[4, 3], [2, 1]], dtype=np.int32)
        # Colors don't consistently map the same way across pairs
        result = _cross_pair_notes([(inp1, out1), (inp2, out2)])
        # Should not crash; result may be None or contain something but no swap detected
        # Just verify it doesn't raise an exception
        assert result is None or isinstance(result, str)

    def test_empty_pairs_returns_none(self):
        assert _cross_pair_notes([]) is None

    def test_single_pair_no_crash(self):
        inp = np.array([[1, 2]], dtype=np.int32)
        out = np.array([[2, 1]], dtype=np.int32)
        # Single pair — swap analysis runs but consistency check trivially passes
        result = _cross_pair_notes([(inp, out)])
        # No crash; result may contain swap note
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# _format_grid_comparison
# ---------------------------------------------------------------------------

class TestFormatGridComparison:
    def test_identity_flag_when_actual_equals_input(self):
        """When actual == input, the identity warning must appear."""
        inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        exp = np.array([[5, 6], [7, 8]], dtype=np.int32)
        actual = inp.copy()  # identity transform
        result = _format_grid_comparison(actual, exp, inp)
        assert "IDENTITY" in result.upper()

    def test_no_identity_flag_when_actual_differs(self):
        """When actual != input, no identity warning."""
        inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        exp = np.array([[5, 6], [7, 8]], dtype=np.int32)
        actual = np.array([[5, 6], [7, 8]], dtype=np.int32)
        result = _format_grid_comparison(actual, exp, inp)
        assert "IDENTITY" not in result.upper()

    def test_none_actual_shows_error_message(self):
        """When actual is None, shows error message."""
        inp = np.array([[1, 2]], dtype=np.int32)
        exp = np.array([[3, 4]], dtype=np.int32)
        result = _format_grid_comparison(None, exp, inp)
        assert "runtime error" in result.lower() or "no output" in result.lower()

    def test_shows_match_yes_when_correct(self):
        """When actual == expected, shows YES match."""
        inp = np.array([[1, 2]], dtype=np.int32)
        exp = np.array([[3, 4]], dtype=np.int32)
        actual = np.array([[3, 4]], dtype=np.int32)
        result = _format_grid_comparison(actual, exp, inp)
        assert "YES" in result

    def test_shows_match_no_when_wrong(self):
        """When actual != expected, shows NO match."""
        inp = np.array([[1, 2]], dtype=np.int32)
        exp = np.array([[3, 4]], dtype=np.int32)
        actual = np.array([[5, 6]], dtype=np.int32)
        result = _format_grid_comparison(actual, exp, inp)
        assert "NO" in result

    def test_large_grid_capped(self):
        """Grids larger than 10×10 are capped without crashing."""
        inp = np.zeros((20, 20), dtype=np.int32)
        exp = np.ones((20, 20), dtype=np.int32)
        actual = np.zeros((20, 20), dtype=np.int32)
        result = _format_grid_comparison(actual, exp, inp)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _structural_note — solid rectangle detection
# ---------------------------------------------------------------------------

class TestStructuralNoteSolidRectangle:
    def test_solid_rectangle_detected(self):
        """A color forming a solid filled bbox should trigger the hint."""
        # 5×5 grid with a 2×3 solid rectangle of color 3 embedded
        inp = np.zeros((5, 5), dtype=np.int32)
        inp[1:3, 1:4] = 3  # solid 2×3 block
        out = np.zeros((5, 5), dtype=np.int32)
        result = _structural_note(inp, out)
        assert result is not None
        assert "SOLID" in result
        assert "3" in result  # color 3 mentioned

    def test_non_solid_not_detected(self):
        """Scattered cells that don't fill their bbox shouldn't trigger."""
        inp = np.zeros((5, 5), dtype=np.int32)
        inp[0, 0] = 1
        inp[4, 4] = 1  # two isolated cells — bbox is 5×5 but only 2 cells filled
        out = np.zeros((5, 5), dtype=np.int32)
        result = _structural_note(inp, out)
        # Should NOT report solid rectangle (2 cells ≠ 25 bbox cells)
        assert result is None or "SOLID" not in result

    def test_full_grid_fill_excluded(self):
        """A color that fills the entire grid (bbox == grid) is excluded (too large)."""
        inp = np.ones((4, 4), dtype=np.int32)
        out = np.ones((4, 4), dtype=np.int32)
        result = _structural_note(inp, out)
        # Full-grid fill is excluded by the bbox_area < inp.size // 2 guard
        assert result is None or "SOLID" not in result


# ---------------------------------------------------------------------------
# _structural_note — diagonal extension detection
# ---------------------------------------------------------------------------

class TestStructuralNoteDiagonalExtension:
    def test_diagonal_adjacent_new_cells_detected(self):
        """New cells that are all diagonally adjacent to input cells → hint."""
        inp = np.zeros((5, 5), dtype=np.int32)
        inp[2, 2] = 1  # single center cell
        out = np.zeros((5, 5), dtype=np.int32)
        out[2, 2] = 1  # keep center
        out[1, 1] = 1  # diagonal neighbor
        out[1, 3] = 1  # diagonal neighbor
        out[3, 1] = 1  # diagonal neighbor
        out[3, 3] = 1  # diagonal neighbor
        result = _structural_note(inp, out)
        assert result is not None
        assert "diagonal" in result.lower() or "DIAGONAL" in result

    def test_orthogonal_new_cells_not_diagonal(self):
        """New cells orthogonally adjacent only — should not trigger diagonal hint."""
        inp = np.zeros((5, 5), dtype=np.int32)
        inp[2, 2] = 1
        out = np.zeros((5, 5), dtype=np.int32)
        out[2, 2] = 1
        out[1, 2] = 1  # up (orthogonal)
        out[3, 2] = 1  # down (orthogonal)
        result = _structural_note(inp, out)
        # These are orthogonal, not diagonal — no diagonal hint
        assert result is None or "DIAGONAL" not in result.upper()


# ---------------------------------------------------------------------------
# Repeated-prediction escalation in solve()
# ---------------------------------------------------------------------------

class TestRepeatedPredictionEscalation:
    """When the Coder produces the same wrong prediction 3× in a row, the
    orchestrator should escalate to the next hypothesis without calling Critic."""

    def _make_task(self):
        train_inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_out = np.array([[5, 6], [7, 8]], dtype=np.int32)
        return {
            "train": [{"input": train_inp, "output": train_out}],
            "test":  [{"input": train_inp}],
        }

    def test_repeated_prediction_skips_critic(self):
        """Three identical predictions should skip the Critic and move to next hyp."""
        from unittest.mock import patch, MagicMock
        from agents.multi_agent import MultiAgent

        # Fixed wrong prediction (always same structure)
        wrong_pred = np.array([[9, 9], [9, 9]], dtype=np.int32)

        with patch("agents.multi_agent.LLMClient") as MockClient:
            mock_llm = MagicMock()
            mock_llm.model = "mock"
            mock_llm.generate.side_effect = [
                # Hypothesizer
                "1. The output inverts colors.\n\n2. Rotate 90 degrees.\n\n3. Flip horizontally and vertically.",
                # 3 Coder responses (all produce same wrong result)
                "```python\ndef transform(input_grid):\n    return input_grid * 0 + 9\n```",
                "```python\ndef transform(input_grid):\n    return input_grid * 0 + 9\n```",
                "```python\ndef transform(input_grid):\n    return input_grid * 0 + 9\n```",
            ]
            MockClient.return_value = mock_llm

            critic_calls = []

            with patch("agents.multi_agent.sandbox.evaluate_code") as mock_eval, \
                 patch("agents.multi_agent.sandbox.compute_spatial_diff", return_value="diff"):
                mock_eval.return_value = {
                    "all_correct": False,
                    "n_correct": 0, "n_total": 1,
                    "pairs": [{"correct": False,
                               "predicted": wrong_pred,
                               "expected": np.array([[5, 6], [7, 8]], dtype=np.int32),
                               "error": None}],
                }

                agent = MultiAgent(max_cycles=6, use_verifier=False, use_decomposer=False)
                # Patch critic to track calls
                orig_analyze = agent._critic.analyze
                def tracking_analyze(*args, **kwargs):
                    critic_calls.append(1)
                    return {"route": "coder", "feedback": "fix it"}
                agent._critic.analyze = tracking_analyze

                agent.solve(self._make_task())

        # After 3 identical predictions the repeated-rut logic fires, skipping Critic
        # So Critic should be called fewer times than total Coder attempts
        assert len(critic_calls) < 3


class TestOrchestratorInit:
    def test_init_computes_max_cycles(self):
        from unittest.mock import patch, MagicMock
        from agents.orchestrator import Orchestrator

        with patch("agents.multi_agent.LLMClient") as MockClient:
            MockClient.return_value = MagicMock(model="m")
            orch = Orchestrator(n_hypotheses=3, max_retries=2)

        # max_cycles = 1 + 3*(1 + 2*2) = 1 + 15 = 16
        assert orch.max_cycles == 16
        assert orch.n_hypotheses == 3
        assert orch.max_retries == 2
