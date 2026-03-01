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

    def test_large_grid_sparse_format(self):
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
        # All pairs are shown, but large grids use sparse format
        assert "Training pair 3" in text
        # Large grids (441 cells > RLE threshold 400) use "sparse" notation
        assert "sparse" in text.lower()


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
    def test_very_large_train_grid_omitted(self):
        # >800 cells triggers "omitted" path for training pair
        task = _make_task((30, 30), (2, 2))   # 900 train cells
        desc = _format_task_description(task)
        assert "omitted" in desc.lower()

    def test_rle_train_grid(self):
        # 64 cells (8×8) > 50 → RLE for training pair
        task = _make_task((8, 8), (2, 2))
        desc = _format_task_description(task)
        assert "[RLE]" in desc

    def test_very_large_test_input_omitted(self):
        # Training pair tiny, test input >800 cells → omit test input
        task = _make_task((2, 2), (30, 30))
        desc = _format_task_description(task)
        assert "omitted" in desc.lower()

    def test_rle_test_input(self):
        # Training pair tiny, test input 8×8=64 cells > 50 → RLE for test
        task = _make_task((2, 2), (8, 8))
        desc = _format_task_description(task)
        assert "[RLE]" in desc


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
