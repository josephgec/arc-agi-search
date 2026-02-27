"""Tests for arc/sandbox.py — sandboxed code execution."""
from __future__ import annotations

import numpy as np
import pytest

from arc.sandbox import execute, evaluate_code, EXECUTION_TIMEOUT, DSL_NAMESPACE


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------

IDENTITY_CODE = "def transform(input_grid):\n    return input_grid.copy()"
RECOLOR_CODE  = "def transform(input_grid):\n    return recolor(input_grid, 1, 2)"
ROTATE_CODE   = "def transform(input_grid):\n    return rotate(input_grid, 1)"


class TestExecute:
    def test_identity_transform(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result, err = execute(IDENTITY_CODE, grid)
        assert err is None
        np.testing.assert_array_equal(result, grid)

    def test_recolor_transform(self):
        grid = np.array([[1, 0], [0, 1]], dtype=np.int32)
        result, err = execute(RECOLOR_CODE, grid)
        assert err is None
        np.testing.assert_array_equal(result, np.array([[2, 0], [0, 2]], dtype=np.int32))

    def test_rotate_transform(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result, err = execute(ROTATE_CODE, grid)
        assert err is None
        assert result is not None
        assert result.shape == (2, 2)

    def test_syntax_error_returns_error_string(self):
        bad_code = "def transform(input_grid):\n    return @@@@"
        result, err = execute(bad_code, np.zeros((2, 2), dtype=np.int32))
        assert result is None
        assert err is not None
        assert "error" in err.lower() or "SyntaxError" in err or "invalid" in err.lower()

    def test_runtime_error_returns_error_string(self):
        bad_code = "def transform(input_grid):\n    raise ValueError('intentional')"
        result, err = execute(bad_code, np.zeros((2, 2), dtype=np.int32))
        assert result is None
        assert err is not None

    def test_no_transform_function_error(self):
        code = "x = 42"
        result, err = execute(code, np.zeros((2, 2), dtype=np.int32))
        assert result is None
        assert err is not None

    def test_stdin_blocked(self):
        code = "def transform(g):\n    input('x')\n    return g"
        result, err = execute(code, np.zeros((2, 2), dtype=np.int32))
        assert result is None
        assert err is not None

    def test_timeout(self):
        code = "def transform(g):\n    while True: pass"
        result, err = execute(code, np.zeros((2, 2), dtype=np.int32), timeout=1.0)
        assert result is None
        assert "timed out" in err.lower() or "timeout" in err.lower()

    def test_non_transform_named_function_is_accepted(self):
        code = "def solve(input_grid):\n    return input_grid.copy()"
        result, err = execute(code, np.array([[5]], dtype=np.int32))
        assert err is None
        np.testing.assert_array_equal(result, np.array([[5]], dtype=np.int32))

    def test_result_dtype_is_int32(self):
        grid = np.array([[1, 2]], dtype=np.int32)
        result, err = execute(IDENTITY_CODE, grid)
        assert err is None
        assert result.dtype == np.int32

    def test_dsl_namespace_available(self):
        # flip is in DSL_NAMESPACE; should execute without import
        code = "def transform(g):\n    return flip(g, axis=1)"
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result, err = execute(code, grid)
        assert err is None
        np.testing.assert_array_equal(result, np.array([[2, 1], [4, 3]], dtype=np.int32))


# ---------------------------------------------------------------------------
# evaluate_code()
# ---------------------------------------------------------------------------

class TestEvaluateCode:
    def test_all_correct(self, identity_task):
        result = evaluate_code(IDENTITY_CODE, identity_task)
        assert result["all_correct"] is True
        assert result["n_correct"] == result["n_total"]

    def test_all_wrong(self, identity_task):
        wrong_code = "def transform(g):\n    return g * 0"
        result = evaluate_code(wrong_code, identity_task)
        # Input has non-zero values, so zeroing gives wrong output
        assert result["n_correct"] < result["n_total"]

    def test_result_keys(self, identity_task):
        result = evaluate_code(IDENTITY_CODE, identity_task)
        assert set(result.keys()) == {"pairs", "n_correct", "n_total", "all_correct"}

    def test_pairs_length(self, identity_task):
        result = evaluate_code(IDENTITY_CODE, identity_task)
        assert len(result["pairs"]) == len(identity_task["train"])

    def test_partial_correct(self, recolor_task):
        # Correct code for this specific recolor task
        correct_code = "def transform(g):\n    return recolor(g, 1, 2)"
        result = evaluate_code(correct_code, recolor_task)
        assert result["all_correct"] is True

    def test_error_in_pair_recorded(self, identity_task):
        crash_code = "def transform(g):\n    raise RuntimeError('boom')"
        result = evaluate_code(crash_code, identity_task)
        assert result["n_correct"] == 0
        for pair in result["pairs"]:
            assert pair["error"] is not None

    def test_n_total_matches_train_pairs(self, identity_task):
        result = evaluate_code(IDENTITY_CODE, identity_task)
        assert result["n_total"] == len(identity_task["train"])


# ---------------------------------------------------------------------------
# DSL_NAMESPACE contents
# ---------------------------------------------------------------------------

class TestDSLNamespace:
    def test_required_functions_present(self):
        required = [
            "np", "numpy", "crop", "rotate", "flip", "translate",
            "scale", "tile", "recolor", "mask", "overlay",
            "flood_fill", "find_objects", "bounding_box", "crop_to_content",
        ]
        for name in required:
            assert name in DSL_NAMESPACE, f"Missing '{name}' from DSL_NAMESPACE"
