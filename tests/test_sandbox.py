"""Tests for arc/sandbox.py — sandboxed code execution."""
from __future__ import annotations

import numpy as np
import pytest

from arc.sandbox import execute, evaluate_code, compute_spatial_diff, EXECUTION_TIMEOUT, DSL_NAMESPACE


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

# ---------------------------------------------------------------------------
# compute_spatial_diff()
# ---------------------------------------------------------------------------

class TestComputeSpatialDiff:
    def test_none_predicted(self):
        exp  = np.array([[1, 0], [0, 1]], dtype=np.int32)
        result = compute_spatial_diff(None, exp)
        assert "no output" in result.lower()

    def test_perfect_match(self):
        grid   = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = compute_spatial_diff(grid, grid)
        assert "match perfectly" in result.lower()

    def test_shape_mismatch_mentions_dimensions(self):
        pred = np.array([[1, 2, 3]], dtype=np.int32)      # 1×3
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)  # 2×2
        result = compute_spatial_diff(pred, exp)
        assert "1" in result and "3" in result or "shape" in result.lower() or "wrong" in result.lower()
        # Must mention dimensions
        assert any(char.isdigit() for char in result)

    def test_same_shape_wrong_cells_mentions_region(self):
        pred = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        exp  = np.array([[1, 0, 2], [0, 0, 0], [1, 0, 2]], dtype=np.int32)
        result = compute_spatial_diff(pred, exp)
        assert "wrong" in result.lower() or "cell" in result.lower()
        # Should mention a region
        assert any(word in result.lower() for word in ["top", "bottom", "left", "right", "middle", "center"])

    def test_shift_detection_mentions_direction(self):
        # Blue object at top of expected, at bottom in predicted → shifted down
        pred = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int32)
        exp  = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        result = compute_spatial_diff(pred, exp)
        assert any(d in result.lower() for d in ["down", "up", "shifted", "rows"])

    def test_1d_array(self):
        pred = np.array([1, 2, 3], dtype=np.int32)
        exp  = np.array([[1, 2, 3]], dtype=np.int32)
        result = compute_spatial_diff(pred, exp)
        assert "non-2d" in result.lower()

    def test_transposed_shape_detected(self):
        # pr==ec and pc==er: 2×3 vs 3×2 → transposed message
        pred = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)     # 2×3
        exp  = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)   # 3×2
        result = compute_spatial_diff(pred, exp)
        assert "transposed" in result.lower()

    def test_wrong_width_double(self):
        # pr==er, pc==2*ec: "Width is 2× expected"
        pred = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)  # 2×4
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)               # 2×2
        result = compute_spatial_diff(pred, exp)
        assert "width" in result.lower() or "col" in result.lower()

    def test_wrong_width_fraction(self):
        # pr==er, ec==2*pc: "Width is 1/2 of expected"
        pred = np.array([[1], [2]], dtype=np.int32)              # 2×1
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)        # 2×2
        result = compute_spatial_diff(pred, exp)
        assert "width" in result.lower() or "col" in result.lower()

    def test_wrong_height_double(self):
        # pc==ec, pr==2*er: "Height is 2× expected"
        pred = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)  # 4×2
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)                   # 2×2
        result = compute_spatial_diff(pred, exp)
        assert "height" in result.lower() or "row" in result.lower()

    def test_wrong_height_fraction(self):
        # pc==ec, er==2*pr: "Height is 1/2 of expected"
        pred = np.array([[1, 2]], dtype=np.int32)                    # 1×2
        exp  = np.array([[1, 2], [3, 4]], dtype=np.int32)            # 2×2
        result = compute_spatial_diff(pred, exp)
        assert "height" in result.lower() or "row" in result.lower()

    def test_overlapping_region_matches_noted(self):
        # Top-left 2×2 of pred matches top-left 2×2 of a 3×3 exp
        pred = np.array([[1, 2], [3, 4]], dtype=np.int32)               # 2×2
        exp  = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype=np.int32)  # 3×3
        result = compute_spatial_diff(pred, exp)
        assert "overlap" in result.lower() or "match" in result.lower()


class TestDSLNamespace:
    def test_required_functions_present(self):
        required = [
            "np", "numpy", "crop", "rotate", "flip", "translate",
            "scale", "tile", "recolor", "mask", "overlay",
            "flood_fill", "find_objects", "bounding_box", "crop_to_content",
            "pad", "symmetrize", "get_color", "get_size", "get_centroid",
            "detect_grid_layout", "find_periodicity", "gravity",
        ]
        for name in required:
            assert name in DSL_NAMESPACE, f"Missing '{name}' from DSL_NAMESPACE"


# ---------------------------------------------------------------------------
# param_search
# ---------------------------------------------------------------------------

from arc.sandbox import param_search


class TestParamSearch:
    """Tests for CPU-parallel parameter sweep."""

    TASK = {
        "train": [
            {
                "input":  np.array([[1, 0, 0]], dtype=np.int32),
                "output": np.array([[2, 0, 0]], dtype=np.int32),
            },
            {
                "input":  np.array([[0, 1, 0]], dtype=np.int32),
                "output": np.array([[0, 2, 0]], dtype=np.int32),
            },
        ]
    }

    GOOD_CODE = """\
PARAM_GRID = dict(src=list(range(10)), dst=list(range(10)))

def transform(grid, src=0, dst=0):
    result = grid.copy()
    result[result == src] = dst
    return result
"""

    def test_finds_correct_params(self):
        params, fitness = param_search(self.GOOD_CODE, self.TASK, timeout=30.0)
        assert fitness == pytest.approx(1.0, abs=1e-6)
        assert params.get("src") == 1
        assert params.get("dst") == 2

    def test_no_param_grid_returns_zero(self):
        code = "def transform(grid):\n    return grid.copy()\n"
        params, fitness = param_search(code, self.TASK, timeout=10.0)
        assert fitness == pytest.approx(0.0)
        assert params == {}

    def test_syntax_error_returns_zero(self):
        params, fitness = param_search("def f(: pass", self.TASK, timeout=5.0)
        assert params == {}
        assert fitness == pytest.approx(0.0)

    def test_returns_best_not_first(self):
        # PARAM_GRID has many combos; best (src=1,dst=2) should win
        params, fitness = param_search(self.GOOD_CODE, self.TASK,
                                       max_combinations=500, timeout=30.0)
        assert fitness == pytest.approx(1.0, abs=1e-6)

    def test_max_combinations_respected(self):
        # Even with only 1 combination tested, function returns without error
        params, fitness = param_search(self.GOOD_CODE, self.TASK,
                                       max_combinations=1, timeout=10.0)
        assert isinstance(params, dict)
        assert 0.0 <= fitness <= 1.0


# ---------------------------------------------------------------------------
# _subprocess_worker direct call tests (no forking — covers subprocess body)
# ---------------------------------------------------------------------------

from arc.sandbox import _subprocess_worker, _param_search_worker
from queue import Queue as ThreadQueue   # threading queue — works in-process without fork


class TestSubprocessWorkerDirect:
    """Call _subprocess_worker directly in-process to get line coverage.

    Uses a threading.Queue (not multiprocessing.Queue) because the fork-context
    Queue uses pipes that require a real subprocess to function correctly.
    """

    def _q(self):
        return ThreadQueue()

    def test_compile_error_reports_error(self):
        q = self._q()
        _subprocess_worker("def f(: pass", [[1, 0]], q)
        status, msg = q.get_nowait()
        assert status == "error"
        assert "Compile error" in msg

    def test_no_transform_function_reports_error(self):
        q = self._q()
        _subprocess_worker("x = 42", [[1, 0]], q)
        status, msg = q.get_nowait()
        assert status == "error"
        assert "transform" in msg.lower()

    def test_runtime_error_reports_error(self):
        q = self._q()
        code = "def transform(g):\n    raise ValueError('boom')\n"
        _subprocess_worker(code, [[1, 0]], q)
        status, msg = q.get_nowait()
        assert status == "error"
        assert "Runtime error" in msg

    def test_non_ndarray_result_converted(self):
        q = self._q()
        code = "def transform(g):\n    return g.tolist()\n"
        _subprocess_worker(code, [[1, 2]], q)
        status, val = q.get_nowait()
        assert status == "ok"
        assert val == [[1, 2]]

    def test_non_transform_user_function_used(self):
        # If no 'transform' found, the last user-defined callable is used
        q = self._q()
        code = "def my_fn(g):\n    return g.copy()\n"
        _subprocess_worker(code, [[5]], q)
        status, val = q.get_nowait()
        assert status == "ok"


class TestParamSearchWorkerDirect:
    """Call _param_search_worker directly in-process."""

    TASK_PAIRS = [
        ([[1, 0]], [[2, 0]]),
    ]

    def _q(self):
        return ThreadQueue()

    def test_no_param_grid_error(self):
        q = self._q()
        code = "def transform(g):\n    return g.copy()\n"
        _param_search_worker(code, self.TASK_PAIRS, q, max_combinations=10)
        status, _ = q.get_nowait()
        assert status == "error"

    def test_compile_error(self):
        q = self._q()
        _param_search_worker("def f(: pass", self.TASK_PAIRS, q, max_combinations=10)
        status, _ = q.get_nowait()
        assert status == "error"

    def test_finds_best_params(self):
        q = self._q()
        code = "PARAM_GRID = dict(v=list(range(5)))\ndef transform(g, v=0):\n    r = g.copy(); r[r==1]=v; return r\n"
        _param_search_worker(code, self.TASK_PAIRS, q, max_combinations=100)
        status, (best_params, best_fitness) = q.get_nowait()
        assert status == "ok"
        assert best_params.get("v") == 2
        assert best_fitness == pytest.approx(1.0, abs=0.01)


class TestParamSearchStdinGuard:
    def test_stdin_code_blocked(self):
        task = {"train": [
            {"input":  np.array([[1]], dtype=np.int32),
             "output": np.array([[1]], dtype=np.int32)},
        ]}
        params, fitness = param_search("x = input('hi')", task)
        assert params == {}
        assert fitness == pytest.approx(0.0)
