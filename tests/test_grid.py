"""Tests for arc/grid.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arc.grid import (
    load_task,
    grids_equal,
    grid_from_list,
    grid_to_list,
    unique_colors,
    background_color,
    COLOR_NAMES,
)


# ---------------------------------------------------------------------------
# load_task
# ---------------------------------------------------------------------------

class TestLoadTask:
    def test_loads_train_and_test(self, tmp_task_file):
        task = load_task(tmp_task_file)
        assert "train" in task
        assert "test" in task

    def test_train_pairs_are_numpy(self, tmp_task_file):
        task = load_task(tmp_task_file)
        for pair in task["train"]:
            assert isinstance(pair["input"], np.ndarray)
            assert isinstance(pair["output"], np.ndarray)
            assert pair["input"].dtype == np.int32

    def test_test_input_is_numpy(self, tmp_task_file):
        task = load_task(tmp_task_file)
        assert isinstance(task["test"][0]["input"], np.ndarray)

    def test_task_without_test_output(self, tmp_path):
        raw = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test":  [{"input": [[1]]}],   # no output
        }
        path = tmp_path / "no_test_out.json"
        path.write_text(json.dumps(raw))
        task = load_task(path)
        assert "output" not in task["test"][0]

    def test_correct_values(self, tmp_path):
        raw = {"train": [{"input": [[3, 5], [7, 9]], "output": [[0]]}], "test": [{"input": [[1]]}]}
        path = tmp_path / "vals.json"
        path.write_text(json.dumps(raw))
        task = load_task(path)
        np.testing.assert_array_equal(task["train"][0]["input"], [[3, 5], [7, 9]])


# ---------------------------------------------------------------------------
# grids_equal
# ---------------------------------------------------------------------------

class TestGridsEqual:
    def test_identical(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert grids_equal(a, a.copy())

    def test_different_values(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1, 3]], dtype=np.int32)
        assert not grids_equal(a, b)

    def test_different_shapes(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1], [2]], dtype=np.int32)
        assert not grids_equal(a, b)

    def test_single_cell(self):
        assert grids_equal(np.array([[5]], dtype=np.int32), np.array([[5]], dtype=np.int32))
        assert not grids_equal(np.array([[5]], dtype=np.int32), np.array([[6]], dtype=np.int32))

    def test_all_zeros(self):
        a = np.zeros((3, 3), dtype=np.int32)
        assert grids_equal(a, a.copy())


# ---------------------------------------------------------------------------
# grid_from_list / grid_to_list
# ---------------------------------------------------------------------------

class TestGridConversion:
    def test_roundtrip(self):
        lst = [[1, 2, 3], [4, 5, 6]]
        grid = grid_from_list(lst)
        assert grid.dtype == np.int32
        assert grid_to_list(grid) == lst

    def test_dtype_is_int32(self):
        grid = grid_from_list([[9]])
        assert grid.dtype == np.int32

    def test_to_list_native_ints(self):
        grid = np.array([[1, 2]], dtype=np.int32)
        result = grid_to_list(grid)
        assert isinstance(result[0][0], int)


# ---------------------------------------------------------------------------
# unique_colors
# ---------------------------------------------------------------------------

class TestUniqueColors:
    def test_sorted_unique(self):
        grid = np.array([[3, 1, 2, 1, 3]], dtype=np.int32)
        assert unique_colors(grid) == [1, 2, 3]

    def test_single_color(self):
        grid = np.zeros((4, 4), dtype=np.int32)
        assert unique_colors(grid) == [0]

    def test_all_colors(self):
        grid = np.arange(10, dtype=np.int32).reshape(2, 5)
        assert unique_colors(grid) == list(range(10))


# ---------------------------------------------------------------------------
# background_color
# ---------------------------------------------------------------------------

class TestBackgroundColor:
    def test_zero_present_always_background(self):
        grid = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.int32)
        assert background_color(grid) == 0

    def test_zero_even_if_minority(self):
        # 0 appears once, 1 appears 8 times — should still return 0
        grid = np.ones((3, 3), dtype=np.int32)
        grid[0, 0] = 0
        assert background_color(grid) == 0

    def test_no_zero_returns_most_frequent(self):
        grid = np.array([[1, 2, 2], [2, 3, 3]], dtype=np.int32)
        # 1×1, 3×2, 2×3 → most frequent is 2
        assert background_color(grid) == 2

    def test_color_names_coverage(self):
        # All 10 colours have names
        assert len(COLOR_NAMES) == 10
