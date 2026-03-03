"""Tests for agents/formatting.py — format_grid_visual()."""
from __future__ import annotations

import numpy as np
import pytest

from agents.formatting import format_grid_visual, LEGEND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# Basic visual format
# ---------------------------------------------------------------------------

class TestFormatGridVisualSmall:
    """Grids ≤10 in both dims: plain symbol rows, no index labels."""

    def test_single_cell(self):
        g = _make_grid([3])
        result = format_grid_visual(g)
        assert "G" in result
        assert LEGEND in result

    def test_3x3_symbols(self):
        g = _make_grid([0, 1, 2], [3, 4, 5], [6, 7, 8])
        result = format_grid_visual(g)
        assert ". B R" in result   # row 0
        assert "G Y X" in result   # row 1
        assert "M O A" in result   # row 2

    def test_color_9_shown_as_W(self):
        g = _make_grid([9])
        assert "W" in format_grid_visual(g)

    def test_all_ten_colors(self):
        g = _make_grid([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        result = format_grid_visual(g)
        for char in ".", "B", "R", "G", "Y", "X", "M", "O", "A", "W":
            assert char in result, f"Expected color char {char!r} in output"

    def test_cells_separated_by_spaces(self):
        g = _make_grid([1, 2])
        line = format_grid_visual(g).split("\n")[0]
        assert line == "B R"

    def test_10x10_no_labels(self):
        """Boundary: exactly 10×10 should have NO row/col index labels."""
        g = np.zeros((10, 10), dtype=np.int32)
        result = format_grid_visual(g)
        # No row labels like "0:" or " 0:"
        assert " 0:" not in result
        assert "0:" not in result.split("\n")[0]  # first line is a data row, not header

    def test_legend_always_present(self):
        for shape in [(1, 1), (5, 3), (10, 10)]:
            g = np.zeros(shape, dtype=np.int32)
            assert LEGEND in format_grid_visual(g)


# ---------------------------------------------------------------------------
# Large grid format (row/col index labels)
# ---------------------------------------------------------------------------

class TestFormatGridVisualLarge:
    """Grids >10 in at least one dim: row index labels + column header."""

    def test_11x11_has_row_labels(self):
        g = np.zeros((11, 11), dtype=np.int32)
        result = format_grid_visual(g)
        # r_width = len("10") = 2 → row 0 formatted as " 0: ."
        assert " 0:" in result

    def test_column_header_present(self):
        g = np.zeros((11, 5), dtype=np.int32)
        lines = format_grid_visual(g).split("\n")
        # First line is column header: "   0 1 2 3 4"
        assert "0" in lines[0] and "4" in lines[0]

    def test_row_label_width_scales(self):
        """100-row grid: row labels are zero-padded to 2 chars."""
        g = np.zeros((100, 3), dtype=np.int32)
        result = format_grid_visual(g)
        # Row 0 → " 0:" (r_width=2, because max label is "99")
        assert " 0:" in result
        # Row 99 → "99:"
        assert "99:" in result

    def test_tall_grid_has_all_row_labels(self):
        g = np.zeros((12, 3), dtype=np.int32)
        result = format_grid_visual(g)
        for r in range(12):
            assert f"{r}:" in result

    def test_legend_present_in_large_grid(self):
        g = np.zeros((15, 15), dtype=np.int32)
        assert LEGEND in format_grid_visual(g)

    def test_content_visible_in_large_grid(self):
        """Non-zero cell should still appear as its colour character."""
        g = np.zeros((12, 12), dtype=np.int32)
        g[0, 0] = 2  # red
        result = format_grid_visual(g)
        assert "R" in result
