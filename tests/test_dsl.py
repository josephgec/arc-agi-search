"""Tests for arc/dsl.py — all pure grid transformation primitives."""
from __future__ import annotations

import numpy as np
import pytest

from arc.dsl import (
    crop, rotate, flip, translate, scale, tile,
    recolor, mask, overlay, flood_fill,
    find_objects, bounding_box, crop_to_content,
    pad, symmetrize,
    get_color, get_size, get_centroid,
    detect_grid_layout, find_periodicity, gravity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def g(*rows) -> np.ndarray:
    """Shorthand: g([1,2],[3,4]) → int32 numpy array."""
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

class TestCrop:
    def test_basic(self):
        grid = g([1, 2, 3], [4, 5, 6], [7, 8, 9])
        result = crop(grid, 0, 1, 2, 3)
        np.testing.assert_array_equal(result, g([2, 3], [5, 6]))

    def test_single_cell(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(crop(grid, 1, 1, 2, 2), g([4]))

    def test_does_not_mutate(self):
        grid = g([1, 2], [3, 4])
        original = grid.copy()
        crop(grid, 0, 0, 1, 1)
        np.testing.assert_array_equal(grid, original)

    def test_full_grid(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(crop(grid, 0, 0, 2, 2), grid)


class TestRotate:
    def test_90_once(self):
        grid = g([1, 2], [3, 4])
        # 90° CCW: top-left becomes bottom-left
        result = rotate(grid, 1)
        np.testing.assert_array_equal(result, g([2, 4], [1, 3]))

    def test_180(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(rotate(grid, 2), g([4, 3], [2, 1]))

    def test_360_identity(self):
        grid = g([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(rotate(grid, 4), grid)

    def test_does_not_mutate(self):
        grid = g([1, 2], [3, 4])
        original = grid.copy()
        rotate(grid)
        np.testing.assert_array_equal(grid, original)


class TestFlip:
    def test_vertical(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(flip(grid, axis=0), g([3, 4], [1, 2]))

    def test_horizontal(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(flip(grid, axis=1), g([2, 1], [4, 3]))

    def test_double_flip_identity(self):
        grid = g([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(flip(flip(grid, 0), 0), grid)

    def test_does_not_mutate(self):
        grid = g([1, 2], [3, 4])
        original = grid.copy()
        flip(grid)
        np.testing.assert_array_equal(grid, original)


class TestTranslate:
    def test_down(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, dr=1, dc=0)
        np.testing.assert_array_equal(result, g([0, 0], [1, 2]))

    def test_right(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, dr=0, dc=1)
        np.testing.assert_array_equal(result, g([0, 1], [0, 3]))

    def test_up(self):
        grid = g([0, 0], [1, 2])
        result = translate(grid, dr=-1, dc=0)
        np.testing.assert_array_equal(result, g([1, 2], [0, 0]))

    def test_custom_fill(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, dr=1, dc=0, fill=9)
        np.testing.assert_array_equal(result, g([9, 9], [1, 2]))

    def test_zero_shift_identity(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(translate(grid, 0, 0), grid)


class TestScale:
    def test_2x(self):
        # scale() uses np.kron which scales BOTH dimensions
        grid = g([1, 2])   # shape (1, 2)
        result = scale(grid, 2)  # → shape (2, 4)
        np.testing.assert_array_equal(result, g([1, 1, 2, 2], [1, 1, 2, 2]))

    def test_2x_2d(self):
        grid = g([1, 2], [3, 4])
        expected = g([1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4])
        np.testing.assert_array_equal(scale(grid, 2), expected)

    def test_1x_identity(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(scale(grid, 1), grid)


class TestTile:
    def test_1x3(self):
        grid = g([1, 2])
        np.testing.assert_array_equal(tile(grid, 1, 3), g([1, 2, 1, 2, 1, 2]))

    def test_2x2(self):
        grid = g([1])
        np.testing.assert_array_equal(tile(grid, 2, 2), g([1, 1], [1, 1]))

    def test_shape(self):
        grid = g([1, 2], [3, 4])
        result = tile(grid, 3, 2)
        assert result.shape == (6, 4)


# ---------------------------------------------------------------------------
# Colour operations
# ---------------------------------------------------------------------------

class TestRecolor:
    def test_basic(self):
        grid = g([1, 0, 1], [0, 1, 0])
        result = recolor(grid, from_color=1, to_color=5)
        np.testing.assert_array_equal(result, g([5, 0, 5], [0, 5, 0]))

    def test_noop_when_absent(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(recolor(grid, 9, 5), grid)

    def test_does_not_mutate(self):
        grid = g([1, 2], [3, 4])
        original = grid.copy()
        recolor(grid, 1, 5)
        np.testing.assert_array_equal(grid, original)


class TestMask:
    def test_zeros_mask_erases(self):
        grid = g([1, 2], [3, 4])
        m    = g([1, 0], [0, 1])
        result = mask(grid, m)
        np.testing.assert_array_equal(result, g([1, 0], [0, 4]))

    def test_all_ones_preserves(self):
        grid = g([1, 2], [3, 4])
        m    = g([1, 1], [1, 1])
        np.testing.assert_array_equal(mask(grid, m), grid)


class TestOverlay:
    def test_transparent_zero_skipped(self):
        base = g([1, 1], [1, 1])
        top  = g([0, 2], [3, 0])
        result = overlay(base, top, transparent=0)
        np.testing.assert_array_equal(result, g([1, 2], [3, 1]))

    def test_full_overlay(self):
        base = g([1, 1], [1, 1])
        top  = g([5, 5], [5, 5])
        np.testing.assert_array_equal(overlay(base, top, transparent=-1), top)


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

class TestFloodFill:
    def test_basic(self):
        grid = g([1, 1, 0], [1, 0, 0], [0, 0, 0])
        result = flood_fill(grid, 0, 0, 7)
        np.testing.assert_array_equal(result[0, 0], 7)
        np.testing.assert_array_equal(result[0, 1], 7)
        np.testing.assert_array_equal(result[1, 0], 7)
        assert result[0, 2] == 0   # not connected

    def test_noop_if_same_color(self):
        grid = g([1, 1], [1, 1])
        result = flood_fill(grid, 0, 0, 1)
        np.testing.assert_array_equal(result, grid)

    def test_does_not_mutate(self):
        grid = g([1, 1], [1, 0])
        original = grid.copy()
        flood_fill(grid, 0, 0, 9)
        np.testing.assert_array_equal(grid, original)

    def test_boundary_not_crossed(self):
        grid = g([1, 0, 2], [0, 0, 0], [3, 0, 4])
        result = flood_fill(grid, 1, 1, 9)
        # All 0s connected to (1,1) become 9; corners unchanged
        assert result[0, 0] == 1
        assert result[0, 2] == 2


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

class TestFindObjects:
    def test_single_object(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        assert objs[0]["color"] == 1
        assert objs[0]["pixels"] == [(1, 1)]

    def test_two_objects(self):
        grid = g([1, 0, 2], [0, 0, 0], [0, 0, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 2
        colors = {o["color"] for o in objs}
        assert colors == {1, 2}

    def test_object_bbox(self):
        grid = g([0, 0, 0], [0, 1, 1], [0, 1, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        r_min, c_min, r_max, c_max = objs[0]["bbox"]
        assert r_min == 1 and r_max == 2
        assert c_min == 1 and c_max == 2

    def test_subgrid_correct(self):
        grid = g([0, 0], [0, 3])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        np.testing.assert_array_equal(objs[0]["subgrid"], g([3]))

    def test_empty_grid(self):
        grid = g([0, 0], [0, 0])
        objs = find_objects(grid, background=0)
        assert objs == []

    def test_auto_background_detection(self):
        # Background is 0 (always, since 0 is present)
        grid = g([0, 1], [0, 1])
        objs = find_objects(grid)
        assert len(objs) == 1


class TestBoundingBox:
    def test_full_content(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        assert bounding_box(grid) == (1, 1, 1, 1)

    def test_specific_color(self):
        grid = g([1, 0, 2], [0, 0, 0], [2, 0, 1])
        r_min, c_min, r_max, c_max = bounding_box(grid, color=2)
        assert r_min == 0 and r_max == 2
        assert c_min == 0 and c_max == 2

    def test_all_background_returns_zero(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        assert bounding_box(grid) == (0, 0, 0, 0)


class TestCropToContent:
    def test_removes_border_zeros(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        result = crop_to_content(grid)
        np.testing.assert_array_equal(result, g([1]))

    def test_all_background_unchanged(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        result = crop_to_content(grid)
        np.testing.assert_array_equal(result, grid)

    def test_no_padding_unchanged(self):
        grid = g([1, 2], [3, 4])
        result = crop_to_content(grid)
        np.testing.assert_array_equal(result, grid)

    def test_asymmetric_padding(self):
        grid = g([0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0])
        result = crop_to_content(grid)
        np.testing.assert_array_equal(result, g([1, 2]))


class TestPad:
    def test_top_padding(self):
        grid = g([1, 2], [3, 4])
        result = pad(grid, top=1)
        np.testing.assert_array_equal(result, g([0, 0], [1, 2], [3, 4]))

    def test_all_sides(self):
        grid = g([1])
        result = pad(grid, top=1, bottom=1, left=1, right=1)
        np.testing.assert_array_equal(result, g([0, 0, 0], [0, 1, 0], [0, 0, 0]))

    def test_custom_fill(self):
        grid = g([1, 2])
        result = pad(grid, right=1, fill=9)
        np.testing.assert_array_equal(result, g([1, 2, 9]))

    def test_no_padding_unchanged(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(pad(grid), grid)


class TestSymmetrize:
    def test_left_right(self):
        grid = g([1, 2, 0, 0], [3, 4, 0, 0])
        result = symmetrize(grid, axis=0)
        np.testing.assert_array_equal(result, g([1, 2, 2, 1], [3, 4, 4, 3]))

    def test_top_bottom(self):
        grid = g([1, 2], [3, 4], [0, 0], [0, 0])
        result = symmetrize(grid, axis=1)
        np.testing.assert_array_equal(result, g([1, 2], [3, 4], [3, 4], [1, 2]))

    def test_preserves_existing_nonzero(self):
        # Existing non-zero in dest half should not be overwritten
        grid = g([1, 0, 5], [3, 0, 0])
        result = symmetrize(grid, axis=0)
        # mirror col 0→col 2: dest col[2] already has 5, keeps 5
        assert result[0, 2] == 5

    def test_both_axes(self):
        grid = g([1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
        result = symmetrize(grid, axis=2)
        assert result.shape == (4, 4)
        # top-left should be preserved
        assert result[0, 0] == 1


# ---------------------------------------------------------------------------
# Object property helpers
# ---------------------------------------------------------------------------

class TestGetColor:
    def test_single_color(self):
        obj = g([0, 3, 3], [0, 3, 0])
        assert get_color(obj) == 3

    def test_most_common_wins(self):
        # 2 appears 3 times, 5 appears 2 times
        obj = g([2, 5, 2], [5, 2, 0])
        assert get_color(obj) == 2

    def test_all_zero_returns_zero(self):
        obj = g([0, 0], [0, 0])
        assert get_color(obj) == 0

    def test_does_not_mutate(self):
        obj = g([1, 2], [3, 0])
        original = obj.copy()
        get_color(obj)
        np.testing.assert_array_equal(obj, original)


class TestGetSize:
    def test_nonzero_count(self):
        obj = g([1, 0, 2], [0, 3, 0])
        assert get_size(obj) == 3

    def test_all_zero(self):
        obj = g([0, 0], [0, 0])
        assert get_size(obj) == 0

    def test_all_nonzero(self):
        obj = g([1, 2], [3, 4])
        assert get_size(obj) == 4

    def test_does_not_mutate(self):
        obj = g([1, 0], [0, 2])
        original = obj.copy()
        get_size(obj)
        np.testing.assert_array_equal(obj, original)


class TestGetCentroid:
    def test_single_cell(self):
        obj = g([0, 0, 0], [0, 5, 0], [0, 0, 0])
        cr, cc = get_centroid(obj)
        assert cr == 1.0 and cc == 1.0

    def test_symmetric_four_corners(self):
        obj = g([1, 0, 1], [0, 0, 0], [1, 0, 1])
        cr, cc = get_centroid(obj)
        assert cr == 1.0 and cc == 1.0

    def test_all_zero_returns_origin(self):
        obj = g([0, 0], [0, 0])
        assert get_centroid(obj) == (0.0, 0.0)

    def test_fractional_centroid(self):
        obj = g([1, 0], [0, 0], [1, 0])
        cr, cc = get_centroid(obj)
        assert cr == 1.0 and cc == 0.0

    def test_does_not_mutate(self):
        obj = g([1, 0], [0, 1])
        original = obj.copy()
        get_centroid(obj)
        np.testing.assert_array_equal(obj, original)


# ---------------------------------------------------------------------------
# Grid structure analysis
# ---------------------------------------------------------------------------

class TestDetectGridLayout:
    def test_horizontal_divider(self):
        # Row 1 is all 5s → divides into 2 row sections, 1 col section
        grid = g([1, 0, 2], [5, 5, 5], [0, 3, 1])
        result = detect_grid_layout(grid)
        assert result == (2, 1)

    def test_vertical_divider(self):
        # Col 1 is all 5s → 1 row section, 2 col sections
        grid = g([1, 5, 2], [0, 5, 3], [4, 5, 1])
        result = detect_grid_layout(grid)
        assert result == (1, 2)

    def test_cross_divider(self):
        # Row 1 and col 1 both all 5s → (2, 2)
        grid = g([1, 5, 2], [5, 5, 5], [3, 5, 4])
        result = detect_grid_layout(grid)
        assert result == (2, 2)

    def test_no_dividers_returns_none(self):
        grid = g([1, 2], [3, 4])
        assert detect_grid_layout(grid) is None

    def test_zero_row_not_a_divider(self):
        # All-zero row is NOT a divider (value must be non-zero)
        grid = g([1, 2], [0, 0], [3, 4])
        assert detect_grid_layout(grid) is None

    def test_does_not_mutate(self):
        grid = g([1, 5, 2], [0, 5, 3])
        original = grid.copy()
        detect_grid_layout(grid)
        np.testing.assert_array_equal(grid, original)


class TestFindPeriodicity:
    def test_row_period_2(self):
        grid = g([1, 2], [3, 4], [1, 2], [3, 4])
        result = find_periodicity(grid)
        assert result is not None
        assert result[0] == 2

    def test_col_period_3(self):
        # Each column repeats with period 3 across columns
        col = np.array([[1, 2, 3, 1, 2, 3]], dtype=np.int32)
        result = find_periodicity(col)
        assert result is not None
        assert result[1] == 3

    def test_non_periodic_returns_none(self):
        grid = g([1, 2, 3], [4, 5, 6])
        assert find_periodicity(grid) is None

    def test_single_row_col_period(self):
        # 1-row grid: row period trivially None, check col period
        grid = np.array([[1, 2, 1, 2]], dtype=np.int32)
        result = find_periodicity(grid)
        assert result is not None
        assert result[1] == 2

    def test_does_not_mutate(self):
        grid = g([1, 2], [1, 2])
        original = grid.copy()
        find_periodicity(grid)
        np.testing.assert_array_equal(grid, original)


# ---------------------------------------------------------------------------
# Gravity / physics
# ---------------------------------------------------------------------------

class TestGravity:
    def test_gravity_down(self):
        grid = g([1, 0], [0, 0], [0, 2])
        result = gravity(grid, "down")
        # col 0: [1,0,0] → [0,0,1]; col 1: [0,0,2] → [0,0,2]
        np.testing.assert_array_equal(result, g([0, 0], [0, 0], [1, 2]))

    def test_gravity_up(self):
        grid = g([0, 0], [1, 0], [0, 2])
        result = gravity(grid, "up")
        # col 0: [0,1,0] → [1,0,0]; col 1: [0,0,2] → [2,0,0]
        np.testing.assert_array_equal(result, g([1, 2], [0, 0], [0, 0]))

    def test_gravity_left(self):
        grid = g([0, 1, 0, 2])
        result = gravity(grid, "left")
        # row: [0,1,0,2] → [1,2,0,0]
        np.testing.assert_array_equal(result, g([1, 2, 0, 0]))

    def test_gravity_right(self):
        grid = g([1, 0, 2, 0])
        result = gravity(grid, "right")
        # row: [1,0,2,0] → [0,0,1,2]
        np.testing.assert_array_equal(result, g([0, 0, 1, 2]))

    def test_already_settled(self):
        grid = g([0, 0], [1, 2])
        result = gravity(grid, "down")
        np.testing.assert_array_equal(result, grid)

    def test_preserves_order_within_column(self):
        # Multiple non-zero cells: order preserved
        grid = g([3, 0], [0, 0], [7, 0])
        result = gravity(grid, "down")
        # col 0: [3,0,7] → zeros fill top, nz=[3,7] go to bottom
        np.testing.assert_array_equal(result, g([0, 0], [3, 0], [7, 0]))

    def test_all_zero(self):
        grid = g([0, 0], [0, 0])
        result = gravity(grid, "down")
        np.testing.assert_array_equal(result, grid)

    def test_invalid_direction_raises(self):
        grid = g([1, 2], [3, 4])
        with pytest.raises(ValueError):
            gravity(grid, "diagonal")

    def test_does_not_mutate(self):
        grid = g([1, 0], [0, 2])
        original = grid.copy()
        gravity(grid, "down")
        np.testing.assert_array_equal(grid, original)
