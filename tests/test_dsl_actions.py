"""Tests for arc.dsl_actions — DSL action enumeration for MCTS."""
from __future__ import annotations

import numpy as np
import pytest

from arc.dsl_actions import (
    DslAction,
    enumerate_actions,
    get_target_colors,
    get_target_shape,
    prune_actions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(inp, out, n_pairs=1):
    """Build a minimal ARC task dict."""
    inp = np.asarray(inp, dtype=np.int32)
    out = np.asarray(out, dtype=np.int32)
    return {
        "train": [{"input": inp.copy(), "output": out.copy()} for _ in range(n_pairs)],
        "test":  [{"input": inp.copy(), "output": out.copy()}],
    }


# ---------------------------------------------------------------------------
# DslAction
# ---------------------------------------------------------------------------

class TestDslAction:
    def test_to_code_fragment_no_args(self):
        a = DslAction("crop_to_content", ())
        assert a.to_code_fragment() == "crop_to_content(grid)"

    def test_to_code_fragment_single_arg(self):
        a = DslAction("rotate", (1,))
        assert a.to_code_fragment() == "rotate(grid, 1)"

    def test_to_code_fragment_multiple_args(self):
        a = DslAction("recolor", (3, 5))
        assert a.to_code_fragment() == "recolor(grid, 3, 5)"

    def test_to_code_fragment_string_arg(self):
        a = DslAction("gravity", ("down", 0))
        assert a.to_code_fragment() == "gravity(grid, 'down', 0)"

    def test_repr_matches_code_fragment(self):
        a = DslAction("flip", (0,))
        assert repr(a) == a.to_code_fragment()

    def test_frozen(self):
        a = DslAction("rotate", (1,))
        with pytest.raises(AttributeError):
            a.name = "flip"

    def test_hashable(self):
        a = DslAction("rotate", (1,))
        b = DslAction("rotate", (1,))
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# get_target_colors / get_target_shape
# ---------------------------------------------------------------------------

class TestTargetHelpers:
    def test_get_target_colors(self):
        task = _make_task([[0, 1]], [[0, 5, 3]])
        colors = get_target_colors(task)
        assert colors == {0, 3, 5}

    def test_get_target_colors_multi_pair(self):
        task = {
            "train": [
                {"input": np.zeros((2, 2), dtype=np.int32),
                 "output": np.array([[1, 2], [0, 0]], dtype=np.int32)},
                {"input": np.zeros((2, 2), dtype=np.int32),
                 "output": np.array([[3, 0], [0, 0]], dtype=np.int32)},
            ],
            "test": [],
        }
        assert get_target_colors(task) == {0, 1, 2, 3}

    def test_get_target_shape_consistent(self):
        task = {
            "train": [
                {"input": np.zeros((3, 3), dtype=np.int32),
                 "output": np.zeros((2, 4), dtype=np.int32)},
                {"input": np.zeros((3, 3), dtype=np.int32),
                 "output": np.zeros((2, 4), dtype=np.int32)},
            ],
            "test": [],
        }
        assert get_target_shape(task) == (2, 4)

    def test_get_target_shape_inconsistent(self):
        task = {
            "train": [
                {"input": np.zeros((3, 3), dtype=np.int32),
                 "output": np.zeros((2, 4), dtype=np.int32)},
                {"input": np.zeros((3, 3), dtype=np.int32),
                 "output": np.zeros((5, 5), dtype=np.int32)},
            ],
            "test": [],
        }
        assert get_target_shape(task) is None


# ---------------------------------------------------------------------------
# enumerate_actions
# ---------------------------------------------------------------------------

class TestEnumerateActions:
    def test_returns_list_of_dsl_actions(self):
        grid = np.array([[1, 2], [3, 0]], dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        assert isinstance(actions, list)
        assert all(isinstance(a, DslAction) for a in actions)

    def test_rotate_actions_always_present(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        names_args = {(a.name, a.args) for a in actions}
        assert ("rotate", (1,)) in names_args
        assert ("rotate", (2,)) in names_args
        assert ("rotate", (3,)) in names_args

    def test_flip_actions_always_present(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        names_args = {(a.name, a.args) for a in actions}
        assert ("flip", (0,)) in names_args
        assert ("flip", (1,)) in names_args

    def test_crop_to_content_always_present(self):
        grid = np.ones((4, 4), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        assert any(a.name == "crop_to_content" for a in actions)

    def test_symmetrize_actions_present(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        sym_axes = {a.args[0] for a in actions if a.name == "symmetrize"}
        assert sym_axes == {0, 1, 2}

    def test_gravity_actions_present(self):
        grid = np.array([[1, 0], [0, 2]], dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        grav_dirs = {a.args[0] for a in actions if a.name == "gravity"}
        assert grav_dirs == {"up", "down", "left", "right"}

    def test_recolor_only_targets_output_colors(self):
        inp = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.array([[0, 5, 7]], dtype=np.int32)
        task = _make_task(inp, out)
        actions = enumerate_actions(inp, task)
        recolor_targets = {a.args[1] for a in actions if a.name == "recolor"}
        target_colors = get_target_colors(task)
        assert recolor_targets.issubset(target_colors)

    def test_scale_excluded_for_large_grids(self):
        grid = np.zeros((20, 20), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        scale_factors = {a.args[0] for a in actions if a.name == "scale"}
        # 20×2 = 40 > 30, so scale(2) should be excluded
        assert 2 not in scale_factors
        assert 3 not in scale_factors

    def test_tile_excluded_for_large_grids(self):
        grid = np.zeros((20, 20), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        assert not any(a.name == "tile" for a in actions)

    def test_translate_actions_present(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        task = _make_task(grid, grid)
        actions = enumerate_actions(grid, task)
        translate_count = sum(1 for a in actions if a.name == "translate")
        assert translate_count == 24  # 5×5 - (0,0) = 24


# ---------------------------------------------------------------------------
# prune_actions
# ---------------------------------------------------------------------------

class TestPruneActions:
    def test_prune_removes_shape_ops_when_shape_correct(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        target = np.zeros((3, 3), dtype=np.int32)
        task = _make_task(grid, target)
        actions = enumerate_actions(grid, task)
        pruned = prune_actions(actions, grid, target)
        shape_ops = {"scale", "tile", "pad", "crop", "crop_to_content"}
        assert not any(a.name in shape_ops for a in pruned)

    def test_prune_removes_color_ops_when_palette_correct(self):
        grid = np.array([[0, 1, 2]], dtype=np.int32)
        target = np.array([[2, 1, 0]], dtype=np.int32)
        task = _make_task(grid, target)
        actions = enumerate_actions(grid, task)
        pruned = prune_actions(actions, grid, target)
        color_ops = {"recolor", "fill_enclosed_regions"}
        assert not any(a.name in color_ops for a in pruned)

    def test_prune_keeps_shape_ops_when_shape_wrong(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        target = np.zeros((6, 6), dtype=np.int32)
        task = _make_task(grid, target)
        actions = enumerate_actions(grid, task)
        pruned = prune_actions(actions, grid, target)
        assert any(a.name == "scale" for a in pruned)

    def test_total_action_cap(self):
        # Create a grid with many colors to force many recolor combos
        grid = np.arange(10, dtype=np.int32).reshape(2, 5)
        # Target also has many colors → many recolor pairs
        target = np.arange(10, dtype=np.int32).reshape(2, 5)[:, ::-1]
        task = _make_task(grid, target)
        actions = enumerate_actions(grid, task)
        pruned = prune_actions(actions, grid, target)
        assert len(pruned) <= 120

    def test_prune_removes_oversized_scale(self):
        grid = np.zeros((16, 16), dtype=np.int32)
        target = np.zeros((20, 20), dtype=np.int32)
        actions = [DslAction("scale", (2,)), DslAction("rotate", (1,))]
        pruned = prune_actions(actions, grid, target)
        # scale(2) on 16×16 → 32×32 > 30, should be removed
        assert not any(a.name == "scale" for a in pruned)
        assert any(a.name == "rotate" for a in pruned)

    def test_prune_removes_oversized_tile(self):
        grid = np.zeros((16, 16), dtype=np.int32)
        target = np.zeros((20, 20), dtype=np.int32)
        actions = [DslAction("tile", (2, 2)), DslAction("rotate", (1,))]
        pruned = prune_actions(actions, grid, target)
        # tile(2,2) on 16×16 → 32×32 > 30, should be removed
        assert not any(a.name == "tile" for a in pruned)
        assert any(a.name == "rotate" for a in pruned)

    def test_prune_removes_oversized_pad(self):
        grid = np.zeros((28, 28), dtype=np.int32)
        target = np.zeros((32, 32), dtype=np.int32)
        actions = [DslAction("pad", (2, 2, 2, 2)), DslAction("rotate", (1,))]
        pruned = prune_actions(actions, grid, target)
        # pad(2,2,2,2) on 28×28 → 32×32 > 30, should be removed
        assert not any(a.name == "pad" for a in pruned)
        assert any(a.name == "rotate" for a in pruned)

    def test_cap_120_shape_priority(self):
        """When >120 actions and shape is wrong, shape ops are prioritized."""
        grid = np.zeros((3, 3), dtype=np.int32)
        target = np.zeros((6, 6), dtype=np.int32)
        # Build >120 actions: scale + many fake actions
        actions = [DslAction("scale", (2,))]
        for i in range(130):
            actions.append(DslAction("rotate", (1,)))
        pruned = prune_actions(actions, grid, target)
        assert len(pruned) <= 120
        # scale should be preserved as a shape op
        assert any(a.name == "scale" for a in pruned)

    def test_cap_120_color_priority(self):
        """When >120 actions, shape correct, palette wrong → color ops prioritized."""
        grid = np.array([[0, 1]], dtype=np.int32)
        target = np.array([[0, 5]], dtype=np.int32)
        # Same shape, different colors. Build >120 actions with recolor.
        actions = [DslAction("recolor", (1, 5))]
        for i in range(130):
            actions.append(DslAction("rotate", (1,)))
        pruned = prune_actions(actions, grid, target)
        assert len(pruned) <= 120
        # recolor should be preserved as a color op
        assert any(a.name == "recolor" for a in pruned)

    def test_cap_120_random_sample(self):
        """When >120 actions, shape and palette both correct → random sample."""
        grid = np.array([[0, 1, 2]], dtype=np.int32)
        target = np.array([[0, 1, 2]], dtype=np.int32)
        # Same shape, same colors. Build >120 non-shape/color actions.
        actions = []
        for i in range(130):
            actions.append(DslAction("translate", (i % 5 - 2, i // 5 % 5 - 2)))
        pruned = prune_actions(actions, grid, target)
        assert len(pruned) <= 120


# ---------------------------------------------------------------------------
# _crop_actions
# ---------------------------------------------------------------------------

class TestCropActions:
    def test_crop_actions_with_consistent_smaller_target(self):
        """Crop actions are generated when target is smaller than grid."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        target = np.array([[1, 1], [1, 1]], dtype=np.int32)
        task = {
            "train": [
                {"input": grid.copy(), "output": target.copy()},
                {"input": grid.copy(), "output": target.copy()},
            ],
            "test": [],
        }
        actions = enumerate_actions(grid, task)
        crop_actions = [a for a in actions if a.name == "crop"]
        assert len(crop_actions) > 0
        # Should include crop based on color 1's bounding box
        assert any(a.args == (1, 1, 3, 3) for a in crop_actions)

    def test_crop_actions_none_when_inconsistent_shape(self):
        """No crop actions when target shapes are inconsistent."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1, 1] = 1
        task = {
            "train": [
                {"input": grid.copy(),
                 "output": np.zeros((2, 2), dtype=np.int32)},
                {"input": grid.copy(),
                 "output": np.zeros((3, 3), dtype=np.int32)},
            ],
            "test": [],
        }
        actions = enumerate_actions(grid, task)
        assert not any(a.name == "crop" for a in actions)

    def test_crop_actions_none_when_target_larger(self):
        """No crop actions when target is larger than grid."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target = np.zeros((5, 5), dtype=np.int32)
        task = {
            "train": [
                {"input": grid.copy(), "output": target.copy()},
                {"input": grid.copy(), "output": target.copy()},
            ],
            "test": [],
        }
        actions = enumerate_actions(grid, task)
        assert not any(a.name == "crop" for a in actions)
