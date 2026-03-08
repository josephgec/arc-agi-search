"""Enumerate DSL actions for MCTS tree expansion.

Each action is a (name, args_dict) tuple that can be applied to a grid.
Argument values are derived from the grid and task context at expansion
time so the tree only explores plausible operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arc.grid import Grid, unique_colors, background_color


# ---------------------------------------------------------------------------
# DslAction dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DslAction:
    """A single DSL operation with concrete arguments."""

    name: str
    args: tuple  # positional args AFTER the grid argument

    def to_code_fragment(self) -> str:
        """Return a Python expression like 'rotate(grid, 1)'."""
        arg_strs = [repr(a) for a in self.args]
        if arg_strs:
            return f"{self.name}(grid, {', '.join(arg_strs)})"
        return f"{self.name}(grid)"

    def __repr__(self) -> str:
        return self.to_code_fragment()


# ---------------------------------------------------------------------------
# Task-level helpers
# ---------------------------------------------------------------------------

def get_target_colors(task: dict) -> set[int]:
    """Union of all unique colors across all training output grids."""
    colors: set[int] = set()
    for pair in task.get("train", []):
        out = pair.get("output")
        if out is not None:
            arr = np.asarray(out, dtype=np.int32)
            colors.update(int(c) for c in np.unique(arr))
    return colors


def get_target_shape(task: dict) -> tuple[int, int] | None:
    """Return the output shape if consistent across all training pairs, else None."""
    shapes: set[tuple[int, int]] = set()
    for pair in task.get("train", []):
        out = pair.get("output")
        if out is not None:
            arr = np.asarray(out, dtype=np.int32)
            shapes.add(arr.shape)
    if len(shapes) == 1:
        return shapes.pop()
    return None


# ---------------------------------------------------------------------------
# Per-action enumeration helpers
# ---------------------------------------------------------------------------

_MAX_OUTPUT_DIM = 30


def _rotate_actions() -> list[DslAction]:
    return [DslAction("rotate", (n,)) for n in (1, 2, 3)]


def _flip_actions() -> list[DslAction]:
    return [DslAction("flip", (ax,)) for ax in (0, 1)]


def _crop_to_content_actions() -> list[DslAction]:
    return [DslAction("crop_to_content", ())]


def _symmetrize_actions() -> list[DslAction]:
    return [DslAction("symmetrize", (ax,)) for ax in (0, 1, 2)]


def _gravity_actions(grid: Grid) -> list[DslAction]:
    bg = int(background_color(grid))
    actions: list[DslAction] = []
    for direction in ("up", "down", "left", "right"):
        actions.append(DslAction("gravity", (direction, 0)))
        if bg != 0:
            actions.append(DslAction("gravity", (direction, bg)))
    return actions


def _fill_enclosed_actions(grid: Grid) -> list[DslAction]:
    colors = sorted(int(c) for c in unique_colors(grid))
    # Existing colors + up to 2 new ones
    new_colors = [c for c in range(10) if c not in colors][:2]
    fill_colors = colors + new_colors
    return [DslAction("fill_enclosed_regions", (fc,)) for fc in fill_colors[:12]]


def _recolor_actions(grid: Grid, task: dict) -> list[DslAction]:
    grid_colors = set(int(c) for c in unique_colors(grid))
    target_colors = get_target_colors(task)
    actions: list[DslAction] = []
    for from_c in sorted(grid_colors):
        for to_c in sorted(target_colors - {from_c}):
            actions.append(DslAction("recolor", (from_c, to_c)))
    return actions


def _scale_actions(grid: Grid, task: dict) -> list[DslAction]:
    rows, cols = grid.shape
    actions: list[DslAction] = []
    for factor in (2, 3):
        if rows * factor <= _MAX_OUTPUT_DIM and cols * factor <= _MAX_OUTPUT_DIM:
            actions.append(DslAction("scale", (factor,)))
    return actions


def _tile_actions(grid: Grid) -> list[DslAction]:
    rows, cols = grid.shape
    actions: list[DslAction] = []
    for nr in (1, 2, 3):
        for nc in (1, 2, 3):
            if (nr, nc) == (1, 1):
                continue
            if rows * nr <= _MAX_OUTPUT_DIM and cols * nc <= _MAX_OUTPUT_DIM:
                actions.append(DslAction("tile", (nr, nc)))
    return actions


def _translate_actions() -> list[DslAction]:
    actions: list[DslAction] = []
    for dr in (-2, -1, 0, 1, 2):
        for dc in (-2, -1, 0, 1, 2):
            if (dr, dc) != (0, 0):
                actions.append(DslAction("translate", (dr, dc)))
    return actions


def _pad_actions(grid: Grid) -> list[DslAction]:
    rows, cols = grid.shape
    actions: list[DslAction] = []
    for p in (1, 2):
        if rows + 2 * p <= _MAX_OUTPUT_DIM and cols + 2 * p <= _MAX_OUTPUT_DIM:
            actions.append(DslAction("pad", (p, p, p, p)))
    return actions


def _crop_actions(grid: Grid, task: dict) -> list[DslAction]:
    """Enumerate crops based on bounding boxes of non-bg colors."""
    target_shape = get_target_shape(task)
    if target_shape is None:
        return []
    tr, tc = target_shape
    rows, cols = grid.shape
    # Only enumerate crops if the target is smaller than the grid
    if tr >= rows and tc >= cols:
        return []

    bg = int(background_color(grid))
    colors = sorted(set(int(c) for c in unique_colors(grid)) - {bg})
    actions: list[DslAction] = []
    seen: set[tuple[int, int, int, int]] = set()
    for color in colors:
        mask = grid == color
        if not np.any(mask):
            continue
        rs = np.any(mask, axis=1)
        cs = np.any(mask, axis=0)
        if not np.any(rs):
            continue
        r_min, r_max = int(np.where(rs)[0][0]), int(np.where(rs)[0][-1])
        c_min, c_max = int(np.where(cs)[0][0]), int(np.where(cs)[0][-1])
        key = (r_min, c_min, r_max + 1, c_max + 1)
        if key not in seen:
            seen.add(key)
            actions.append(DslAction("crop", (r_min, c_min, r_max + 1, c_max + 1)))
        if len(actions) >= 10:
            break
    return actions


# ---------------------------------------------------------------------------
# Main enumeration + pruning
# ---------------------------------------------------------------------------

def enumerate_actions(grid: Grid, task: dict) -> list[DslAction]:
    """Return all plausible DSL actions for the given intermediate grid and task."""
    actions: list[DslAction] = []
    actions.extend(_rotate_actions())
    actions.extend(_flip_actions())
    actions.extend(_crop_to_content_actions())
    actions.extend(_symmetrize_actions())
    actions.extend(_gravity_actions(grid))
    actions.extend(_fill_enclosed_actions(grid))
    actions.extend(_recolor_actions(grid, task))
    actions.extend(_scale_actions(grid, task))
    actions.extend(_tile_actions(grid))
    actions.extend(_translate_actions())
    actions.extend(_pad_actions(grid))
    actions.extend(_crop_actions(grid, task))
    return actions


def prune_actions(
    actions: list[DslAction],
    grid: Grid,
    target_grid: Grid,
) -> list[DslAction]:
    """Apply fast heuristic filters to reduce the action set."""
    grid_shape = grid.shape
    target_shape = target_grid.shape
    shape_correct = grid_shape == target_shape

    grid_colors = set(int(c) for c in unique_colors(grid))
    target_colors = set(int(c) for c in unique_colors(target_grid))
    palette_correct = grid_colors == target_colors

    _shape_changing_ops = {"scale", "tile", "pad", "crop", "crop_to_content"}
    _color_changing_ops = {"recolor", "fill_enclosed_regions"}

    pruned: list[DslAction] = []
    for action in actions:
        # Remove shape-changing ops when shape already correct
        if shape_correct and action.name in _shape_changing_ops:
            continue
        # Remove color-changing ops when palette already correct
        if palette_correct and action.name in _color_changing_ops:
            continue
        # Remove any action that would produce a grid larger than 30×30
        if action.name == "scale" and action.args:
            factor = action.args[0]
            if (grid_shape[0] * factor > _MAX_OUTPUT_DIM
                    or grid_shape[1] * factor > _MAX_OUTPUT_DIM):
                continue
        if action.name == "tile" and len(action.args) >= 2:
            nr, nc = action.args[0], action.args[1]
            if (grid_shape[0] * nr > _MAX_OUTPUT_DIM
                    or grid_shape[1] * nc > _MAX_OUTPUT_DIM):
                continue
        if action.name == "pad" and len(action.args) >= 4:
            top, bottom, left, right = action.args[:4]
            if (grid_shape[0] + top + bottom > _MAX_OUTPUT_DIM
                    or grid_shape[1] + left + right > _MAX_OUTPUT_DIM):
                continue
        pruned.append(action)

    # Cap at 120 actions — prioritize shape-fixing if shape wrong, else random sample
    if len(pruned) > 120:
        rng = np.random.default_rng(42)
        if not shape_correct:
            shape_ops = [a for a in pruned if a.name in _shape_changing_ops]
            other_ops = [a for a in pruned if a.name not in _shape_changing_ops]
            rng.shuffle(other_ops)
            pruned = shape_ops + other_ops[:120 - len(shape_ops)]
        elif not palette_correct:
            color_ops = [a for a in pruned if a.name in _color_changing_ops]
            other_ops = [a for a in pruned if a.name not in _color_changing_ops]
            rng.shuffle(other_ops)
            pruned = color_ops + other_ops[:120 - len(color_ops)]
        else:
            rng.shuffle(pruned)
            pruned = pruned[:120]

    return pruned
