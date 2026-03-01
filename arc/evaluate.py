"""Evaluation harness for ARC-AGI tasks.

Provides utilities for running a transform function against task training pairs
and aggregating correctness metrics across a directory of task files.

Also provides calculate_continuous_fitness() — a smooth [0, 1] fitness
signal used by the PSO swarm to guide optimization beyond the binary pass/fail.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from .grid import Grid, grids_equal, load_task, unique_colors, background_color


TransformFn = Callable[[Grid], Grid]


# ---------------------------------------------------------------------------
# Object counting helper
# ---------------------------------------------------------------------------

def _count_objects_total(grid: Grid) -> int:
    """Count 4-connected components of non-background cells."""
    if grid.size == 0:
        return 0
    bg = background_color(grid)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != bg and not visited[r, c]:
                count += 1
                color = grid[r, c]
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or grid[cr, cc] != color:
                        continue
                    visited[cr, cc] = True
                    stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
    return count


# ---------------------------------------------------------------------------
# Standard binary evaluation
# ---------------------------------------------------------------------------

def evaluate_task(task: dict, transform_fn: TransformFn) -> dict:
    """Apply transform_fn to every training pair and return per-pair results.

    Exceptions raised by transform_fn are caught and recorded as failures so
    that a single bad pair does not abort the whole evaluation.

    Returns:
        {
            'pairs':       list of per-pair result dicts
                           {'correct': bool, 'predicted': Grid,
                            'expected': Grid, 'error': str | None}
            'all_correct': True only if every pair is correct
            'n_correct':   number of correct pairs
            'n_total':     total number of pairs
        }
    """
    pairs = []
    for pair in task["train"]:
        inp = pair["input"]
        expected = pair["output"]
        try:
            predicted = transform_fn(inp)
            correct = grids_equal(predicted, expected)
            pairs.append(
                {"correct": correct, "predicted": predicted, "expected": expected, "error": None}
            )
        except Exception as e:
            pairs.append(
                {
                    "correct": False,
                    "predicted": None,
                    "expected": expected,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                }
            )

    n_correct = sum(p["correct"] for p in pairs)
    return {
        "pairs": pairs,
        "all_correct": n_correct == len(pairs),
        "n_correct": n_correct,
        "n_total": len(pairs),
    }


def evaluate_directory(
    task_dir: str | Path,
    transform_fn: TransformFn,
) -> dict:
    """Evaluate transform_fn against all task JSON files in task_dir.

    Returns:
        {
            'tasks':    list of per-task dicts {path, all_correct, n_correct, n_total}
            'n_solved': number of tasks where all pairs were correct
            'n_tasks':  total number of tasks attempted
        }
    """
    task_dir = Path(task_dir)
    results = []
    for path in sorted(task_dir.glob("*.json")):
        task = load_task(path)
        result = evaluate_task(task, transform_fn)
        results.append({"path": str(path), **result})

    n_solved = sum(r["all_correct"] for r in results)
    return {"tasks": results, "n_solved": n_solved, "n_tasks": len(results)}


# ---------------------------------------------------------------------------
# Continuous fitness for PSO
# ---------------------------------------------------------------------------

def calculate_continuous_fitness(
    pred_grid: Grid | None,
    target_grid: Grid,
    progress: float | None = None,
) -> float:
    """Compute a smooth fitness score in [0.0, 1.0] comparing pred to target.

    Used by the PSO swarm to obtain a gradient signal beyond binary pass/fail.

    When progress is None (default), uses fixed 3-component weights:
      - Dimension score  (20%): Does the output have the right shape?
      - Color palette    (30%): Jaccard index of unique colour sets.
      - Pixel accuracy   (50%): Fraction of correct pixels (or IoU overlap
                                region score when dimensions differ).

    When progress ∈ [0, 1], uses 4-component curriculum weights that
    transition from shape/palette matching early to pixel exactness late:
      - Dimension score  (30%→10%)
      - Color palette    (30%→15%)
      - Object count     (20%→10%)
      - Pixel accuracy   (20%→65%)

    Args:
        pred_grid:   The predicted output grid (may be None on hard failure).
        target_grid: The ground-truth output grid.
        progress:    Iteration progress in [0, 1], or None for fixed weights.

    Returns:
        Fitness in [0.0, 1.0].  Returns 0.0 if pred_grid is None.
    """
    if pred_grid is None:
        return 0.0

    # Guard against 1-D outputs (malformed transform return)
    if pred_grid.ndim != 2:
        return 0.0

    # -- Dimension score (20%) ------------------------------------------
    pr, pc = pred_grid.shape
    tr, tc = target_grid.shape

    if pr == tr and pc == tc:
        dim_score = 1.0
    else:
        # Penalise proportional to how wrong each dimension is
        row_ratio = min(pr, tr) / max(pr, tr) if max(pr, tr) > 0 else 0.0
        col_ratio = min(pc, tc) / max(pc, tc) if max(pc, tc) > 0 else 0.0
        dim_score = row_ratio * col_ratio

    # -- Color palette score (30%) – Jaccard index ----------------------
    pred_colors   = set(unique_colors(pred_grid))
    target_colors = set(unique_colors(target_grid))
    intersection  = len(pred_colors & target_colors)
    union         = len(pred_colors | target_colors)
    color_score   = intersection / union if union > 0 else 0.0

    # -- Pixel accuracy (50%) -------------------------------------------
    if pr == tr and pc == tc:
        # Perfect shape match: straightforward cell-wise accuracy
        pixel_score = float(np.sum(pred_grid == target_grid)) / (tr * tc)
    else:
        # Shapes differ: score over the overlapping sub-region, penalised
        # by the total target area (to punish wrong-size outputs).
        min_r = min(pr, tr)
        min_c = min(pc, tc)
        overlap_pred   = pred_grid[:min_r, :min_c]
        overlap_target = target_grid[:min_r, :min_c]
        matches        = int(np.sum(overlap_pred == overlap_target))
        pixel_score    = matches / (tr * tc)   # denominator = full target area

    if progress is None:
        return 0.20 * dim_score + 0.30 * color_score + 0.50 * pixel_score

    # -- Object count score (curriculum mode only) -----------------------
    pred_count   = _count_objects_total(pred_grid)
    target_count = _count_objects_total(target_grid)
    if max(pred_count, target_count) == 0:
        object_score = 1.0
    else:
        object_score = min(pred_count, target_count) / max(pred_count, target_count)

    p = max(0.0, min(1.0, progress))
    w_dim   = 0.30 - 0.20 * p   # 30%→10%
    w_color = 0.30 - 0.15 * p   # 30%→15%
    w_obj   = 0.20 - 0.10 * p   # 20%→10%
    w_pixel = 0.20 + 0.45 * p   # 20%→65%
    return w_dim * dim_score + w_color * color_score + w_obj * object_score + w_pixel * pixel_score
