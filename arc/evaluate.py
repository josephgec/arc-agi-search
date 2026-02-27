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

from .grid import Grid, grids_equal, load_task, unique_colors


TransformFn = Callable[[Grid], Grid]


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

def calculate_continuous_fitness(pred_grid: Grid | None, target_grid: Grid) -> float:
    """Compute a smooth fitness score in [0.0, 1.0] comparing pred to target.

    Used by the PSO swarm to obtain a gradient signal beyond binary pass/fail.

    Component weights:
      - Dimension score  (20%): Does the output have the right shape?
      - Color palette    (30%): Jaccard index of unique colour sets.
      - Pixel accuracy   (50%): Fraction of correct pixels (or IoU overlap
                                region score when dimensions differ).

    Args:
        pred_grid:   The predicted output grid (may be None on hard failure).
        target_grid: The ground-truth output grid.

    Returns:
        Fitness in [0.0, 1.0].  Returns 0.0 if pred_grid is None.
    """
    if pred_grid is None:
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

    return 0.20 * dim_score + 0.30 * color_score + 0.50 * pixel_score
