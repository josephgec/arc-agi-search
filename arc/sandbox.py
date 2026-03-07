"""Sandboxed execution and evaluation of LLM-generated transform functions.

This module is the single source of truth for running untrusted code safely.
It is designed to be backend-agnostic so any future agent can share the same
hardened execution layer without duplicating timeout or namespace logic.

Public API
----------
execute(code, input_grid, timeout) -> (Grid | None, str | None)
    Run a transform function in a child process with a hard wall-clock limit.

evaluate_code(code, task, timeout) -> dict
    Apply execute() to every training pair and return correctness statistics.

shutdown_pool() -> None
    Shut down the persistent worker pool (call at program exit or in tests).

Constants
---------
EXECUTION_TIMEOUT       Hard wall-clock limit in seconds (default 10 s).
MAX_PARAM_COMBINATIONS  Upper bound on PARAM_GRID combinations swept (default 5000).
DSL_NAMESPACE           The exec namespace seeded with DSL helpers and numpy.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import concurrent.futures.process as _cf_process  # BrokenProcessPool lives here in Python 3.9
_BrokenProcessPool = getattr(cf, "BrokenProcessPool", _cf_process.BrokenProcessPool)
import multiprocessing as mp
import threading
from typing import Any

import numpy as np

from arc.grid import Grid, grids_equal
from arc.dsl import (
    crop, rotate, flip, translate, scale, tile,
    recolor, mask, overlay, flood_fill,
    find_objects, bounding_box, crop_to_content,
    pad, symmetrize,
    get_color, get_size, get_centroid,
    detect_grid_layout, find_periodicity,
    fill_enclosed_regions, gravity,
)


def safe_neighbors(grid: np.ndarray, r: int, c: int, size: int = 1) -> np.ndarray:
    """Return the neighbourhood of cell (r, c), clamped to grid boundaries.

    Equivalent to grid[r-size:r+size+1, c-size:c+size+1] but safe at edges.
    Use instead of grid[r-1:r+2, c-1:c+2] to avoid index wrapping at row 0
    or the last row/column.

    Args:
        grid: 2-D numpy array.
        r:    Row index of the centre cell.
        c:    Column index of the centre cell.
        size: Half-width of the neighbourhood (default 1 → 3×3 patch).

    Returns:
        A sub-array of shape up to (2*size+1, 2*size+1).
    """
    r_min = max(0, r - size)
    r_max = min(grid.shape[0], r + size + 1)
    c_min = max(0, c - size)
    c_max = min(grid.shape[1], c + size + 1)
    return grid[r_min:r_max, c_min:c_max]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EXECUTION_TIMEOUT: float = 10.0
MAX_PARAM_COMBINATIONS: int = 5000

DSL_NAMESPACE: dict[str, Any] = {
    "np":              np,
    "numpy":           np,
    "crop":            crop,
    "rotate":          rotate,
    "flip":            flip,
    "translate":       translate,
    "scale":           scale,
    "tile":            tile,
    "recolor":         recolor,
    "mask":            mask,
    "overlay":         overlay,
    "flood_fill":      flood_fill,
    "find_objects":    find_objects,
    "bounding_box":    bounding_box,
    "crop_to_content": crop_to_content,
    "pad":                 pad,
    "symmetrize":          symmetrize,
    "get_color":           get_color,
    "get_size":            get_size,
    "get_centroid":        get_centroid,
    "detect_grid_layout":  detect_grid_layout,
    "find_periodicity":    find_periodicity,
    "fill_enclosed_regions": fill_enclosed_regions,
    "gravity":               gravity,
    "safe_neighbors":        safe_neighbors,
}


# ---------------------------------------------------------------------------
# Persistent worker pool
# ---------------------------------------------------------------------------
# Using a ProcessPoolExecutor avoids the ~100 ms spawn overhead that
# occurred when a new Process was forked for every execute() call.
# With 6 particles × 10 candidates × 3-5 training pairs the savings
# are substantial (~18-30 s per PSO iteration on cold-start macOS).
#
# fork context: child inherits parent memory (DSL_NAMESPACE, numpy, etc.)
# so the worker functions require no re-import of heavy dependencies.
#
# Timeout semantics: future.result(timeout=t) raises TimeoutError in the
# calling thread.  The underlying worker process continues until it finishes
# naturally; the pool spawns a fresh replacement for the next task.  With
# max_workers=8 and a 10 s execution cap this is acceptable — stuck workers
# consume one slot briefly before being recycled.

_POOL_MAX_WORKERS = 8
_POOL: cf.ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> cf.ProcessPoolExecutor:
    """Return the shared ProcessPoolExecutor, creating it on first call."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = cf.ProcessPoolExecutor(
                max_workers=_POOL_MAX_WORKERS,
                mp_context=mp.get_context("fork"),
            )
    return _POOL


def shutdown_pool() -> None:
    """Shut down the persistent worker pool.

    Called automatically at process exit (registered with atexit).
    Safe to call even if the pool was never created or already shut down.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None


# Ensure the pool is cleaned up when the process exits so that worker
# processes don't linger and block pytest / script exit.
atexit.register(shutdown_pool)


# ---------------------------------------------------------------------------
# Subprocess workers
# ---------------------------------------------------------------------------

def _subprocess_worker(code: str, grid_list: list) -> tuple[str, Any]:
    """Execute untrusted transform code against one input grid.

    Designed to run inside a ProcessPoolExecutor worker process (fork).
    Returns ("ok", result_list) on success or ("error", message) on failure.
    """
    import contextlib
    import io
    import traceback

    import numpy as np

    namespace = dict(DSL_NAMESPACE)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102
    except Exception as e:
        return ("error", f"Compile error: {type(e).__name__}: {e}")

    transform_fn = namespace.get("transform")
    if transform_fn is None:
        # Some LLMs name the function differently — check common aliases
        for alias in ("transform_grid", "solve", "process"):
            if alias in namespace and callable(namespace[alias]):
                transform_fn = namespace[alias]
                break
    if transform_fn is None:
        user_fns = [
            v for k, v in namespace.items()
            if callable(v)
            and k not in DSL_NAMESPACE
            and not k.startswith("_")
            and k != "__builtins__"
        ]
        if user_fns:
            transform_fn = user_fns[-1]
        else:
            return ("error", "No `transform` function found in generated code.")

    input_grid = np.array(grid_list, dtype=np.int32)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = transform_fn(input_grid.copy())
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.int32)
        return ("ok", result.astype(np.int32).tolist())
    except Exception as e:
        return ("error", f"Runtime error: {type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute(
    code: str,
    input_grid: Grid,
    timeout: float = EXECUTION_TIMEOUT,
) -> tuple[Grid | None, str | None]:
    if "input(" in code or "sys.stdin" in code:
        return None, "Code uses input()/stdin — not allowed; grid is passed as argument."

    try:
        future = _get_pool().submit(_subprocess_worker, code, input_grid.tolist())
        status, value = future.result(timeout=timeout)
    except cf.TimeoutError:
        return None, (
            f"Execution timed out after {timeout}s "
            "(infinite loop or excessive computation)"
        )
    except _BrokenProcessPool:
        shutdown_pool()
        return None, "Worker pool crashed; pool has been reset."
    except Exception as exc:
        return None, f"Execution error: {exc}"

    if status == "ok":
        return np.array(value, dtype=np.int32), None
    return None, value


def evaluate_code(
    code: str,
    task: dict,
    timeout: float = EXECUTION_TIMEOUT,
) -> dict:
    pairs = []
    for pair in task["train"]:
        output, error = execute(code, pair["input"], timeout)
        if error:
            pairs.append({
                "correct":   False,
                "predicted": None,
                "expected":  pair["output"],
                "error":     error,
            })
        else:
            correct = grids_equal(output, pair["output"])
            pairs.append({
                "correct":   correct,
                "predicted": output,
                "expected":  pair["output"],
                "error":     None,
            })

    n_correct = sum(p["correct"] for p in pairs)
    return {
        "pairs":       pairs,
        "n_correct":   n_correct,
        "n_total":     len(pairs),
        "all_correct": n_correct == len(pairs),
    }


# ---------------------------------------------------------------------------
# CPU parameter search
# ---------------------------------------------------------------------------

def _param_search_worker(
    code: str,
    train_pairs: list[tuple[list, list]],
    max_combinations: int,
) -> tuple[str, Any]:
    """Subprocess target: sweep PARAM_GRID and return (best_params, best_fitness).

    Returns ("ok", (best_params, best_fitness)) or ("error", message).
    """
    import contextlib
    import io
    import itertools

    import numpy as np

    from arc.evaluate import calculate_continuous_fitness

    namespace = dict(DSL_NAMESPACE)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return ("error", f"Compile error: {exc}")

    param_grid = namespace.get("PARAM_GRID", {})
    transform_fn = namespace.get("transform")
    if not param_grid or transform_fn is None:
        return ("error", "Code must define PARAM_GRID and transform(grid, **params)")

    param_names  = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    best_params:  dict  = {}
    best_fitness: float = -1.0

    for combo in itertools.islice(itertools.product(*param_values), max_combinations):
        params = dict(zip(param_names, combo))
        total  = 0.0
        for inp_list, out_list in train_pairs:
            inp_arr = np.array(inp_list, dtype=np.int32)
            out_arr = np.array(out_list, dtype=np.int32)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pred = transform_fn(inp_arr.copy(), **params)
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred, dtype=np.int32)
                total += calculate_continuous_fitness(pred, out_arr)
            except Exception:
                pass
        fitness = total / max(len(train_pairs), 1)
        if fitness > best_fitness:
            best_fitness = fitness
            best_params  = params
            if best_fitness >= 1.0:
                break

    return ("ok", (best_params, best_fitness))


def param_search(
    code: str,
    task: dict,
    timeout: float = 30.0,
    max_combinations: int = MAX_PARAM_COMBINATIONS,
) -> tuple[dict, float]:
    """Sweep all combinations of PARAM_GRID defined in ``code``.

    The LLM-generated code must define:

      PARAM_GRID = {'param_name': [v1, v2, ...], ...}
      def transform(grid, **params):
          ...

    All combinations (up to ``max_combinations``) are tested against the
    task's training pairs inside a single sandboxed subprocess.  The
    subprocess is killed after ``timeout`` seconds.

    Returns:
        (best_params, best_fitness) — the parameter dict that achieved the
        highest average continuous fitness across training pairs, and its
        score.  Returns ({}, 0.0) on timeout or any error.
    """
    if "input(" in code or "sys.stdin" in code:
        return {}, 0.0

    train_pairs = [
        (pair["input"].tolist(), pair["output"].tolist())
        for pair in task["train"]
    ]

    try:
        future = _get_pool().submit(
            _param_search_worker, code, train_pairs, max_combinations
        )
        status, value = future.result(timeout=timeout)
    except cf.TimeoutError:
        return {}, 0.0
    except _BrokenProcessPool:
        shutdown_pool()
        return {}, 0.0
    except Exception:
        return {}, 0.0

    if status == "ok":
        best_params, best_fitness = value
        return best_params, float(best_fitness)
    return {}, 0.0


# ---------------------------------------------------------------------------
# Spatial diff helpers
# ---------------------------------------------------------------------------

_SPATIAL_COLOR_NAMES: dict[int, str] = {
    0: "black", 1: "blue",    2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "fuschia", 7: "orange", 8: "azure",  9: "maroon",
}


def _bbox_str(positions: np.ndarray) -> str:
    """Return a human-readable bounding-box string for a set of (row, col) positions."""
    if len(positions) == 1:
        r, c = int(positions[0, 0]), int(positions[0, 1])
        return f"({r},{c})"
    r1 = int(positions[:, 0].min())
    r2 = int(positions[:, 0].max())
    c1 = int(positions[:, 1].min())
    c2 = int(positions[:, 1].max())
    return f"rows {r1}–{r2}, cols {c1}–{c2}"


def compute_spatial_diff(predicted, expected) -> str:
    """Return a spatially-grounded natural-language description of prediction errors.

    Args:
        predicted: The predicted output grid (np.ndarray or None).
        expected:  The expected output grid (np.ndarray or list).

    Returns:
        A string describing the spatial nature of the error.
    """
    expected = np.asarray(expected, dtype=np.int32)

    if predicted is None:
        return "The transform produced no output (code crashed or timed out)."

    predicted = np.asarray(predicted, dtype=np.int32)

    if predicted.ndim != 2:
        return "The transform returned a non-2D array."

    er, ec = expected.shape
    pr, pc = predicted.shape

    if (pr, pc) != (er, ec):
        parts: list[str] = []
        if pr == ec and pc == er and (pr, pc) != (er, ec):
            parts.append(
                f"The output appears transposed: got {pr}×{pc} but expected {er}×{ec}."
            )
        elif pr == er and pc != ec:
            parts.append(
                f"Height is correct ({pr} rows), but width is wrong: "
                f"got {pc} cols, expected {ec} cols."
            )
            if ec != 0 and pc % ec == 0:
                parts.append(f"Width is {pc // ec}× expected.")
            elif ec != 0 and ec % pc == 0:
                parts.append(f"Width is 1/{ec // pc} of expected.")
        elif pc == ec and pr != er:
            parts.append(
                f"Width is correct ({pc} cols), but height is wrong: "
                f"got {pr} rows, expected {er} rows."
            )
            if er != 0 and pr % er == 0:
                parts.append(f"Height is {pr // er}× expected.")
            elif er != 0 and er % pr == 0:
                parts.append(f"Height is 1/{er // pr} of expected.")
        else:
            parts.append(f"Wrong shape: got {pr}×{pc}, expected {er}×{ec}.")

        # Check if the overlapping region matches perfectly
        min_r = min(pr, er)
        min_c = min(pc, ec)
        if min_r > 0 and min_c > 0:
            if (predicted[:min_r, :min_c] == expected[:min_r, :min_c]).all():
                parts.append(
                    f"However, the overlapping {min_r}×{min_c} region matches perfectly."
                )

        return " ".join(parts)

    # Same shape — find wrong cells
    wrong_mask  = predicted != expected
    wrong_cells = np.argwhere(wrong_mask)

    if len(wrong_cells) == 0:
        return "The prediction and expected output match perfectly."

    bbox = _bbox_str(wrong_cells)

    # Determine spatial quadrant of the errors
    r_mid_lo = er // 3
    r_mid_hi = 2 * er // 3
    c_mid_lo = ec // 3
    c_mid_hi = 2 * ec // 3

    r_center = float(wrong_cells[:, 0].mean())
    c_center = float(wrong_cells[:, 1].mean())

    v_pos = "top" if r_center < r_mid_lo else ("bottom" if r_center > r_mid_hi else "middle")
    h_pos = "left" if c_center < c_mid_lo else ("right" if c_center > c_mid_hi else "center")
    quadrant = f"{v_pos}-{h_pos}"

    parts = [
        f"{len(wrong_cells)} cell(s) are wrong in the {quadrant} region (at {bbox})."
    ]

    # Per-color shift / absence detection
    wrong_exp_colors = np.unique(expected[wrong_mask])
    for color in wrong_exp_colors:
        color_name    = _SPATIAL_COLOR_NAMES.get(int(color), str(color))
        pred_positions = np.argwhere(predicted == color)

        if len(pred_positions) == 0:
            # Color absent from predicted output
            absent_positions = np.argwhere((expected == color) & wrong_mask)
            absent_bbox      = _bbox_str(absent_positions)
            rep_colors       = np.unique(
                predicted[absent_positions[:, 0], absent_positions[:, 1]]
            )
            rep_names = [_SPATIAL_COLOR_NAMES.get(int(c), str(c)) for c in rep_colors]
            parts.append(
                f"  - {color_name} region (at {absent_bbox}) is absent; "
                f"predicted has {'/'.join(rep_names)} there."
            )
        else:
            # Compare expected vs predicted centroids
            exp_positions   = np.argwhere(expected == color)
            exp_r = float(exp_positions[:, 0].mean())
            exp_c = float(exp_positions[:, 1].mean())
            pred_r = float(pred_positions[:, 0].mean())
            pred_c = float(pred_positions[:, 1].mean())

            dr = pred_r - exp_r
            dc = pred_c - exp_c

            if abs(dr) > 0.1 or abs(dc) > 0.1:
                row_dir = "down" if dr > 0 else "up"
                col_dir = "right" if dc > 0 else "left"
                parts.append(
                    f"  - {color_name} object shifted {abs(dr):.1f} rows {row_dir} "
                    f"and {abs(dc):.1f} cols {col_dir}."
                )

    return "\n".join(parts)
