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

Constants
---------
EXECUTION_TIMEOUT   Hard wall-clock limit in seconds (default 10 s).
DSL_NAMESPACE       The exec namespace seeded with DSL helpers and numpy.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from arc.grid import Grid, grids_equal
from arc.dsl import (
    crop, rotate, flip, translate, scale, tile,
    recolor, mask, overlay, flood_fill,
    find_objects, bounding_box, crop_to_content,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EXECUTION_TIMEOUT: float = 10.0

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
}


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

def _subprocess_worker(code: str, grid_list: list, out_queue: mp.Queue) -> None:
    import contextlib
    import io
    import traceback

    import numpy as np

    namespace = dict(DSL_NAMESPACE)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102
    except Exception as e:
        out_queue.put(("error", f"Compile error: {type(e).__name__}: {e}"))
        return

    transform_fn = namespace.get("transform")
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
            out_queue.put(("error", "No `transform` function found in generated code."))
            return

    input_grid = np.array(grid_list, dtype=np.int32)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = transform_fn(input_grid.copy())
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.int32)
        out_queue.put(("ok", result.astype(np.int32).tolist()))
    except Exception as e:
        out_queue.put(("error", f"Runtime error: {type(e).__name__}: {e}\n{traceback.format_exc()}"))


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

    out_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_subprocess_worker,
        args=(code, input_grid.tolist(), out_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, (
            f"Execution timed out after {timeout}s "
            "(infinite loop or excessive computation)"
        )

    if out_queue.empty():
        return None, f"Execution process exited unexpectedly (exit code {proc.exitcode})"

    status, value = out_queue.get()
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
