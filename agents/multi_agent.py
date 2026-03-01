"""Multi-agent orchestrator for ARC-AGI puzzles.

Coordinates three specialized agents in a feedback loop:

  Hypothesizer → generates 3 competing natural-language transformation rules
  Coder        → translates one rule into executable Python DSL code
  Critic       → diagnoses failures and routes feedback to the right agent

The loop runs up to ``max_cycles`` total agent calls.  On each cycle either:
  - A correct solution is found (success), or
  - The Critic routes to the Hypothesizer (try next/new hypothesis), or
  - The Critic routes to the Coder (fix the current implementation).
"""
from __future__ import annotations

import re

import numpy as np

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder, Critic, Decomposer, Verifier, ROUTE_HYPOTHESIZER


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DENSE_GRID_THRESHOLD  =  50   # <=50 cells: dense [[v,...],...]
_RLE_GRID_THRESHOLD    = 400   # 51-400: RLE per row
_SPARSE_GRID_THRESHOLD = 800   # 401-800: sparse {(r,c)=v,...}
                                # >800: omitted

_COLOR_NAMES = {
    0: "black", 1: "blue",   2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "fuschia", 7: "orange", 8: "azure",  9: "maroon",
}


# ---------------------------------------------------------------------------
# Grid / task formatting helpers (also used by pso_orchestrator)
# ---------------------------------------------------------------------------

def _grid_to_str(grid) -> str:
    return (
        "["
        + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in grid.tolist())
        + "]"
    )


def _grid_to_sparse(grid) -> str:
    """Compact representation: list only non-zero (foreground) cells."""
    rows, cols = grid.shape
    cells = []
    for r in range(rows):
        for c in range(cols):
            v = int(grid[r, c])
            if v != 0:
                cells.append(f"({r},{c})={v}")
    if not cells:
        return "(empty — all zeros)"
    return "{" + ", ".join(cells) + "}"


def _grid_to_rle(grid) -> str:
    """Encode grid as run-length encoding per row.

    Each run encoded as 'value(colorname)xcount'; singletons omit 'xcount'.
    Example: [0,0,2,2,0] -> '0(black)x3 2(red)x2 0(black)'
    Rows separated by newline + two spaces for readability.
    """
    row_strs = []
    for row in grid.tolist():
        runs = []
        if not row:
            row_strs.append("")
            continue
        cur_val = row[0]
        count = 1
        for v in row[1:]:
            if v == cur_val:
                count += 1
            else:
                name = _COLOR_NAMES.get(cur_val, str(cur_val))
                token = f"{cur_val}({name})" if count == 1 else f"{cur_val}({name})x{count}"
                runs.append(token)
                cur_val = v
                count = 1
        name = _COLOR_NAMES.get(cur_val, str(cur_val))
        token = f"{cur_val}({name})" if count == 1 else f"{cur_val}({name})x{count}"
        runs.append(token)
        row_strs.append(" ".join(runs))
    return "\n  ".join(row_strs)


def _block_analysis(inp, out) -> str | None:
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    br, bc = oh // ih, ow // iw
    lines  = [f"  (Output divided into {br}×{bc} blocks, each {ih}×{iw}:)"]
    for r in range(br):
        for c in range(bc):
            block    = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
            cell_val = inp[r, c] if r < ih and c < iw else "?"
            if (block == 0).all():
                content = "all zeros"
            else:
                content = _grid_to_str(block)
                if block.shape == inp.shape and (block == inp).all():
                    content += " (= input)"
            lines.append(f"  block({r},{c}): input[{r}][{c}]={cell_val} → {content}")
    return "\n".join(lines)


def _diff_annotation(inp: np.ndarray, out: np.ndarray) -> str | None:
    """For same-shape pairs, briefly describe which cells changed."""
    if inp.shape != out.shape:
        return None
    diff = np.argwhere(inp != out)
    n = len(diff)
    if n == 0:
        return "  (no cells changed)"
    if n > 20:
        return f"  ({n} cells changed)"
    parts = [f"[{r},{c}] {inp[r,c]}→{out[r,c]}" for r, c in diff[:8]]
    suffix = "…" if n > 8 else ""
    return f"  ({n} cells changed: {', '.join(parts)}{suffix})"


def _count_objects(grid: np.ndarray) -> dict[int, int]:
    """Return {color: n_connected_components} for each foreground color."""
    from collections import deque
    result: dict[int, int] = {}
    for color in np.unique(grid):
        if color == 0:
            continue
        visited = np.zeros(grid.shape, dtype=bool)
        n_comp = 0
        positions = list(zip(*np.where(grid == color)))
        pos_set = set(positions)
        for start in positions:
            if visited[start]:
                continue
            n_comp += 1
            queue = deque([start])
            while queue:
                r, c = queue.popleft()
                if visited[r, c]:
                    continue
                visited[r, c] = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in pos_set and not visited[nr, nc]:
                        queue.append((nr, nc))
        result[int(color)] = n_comp
    return result


def _structural_note(inp: np.ndarray, out: np.ndarray) -> str | None:
    """Detect and describe structural patterns in the pair (diagonal, periodic, etc.)."""
    notes = []

    in_colors  = set(int(v) for v in np.unique(inp))
    out_colors = set(int(v) for v in np.unique(out))

    # New colors appearing only in output (flood-fill / object-completion hint)
    new_colors = out_colors - in_colors
    if new_colors:
        c_list = sorted(new_colors)
        notes.append(
            f"  [Structural] Output introduces NEW color(s) {c_list} not present in input — "
            "likely flood_fill, enclosed-region fill, or object completion."
        )

    # Colors removed in output (masking / filtering hint)
    removed_colors = in_colors - out_colors
    if removed_colors and 0 not in removed_colors:  # ignore background=0 disappearing
        c_list = sorted(removed_colors)
        notes.append(
            f"  [Structural] Color(s) {c_list} present in input but absent from output — "
            "possibly masked, filtered, or merged."
        )

    # Object count per color (for multi-object tasks)
    total_cells = inp.size
    if total_cells <= 400:   # only for smaller grids where counting is meaningful
        in_obj  = _count_objects(inp)
        out_obj = _count_objects(out)
        total_in_obj = sum(in_obj.values())
        if total_in_obj > 1:
            obj_desc = ", ".join(f"color {c}: {n}" for c, n in sorted(in_obj.items()))
            notes.append(
                f"  [Structural] Input has {total_in_obj} distinct connected objects ({obj_desc})."
            )
        # Note if object count changes
        total_out_obj = sum(out_obj.values())
        if total_in_obj > 0 and total_out_obj != total_in_obj:
            notes.append(
                f"  [Structural] Object count changes: {total_in_obj} → {total_out_obj} objects."
            )

    # Shape change analysis
    ir, ic = inp.shape
    or_, oc = out.shape
    if ir != or_ or ic != oc:
        # Check for integer scale factor
        if or_ % ir == 0 and oc % ic == 0 and or_ // ir == oc // ic:
            sf = or_ // ir
            notes.append(f"  [Structural] Output is {sf}× scaled version of input.")
        elif ir % or_ == 0 and ic % oc == 0 and ir // or_ == ic // oc:
            sf = ir // or_
            notes.append(f"  [Structural] Output is 1/{sf} scaled (cropped/compressed) version of input.")
        else:
            notes.append(
                f"  [Structural] Shape changes: ({ir}×{ic}) → ({or_}×{oc})."
            )

    # Divider detection: a full column or row of a single unique color
    ir, ic = inp.shape
    for col in range(ic):
        col_vals = set(inp[:, col].tolist())
        if len(col_vals) == 1 and list(col_vals)[0] != 0:
            div_color = list(col_vals)[0]
            left  = inp[:, :col]
            right = inp[:, col + 1:]
            if left.size > 0 and right.size > 0:
                notes.append(
                    f"  [Structural] Input has a vertical DIVIDER at column {col} (all cells = {div_color}). "
                    f"Left side is {ir}×{col}, right side is {ir}×{ic - col - 1}. "
                    "Output may depend on comparing or combining the two halves."
                )
            break  # only report first divider

    for row in range(ir):
        row_vals = set(inp[row, :].tolist())
        if len(row_vals) == 1 and list(row_vals)[0] != 0:
            div_color = list(row_vals)[0]
            top    = inp[:row, :]
            bottom = inp[row + 1:, :]
            if top.size > 0 and bottom.size > 0:
                notes.append(
                    f"  [Structural] Input has a horizontal DIVIDER at row {row} (all cells = {div_color}). "
                    f"Top part is {row}×{ic}, bottom part is {ir - row - 1}×{ic}. "
                    "Output may depend on comparing or combining the two halves."
                )
            break

    # Anti-diagonal groups in input (r+c = const): detect if non-zero cells share diagonals
    # Only report when there are multiple distinct colors (otherwise trivially true).
    nz = np.argwhere(inp != 0)
    if 2 <= len(nz) <= 50:
        diag_vals: dict[int, set] = {}
        for r, c in nz:
            k = int(r + c)
            diag_vals.setdefault(k, set()).add(int(inp[r, c]))
        # Each diagonal group must have exactly 1 unique color AND at least 2 different
        # colors exist across diagonals (otherwise it's trivially "all cells are color X").
        all_colors_on_diags = {list(v)[0] for v in diag_vals.values()}
        if (diag_vals
                and all(len(v) == 1 for v in diag_vals.values())
                and len(all_colors_on_diags) >= 2):
            sorted_diags = sorted(diag_vals.items())
            desc = ", ".join(f"k={k}→{list(v)[0]}" for k, v in sorted_diags)
            notes.append(f"  [Structural] Input: anti-diagonals (r+c=k) have consistent colors: {desc}")

            # Check if these diagonal colors follow a cycling pattern of period N
            ks     = [k for k, _ in sorted_diags]
            colors = [list(v)[0] for _, v in sorted_diags]
            n_diags = len(ks)
            for period in range(2, min(n_diags + 1, 7)):
                # Build the cycle: color at position k = cycle[(k - k_min) % period]
                k_min = ks[0]
                cycle = [None] * period
                consistent = True
                for k, col in zip(ks, colors):
                    idx = (k - k_min) % period
                    if cycle[idx] is None:
                        cycle[idx] = col
                    elif cycle[idx] != col:
                        consistent = False
                        break
                if consistent and all(c is not None for c in cycle):
                    notes.append(
                        f"  [Structural] Anti-diagonal colors CYCLE with period {period}: "
                        f"color at (r+c)%{period} maps as "
                        + ", ".join(f"{i}→{c}" for i, c in enumerate(cycle))
                        + f".  Hint: output[r][c] = cycle[(r+c - {k_min}) % {period}]."
                    )
                    break

    # Row-period detection: input rows cycle with period P; output extends that cycle.
    # Only when output has more rows than input (extension pattern).
    ir, ic = inp.shape
    or_, oc = out.shape
    if or_ > ir and ic == oc:
        # Find minimal row period P in input
        for P in range(1, ir + 1):
            if all((inp[r] == inp[r % P]).all() for r in range(ir)):
                # Confirm output continues the same cycle (with possible color remapping)
                # Map colors: e.g. 1→2 is common; check if out[r] == remap(inp[r % P])
                # Simple check: out rows are consistent with cycling inp rows (any remap)
                cycle_rows = [inp[r % P] for r in range(or_)]
                # Check if each out row matches the corresponding cycle row up to color remap
                def rows_same_pattern(r1, r2):
                    if r1.shape != r2.shape: return False
                    # same non-zero pattern (positions where r1>0 == positions where r2>0)
                    return ((r1 > 0) == (r2 > 0)).all()
                if all(rows_same_pattern(out[r], cycle_rows[r]) for r in range(or_)):
                    notes.append(
                        f"  [Structural] Input rows cycle with period {P}; output extends "
                        f"the cycle from {ir} to {or_} rows (possibly with color replacement)."
                    )
                    break

    # Periodic output pattern: check if output[r][c] = output[r % T][c % T] for small T
    if out.shape[0] == out.shape[1] == inp.shape[0]:
        n = out.shape[0]
        for T in range(1, min(n, 7)):
            if n % T == 0:
                tile = out[:T, :T]
                tiled = np.tile(tile, (n // T, n // T))
                if (tiled == out).all():
                    notes.append(f"  [Structural] Output has a repeating {T}×{T} tile pattern")
                    break

    return "\n".join(notes) if notes else None


def _format_training_examples(task: dict) -> str:
    lines = ["Training examples (use these to verify your implementation):"]
    for i, pair in enumerate(task["train"]):
        inp, out  = pair["input"], pair["output"]
        ih, iw    = inp.shape
        oh, ow    = out.shape
        max_cells = max(inp.size, out.size)
        lines.append(f"Example {i + 1}: input ({ih}×{iw}) → output ({oh}×{ow})")
        if max_cells > _SPARSE_GRID_THRESHOLD:
            lines.append(f"  Input:  [large grid — {inp.size} cells, see task description]")
            lines.append(f"  Output: [large grid — {out.size} cells, see task description]")
        elif max_cells > _RLE_GRID_THRESHOLD:
            lines.append(f"  Input  [sparse]: {_grid_to_sparse(inp)}")
            lines.append(f"  Output [sparse]: {_grid_to_sparse(out)}")
        elif max_cells > _DENSE_GRID_THRESHOLD:
            lines.append(f"  Input  [RLE]:\n    {_grid_to_rle(inp)}")
            lines.append(f"  Output [RLE]:\n    {_grid_to_rle(out)}")
        else:
            lines.append(f"  Input:  {_grid_to_str(inp)}")
            lines.append(f"  Output: {_grid_to_str(out)}")
        ann = _diff_annotation(inp, out)
        if ann:
            lines.append(ann)
        sn = _structural_note(inp, out)
        if sn:
            lines.append(sn)
    return "\n".join(lines)


def _output_shape_constraint(task: dict) -> str:
    """Return a shape hint for LLM: hard constraint when fixed, guidance when variable."""
    pairs      = task["train"]
    in_shapes  = [tuple(p["input"].shape)  for p in pairs]
    out_shapes = [tuple(p["output"].shape) for p in pairs]

    if len(set(out_shapes)) == 1:
        oh, ow = out_shapes[0]
        return (
            f"\n⚠ HARD CONSTRAINT: transform() MUST return an array of shape "
            f"({oh}, {ow}) — i.e. exactly {oh} rows × {ow} columns.  "
            "Any other output size is WRONG regardless of content."
        )

    # Variable output sizes — give rich guidance
    lines = ["\n⚠ Note: output size VARIES across training pairs — it must be computed "
             "dynamically from the input content, not hardcoded."]
    for i, (ish, osh) in enumerate(zip(in_shapes, out_shapes)):
        lines.append(f"  Pair {i+1}: input {ish[0]}×{ish[1]} → output {osh[0]}×{osh[1]}")
        if ish == osh:
            lines.append("    (same shape as input)")
        elif osh[0] < ish[0] or osh[1] < ish[1]:
            lines.append("    (output is SMALLER than input — likely a crop or selection)")
        elif osh[0] > ish[0] or osh[1] > ish[1]:
            lines.append("    (output is LARGER than input — likely expanded/tiled)")
    lines.append("Study how each output size relates to what's in the corresponding input.")
    return "\n".join(lines)


def _format_task_description(task: dict) -> str:
    """Format task description showing ALL training pairs.

    Dispatch by max cells in a pair:
      >800  : omit both grids
      >400  : sparse {(r,c)=v,...}
      > 50  : RLE per row
      <=50  : dense [[v,...],...]
    """
    all_pairs = task["train"]
    lines = ["Here is an ARC-AGI puzzle.\n"]

    for i, pair in enumerate(all_pairs):
        inp, out = pair["input"], pair["output"]
        ih, iw   = inp.shape
        oh, ow   = out.shape
        in_cells  = inp.size
        out_cells = out.size
        max_cells = max(in_cells, out_cells)

        lines.append(f"### Training pair {i + 1}")

        if max_cells > _SPARSE_GRID_THRESHOLD:
            lines.append(
                f"Input  ({ih}\u00d7{iw}): [omitted \u2014 {in_cells} cells, too large to display]"
            )
            lines.append(
                f"Output ({oh}\u00d7{ow}): [omitted \u2014 {out_cells} cells, too large to display]"
            )
        elif max_cells > _RLE_GRID_THRESHOLD:
            # Sparse format: list non-zero cells only
            lines.append(f"Input  ({ih}\u00d7{iw}) sparse: {_grid_to_sparse(inp)}")
            lines.append(f"Output ({oh}\u00d7{ow}) sparse: {_grid_to_sparse(out)}")
        elif max_cells > _DENSE_GRID_THRESHOLD:
            # RLE format: human-readable run-length encoding
            lines.append(f"Input  ({ih}\u00d7{iw}) [RLE]:\n  {_grid_to_rle(inp)}")
            lines.append(f"Output ({oh}\u00d7{ow}) [RLE]:\n  {_grid_to_rle(out)}")
        else:
            lines.append(f"Input  ({ih}\u00d7{iw}):\n{_grid_to_str(inp)}")
            lines.append(f"Output ({oh}\u00d7{ow}):\n{_grid_to_str(out)}")
            ba = _block_analysis(inp, out)
            if ba:
                lines.append(ba)
        lines.append("")

    test_inp = task["test"][0]["input"]
    th, tw   = test_inp.shape
    test_cells = test_inp.size

    if test_cells > _SPARSE_GRID_THRESHOLD:
        lines.append(
            f"### Test input ({th}\u00d7{tw}): [omitted \u2014 {test_cells} cells, too large to display]"
        )
    elif test_cells > _RLE_GRID_THRESHOLD:
        lines.append(f"### Test input ({th}\u00d7{tw}) sparse:\n{_grid_to_sparse(test_inp)}")
    elif test_cells > _DENSE_GRID_THRESHOLD:
        lines.append(f"### Test input ({th}\u00d7{tw}) [RLE]:\n  {_grid_to_rle(test_inp)}")
    else:
        lines.append(f"### Test input ({th}\u00d7{tw}):\n{_grid_to_str(test_inp)}")

    lines.append(_output_shape_constraint(task))
    lines.append("\nStudy the training pairs and identify the transformation rule.")
    return "\n".join(lines)


def _format_eval_diff(eval_result: dict, max_pairs: int = 2, max_mismatches: int = 8) -> str:
    """Format a concise diff of sandbox evaluation failures for LLM feedback."""
    lines = []
    for i, pair in enumerate(eval_result.get("pairs", [])[:max_pairs]):
        if pair.get("error"):
            lines.append(f"Pair {i + 1}: runtime error — {pair['error'][:120]}")
            continue
        pred = pair.get("predicted")
        exp  = pair.get("expected")
        if pred is None or exp is None:
            lines.append(f"Pair {i + 1}: no prediction")
            continue
        if pred.shape != exp.shape:
            lines.append(
                f"Pair {i + 1}: wrong shape — got {pred.shape[0]}×{pred.shape[1]}, "
                f"expected {exp.shape[0]}×{exp.shape[1]}"
            )
        else:
            wrong = int(np.sum(pred != exp))
            total = int(exp.size)
            lines.append(f"Pair {i + 1}: {wrong}/{total} cells wrong")
            mismatches = np.argwhere(pred != exp)[:max_mismatches]
            for r, c in mismatches:
                got_name = _COLOR_NAMES.get(int(pred[r, c]), str(pred[r, c]))
                exp_name = _COLOR_NAMES.get(int(exp[r, c]),  str(exp[r, c]))
                lines.append(f"  [{r},{c}]: got {pred[r,c]}({got_name}), expected {exp[r,c]}({exp_name})")
            if wrong > max_mismatches:
                lines.append(f"  … and {wrong - max_mismatches} more wrong cells")
    return "\n".join(lines) if lines else "No diff available"


# ---------------------------------------------------------------------------
# Code extraction / response cleaning
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _truncate_to_valid_function(text: str) -> str:
    lines   = text.splitlines()
    in_func = False
    result  = []
    for line in lines:
        if line.startswith("def "):
            in_func = True
        if in_func:
            stripped = line.rstrip()
            if stripped and not stripped[0].isspace() and not stripped.startswith("def "):
                break
            result.append(line)
    return "\n".join(result).rstrip() if result else text


def _extract_code(text: str) -> str | None:
    text = _strip_thinking(text)
    if not text.strip():
        return None

    # 1. ```python … ``` — last block preferred
    matches = list(re.finditer(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip()
        return _truncate_to_valid_function(candidate) if "def " in candidate else candidate

    # 2. ``` … ``` with a def — last qualifying block
    for m in reversed(list(re.finditer(r"```\s*(.*?)\s*```", text, re.DOTALL))):
        candidate = m.group(1).strip()
        if "def " in candidate:
            return _truncate_to_valid_function(candidate)

    # 3. Bare def transform( — last occurrence
    all_transforms = list(re.finditer(r"def transform\(", text))
    if all_transforms:
        return _truncate_to_valid_function(text[all_transforms[-1].start():])

    # 4. Any bare def — last occurrence
    all_defs = list(re.finditer(r"def \w+\(", text))
    if all_defs:
        return _truncate_to_valid_function(text[all_defs[-1].start():])

    # 5. import numpy … def (no fence)
    if "import numpy" in text and "def " in text:
        start = text.rfind("import numpy")
        return _truncate_to_valid_function(text[start:])

    return None


# ---------------------------------------------------------------------------
# Hypothesis parsing
# ---------------------------------------------------------------------------

_MIN_HYPOTHESIS_CHARS = 80

def _parse_hypotheses(text: str, max_n: int | None = None) -> list[str]:
    text     = _strip_thinking(text)
    stripped = text.strip()

    paragraphs = re.split(r"\n\s*\n", stripped)
    hyp_paras  = [p.strip() for p in paragraphs
                  if re.match(r"^[1-9]\.\s", p.strip())]
    if len(hyp_paras) >= 2:
        hypotheses = hyp_paras
    else:
        parts      = re.split(r"(?m)^(?=[1-9]\.\s)", stripped)
        hypotheses = [p.strip() for p in parts if p.strip()]
        if len(hypotheses) < 2:
            return [stripped] if stripped else []

    hypotheses = [h for h in hypotheses if len(h) >= _MIN_HYPOTHESIS_CHARS]

    if max_n is not None:
        hypotheses = hypotheses[:max_n]

    return hypotheses if hypotheses else [stripped]


# ---------------------------------------------------------------------------
# Error / diff formatting for the Critic
# ---------------------------------------------------------------------------

def _format_error_info(eval_result: dict) -> str:
    lines = [f"{eval_result['n_correct']}/{eval_result['n_total']} training pairs correct."]
    for i, pair in enumerate(eval_result["pairs"]):
        if pair["error"]:
            lines.append(f"Pair {i + 1} error: {pair['error']}")
        elif not pair["correct"]:
            lines.append(f"Pair {i + 1}: produced wrong output (no exception).")
    return "\n".join(lines)


def _diff_summary(expected, predicted, max_show: int = 12) -> str:
    if predicted is None:
        return "(no output produced)"
    if expected.shape != predicted.shape:
        return f"Shape mismatch: expected {expected.shape}, got {predicted.shape}."
    diffs = list(zip(*np.where(expected != predicted)))
    if not diffs:
        return "(no differences)"
    lines = []
    for r, c in diffs[:max_show]:
        ev = int(expected[r, c])
        pv = int(predicted[r, c])
        lines.append(
            f"  [{r},{c}] expected {ev} ({_COLOR_NAMES.get(ev, ev)}), "
            f"got {pv} ({_COLOR_NAMES.get(pv, pv)})"
        )
    if len(diffs) > max_show:
        lines.append(f"  … and {len(diffs) - max_show} more differences")
    return "\n".join(lines)


def _format_diff(eval_result: dict) -> str:
    _FULL_GRID_CELL_LIMIT = 200
    parts = []
    first_fail = True
    for i, pair in enumerate(eval_result["pairs"]):
        if not pair["correct"]:
            diff = _diff_summary(pair["expected"], pair["predicted"])
            section = f"Pair {i + 1}:\n{diff}"
            if first_fail and pair["predicted"] is not None:
                exp = pair["expected"]
                pred = pair["predicted"]
                if exp.size <= _FULL_GRID_CELL_LIMIT:
                    section += (
                        f"\n  Full expected:  {_grid_to_str(exp)}"
                        f"\n  Full predicted: {_grid_to_str(pred)}"
                    )
                first_fail = False
            parts.append(section)
    return "\n\n".join(parts) if parts else "(all pairs correct)"


def _format_spatial_diff(eval_result: dict) -> str:
    """Use compute_spatial_diff for each failing pair, returning a combined description."""
    parts = []
    for i, pair in enumerate(eval_result.get("pairs", [])):
        if pair.get("correct"):
            continue
        pred = pair.get("predicted")
        exp  = pair.get("expected")
        if exp is None:
            continue
        diff = sandbox.compute_spatial_diff(pred, exp)
        parts.append(f"Pair {i + 1}:\n{diff}")
    return "\n\n".join(parts) if parts else "(all pairs correct)"


# ---------------------------------------------------------------------------
# MultiAgent orchestrator
# ---------------------------------------------------------------------------

class MultiAgent:
    def __init__(
        self,
        backend:                  str        = "ollama",
        model:                    str | None = None,
        hypothesizer_model:       str | None = None,
        coder_model:              str | None = None,
        critic_model:             str | None = None,
        decomposer_model:         str | None = None,
        verifier_model:           str | None = None,
        hypothesizer_temperature: float      = 0.6,
        coder_temperature:        float      = 0.1,
        critic_temperature:       float      = 0.2,
        hypothesizer_max_tokens:  int        = 8192,
        coder_max_tokens:         int        = 4096,
        critic_max_tokens:        int        = 4096,
        decomposer_max_tokens:    int        = 4096,
        verifier_max_tokens:      int        = 4096,
        timeout:                  float      = 120.0,
        debug:                    bool       = False,
        max_cycles:               int        = 9,
        use_decomposer:           bool       = True,
        use_verifier:             bool       = True,
    ) -> None:
        def _make_client(role_model: str | None, temperature: float, max_tokens: int) -> LLMClient:
            return LLMClient(
                backend=backend,
                model=role_model or model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                debug=debug,
            )

        hyp_client   = _make_client(hypothesizer_model, hypothesizer_temperature, hypothesizer_max_tokens)
        cod_client   = _make_client(coder_model,        coder_temperature,        coder_max_tokens)
        cri_client   = _make_client(critic_model,       critic_temperature,       critic_max_tokens)
        decomp_client = _make_client(decomposer_model,  0.3,                      decomposer_max_tokens)
        verif_client  = _make_client(verifier_model,    0.1,                      verifier_max_tokens)

        self._hypothesizer            = Hypothesizer(hyp_client)
        self._coder                   = Coder(cod_client)
        self._critic                  = Critic(cri_client)
        self._decomposer              = Decomposer(decomp_client)
        self._verifier                = Verifier(verif_client)
        self.use_decomposer           = use_decomposer
        self.use_verifier             = use_verifier
        self.max_cycles               = max_cycles
        self.debug                    = debug
        self.backend                  = backend
        self.hypothesizer_model       = hyp_client.model
        self.coder_model              = cod_client.model
        self.critic_model             = cri_client.model
        self.hypothesizer_temperature = hypothesizer_temperature
        self.coder_temperature        = coder_temperature
        self.critic_temperature       = critic_temperature
        self.hypothesizer_max_tokens  = hypothesizer_max_tokens
        self.coder_max_tokens         = coder_max_tokens
        self.critic_max_tokens        = critic_max_tokens
        self.model                    = self.hypothesizer_model

    def solve(self, task: dict) -> dict:
        log:            list[dict] = []
        best_code:      str | None = None
        best_n_correct: int        = -1
        cycle:          int        = 0

        test_pair             = task.get("test", [{}])[0]
        has_test_ground_truth = "output" in test_pair

        task_description: str        = _format_task_description(task)
        training_examples: str       = _format_training_examples(task)
        hypotheses:        list[str] = []
        hyp_index:         int       = 0
        prev_hyp_index:    int       = -1   # tracks hypothesis transitions
        hyp_feedback:      str | None = None
        coder_feedback:    str | None = None
        prev_n_correct:    int        = -1
        no_improve_count:  int        = 0
        coder_attempt:     int        = 0
        decomp_tried:      bool       = False
        verifier_attempts: int        = 0

        while cycle < self.max_cycles:

            if not hypotheses or hyp_index >= len(hypotheses):
                cycle += 1
                if cycle > self.max_cycles:
                    break
                try:
                    hyp_response = self._hypothesizer.generate(task_description, hyp_feedback)
                except Exception as e:
                    log.append({"cycle": cycle, "agent": "hypothesizer", "error": str(e)})
                    break

                hyp_feedback = None
                hypotheses   = _parse_hypotheses(hyp_response, max_n=3)
                hyp_index    = 0

                log.append({
                    "cycle": cycle, "agent": "hypothesizer",
                    "n_hypotheses": len(hypotheses),
                })
                if self.debug:
                    print(f"[debug] Hypothesizer: {len(hypotheses)} hypothesis(es)")

                # Guard: if the model response was truncated (only thinking
                # tokens, nothing parseable), treat it like a transient error
                # and skip this cycle rather than crashing with IndexError.
                if not hypotheses:
                    log.append({"cycle": cycle, "agent": "hypothesizer",
                                "error": "no_parseable_hypotheses"})
                    continue

            # Guard: hyp_index must be in bounds (defensive, belt-and-suspenders)
            if hyp_index >= len(hypotheses):
                hyp_index = 0

            # Reset per-hypothesis counters whenever we transition to a new
            # hypothesis (hyp_index changed) or just got fresh hypotheses from
            # the Hypothesizer.  This ensures no_improve_count starts from zero
            # for every hypothesis, so stuck-detection works correctly.
            if hyp_index != prev_hyp_index:
                coder_attempt    = 0
                prev_n_correct   = -1
                no_improve_count = 0
                prev_hyp_index   = hyp_index

            current_hypothesis = hypotheses[hyp_index]
            coder_attempt += 1

            temperature = min(self.coder_temperature + (coder_attempt - 1) * 0.3, 0.9)

            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                code_response = self._coder.generate(
                    current_hypothesis, coder_feedback,
                    training_context=training_examples,
                    temperature=temperature,
                )
            except Exception as e:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": str(e),
                })
                hyp_index     += 1
                coder_feedback = None
                coder_attempt  = 0
                decomp_tried   = False
                continue

            coder_feedback = None

            clean  = _strip_thinking(code_response)
            code   = _extract_code(clean) or _extract_code(code_response)

            if self.debug:
                print(
                    f"[debug] Coder (hyp {hyp_index}): "
                    f"response={len(code_response)} chars, code={code is not None}"
                )

            if code is None:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": "no_code_block",
                })
                hyp_index    += 1
                decomp_tried  = False
                continue

            eval_result = sandbox.evaluate_code(code, task)
            n_correct   = eval_result["n_correct"]

            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code      = code

            log.append({
                "cycle":            cycle,
                "agent":            "coder",
                "hypothesis_index": hyp_index,
                "n_correct":        n_correct,
                "n_total":          eval_result["n_total"],
                "all_correct":      eval_result["all_correct"],
            })

            if eval_result["all_correct"]:
                if self.use_verifier and verifier_attempts < 2:
                    try:
                        ver_result = self._verifier.verify(
                            code, task_description, training_examples, eval_result
                        )
                    except Exception:
                        ver_result = {"passes": True, "issues": "", "suggestion": ""}

                    log.append({
                        "cycle":   cycle,
                        "agent":   "verifier",
                        "verdict": "pass" if ver_result["passes"] else "fail",
                    })

                    if not ver_result["passes"]:
                        verifier_attempts += 1
                        coder_feedback = (
                            f"The Verifier flagged potential issues:\n"
                            f"ISSUES: {ver_result['issues']}\n"
                            f"SUGGESTION: {ver_result['suggestion']}"
                        )
                        # Continue the loop so the Coder can fix the fragility
                        continue

                return {
                    "success":      True,
                    "test_correct": self._evaluate_test(best_code, test_pair) if has_test_ground_truth else None,
                    "code":         best_code,
                    "n_cycles":     cycle,
                    "log":          log,
                }

            if n_correct <= prev_n_correct:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_n_correct = n_correct

            _hypothesis_stuck = (n_correct == 0 and no_improve_count >= 2)
            if _hypothesis_stuck:
                if self.use_decomposer and not decomp_tried:
                    if self.debug:
                        print(f"[debug] Stuck — invoking Decomposer")
                    try:
                        decomp_result = self._decomposer.decompose(
                            task_description,
                            training_examples,
                            stuck_approaches=current_hypothesis,
                        )
                    except Exception:
                        decomp_result = None

                    if decomp_result:
                        hypotheses[hyp_index] = decomp_result
                        log.append({"cycle": cycle, "agent": "decomposer"})

                    coder_attempt    = 0
                    no_improve_count = 0
                    decomp_tried     = True
                else:
                    if self.debug:
                        print(f"[debug] Stuck at 0/{eval_result['n_total']} — skipping Critic, next hyp")
                    hyp_index    += 1
                    coder_attempt = 0
                    decomp_tried  = False
                continue

            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                critic_result = self._critic.analyze(
                    current_hypothesis, code,
                    _format_error_info(eval_result),
                    _format_spatial_diff(eval_result),
                )
            except Exception as e:
                log.append({"cycle": cycle, "agent": "critic", "error": str(e)})
                hyp_index    += 1
                coder_attempt = 0
                continue

            log.append({
                "cycle":    cycle,
                "agent":    "critic",
                "route":    critic_result["route"],
                "feedback": critic_result["feedback"],
            })
            if self.debug:
                print(f"[debug] Critic → {critic_result['route']}")

            if critic_result["route"] == ROUTE_HYPOTHESIZER:
                hyp_feedback  = critic_result["feedback"]
                hyp_index    += 1
                coder_attempt = 0
                decomp_tried  = False
            else:
                coder_feedback = critic_result["feedback"]

        return {
            "success":      False,
            "test_correct": (
                self._evaluate_test(best_code, test_pair)
                if has_test_ground_truth and best_code else None
            ),
            "code":         best_code,
            "n_cycles":     cycle,
            "log":          log,
        }

    def predict(self, task: dict) -> Grid | None:
        result = self.solve(task)
        if not result["code"]:
            return None
        out, _ = sandbox.execute(result["code"], task["test"][0]["input"])
        return out

    def _evaluate_test(self, code: str, test_pair: dict) -> bool:
        out, err = sandbox.execute(code, test_pair["input"])
        if err or out is None:
            return False
        return grids_equal(out, test_pair["output"])
