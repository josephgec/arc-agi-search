"""Visual grid formatting for ARC-AGI agent prompts.

Single-character colour codes let LLMs perceive spatial structure directly
instead of reasoning over raw integer arrays.  All roles use this format;
the Coder additionally receives the raw list-of-lists for numpy authoring.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

_COLOR_CHARS: dict[int, str] = {
    0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
    5: "X", 6: "M", 7: "O", 8: "A", 9: "W",
}

LEGEND = (
    "(. black, B blue, R red, G green, Y yellow, "
    "X grey, M magenta, O orange, A azure, W maroon)"
)

# Grids with any dimension > this threshold get row/column index labels
_LABEL_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_grid_visual(grid: np.ndarray) -> str:
    """Return a human-readable single-character visual of an ARC grid.

    Grids where both dimensions are ≤ 10: plain symbol rows, one space
    between cells.

    Grids with any dimension > 10: row index labels on the left and a
    column index header on top for spatial orientation.

    A colour legend is always appended as the final line.

    Colour map:
        0→.  1→B  2→R  3→G  4→Y  5→X  6→M  7→O  8→A  9→W

    Args:
        grid: 2-D numpy array of dtype int32 with values 0–9.

    Returns:
        Multi-line string ending with the legend.

    Example (3×3)::

        O . O
        O O .
        . O .
        (. black, B blue, ...)

    Example (12×5, large → labels)::

           0 1 2 3 4
         0: . . . . .
         1: . . . . .
         ...
        (. black, B blue, ...)
    """
    rows, cols = grid.shape

    def cell(v: int) -> str:
        return _COLOR_CHARS.get(int(v), str(v))

    lines: list[str] = []

    if rows <= _LABEL_THRESHOLD and cols <= _LABEL_THRESHOLD:
        for row in grid.tolist():
            lines.append(" ".join(cell(v) for v in row))
    else:
        # Width of the widest row label, e.g. rows=100 → "99:" = 3+1 chars
        r_width = len(str(rows - 1))
        # Column header — indent to align with row content ("N: " prefix)
        indent = " " * (r_width + 2)
        lines.append(indent + " ".join(str(c) for c in range(cols)))
        for r, row in enumerate(grid.tolist()):
            row_str = " ".join(cell(v) for v in row)
            lines.append(f"{r:{r_width}}: {row_str}")

    lines.append(LEGEND)
    return "\n".join(lines)
