"""DSL of common ARC grid transformation primitives.

Every function takes and returns numpy int32 arrays (Grid).
All operations are pure — they return a new array and never mutate the input.
"""
from __future__ import annotations

import numpy as np
from .grid import Grid, background_color


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def crop(grid: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
    """Return the sub-grid at rows [r1:r2], cols [c1:c2] (exclusive end)."""
    return grid[r1:r2, c1:c2].copy()


def rotate(grid: Grid, n: int = 1) -> Grid:
    """Rotate the grid 90° counter-clockwise n times."""
    return np.rot90(grid, n).copy()


def flip(grid: Grid, axis: int = 0) -> Grid:
    """Flip the grid along an axis: 0 = vertical (up/down), 1 = horizontal (left/right)."""
    return np.flip(grid, axis=axis).copy()


def translate(grid: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    """Shift the grid by (dr rows, dc cols), filling vacated cells with `fill`.

    Positive dr shifts down; positive dc shifts right.
    Cells that scroll off an edge are discarded (no wrapping).
    """
    result = np.full_like(grid, fill)
    rows, cols = grid.shape

    src_r = max(0, -dr), max(0, min(rows, rows - dr))
    dst_r = max(0,  dr), max(0, min(rows, rows + dr))

    src_c = max(0, -dc), max(0, min(cols, cols - dc))
    dst_c = max(0,  dc), max(0, min(cols, cols + dc))

    result[dst_r[0]:dst_r[1], dst_c[0]:dst_c[1]] = grid[src_r[0]:src_r[1], src_c[0]:src_c[1]]
    return result


def scale(grid: Grid, factor: int) -> Grid:
    """Scale up a grid by an integer factor (each cell becomes a factor×factor block)."""
    return np.kron(grid, np.ones((factor, factor), dtype=np.int32))


def tile(grid: Grid, n_rows: int, n_cols: int) -> Grid:
    """Repeat the grid n_rows times vertically and n_cols times horizontally."""
    return np.tile(grid, (n_rows, n_cols))


# ---------------------------------------------------------------------------
# Colour operations
# ---------------------------------------------------------------------------

def recolor(grid: Grid, from_color: int, to_color: int) -> Grid:
    """Replace every occurrence of from_color with to_color."""
    result = grid.copy()
    result[result == from_color] = to_color
    return result


def mask(grid: Grid, mask_grid: Grid, fill: int = 0) -> Grid:
    """Zero out (fill) cells where mask_grid == 0, keeping all other cells."""
    result = grid.copy()
    result[mask_grid == 0] = fill
    return result


def overlay(base: Grid, top: Grid, transparent: int = 0) -> Grid:
    """Overlay `top` onto `base`, skipping cells where top == transparent."""
    result = base.copy()
    result[top != transparent] = top[top != transparent]
    return result


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

def flood_fill(grid: Grid, row: int, col: int, new_color: int) -> Grid:
    """Flood-fill starting at (row, col), replacing all connected same-color cells.

    Uses 4-connectivity (up/down/left/right).  Returns grid unchanged if the
    target cell already has new_color.
    """
    result = grid.copy()
    target = int(result[row, col])
    if target == new_color:
        return result

    rows, cols = result.shape
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        if result[r, c] != target:
            continue
        result[r, c] = new_color
        stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
    return result


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def find_objects(grid: Grid, background: int | None = None) -> list[dict]:
    """Find connected foreground objects in the grid.

    Uses depth-first search with 4-connectivity to label connected components.
    Cells with the background colour are ignored.

    Args:
        grid:       The input grid.
        background: The colour to treat as background.  If None, the most
                    frequent colour is used (see background_color()).

    Returns:
        A list of object dicts, each containing:
            color    — colour value of the object
            pixels   — list of (row, col) tuples for every cell in the object
            bbox     — (r_min, c_min, r_max, c_max)  (inclusive)
            subgrid  — smallest Grid that fits the object (background cells = 0)
    """
    if background is None:
        background = background_color(grid)

    objects = []
    visited = np.zeros_like(grid, dtype=bool)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if color == background or visited[r, c]:
                continue

            pixels: list[tuple[int, int]] = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr, cc] or grid[cr, cc] != color:
                    continue
                visited[cr, cc] = True
                pixels.append((cr, cc))
                stack.extend([(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)])

            if not pixels:
                continue

            row_coords = [p[0] for p in pixels]
            col_coords = [p[1] for p in pixels]
            r_min, r_max = min(row_coords), max(row_coords)
            c_min, c_max = min(col_coords), max(col_coords)

            subgrid = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=np.int32)
            for pr, pc in pixels:
                subgrid[pr - r_min, pc - c_min] = color

            objects.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "bbox": (r_min, c_min, r_max, c_max),
                    "subgrid": subgrid,
                }
            )

    return objects


def bounding_box(grid: Grid, color: int | None = None) -> tuple[int, int, int, int]:
    """Return (r_min, c_min, r_max, c_max) of non-background (or specific color) cells.

    If color is given, finds the bounding box of that colour only.
    Otherwise finds the bounding box of all non-background cells.
    Coordinates are inclusive on both ends.
    """
    if color is not None:
        mask_arr = grid == color
    else:
        bg = background_color(grid)
        mask_arr = grid != bg

    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)

    if not np.any(rows):
        return 0, 0, 0, 0

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return int(r_min), int(c_min), int(r_max), int(c_max)


def crop_to_content(grid: Grid) -> Grid:
    """Crop the grid to the tight bounding box of all non-background content.

    Returns the original grid unchanged if the grid is entirely background.
    """
    bg = background_color(grid)
    if (grid == bg).all():
        return grid.copy()
    r_min, c_min, r_max, c_max = bounding_box(grid)
    return crop(grid, r_min, c_min, r_max + 1, c_max + 1)


def pad(grid: Grid, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, fill: int = 0) -> Grid:
    """Add padding around the grid.

    Args:
        grid:   Input grid.
        top:    Rows to prepend above.
        bottom: Rows to append below.
        left:   Columns to prepend on the left.
        right:  Columns to append on the right.
        fill:   Value for new cells (default 0 / background).
    """
    rows, cols = grid.shape
    new_h = rows + top + bottom
    new_w = cols + left + right
    result = np.full((new_h, new_w), fill, dtype=np.int32)
    result[top:top + rows, left:left + cols] = grid
    return result


def symmetrize(grid: Grid, axis: int = 1) -> Grid:
    """Make the grid symmetric by reflecting it about a central axis.

    axis=0: vertical symmetry (reflect left half → right half).
        The right half is replaced by the mirror of the left half.
    axis=1: horizontal symmetry (reflect top half → bottom half).
        The bottom half is replaced by the mirror of the top half.
    axis=2: both axes (left-right then top-bottom).

    Only overwrites background (0) cells in the destination half to avoid
    clobbering existing content.
    """
    result = grid.copy()

    if axis in (0, 2):
        # Left → right mirror
        cols = result.shape[1]
        for c in range(cols // 2):
            mirror = cols - 1 - c
            col_src  = result[:, c]
            col_dest = result[:, mirror]
            result[:, mirror] = np.where(col_dest == 0, col_src, col_dest)

    if axis in (1, 2):
        # Top → bottom mirror
        rows = result.shape[0]
        for r in range(rows // 2):
            mirror = rows - 1 - r
            row_src  = result[r, :]
            row_dest = result[mirror, :]
            result[mirror, :] = np.where(row_dest == 0, row_src, row_dest)

    return result


# ---------------------------------------------------------------------------
# Object property helpers
# ---------------------------------------------------------------------------

def get_color(obj: Grid) -> int:
    """Return the most common non-zero color in obj; 0 if obj is all-zero."""
    flat = obj.flatten(); nz = flat[flat != 0]
    if nz.size == 0: return 0
    values, counts = np.unique(nz, return_counts=True)
    return int(values[np.argmax(counts)])


def get_size(obj: Grid) -> int:
    """Return the count of non-zero cells in obj."""
    return int(np.count_nonzero(obj))


def get_centroid(obj: Grid) -> tuple:
    """Return (row, col) centroid of non-zero cells; (0.0, 0.0) if all-zero."""
    pos = np.argwhere(obj != 0)
    if pos.size == 0: return (0.0, 0.0)
    return (float(pos[:, 0].mean()), float(pos[:, 1].mean()))


# ---------------------------------------------------------------------------
# Grid structure analysis
# ---------------------------------------------------------------------------

def detect_grid_layout(grid: Grid) -> tuple | None:
    """Detect sub-grid structure from row/col divider lines (full rows/cols of same non-zero value).
    Returns (n_row_sections, n_col_sections) or None."""
    rows, cols = grid.shape
    def _dividers(n, get_line):
        d = []
        for i in range(n):
            u = np.unique(get_line(i))
            if u.size == 1 and u[0] != 0: d.append(i)
        return len(d) + 1 if d else None
    rs = _dividers(rows, lambda i: grid[i, :])
    cs = _dividers(cols, lambda i: grid[:, i])
    if rs is None and cs is None: return None
    return (rs or 1, cs or 1)


def find_periodicity(grid: Grid) -> tuple | None:
    """Return the smallest (row_period, col_period) for exact tiling, or None.
    Only periods strictly less than the full dimension are considered."""
    rows, cols = grid.shape
    def _min_period(n, get_slice):
        for p in range(1, n):
            if n % p != 0: continue
            ref = [get_slice(i) for i in range(p)]
            if all(np.array_equal(get_slice(i), ref[i % p]) for i in range(n)):
                return p
        return None
    rp = _min_period(rows, lambda i: grid[i, :])
    cp = _min_period(cols, lambda j: grid[:, j])
    if rp is None and cp is None: return None
    return (rp or rows, cp or cols)


# ---------------------------------------------------------------------------
# Gravity / physics
# ---------------------------------------------------------------------------

def gravity(grid: Grid, direction: str = "down") -> Grid:
    """Slide all non-zero cells toward an edge; zeros fill the vacated side.
    direction: "down" | "up" | "left" | "right". Each column/row slides independently."""
    result = grid.copy()
    rows, cols = result.shape
    if direction in ("down", "up"):
        for c in range(cols):
            col = result[:, c]; nz = col[col != 0]
            zeros = np.zeros(rows - len(nz), dtype=np.int32)
            result[:, c] = np.concatenate([zeros, nz] if direction == "down" else [nz, zeros])
    elif direction in ("left", "right"):
        for r in range(rows):
            row = result[r, :]; nz = row[row != 0]
            zeros = np.zeros(cols - len(nz), dtype=np.int32)
            result[r, :] = np.concatenate([nz, zeros] if direction == "left" else [zeros, nz])
    else:
        raise ValueError(f"direction must be up/down/left/right; got {direction!r}")
    return result
