"""Single source of truth for the DSL function reference string.

Imported by all system-prompt locations so every agent sees the same list.

IMPORTANT: No curly braces in _DSL_REFERENCE — system prompts that call
.format(k=k) would misinterpret stray braces.
"""

_DSL_REFERENCE = """\
Available DSL primitives (already imported — do NOT re-import):

GEOMETRIC TRANSFORMS
  crop(grid, r1, c1, r2, c2) -> Grid      Sub-grid at rows [r1:r2], cols [c1:c2].
  rotate(grid, n=1) -> Grid               Rotate 90 deg CCW n times.
  flip(grid, axis=0) -> Grid              Flip: axis=0 vertical, axis=1 horizontal.
  translate(grid, dr, dc, fill=0) -> Grid Shift by (dr rows, dc cols).
  scale(grid, factor) -> Grid             Scale up by integer factor.
  tile(grid, n_rows, n_cols) -> Grid      Repeat grid n_rows x n_cols times.

COLOUR OPERATIONS
  recolor(grid, from_color, to_color) -> Grid   Replace one color with another.
  mask(grid, mask_grid, fill=0) -> Grid          Zero out cells where mask_grid==0.
  overlay(base, top, transparent=0) -> Grid      Overlay top onto base.

FLOOD FILL
  flood_fill(grid, row, col, new_color) -> Grid  4-connected flood fill.

OBJECT DETECTION
  find_objects(grid, background=None) -> list[dict]
      Each dict: color, pixels, bbox (r_min,c_min,r_max,c_max), subgrid.
  bounding_box(grid, color=None) -> (r_min,c_min,r_max,c_max)
      Bounding box of non-background or specific-color cells.
  crop_to_content(grid) -> Grid           Tight crop to non-background cells.

PADDING / SYMMETRY
  pad(grid, top=0, bottom=0, left=0, right=0, fill=0) -> Grid  Add border.
  symmetrize(grid, axis=1) -> Grid        Mirror: axis=0 left-right, 1 top-bottom, 2 both.

OBJECT PROPERTY HELPERS
  get_color(obj) -> int          Most common non-zero color; 0 if all-zero.
  get_size(obj) -> int           Count of non-zero cells.
  get_centroid(obj) -> (float,float)  Row,col centroid of non-zero cells.

GRID STRUCTURE ANALYSIS
  detect_grid_layout(grid) -> (n_rows,n_cols)|None   Sub-grid sections from divider lines.
  find_periodicity(grid) -> (rp,cp)|None             Smallest exact tiling period.

ENCLOSED REGION FILL
  fill_enclosed_regions(grid, fill_color, bg_color=None) -> Grid
      Fill background cells that are completely enclosed by non-background cells.
      Uses BFS from all border cells; any background cell unreachable from the
      border is "enclosed" and gets painted fill_color.  bg_color defaults to
      background_color(grid) (0 when any zero is present).
      Example: a ring of 1s surrounding interior 0s -> interior 0s become fill_color.

PHYSICS / GRAVITY
  gravity(grid, direction="down", bg_color=0) -> Grid
      Slide non-background cells to edge; bg_color fills vacated cells.
      direction: "down" | "up" | "left" | "right". Each column/row slides independently.
      bg_color: color treated as empty space (default 0).

SAFE INDEXING
  safe_neighbors(grid, r, c, size=1) -> Grid
      Return the neighbourhood of (r,c) clamped to grid boundaries.
      Equivalent to grid[r-size:r+size+1, c-size:c+size+1] but safe at edges.
      ALWAYS use this instead of grid[r-1:r+2, c-1:c+2] to avoid index
      wrapping when r=0 or c=0 (Python negative indices return wrong data).

numpy is also available as np and numpy.

PARAMETERIZED OPTIMIZATION (optional — CPU sweeps constants automatically):
  If your transform depends on unknown constants (target colour, shift amount,
  threshold…), define PARAM_GRID and accept **params in the signature.
  The solver will test all combinations against training pairs instantly.

  Example:
    PARAM_GRID = dict(
        target_color=list(range(10)),
        fill_color=list(range(10)),
    )
    def transform(grid, target_color=0, fill_color=1):
        result = grid.copy()
        result[result == target_color] = fill_color
        return result

  Rules:
  - PARAM_GRID must be a plain dict of name -> list of values (use dict() not literal braces).
  - transform must have keyword arguments with sensible defaults
    so it still runs as transform(grid) when params are not swept.
  - Keep each list short (<=10 values) to avoid combinatorial explosion.
"""
