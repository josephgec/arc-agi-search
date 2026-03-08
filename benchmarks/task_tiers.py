"""Curated ARC task tiers for benchmarking MCTS and other solvers.

Each tier contains tasks verified to be solvable by a specific number of DSL
operations.  Tasks were identified by exhaustive search over the training set.

Tier 1 — Single geometric op (rotate, flip)
Tier 2 — Single shape-changing op (scale, tile, crop_to_content)
Tier 3 — Single cell-level op (gravity, symmetrize, fill_enclosed)
Tier 4 — Two-operation compositions
Tier 5 — Three-operation compositions or complex patterns
Tier 6 — Hard / currently unsolvable by pure DSL search
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """A benchmark task with its expected solution metadata."""

    task_id: str
    tier: int
    expected_ops: int          # minimum DSL operations needed
    description: str           # human-readable transformation
    solution_hint: str = ""    # e.g. "rotate(2)" — for validation only


# ---------------------------------------------------------------------------
# Tier 1 — Single geometric transform (rotate / flip)
# ---------------------------------------------------------------------------

TIER_1: list[TaskSpec] = [
    TaskSpec("3c9b0459", 1, 1, "Rotate 180°", "rotate(2)"),
    TaskSpec("6150a2bd", 1, 1, "Rotate 180°", "rotate(2)"),
    TaskSpec("ed36ccf7", 1, 1, "Rotate 90° CCW", "rotate(1)"),
    TaskSpec("67a3c6ac", 1, 1, "Flip horizontal", "flip(1)"),
    TaskSpec("68b16354", 1, 1, "Flip vertical", "flip(0)"),
]

# ---------------------------------------------------------------------------
# Tier 2 — Single shape-changing op (scale, tile, crop_to_content)
# ---------------------------------------------------------------------------

TIER_2: list[TaskSpec] = [
    TaskSpec("9172f3a0", 2, 1, "Scale ×3", "scale(3)"),
    TaskSpec("c59eb873", 2, 1, "Scale ×2", "scale(2)"),
    TaskSpec("a416b8f3", 2, 1, "Tile 1×2", "tile(1, 2)"),
    TaskSpec("1cf80156", 2, 1, "Crop to non-bg content", "crop_to_content()"),
]

# ---------------------------------------------------------------------------
# Tier 3 — Single cell-level op (gravity, symmetrize, fill_enclosed)
# ---------------------------------------------------------------------------

TIER_3: list[TaskSpec] = [
    TaskSpec("1e0a9b12", 3, 1, "Gravity down", "gravity('down')"),
    TaskSpec("3906de3d", 3, 1, "Gravity up", "gravity('up')"),
    TaskSpec("496994bd", 3, 1, "Symmetrize top→bottom", "symmetrize(1)"),
    TaskSpec("00d62c1b", 3, 1, "Fill enclosed regions color 4", "fill_enclosed_regions(4)"),
    TaskSpec("a5313dff", 3, 1, "Fill enclosed regions color 1", "fill_enclosed_regions(1)"),
]

# ---------------------------------------------------------------------------
# Tier 4 — Two-operation compositions
# ---------------------------------------------------------------------------

TIER_4: list[TaskSpec] = [
    # These are tasks where the output requires exactly 2 DSL ops
    TaskSpec("d10ecb37", 4, 2, "Crop + recolor (solved by multi-agent)", ""),
    TaskSpec("8be77c9e", 4, 2, "Flip + vstack (solved by multi-agent)", ""),
    TaskSpec("007bbfb7", 4, 2, "Self-tiling 3×3 (needs mask-aware tile)", ""),
    TaskSpec("0d3d703e", 4, 2, "Multi-recolor chain (1→4, 2→5, 3→6)", ""),
]

# ---------------------------------------------------------------------------
# Tier 5 — Three+ ops or complex patterns
# ---------------------------------------------------------------------------

TIER_5: list[TaskSpec] = [
    TaskSpec("025d127b", 5, 3, "Symmetry + fill gaps", ""),
    TaskSpec("0ca9ddb6", 5, 3, "Object detection + recolor by property", ""),
    TaskSpec("05f2a901", 5, 3, "Grid layout extraction + assembly", ""),
]

# ---------------------------------------------------------------------------
# Tier 6 — Hard / currently unsolvable by pure DSL
# ---------------------------------------------------------------------------

TIER_6: list[TaskSpec] = [
    TaskSpec("045e512c", 6, 4, "Complex sorting + gravity", ""),
    TaskSpec("0520fde7", 6, 4, "Pattern completion with context", ""),
    TaskSpec("0e206a2e", 6, 4, "Object counting + assembly", ""),
]

# ---------------------------------------------------------------------------
# Convenience groupings
# ---------------------------------------------------------------------------

ALL_TIERS: dict[int, list[TaskSpec]] = {
    1: TIER_1,
    2: TIER_2,
    3: TIER_3,
    4: TIER_4,
    5: TIER_5,
    6: TIER_6,
}

SMOKE_TEST: list[TaskSpec] = [
    TIER_1[0],   # 3c9b0459 — rotate(2)
    TIER_2[0],   # 9172f3a0 — scale(3)
    TIER_3[0],   # 1e0a9b12 — gravity(down)
]


def get_tier(tier: int) -> list[TaskSpec]:
    """Return tasks for a given tier number (1–6)."""
    return ALL_TIERS.get(tier, [])


def get_all_tasks() -> list[TaskSpec]:
    """Return all benchmark tasks across all tiers."""
    tasks: list[TaskSpec] = []
    for tier_tasks in ALL_TIERS.values():
        tasks.extend(tier_tasks)
    return tasks
