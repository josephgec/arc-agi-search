# MCTS Benchmark Results

Date: 2026-03-07
Solver: MCTSSolver (LLM-free DSL composition)
Params: depth=5, iters=2000, time=120s, exploration=1.414, rollout_depth=3

## Summary

| Tier | Tasks | Solved | Test OK | Avg Fitness | Avg Time |
|------|-------|--------|---------|-------------|----------|
| 1 (geometric) | 5 | 1/5 (20%) | 2/5 (40%) | 0.868 | 2.6s |
| 2 (shape-change) | 4 | 2/4 (50%) | 4/4 (100%) | 0.870 | 2.0s |
| 3 (cell-level) | 5 | 0/5 (0%) | 0/5 (0%) | 0.888 | 3.4s |
| 4 (two-op) | 4 | 0/4 (0%) | 0/4 (0%) | 0.637 | 3.3s |
| 5 (three-op) | 3 | 0/3 (0%) | 0/3 (0%) | 0.835 | 2.9s |
| 6 (hard) | 3 | 0/3 (0%) | 0/3 (0%) | 0.745 | 3.6s |
| **Total** | **24** | **3/24 (12.5%)** | **6/24 (25%)** | **0.814** | **2.9s** |

## Key Findings

### What MCTS solves well
- **Scale ×2** (c59eb873): Solved at iter 1236, 1.8s. Direct DSL op maps cleanly.
- **Crop to content** (1cf80156): Solved at iter 749, 1.3s. Found via translate+pad composition.
- **Gravity-like** (68b16354): Solved via `gravity('down')` at iter 659, 1.1s.

### Near-misses (fitness > 0.9 but not solved)
- **Gravity down** (1e0a9b12): 0.956 — found `gravity('up',0)` twice instead of `gravity('down',0)` once
- **Fill enclosed** (a5313dff): 0.926 — found `fill_enclosed_regions(1)` but only on some pairs
- **Symmetrize** (496994bd): 0.912 — found translate approximation instead of exact symmetrize
- **Rotate 180°** (6150a2bd): 0.920 — found `rotate(2)` but rollout added extra ops lowering fitness
- **Scale ×3** (9172f3a0): 0.906 — found `scale(3)` but not consistently across all pairs

### What MCTS struggles with
- **Tile 1×2** (a416b8f3): Only 0.572. Tile action found early but not explored deeply enough.
- **Rotation with multiple training pairs**: Gets close but rollout noise prevents convergence to exact solution.
- **Multi-recolor chains** (0d3d703e): Only 0.481. Needs 3 sequential recolors (1→4, 2→5, 3→6); found 2 within budget.
- **All tier 4-6 tasks** (0/10): Two-op and multi-op compositions never fully solved.

### Tier 4-6 highlights
- **025d127b** (tier 5, symmetry+fill): 0.928 fitness — closest near-miss across all hard tasks.
- **007bbfb7** (tier 4, self-tile): 0.790 — found `scale(3)` as approximation but task needs mask-aware tiling.
- **045e512c** (tier 6, sorting+gravity): 0.822 — decent fitness but needs custom logic beyond DSL.

## Observations

1. **Average fitness is high** (0.876) — MCTS finds near-solutions quickly but struggles with the last mile to perfection.

2. **Stochastic rollouts add noise**: Random rollout extensions from a correct depth-1 node often worsen the pipeline, leading the search away from the clean single-op solution.

3. **Test correctness exceeds train solve rate** (43% vs 21%): The fitness function is strict on training pairs. Some solutions that aren't perfect on train still generalize to test.

4. **Speed is excellent**: Average 2.7s per task with 2000 iterations. The search is CPU-bound and fast.

5. **Progressive widening works**: Node counts (800-1400) stay well below the theoretical maximum, focusing search on promising branches.

## Bottleneck Analysis

The main bottleneck is **exploitation depth**: MCTS finds the right single action (e.g., `rotate(2)`, `scale(3)`) but:
- Rollouts extend beyond the correct pipeline, mixing in noise
- The fitness function gives high but imperfect scores for correct transforms when averaged across multiple training pairs
- The search doesn't distinguish "exact match on 2/3 pairs" from "close match on 3/3 pairs"

## Improvement Ideas

1. **Evaluate partial pipelines immediately** — check if the current action sequence (without rollout) already solves, before extending
2. **Depth-1 exhaustive sweep** — try every action at depth 1 first, since many ARC tasks are single-op
3. **Per-pair fitness tracking** — report which pairs are solved vs. which need work
4. **Reduce rollout noise** — lower rollout_depth from 3→1 for the first 500 iterations
