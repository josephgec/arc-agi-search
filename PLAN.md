# Implementation Plan — PSO Performance Overhaul

Target: 15-20% solve rate, <300s/task

## Baseline
1/10 solved (10%), 569s avg/task, code extraction fails ~50% of Coder calls.

## PHASE 1 — Strip Dead Embedding Weight (Speed)
- [x] 1a: Remove embedding calls from `_particle_step` (velocity update, cosine selection)
- [x] 1b: Remove embedding calls from stagnation/reinit in `solve()`
- [x] 1c: Update tests (all 658 pass, 95% coverage, pso_orchestrator 86%)
GATE: `python -m pytest -x` — PASSED

## PHASE 2 — Adaptive K + Single-Pair Pre-filter (Speed + Accuracy)
- [x] 2a: Adaptive K candidates — more when fitness low, fewer when near-solved
- [x] 2b: Single-pair pre-filter — eval cheapest pair first, skip candidate if it fails
GATE: `python -m pytest -x` — PASSED (662 tests, 95% coverage)

## PHASE 3 — Better PSOCoder Prompts (Accuracy)
- [x] 3a: Visual grid comparisons in eval diff (predicted vs expected grids)
- [x] 3b: Trim Hypothesizer pattern categories (7 concise categories)
GATE: `python -m pytest -x` — PASSED (663 tests, 95% coverage)

## PHASE 4 — Improved Fitness Function (Accuracy)
- [x] 4a: Per-color positional IoU scoring (35% weight in fixed mode, 25%→20% curriculum)
GATE: `python -m pytest -x` — PASSED (663 tests)

## PHASE 5 — Refinement + Stagnation (Accuracy)
- [x] 5a: Lower refinement threshold 0.85→0.60 + max_attempts 5→8
- [x] 5b: Per-particle stagnation (reinit after 3 flat iters, fresh hypothesis)
GATE: `python -m pytest -x` — PASSED (663 tests)

## PHASE 6 — Ensemble + Few-Shot (Accuracy)
- [x] 6a: PSO particle ensemble (try all unique particle codes on test input)
- [ ] 6b: Cross-task few-shot pattern library (deferred)
GATE: `python -m pytest -x` — PASSED (665 tests, 94% coverage)

## Run Results
- **Baseline**: 1/10 (10%), 569s avg, code extraction fails ~50%
- **Post Phase 1-6**: 2/10 (20%), 770s avg, code=True 7/10 first attempts
  - Solved tasks: d10ecb37 (236s), 8be77c9e (266s)
  - Remaining bottleneck: code=False on 27-34K char responses (thinking exhausts token budget)
  - coder_max_tokens increased 4096→8192 to improve extraction rate

## Next Steps (if continuing)
- Increase coder_max_tokens further or add "BE BRIEF" instruction to Coder
- Use qwen2.5-coder:7b for Coder role (non-thinking model, faster)
- Reduce hypothesizer_max_tokens to save budget for Coder
- Add retry with lower temperature on code=False
