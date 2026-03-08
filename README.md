# ARC-AGI PSO Swarm Solver

A hybrid **Particle Swarm Optimization + LLM** architecture for the [ARC-AGI challenge](https://github.com/fchollet/ARC-AGI). Standard LLM agent loops get stuck in repetitive generation cycles (local minima). This system escapes that trap by coupling PSO swarm topology (pbest/gbest sharing, stagnation reinit, crossover) with LLM code generation via a **Mutation-and-Select bridge**.

---

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Repository Structure](#repository-structure)
3. [Architecture Overview](#architecture-overview)
4. [Module Deep-Dives](#module-deep-dives)
   - [arc/ — Core Library](#arc--core-library)
   - [agents/ — Agent Layer](#agents--agent-layer)
5. [The PSO Algorithm in Detail](#the-pso-algorithm-in-detail)
6. [The Mutation-and-Select Bridge](#the-mutation-and-select-bridge)
7. [Continuous Fitness Function](#continuous-fitness-function)
8. [Workflow Diagrams](#workflow-diagrams)
9. [Pseudocode](#pseudocode)
10. [Running the Solver](#running-the-solver)
11. [Running Tests](#running-tests)
12. [MCTS Solver & Benchmarks](#mcts-solver--benchmarks)

---

## The Core Idea

Standard single-agent LLM loops produce one solution at a time and get stuck in local minima — generating minor variations of the same wrong code. PSO escapes this by maintaining a **swarm of N particles**, each with a specialist role, exploring different regions of the solution space simultaneously.

**How the swarm works:**

```
N particles, each with a specialist role (geometric, color, pattern, ...)
      │
      ▼
Each particle generates K candidate mutations
  informed by pbest_code (personal best) and gbest_code (global best)
      │
      ▼
Sandbox evaluates all K candidates → select highest-fitness winner
      │
      ▼
Update pbest / gbest → share knowledge across swarm → iterate
```

This decouples **search direction** (PSO swarm topology sharing pbest/gbest) from **solution generation** (LLM code mutations). The swarm systematically explores the space of programs rather than randomly sampling. Diversity comes from specialist roles, stagnation recovery via crossover and reinitialisation, and parallel exploration across particles.

---

## Repository Structure

```
arc-agi-search/
│
├── arc/                        ← Core library (backend-agnostic)
│   ├── grid.py                 ← Grid type, I/O, colour utilities
│   ├── dsl.py                  ← Pure transformation primitives
│   ├── dsl_actions.py          ← DSL action enumeration for MCTS tree expansion
│   ├── sandbox.py              ← Hardened subprocess code execution
│   └── evaluate.py             ← Binary eval + continuous fitness
│
├── agents/                     ← Agent layer
│   ├── llm_client.py           ← Unified LLM + embedding API
│   ├── roles.py                ← Hypothesizer, Coder, Critic, Decomposer, Verifier, PSOCoder
│   ├── dsl_reference.py        ← DSL primitives reference injected into LLM context
│   ├── formatting.py           ← Visual grid formatting (single-char colour codes for LLM prompts)
│   ├── multi_agent.py          ← Hypothesizer → Coder → Critic loop (with Decomposer + Verifier)
│   ├── orchestrator.py         ← Candidate-pooling wrapper (exposes fitness/perfect fields)
│   ├── ensemble.py             ← Pixel-weighted majority-vote ensemble with self-correction
│   ├── single_agent.py         ← Single-agent baseline
│   ├── pso_orchestrator.py     ← ★ PSO swarm solver (main contribution)
│   └── mcts_solver.py          ← ★ LLM-free MCTS over DSL pipelines
│
├── benchmarks/                 ← Performance benchmark infrastructure
│   ├── task_tiers.py           ← 24 curated tasks across 6 difficulty tiers
│   ├── run_benchmark.py        ← Benchmark runner with per-task timeout
│   ├── analyze_results.py      ← Post-hoc analysis (histograms, speed, head-to-head)
│   └── RESULTS.md              ← Benchmark findings
│
├── tests/                      ← 774 tests, 94% coverage
│
├── data/
│   ├── training/               ← 400 ARC training tasks (JSON)
│   └── evaluation/             ← 400 ARC evaluation tasks (JSON)
│
├── run_pso.py                  ← CLI entry point for PSO solver
├── run_multi_agent.py          ← Unified CLI (--strategy flag)
├── run_batch_test.py           ← Quick 5-task batch smoke test
├── run_single_test.py          ← Debug single-task run with verbose cycle tracking
├── start_ollama.sh             ← Launch Ollama with tiered model selection
└── requirements.txt
```

---

## Architecture Overview

The system couples PSO swarm topology with LLM code generation. N particles with specialist roles generate K candidate mutations each, informed by shared pbest/gbest code. Candidates are evaluated in the sandbox and the highest-fitness winner is selected (greedy).

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PSO SWARM TOPOLOGY                            │
│                                                                     │
│   Particle 1 (geometric)  ──► K candidates ──► sandbox eval         │
│   Particle 2 (color)      ──► K candidates ──► sandbox eval         │
│   Particle 3 (pattern)    ──► K candidates ──► sandbox eval         │
│   Particle 4 (object)     ──► K candidates ──► sandbox eval         │
│   Particle 5 (rule)       ──► K candidates ──► sandbox eval         │
│   Particle 6 (hybrid)     ──► K candidates ──► sandbox eval         │
│                                                                     │
│   Each informed by:  pbest_code + gbest_code + eval diffs           │
│   Selection:         highest fitness wins (greedy)                  │
│   Sharing:           gbest snapshot broadcast each iteration        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                         CODE SPACE                                  │
│                                                                     │
│   "def transform(g):        "def transform(g):                      │
│       return rotate(g)"         return recolor(g, 1, 2)"           │
│                                                                     │
│   LLM generates K candidates ──► sandbox evaluates ──► fitness     │
└─────────────────────────────────────────────────────────────────────┘
```

### Solving Strategies

| Strategy | Class | When to use |
|---|---|---|
| `pso` | `PSOOrchestrator` | Best exploration; avoids local minima |
| `multi` | `MultiAgent` | Fast; good for well-structured tasks |
| `ensemble` | `Ensemble` | Highest reliability via majority voting |
| `single` | `SingleAgent` | Quick baseline check |
| `two_phase` | `TwoPhaseOrchestrator` | Fast MultiAgent warm-up → seeded PSO; best of both |
| `mcts` | `MCTSSolver` | LLM-free; fast DSL pipeline search (~3s/task) |

---

## Module Deep-Dives

### arc/ — Core Library

#### `arc/grid.py` — Data Layer

Every grid is an `np.ndarray` of shape `(H, W)` with `dtype=int32`. Values 0–9 map to ARC colours.

```
Grid = np.ndarray   # int32, shape (H, W), values in [0, 9]

COLOR_NAMES = {0: "black", 1: "blue", 2: "red", 3: "green",
               4: "yellow", 5: "grey", 6: "magenta", 7: "orange",
               8: "azure", 9: "maroon"}
```

Key functions:

| Function | Description |
|---|---|
| `load_task(path)` | Load a JSON task file → `{train: [...], test: [...]}` |
| `grids_equal(a, b)` | True iff shape and all values match |
| `unique_colors(grid)` | Sorted list of distinct colour values |
| `background_color(grid)` | 0 (black) if present; else most-frequent. Intentional deviation from ARC-AGI's "most-frequent" convention — 0 is the canvas colour in >95% of tasks, and hardcoding avoids misclassification on tasks where a non-zero minority colour is structurally the background |

---

#### `arc/dsl.py` — Transformation Primitives

All functions are **pure** — they return a new array and never mutate the input.

```
Geometric:   crop · rotate · flip · translate · scale · tile
Colour:      recolor · mask · overlay
Fill:        flood_fill · fill_enclosed_regions
Detection:   find_objects · bounding_box · crop_to_content
Physics:     gravity(grid, direction, bg_color=0)
Padding:     pad · symmetrize
Properties:  get_color · get_size · get_centroid
Analysis:    detect_grid_layout · find_periodicity
```

These are the only allowed operations inside generated `transform()` functions. They are injected into the sandbox execution namespace automatically.

**`fill_enclosed_regions(grid, fill_color, bg_color=None)`** — BFS from all border cells; any background cell unreachable from the border is *enclosed* and gets painted `fill_color`. Handles rings, thick borders, and multi-region interiors. `bg_color` defaults to `background_color(grid)`.

**`gravity(grid, direction, bg_color=0)`** — Slides all non-background cells toward an edge (`"up"`, `"down"`, `"left"`, `"right"`). The `bg_color` parameter (default `0`) lets gravity work correctly on grids whose empty colour is not black.

**`pad(grid, top, bottom, left, right, fill)`** — Add padding around the grid with the specified fill value.

**`symmetrize(grid, axis)`** — Mirror the grid: axis=0 reflects left→right, axis=1 reflects top→bottom, axis=2 both. Only overwrites background (0) cells in the destination half to avoid clobbering existing content.

**`get_color(obj)`** — Most common non-zero color in a subgrid; 0 if all-zero.

**`get_size(obj)`** — Count of non-zero cells.

**`get_centroid(obj)`** — `(row, col)` centroid of non-zero cells.

**`detect_grid_layout(grid)`** — Detect sub-grid structure from divider lines (full rows/cols of a single non-zero value). Returns `(n_row_sections, n_col_sections)` or `None`.

**`find_periodicity(grid)`** — Smallest `(row_period, col_period)` for exact tiling, or `None`.

---

#### `arc/sandbox.py` — Hardened Execution

Generated code is untrusted and could contain infinite loops, import statements, or crashes. Every execution runs in a **separate child process** with a hard wall-clock timeout via a persistent `ProcessPoolExecutor`.

```
┌─ Parent process ────────────────────────────────────────────┐
│                                                             │
│  _POOL = ProcessPoolExecutor(max_workers=8, fork)           │
│       (created lazily; shared across all execute() calls)  │
│                                                             │
│  execute(code, input_grid, timeout=10s)                     │
│       │                                                     │
│       ├── pool.submit(_subprocess_worker, code, grid)       │
│       ├── future.result(timeout=10s)                        │
│       │       worker: inject DSL_NAMESPACE, exec, transform │
│       │       returns (status, value) tuple                 │
│       └── TimeoutError / BrokenProcessPool → handled        │
│                                                             │
│  evaluate_code(code, task) ──► {pairs, n_correct, ...}      │
│  shutdown_pool() ──► atexit-registered; safe to call ×N     │
└─────────────────────────────────────────────────────────────┘
```

Using a persistent pool eliminates the 50–200 ms process-spawn overhead on every sandbox call. Workers are forked from the parent so the DSL namespace is inherited without re-importing.

**`safe_neighbors(grid, r, c, size=1)`** — Neighbourhood of `(r,c)` clamped to grid boundaries. Returns a sub-array of shape up to `(2*size+1, 2*size+1)`. Use instead of `grid[r-1:r+2, c-1:c+2]` to avoid negative index wrapping bugs at row 0 or the last row/column.

Security guards:
- `input()` / `sys.stdin` usage → rejected before spawning
- Hard timeout via `future.result(timeout=t)` → timed-out worker replaced automatically by the pool
- `BrokenProcessPool` → caught; pool reset via `shutdown_pool()`
- Crash and infinite-loop isolation via child process + hard timeout. Network and filesystem access are **not** explicitly blocked — trust-boundary enforcement is the responsibility of the deployment environment

**Parameterized CPU Search** (`param_search`): if generated code defines a `PARAM_GRID = dict(...)` mapping parameter names to lists of values, `param_search` sweeps all combinations (up to `MAX_PARAM_COMBINATIONS = 5000`) in a single sandboxed subprocess, returning the best `(params, fitness)` pair — offloading constant-guessing from the LLM to the CPU:

```python
# LLM writes this pattern; CPU sweeps all 100 combinations automatically
PARAM_GRID = dict(target_color=list(range(10)), fill_color=list(range(10)))

def transform(grid, target_color=0, fill_color=1):
    return recolor(grid, target_color, fill_color)
```

---

#### `arc/evaluate.py` — Fitness Functions

Two evaluation modes:

**Binary** (`evaluate_task`): used by the existing multi-agent and ensemble strategies.

**Continuous** (`calculate_continuous_fitness`): used by PSO to get a gradient signal beyond pass/fail. See [Continuous Fitness Function](#continuous-fitness-function) for full details.

```
fitness = 0.15 × dim_score + 0.20 × color_score + 0.30 × pixel_score + 0.35 × color_iou_score

dim_score   = min(H_pred,H_target)/max(...) × min(W_pred,W_target)/max(...)
color_score = |colors_pred ∩ colors_target| / |colors_pred ∪ colors_target|
pixel_score = correct_cells / total_target_cells
color_iou_score = mean per-color positional IoU (same-shape only; 0.0 otherwise)
```

| Case | Fitness |
|---|---|
| Prediction is `None` (crash) | `0.0` |
| Perfect pixel-perfect match | `1.0` |
| Right colours, right positions for most colors | `0.7–0.9` |
| Right colours, wrong positions | `0.2–0.5` |
| Wrong shape | `< 0.5` |

**Curriculum-Weighted Mode** (when `progress` is passed): uses 5 components with weights that shift from shape/palette matching early to pixel exactness late:
- Dimension score (20%→10%), Color palette (20%→10%), Object count (15%→10%), Pixel accuracy (20%→50%), Per-color IoU (25%→20%)
- Object count uses 8-connectivity (`_count_objects_total`)
- Used by PSO to gradually increase pressure for pixel-perfect solutions

**AST Complexity Penalty** (`calculate_complexity_penalty`): deducted from fitness to penalise memorisation code. Two additive terms, overall cap `0.50`:

| Term | Trigger | Per-node penalty |
|---|---|---|
| `If` nodes | Heavy branching | +0.005 |
| Large literals (>5 elements) | Hardcoded arrays/dicts | +0.02 |
| `Compare` nodes | Raw coordinate comparisons | +0.002 |
| **Literal-data ratio** | `sum(repr lengths of ast.Constant nodes) / len(code) > 0.40` | `+min(0.50, ratio)` |

The literal-data ratio term specifically targets lookup-table memorisation — code whose body is dominated by hardcoded grid data rather than transformation logic. A memoriser accumulates enough penalty that its net fitness stays below `1.0`, preventing it from displacing genuinely correct `gbest` code. A `SyntaxError` returns a flat `0.10` penalty.

```
final_fitness = max(0.0, raw_fitness - complexity_penalty)
```

**8-Way Connectivity**: Object counting (`_count_objects_total`) uses all 8 neighbours (4 cardinal + 4 diagonal), so cross-shaped and X-shaped objects are correctly counted as a single connected component.

---

### agents/ — Agent Layer

#### `agents/llm_client.py` — Unified LLM Interface

Supports two backends behind a single `generate()` call:

```
LLMClient(backend="ollama"|"anthropic", model=..., temperature=..., ...)

  .generate(system, messages)             → str         # text completion
  .embed_code(code_str)                   → np.ndarray  # L2-normalised float32 vector
  .batch_generate(requests, max_workers)  → list[str]   # parallel completions
```

`batch_generate` uses `ThreadPoolExecutor` to fire N completions in parallel, saturating all `OLLAMA_NUM_PARALLEL` slots simultaneously:

**Embedding** always uses Ollama's `/api/embeddings` endpoint regardless of which chat backend is selected:

```
POST http://localhost:11434/api/embeddings
{ "model": "nomic-embed-text", "prompt": "<code>" }
→ { "embedding": [0.12, -0.34, ...] }   # 768-dimensional vector

After retrieval: vec = vec / ||vec||₂     # L2 normalise to unit sphere
```

---

#### `agents/roles.py` — Agent Roles

Six role classes, each wrapping an `LLMClient` with a purpose-built system prompt:

```
Hypothesizer   generates 3 competing natural-language hypotheses about the
               transformation rule. System prompt includes seven pattern
               categories: MOVEMENT/ATTRACTION, COLOR PERMUTATION/SWAP,
               OUTPUT SYMMETRY, BLOCK SELECTION, OBJECT OPERATIONS,
               PATTERN FILL, and GRAVITY/PROJECTION. Categories reference
               [Structural] hints injected by the orchestrator so the
               model can detect swap tables, mirrored outputs, and block
               selection.

Coder          translates one hypothesis into a Python transform() function
               using only the DSL primitives.
               prior_failures= parameter: a rolling window of up to 3
               (code_snippet, critic_feedback) pairs from previous failed
               attempts under the same hypothesis, injected into the prompt
               as explicit negative examples so the model avoids repeating
               broken patterns.
               Includes a DEFENSIVE CODING section: guard find_objects()
               with `if not objects: return input_grid.copy()` and any
               filtered list with a fallback to avoid max()/min() on empty
               collections.

Critic         reads the error diff and decides:
               ROUTE: hypothesizer  ← hypothesis is fundamentally wrong
               ROUTE: coder         ← implementation bug, same hypothesis
               Now also receives a grid-comparison block (Input / Expected /
               Actual side-by-side, capped at 10×10) and an explicit
               ⚠ IDENTITY TRANSFORM warning when the code returns the input
               unchanged, enabling detection of no-op bugs.

Decomposer     fires when the hypothesis is stuck (no_improve_count ≥ 2 AND
               n_correct == 0) AND there has been partial progress earlier
               in the solve (best_n_correct > 0). Decomposes the task into
               sub-goals to break out of a local-minimum hypothesis.
               Only fires once per hypothesis (decomp_tried flag).
               When best_n_correct == 0, the Decomposer is skipped and
               the system escalates directly to the next hypothesis.

Verifier       gates success — re-reads the code and training pairs and
               confirms all_correct before the loop exits; fail-safe
               (returns passes=True) on any malformed or missing response

PSOCoder       generates K distinct Python functions that blend the logic of
               pbest and gbest.
               failed_examples= parameter: list of (snippet, fitness, error)
               tuples shown before the reference code so the model avoids
               repeating candidates that already proved ineffective.
               Capped at the 5 most recent failures per particle (rolling
               window).
               Also includes DEFENSIVE CODING guards (same as Coder).
```

---

#### `agents/multi_agent.py` — Multi-Agent Orchestrator

A **state machine** that runs up to `max_cycles` total LLM calls (CLI: `--max-cycles`, default 9). Phases 3–4 added Decomposer, Verifier, spatial diffs, and near-miss candidate collection:

```
┌────────────────┐
│  Hypothesizer  │◄──── feedback (if critic says: try new hypothesis)
│  generates 3   │◄──── [Cross-pair analysis] hints (color swaps, symmetry,
│  hypotheses    │       scale ratio, block selection — see below)
└───────┬────────┘
        │ hyp[0]
        ▼
┌────────────────┐     all correct?
│     Coder      │────────────────► Verifier ──► DONE (or Coder retries)
│  writes code   │
└───────┬────────┘
        │ fails
        ▼
┌────────────────┐     ROUTE=hypothesizer ──► advance hyp index
│     Critic     │     (spatial diff + grid comparison + identity warning)
│  diagnoses     │     ROUTE=coder ──────────► Coder retries with feedback
└───────┬────────┘
        │ stagnation (no_improve_count ≥ 2 AND n_correct == 0
        │             AND best_n_correct > 0)
        ▼
┌────────────────┐
│  Decomposer    │──── breaks task into sub-goals ──► Hypothesizer restart
└────────────────┘     (only fires once per hypothesis; skipped when
                        best_n_correct == 0 — escalates to next hyp instead)
```

**Cross-pair analysis** (`_cross_pair_notes`): runs once per task before the first Hypothesizer call and appends a `[Cross-pair analysis]` block to the task description. Detects four cross-example signals:

| Signal | What it catches |
|---|---|
| **Fixed color swap** | `A→B` in pair 1, `B→A` in pair 2 → `FIXED SWAP: A↔B` hint |
| **Output symmetry** | All outputs identical under `np.fliplr`/`np.flipud` → 4-way/H/V hint |
| **Scale ratio** | All pairs `H×W → H·N × W·N` for the same integer N → "N× scale" hint |
| **Block selection** | Output matches one of K equal horizontal blocks in the input → "selects a block" hint |

**Critic grid comparison** (`_format_grid_comparison`): appended to the Critic's diff context for training pair 0. Renders Input / Expected / Actual grids (capped at 10×10) and prepends `⚠ CODE APPEARS TO BE IDENTITY TRANSFORM` when `actual == input`, enabling detection of no-op bugs where the code returns the input unchanged.

`compute_spatial_diff()` produces natural-language feedback (e.g. *"object shifted 2 rows down"*, *"bottom-right region: 3 wrong cells (expected blue, got red)"*) that the Critic uses to generate targeted fix instructions.

---

#### `agents/orchestrator.py` — Pooling Orchestrator

Extends `MultiAgent` with candidate collection for the Ensemble layer. Derives `max_cycles` from `1 + n_hypotheses × (1 + max_retries × 2)` (the leading 1 accounts for the initial Hypothesizer call). Its `solve()` method calls the parent loop but collects every correct code string (with fitness scores) into a `candidates` list so `Ensemble` can run majority voting across them.

---

#### `agents/single_agent.py` — Single-Agent Baseline

One-shot hypothesize → code solver with **no Critic feedback loop**. Generates 3 hypotheses, tries each once with the Coder, and returns the best result. Useful as a performance baseline and quick sanity check. Constructor takes the same LLM config as `MultiAgent` (`backend`, `model`, `temperature`, `timeout`).

---

#### `agents/dsl_reference.py` — DSL Reference String

Single source of truth for the DSL function reference injected into all agent system prompts. Every agent (Coder, PSOCoder, Hypothesizer) sees the same function signatures and descriptions. Deliberately avoids curly braces so it can be safely used inside `.format()` calls.

---

#### `agents/pso_orchestrator.py` — PSO Swarm Solver

The core contribution. See [The PSO Algorithm in Detail](#the-pso-algorithm-in-detail) below.

Key additions beyond the base PSO loop:

**Parallel particle updates** — each iteration runs all N particles concurrently inside a `ThreadPoolExecutor(max_workers=N)`. Ollama HTTP calls release the GIL, so all N LLM mutation requests are in-flight simultaneously. A gbest snapshot is taken before each iteration; Step 7 (global best update) reconciles in the main thread after all futures complete. An early-exit cancels queued futures the moment any particle reaches `fitness ≥ 1.0`.

**Hypothesis-guided initialization** — before generating initial code for each particle, the Hypothesizer produces N hypotheses (at least 3, or `n_particles` if larger). Each particle receives a different hypothesis matching its specialist role, guiding its initial solution toward a diverse starting point. This replaces generic role-only prompting with task-specific reasoning.

**Adaptive K** — the number of mutation candidates per particle is adjusted each iteration based on personal best fitness:
- `pbest_fitness ≥ 0.7` → `k_eff = max(2, k/2)` (exploit: fewer, focused candidates)
- `pbest_fitness < 0.7` → `k_eff = k` (explore: full candidate budget)

**Single-pair pre-filter** — when a task has 2+ training pairs, each candidate is first tested on only the first pair. Candidates that crash or score zero fitness are skipped, avoiding the cost of full multi-pair evaluation.

**Per-particle stagnation** — independent of global stagnation, any particle whose personal best hasn't improved for 3 consecutive iterations is reinitialized with a fresh hypothesis from the Hypothesizer. This prevents a single particle from wasting LLM budget on an unproductive region while the rest of the swarm makes progress.

**Crossover on stagnation** — when the swarm stagnates (gbest unchanged for `stagnation_limit` iterations), the solver attempts LLM-based crossover before reinitializing. It asks the LLM to study what each parent does correctly and write a single `transform` that merges the strongest logic from both — not just picking one parent. If crossover produces a better solution, stagnation resets; otherwise, the worst particle is reinitialized.

**Phase 3 — Targeted Refinement** (fitness ≥ 0.60): when the best solution is promising but not perfect, the solver runs up to 8 focused fix attempts. Each attempt provides the LLM with:
- Full diff context for all training pairs (up to 24 mismatches shown)
- For near-perfect solutions (≥ 0.97): full predicted vs expected grid comparisons
- Pass/fail contrast hints (which pairs pass and which fail)
- Structural hints from training data

Terminates early if no improvement is seen for 2 consecutive attempts.

**Phase 4 — Particle Ensemble Fallback**: if the gbest code doesn't solve the test input, the solver tries every unique code from all particles' personal bests (sorted by fitness, best first) on the test input. The first code that produces a correct test output is returned as the solution. This recovers cases where a non-gbest particle found a solution that generalises better to the test input.

**Staleness early-exit** — if `gbest_fitness` hasn't improved by more than `0.01` for 3 consecutive iterations, the loop terminates early and logs `"[pso] Stagnation detected after {n} iterations, terminating"`. This is independent of the existing particle-reinit stagnation mechanism (which uses a tighter `1e-6` threshold and triggers crossover/reinit instead of stopping).

**`seed_particles(initial_codes)`** — call before `solve()` to pre-seed the first N particles with provided code strings instead of LLM-generated initialisation. Used by `TwoPhaseOrchestrator` to hand the MultiAgent's best code directly to PSO. Empty strings fall back to normal LLM init; the seed list is consumed (reset to `[]`) at the start of each `solve()` call.

**Behavioral Embedding** (default `embed_mode="behavioral"`): instead of embedding the Python source code, the system runs the code against all training pairs and embeds a textual representation of the predicted outputs. This means particles are positioned in embedding space based on *what they produce*, not *how they're written* — two syntactically different programs that produce the same outputs will occupy the same point. Small grids (≤10×10) are embedded as full matrices; larger grids use a sparse active-coordinate format to avoid token blowout. Results are cached per unique behavior string. Fallback: if behavioral embedding fails (e.g. sandbox crash), falls back to embedding the code text directly.

---

#### `run_multi_agent.py` — `TwoPhaseOrchestrator`

A coordination wrapper that combines fast multi-agent warm-up with PSO refinement:

```
Phase 1 — MultiAgent(max_cycles=5, model=phase1_model)
    fast 7b model; low token budget; quick pattern check
          │
          ├─ success? ──► return result immediately (PSO never runs)
          │
          └─ failed? ──► pso.seed_particles([best_code_from_phase1])
                               │
                         Phase 2 — PSOOrchestrator(full reasoner model)
                               starts from a warmed-up position rather
                               than random initialisation
```

Activated with `--strategy two_phase`. Use `--coder-model qwen2.5-coder:7b` to set the fast phase-1 model while keeping the 32b reasoner for phase 2.

---

#### `agents/ensemble.py` — Pixel-Weighted Majority-Vote Ensemble

Runs `Orchestrator` up to `max_runs` times, pools **perfect** and **near-miss** programs, then resolves the test prediction via per-pixel weighted voting and an optional Critic self-correction loop:

**Candidate pools:**

| Pool | Condition | Vote weight |
|---|---|---|
| `perfect_pool` | `all_correct == True` on training pairs | `1.0` |
| `near_miss_pool` | fitness ≥ `near_miss_threshold` (default 0.85) | `near_miss_weight` (default 0.5) |

**Pixel-level voting** (`_pixel_majority_vote`): at each `(row, col)` position, the colour with the highest total weight wins. Grids whose shape differs from the majority shape are ignored.

**Self-correction loop** (`use_correction=True`, up to `max_corrections=2` rounds):

```
Run 1 → code_A (perfect,  weight=1.0) ──┐
Run 2 → code_B (near-miss, weight=0.5) ──┤  pixel-weighted vote per cell
Run 3 → code_A (duplicate — skip)       │
Run 4 → code_C (perfect,  weight=1.0) ──┘
                                          │
              Prediction = pixel majority vote
                                          │
              Critic LLM: "Is this consistent with the training rule?"
                                          │
           ACCEPT ──► return prediction   │   REJECT ──► exclude matching
                                              outputs, re-vote (capped at
                                              max_corrections iterations)
```

Outputs the final grid along with a `corrections_done` count and `vote_summary`.

---

## The PSO Algorithm in Detail

### Particles

Each of the N particles represents one candidate solution. It maintains:

```python
@dataclass
class Particle:
    code:          str          # current Python transform() function
    pos:           np.ndarray   # behavioral embedding (for position tracking)
    velocity:      np.ndarray   # currently unused (hardcoded to zeros)
    fitness:       float        # continuous fitness in [0, 1]
    pbest_code:    str          # best code this particle ever found
    pbest_pos:     np.ndarray   # embedding of pbest
    pbest_fitness: float        # fitness of pbest
    stagnation_iters: int       # consecutive iters with no pbest improvement
    failed_history:   list      # rolling window of failed candidates (max 5)
```

### Particle Roles

Six specialist roles seed diverse initial solutions:

| Role | Cognitive bias |
|---|---|
| `geometric_specialist` | Rotation, flipping, translation, scaling |
| `color_specialist` | Recoloring, masking, palette operations |
| `pattern_analyst` | Periodic tiling, symmetry, repetition |
| `object_tracker` | Connected-component analysis, bounding boxes |
| `rule_abstractor` | Minimal abstract rules, clean generalisation |
| `hybrid_solver` | Holistic, combines all of the above |

### How Particles Update

Each iteration, particles generate candidates by asking the LLM to blend their personal best code (`pbest_code`) with the global best code (`gbest_code`). The LLM acts as the mutation operator — there is no vector arithmetic on embeddings for candidate selection. Selection is purely fitness-greedy: the candidate with the highest sandbox fitness wins.

**Adaptive inertia**: `w` decreases linearly from `w` to `w×0.4` over the iteration budget (e.g. 0.50 → 0.20). Early iterations explore; later ones exploit local optima.

Default hyperparameters: `w=0.5` (initial, linearly annealed to `w×0.4`), `c1=1.5`, `c2=1.5`

---

## The Mutation-and-Select Bridge

This is the mechanism that connects PSO swarm structure with LLM code generation:

```
Step 1 — PSOCoder generates K candidates
         Each is a blend of pbest_code and gbest_code
         guided by a prompt that shares both solutions,
         their fitness scores, eval diffs, and failed history

Step 2 — Pre-filter on first training pair
         Candidates that crash or score zero on pair 0
         are skipped (saves full multi-pair eval cost)

Step 3 — Full sandbox evaluation of surviving candidates
         Each candidate is run against all training pairs
         fitness = calculate_continuous_fitness(pred, target)

Step 4 — Select highest-fitness candidate (greedy)
         best = max(candidates, key=fitness)

Step 5 — Update particle state
         particle.code = best_candidate
         if fitness > pbest: update pbest
         if fitness > gbest: update gbest (after all particles finish)
```

---

## Continuous Fitness Function

Standard ARC evaluation is binary (pass/fail). PSO needs a gradient to know if a particle is moving in the right direction. `calculate_continuous_fitness` decomposes correctness into four orthogonal signals:

```
              ┌── 15% ──┐  ┌── 20% ──┐  ┌── 30% ──┐  ┌──── 35% ────┐
fitness =      dim_score   color_score   pixel_score   color_iou_score

dim_score:
  if pred.shape == target.shape → 1.0
  else → (min_rows/max_rows) × (min_cols/max_cols)
  (penalises wrong size proportionally, not binary)

color_score (Jaccard index):
  intersection = |colors(pred) ∩ colors(target)|
  union        = |colors(pred) ∪ colors(target)|
  score        = intersection / union
  (rewards using the right colour palette even if positions are wrong)

pixel_score:
  if same shape → correct_cells / total_cells
  else          → overlap_matches / total_target_cells
  (penalises wrong-size outputs via denominator)

color_iou_score (per-color positional IoU):
  For each unique color in the target grid, compute the IoU (intersection
  over union) of that color's position mask between predicted and target grids.
  Score is the mean IoU across all target colors.
  Only computed when shapes match; 0.0 otherwise.
```

Example progression during PSO:

```
Iteration 0  fitness=0.12  (random code, wrong shape, wrong colors)
Iteration 1  fitness=0.31  (right colors starting to appear)
Iteration 2  fitness=0.54  (right shape now, pixels ~50% correct)
Iteration 3  fitness=0.83  (logic mostly right, edge cases wrong)
Iteration 4  fitness=1.00  (solved ✓)
```

---

## Workflow Diagrams

### Full PSO Solve Workflow

```
                         ARC Task (JSON)
                              │
                    ┌─────────▼──────────┐
                    │  Format task for   │
                    │  LLM context       │
                    └─────────┬──────────┘
                              │
              ┌───────────────▼──────────────────┐
              │         PHASE 1: INIT              │
              │                                   │
              │  Hypothesizer → N hypotheses       │
              │                                   │
              │  for each particle i in [1..N]:   │
              │    code_i  ← LLM.generate(        │
              │               role_i prompt,      │
              │               hypothesis_i)       │
              │    fit_i   ← evaluate(code_i)     │
              │    pbest_i ← (code_i, fit_i)      │
              │                                   │
              │  gbest ← argmax_i(fit_i)           │
              └───────────────┬──────────────────┘
                              │
                    fit==1.0? ├──YES──► DONE (return gbest)
                              │
                         NO   ▼
              ┌───────────────────────────────────┐
              │  PHASE 2: PSO ITERATION LOOP       │
              │  for iteration in [1..MAX_ITERS]:  │
              │                                   │
              │    snapshot gbest for all particles│
              │                                   │
              │    for each particle i (parallel): │
              │      ┌─────────────────────────┐  │
              │      │  ADAPTIVE K             │  │
              │      │  k_eff = k/2 if         │  │
              │      │    pbest_fit≥0.7 else k │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  LLM MUTATION (k_eff)   │  │
              │      │  candidates ← PSOCoder( │  │
              │      │    pbest_code,           │  │
              │      │    gbest_code,           │  │
              │      │    eval_diffs,           │  │
              │      │    failed_history)       │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  EVALUATE & SELECT      │  │
              │      │  pre-filter on pair 0   │  │
              │      │  full eval survivors    │  │
              │      │  best = max(fitness)    │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  UPDATE PARTICLE        │  │
              │      │  if fit > pbest: update │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │    update gbest (main thread)     │
              │    stagnation → crossover/reinit  │
              │    per-particle stagnation check  │
              │    staleness early-exit check     │
              │                                   │
              │         fit==1.0? ├─YES─► DONE    │
              └───────────────────┘               │
                              │                   │
                    (budget   ▼   exhausted)       │
              ┌───────────────────────────────────┐
              │  PHASE 3: REFINEMENT (≥0.60)       │
              │  Up to 8 targeted fix attempts     │
              └───────────────┬──────────────────┘
                              │
              ┌───────────────▼──────────────────┐
              │  PHASE 4: PARTICLE ENSEMBLE       │
              │  Try all unique particle codes    │
              │  on test input (best-first)       │
              └───────────────┬──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Final prediction  │
                    └────────────────────┘
```

### Multi-Agent Workflow

```
 Task ──► Hypothesizer ──► [hyp_1, hyp_2, hyp_3]
                                     │
                          ┌──────────▼──────────┐
                          │   Coder(hyp_1)       │
                          │   tries to implement │
                          └──────────┬──────────┘
                                     │
                              ┌──────▼──────┐
                              │  Sandbox    │
                              │  evaluate   │
                              └──────┬──────┘
                                     │
                          all correct?├──YES──► DONE
                                     │
                                  NO ▼
                          ┌──────────▼──────────┐
                          │      Critic          │
                          │  reads diff + errors │
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │ ROUTE=coder                          │ ROUTE=hypothesizer
                  ▼                                      ▼
         Coder retries with                    advance to hyp_2
         specific feedback                     (or re-run Hypothesizer
                                                with critic feedback)
```

### Ensemble Workflow

```
  ┌─ Run 1: Orchestrator ──► perfect code A    (weight=1.0) ──┐
  ├─ Run 2: Orchestrator ──► near-miss code B  (weight=0.5) ───┤
  ├─ Run 3: Orchestrator ──► perfect code A    (duplicate—skip) │
  └─ Run 4: Orchestrator ──► perfect code C    (weight=1.0) ──┘
                                                                │
              execute A, B, C on test_input                     │
              A→grid_X  B→grid_Y  C→grid_X                      │
                                                                ▼
              pixel-weighted vote per cell:
                cell (0,0): X=2.0, Y=0.5  ──► colour from grid_X
                cell (1,2): X=1.0, Y=1.5  ──► colour from grid_Y

              prediction = assembled grid
                                                                │
              Critic: "Consistent with training rule?"          ▼
                ACCEPT ──► final answer
                REJECT ──► exclude matching outputs, re-vote (max 2×)
```

---

## Pseudocode

### PSO Main Loop

```python
# ── PHASE 1: INITIALISATION ─────────────────────────────────────────────────
hypotheses = Hypothesizer.generate(task, n=max(3, N_PARTICLES))

for i in range(N_PARTICLES):
    particle[i].code     = LLM.generate(role_prompt[i], task, hypothesis=hypotheses[i])
    particle[i].fitness  = continuous_fitness(sandbox(particle[i].code))
    particle[i].pbest    = copy(particle[i])

gbest = particle with highest fitness

# ── PHASE 2: ITERATION LOOP ─────────────────────────────────────────────────
for iteration in range(MAX_ITERATIONS):
    progress = iteration / (MAX_ITERATIONS - 1)
    w_eff = W * (1.0 - 0.6 * progress)      # adaptive inertia: 0.50 → 0.20

    gbest_snapshot = gbest                    # all particles see same gbest

    for i in range(N_PARTICLES):  # (parallel via ThreadPoolExecutor)

        # Adaptive K
        if pbest[i].fitness >= 0.7:
            k_eff = max(2, K // 2)            # exploit: fewer candidates
        else:
            k_eff = K                          # explore: full budget

        # LLM generates k_eff mutations
        candidates = PSOCoder.generate_mutations(
            k              = k_eff,
            pbest_code     = pbest[i].code,
            gbest_code     = gbest_snapshot.code,
            eval_diffs     = format_eval_diff(particle[i].last_eval),
            failed_history = particle[i].failed_history[-5:],
        )

        # Pre-filter + evaluate
        for c in candidates:
            if prefilter_fitness(c, pair_0) == 0: skip
            fitness[c] = full_sandbox_eval(c)

        best = max(candidates, key=fitness)    # greedy selection

        # Update particle
        particle[i].code    = best
        particle[i].fitness = fitness[best]
        if fitness[best] > pbest[i].fitness:
            pbest[i] = (best, fitness[best])
            particle[i].stagnation_iters = 0
        else:
            particle[i].stagnation_iters += 1

    # Update global best (main thread)
    gbest = max(all particles, key=pbest.fitness)

    if gbest.fitness == 1.0:
        break   # solved all training pairs

    # Stagnation → crossover → reinit worst particle
    # Per-particle stagnation → reinit stuck particles (3+ flat iters)
    # Staleness early-exit (3 iters with <0.01 gbest improvement)

# ── PHASE 3: REFINEMENT (fitness ≥ 0.60) ────────────────────────────────────
if 0.60 <= gbest.fitness < 1.0:
    gbest = refinement_phase(gbest, max_attempts=8)

# ── PHASE 4: PARTICLE ENSEMBLE ──────────────────────────────────────────────
if gbest fails on test_input:
    for alt_code in unique_particle_codes(sorted by fitness):
        if sandbox(alt_code, test_input) == test_output:
            return alt_code

return sandbox.execute(gbest.code, test_input)
```

### Continuous Fitness Calculation

```python
def continuous_fitness(pred, target):
    if pred is None:
        return 0.0

    # Dimension score (15%)
    if pred.shape == target.shape:
        dim_score = 1.0
    else:
        dim_score = (min(pred.rows, target.rows) / max(pred.rows, target.rows)
                   * min(pred.cols, target.cols) / max(pred.cols, target.cols))

    # Colour palette score (20%) — Jaccard index
    pred_colors   = set(unique(pred))
    target_colors = set(unique(target))
    color_score   = |pred_colors ∩ target_colors| / |pred_colors ∪ target_colors|

    # Pixel accuracy (30%)
    if pred.shape == target.shape:
        pixel_score = count(pred == target) / target.size
    else:
        overlap     = pred[:min_rows, :min_cols]
        pixel_score = count(overlap == target[:min_rows, :min_cols]) / target.size

    # Per-color positional IoU (35%) — same-shape only
    if pred.shape == target.shape:
        ious = []
        for color in unique(target):
            inter = count(pred==color AND target==color)
            union = count(pred==color OR  target==color)
            ious.append(inter / union if union > 0 else 1.0)
        color_iou_score = mean(ious)
    else:
        color_iou_score = 0.0

    return 0.15 * dim_score + 0.20 * color_score + 0.30 * pixel_score + 0.35 * color_iou_score
```

### Critic Routing (Multi-Agent)

```python
def critic_analyze(hypothesis, code, error_info, diff):
    response = LLM.generate(CRITIC_SYSTEM_PROMPT,
                            hypothesis, code, error_info, diff)

    if "ROUTE: hypothesizer" in response:
        return Route.HYPOTHESIZER, extract_feedback(response)
    else:
        return Route.CODER, extract_feedback(response)

# In the orchestration loop:
route, feedback = critic_analyze(...)

if route == Route.HYPOTHESIZER:
    hyp_index += 1          # try next hypothesis
    hyp_feedback = feedback  # pass feedback to next Hypothesizer call
else:
    coder_feedback = feedback  # Coder retries same hypothesis
    temperature   += 0.3       # raise temperature to encourage variation
```

### PSOCoder Mutation Prompt (Pseudocode)

```
SYSTEM:
  You are particle {role_name} in a PSO loop.
  Generate {K} distinct transform() functions blending pbest and gbest.

USER:
  [task description + training examples]

  Personal best (fitness={pbest_fitness:.3f}):
    def transform(input_grid):
        [pbest_code]

  Global best (fitness={gbest_fitness:.3f}):
    def transform(input_grid):
        [gbest_code]

  [per-pair fitness breakdown]
  [eval diff of current failures]
  [failed history — do NOT repeat these]

  Generate {K} variations. Each in its own ```python``` block.
  Named transform_1 ... transform_{K}.

→ LLM returns K code blocks
→ normalise all function names to "transform"
→ sandbox eval → select highest fitness (greedy)
```

---

## Running the Solver

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
# Note: scipy is listed as an optional dependency but not currently used.

# Start Ollama — tiered model selection based on available VRAM
TIER=small  ./start_ollama.sh      # ~8 GB  — deepseek-r1:8b  + qwen2.5-coder:7b
TIER=medium ./start_ollama.sh      # ~16 GB — deepseek-r1:14b + qwen2.5-coder:14b
TIER=large  ./start_ollama.sh      # ~32 GB — deepseek-r1:32b + qwen2.5-coder:32b
TIER=ultra  ./start_ollama.sh      # deepseek-r1:32b (~20 GB at Q4) + qwen2.5-coder:7b (~5 GB) (default)
                                   #   optimal for ≥64 GB machines — ~39 GB KV cache headroom with parallel=1
# Also pulls: nomic-embed-text (embeddings, 274 MB)
# Prints ready-to-paste CLI invocations for each strategy after startup

# Performance flags (set automatically by start_ollama.sh):
#   OLLAMA_FLASH_ATTENTION=1     — fused attention kernel for Apple Silicon
#   OLLAMA_KV_CACHE_TYPE=q8_0   — halves KV memory vs f16 with negligible quality loss
#                                  frees ~6 GB on 64 GB machines for an extra parallel slot
#   OLLAMA_NUM_PARALLEL=1       — MUST be 1 for deepseek-r1:32b on 64 GB (see start_ollama.sh)
#   OLLAMA_KEEP_ALIVE=-1        — models stay loaded; avoids 30-60s reload penalty
#   OLLAMA_MAX_LOADED_MODELS=2  — keep reasoner + coder both loaded in VRAM
#                                  avoids 30-60s model swap penalty
```

### PSO Solver

> `run_pso.py` exposes PSO-specific flags (hyperparameters, swarm size) directly.
> `run_multi_agent.py --strategy pso` wraps the same solver under the unified CLI
> with `--pso-*` prefixed flags. Either entry point is equivalent.

```bash
# Solve a single task with defaults (6 particles, 10 iterations)
python run_pso.py --task data/training/007bbfb7.json

# Verbose mode shows per-particle fitness at each iteration
python run_pso.py --task data/training/007bbfb7.json --debug

# Custom swarm hyperparameters
python run_pso.py --task data/training/007bbfb7.json \
    --n-particles 6      \   # number of particles (max 6)
    --max-iterations 15  \   # PSO iteration budget
    --k-candidates 10    \   # LLM mutations per particle per iteration
    --w 0.5              \   # inertia weight (initial; annealed to w×0.4)
    --c1 1.5             \   # cognitive coefficient
    --c2 1.5             \   # social coefficient
    --temperature 0.7    \   # LLM sampling temperature
    --debug

# Batch evaluation — solve up to 50 tasks, save results
python run_pso.py --task-dir data/training/ \
    --max-tasks 50 \
    --output results/pso_run.json

# Batch evaluation with per-task timeout
python run_pso.py --task-dir data/training/ \
    --max-tasks 50 \
    --task-timeout 300 \
    --output results/pso_run.json

# Use Anthropic backend for generation (embeddings still via Ollama)
python run_pso.py --task data/training/007bbfb7.json \
    --backend anthropic \
    --model claude-sonnet-4-6
```

**CLI Parameters:**

| Flag | Default | Description |
|---|---|---|
| `--n-particles` | 6 | Number of swarm particles (max 6) |
| `--max-iterations` | 10 | PSO iteration budget |
| `--k-candidates` | 10 | LLM mutation candidates per particle per iteration |
| `--w` | 0.5 | PSO inertia weight (initial; annealed to w×0.4) |
| `--c1` | 1.5 | Cognitive coefficient (pull toward personal best) |
| `--c2` | 1.5 | Social coefficient (pull toward global best) |
| `--fitness-alpha` | 0.4 | Reserved parameter; currently unused in candidate selection |
| `--task-timeout` | None | (batch only) Max seconds per task before skipping |

### Unified CLI (all strategies)

```bash
# PSO (default recommended)
python run_multi_agent.py --task data/training/007bbfb7.json --strategy pso

# Multi-agent (Hypothesizer → Coder → Critic loop)
python run_multi_agent.py --task data/training/007bbfb7.json --strategy multi

# Ensemble (majority voting across multiple runs)
python run_multi_agent.py --task data/training/007bbfb7.json --strategy ensemble \
    --ensemble-runs 5 --ensemble-candidates 3

# Single-agent baseline
python run_multi_agent.py --task data/training/007bbfb7.json --strategy single

# Two-phase: fast 7b MultiAgent warm-up, then seeded PSO if unsolved
python run_multi_agent.py --task data/training/007bbfb7.json --strategy two_phase \
    --coder-model qwen2.5-coder:7b
```

### Output Format

```json
{
  "task": "data/training/007bbfb7.json",
  "success": true,
  "gbest_fitness": 1.0,
  "test_correct": true,
  "elapsed_s": 47.3,
  "iterations": 3,
  "code": "def transform(input_grid):\n    return rotate(input_grid, 1)",
  "prediction": [[0, 1, 2], [3, 4, 5]]
}
```

---

## Running Tests

```bash
# Run all 774 tests
python -m pytest

# With coverage report
python -m pytest --cov=arc --cov=agents --cov-report=term-missing

# Run a specific module
python -m pytest tests/test_pso_orchestrator.py -v

# Run a specific test
python -m pytest tests/test_evaluate.py::TestContinuousFitness::test_perfect_match_is_one -v

# Run only Phase 4 ensemble tests
python -m pytest tests/test_ensemble.py -v \
  -k "PixelMajority or CheckPrediction or CandidateFiltering or SelfCorrection"
```

Tests are fully offline — all LLM calls are mocked via `unittest.mock`. No Ollama server required to run the test suite. Coverage numbers below are approximate; run `pytest --cov` for current figures.

### Test Coverage

| Module | Coverage |
|---|---|
| `arc/evaluate.py` | 98% |
| `arc/grid.py` | 100% |
| `arc/dsl.py` | 99% |
| `arc/dsl_actions.py` | 97% |
| `arc/sandbox.py` | 64% |
| `agents/roles.py` | 97% |
| `agents/ensemble.py` | 99% |
| `agents/single_agent.py` | 95% |
| `agents/orchestrator.py` | 85% |
| `agents/multi_agent.py` | 87% |
| `agents/pso_orchestrator.py` | 82% |
| `agents/mcts_solver.py` | 99% |
| `agents/llm_client.py` | 86% |
| `agents/dsl_reference.py` | 100% |
| **Total** | **94%** |

> `arc/sandbox.py` subprocess worker bodies (`_subprocess_worker`, `_param_search_worker`) are tested by calling them directly in-process. The persistent `ProcessPoolExecutor` is registered with `atexit` so pytest exits cleanly without hanging. Coverage is lower on this module because some error-handling paths (broken pool recovery, spatial diff edge cases) are exercised only under real multiprocessing conditions.

---

## Key Design Decisions

**Why behavioral embeddings?** Position tracking embeds what code *does* (its output on training pairs) rather than the code text, so similar-behaving programs cluster together. Two syntactically different programs that produce the same outputs occupy the same point in embedding space.

**Why fitness-greedy selection?** Direct fitness maximization gives stronger signal than embedding proximity, especially when the embedding space doesn't perfectly correlate with solution quality. The system selects the candidate with the highest sandbox fitness rather than the one closest to a target vector.

**Why adaptive K?** High-fitness particles need fewer candidates (exploitation); low-fitness particles get the full budget (exploration). This saves LLM budget where it matters least.

**Why parallel particle updates?** Ollama HTTP calls release the GIL, so all N particles can query the LLM simultaneously via `ThreadPoolExecutor`. A gbest snapshot is taken at iteration start so particles don't interfere.

**Why nomic-embed-text?** It is a 768-dimensional open-weight model that runs locally via Ollama, producing high-quality code embeddings without API costs or rate limits.

**Why 6 fixed roles?** Diversity in the initial population is critical for PSO to avoid premature convergence. Six specialist roles (geometric, colour, pattern, object, rule, hybrid) cover the main reasoning strategies seen across ARC tasks. More roles would increase LLM cost without proportional benefit.

---

## MCTS Solver & Benchmarks

### MCTS Solver (`agents/mcts_solver.py`)

An **LLM-free** solver that builds DSL pipelines incrementally via Monte Carlo Tree Search. Each tree level adds one DSL operation (rotate, flip, scale, recolor, etc.), composing multi-step transformations.

- **Selection**: UCB1 with configurable exploration constant
- **Expansion**: Progressive widening (C=4, α=0.5) over enumerated DSL actions
- **Simulation**: Random rollouts up to `rollout_depth` steps
- **Backpropagation**: Max-fitness tracking (not average)
- **Evaluation**: Sandbox execution + continuous fitness against all training pairs

```bash
# Run MCTS on a single task
python run_multi_agent.py --task data/training/007bbfb7.json --strategy mcts --debug

# With custom parameters
python run_multi_agent.py --task data/training/007bbfb7.json --strategy mcts \
    --mcts-max-iterations 5000 --mcts-max-time 300 --mcts-max-depth 7
```

### Benchmark Infrastructure (`benchmarks/`)

24 curated tasks across 6 difficulty tiers, verified by exhaustive search over the training set:

| Tier | Description | Tasks | Example |
|------|------------|-------|---------|
| 1 | Single geometric op (rotate, flip) | 5 | `3c9b0459` — rotate 180° |
| 2 | Single shape-change (scale, tile, crop) | 4 | `c59eb873` — scale ×2 |
| 3 | Single cell-level op (gravity, symmetrize, fill) | 5 | `1e0a9b12` — gravity down |
| 4 | Two-operation compositions | 4 | `0d3d703e` — multi-recolor chain |
| 5 | Three+ operations | 3 | `025d127b` — symmetry + fill |
| 6 | Hard / beyond pure DSL | 3 | `045e512c` — sorting + gravity |

```bash
# Smoke test (3 tasks)
python -m benchmarks.run_benchmark --smoke --debug

# Full tier-1 benchmark
python -m benchmarks.run_benchmark --tiers 1 --debug

# All tiers, save results
python -m benchmarks.run_benchmark --tiers 1 2 3 4 5 6 --output results.json --debug

# Analyze results
python -m benchmarks.analyze_results results.json

# Compare two runs
python -m benchmarks.analyze_results new.json --compare baseline.json
```

### Benchmark Results (default params: 2000 iters, depth 5)

| Tier | Solved | Avg Fitness | Avg Time |
|------|--------|-------------|----------|
| 1 (geometric) | 1/5 (20%) | 0.868 | 2.6s |
| 2 (shape-change) | 2/4 (50%) | 0.870 | 2.0s |
| 3 (cell-level) | 0/5 (0%) | 0.888 | 3.4s |
| 4 (two-op) | 0/4 (0%) | 0.637 | 3.3s |
| 5 (three-op) | 0/3 (0%) | 0.835 | 2.9s |
| 6 (hard) | 0/3 (0%) | 0.745 | 3.6s |
| **Total** | **3/24 (12.5%)** | **0.814** | **2.9s** |

See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for detailed analysis including near-misses, bottleneck analysis, and improvement ideas.
