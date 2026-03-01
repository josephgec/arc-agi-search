# ARC-AGI PSO Swarm Solver

A hybrid **Particle Swarm Optimization + LLM** architecture for the [ARC-AGI challenge](https://github.com/fchollet/ARC-AGI). Standard LLM agent loops get stuck in repetitive generation cycles (local minima). This system escapes that trap by coupling a *continuous* mathematical optimizer (PSO) with a *discrete* LLM code generator via a **Generate-and-Project bridge**.

---

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Repository Structure](#repository-structure)
3. [Architecture Overview](#architecture-overview)
4. [Module Deep-Dives](#module-deep-dives)
   - [arc/ — Core Library](#arc--core-library)
   - [agents/ — Agent Layer](#agents--agent-layer)
5. [The PSO Algorithm in Detail](#the-pso-algorithm-in-detail)
6. [The Generate-and-Project Bridge](#the-generate-and-project-bridge)
7. [Continuous Fitness Function](#continuous-fitness-function)
8. [Workflow Diagrams](#workflow-diagrams)
9. [Pseudocode](#pseudocode)
10. [Running the Solver](#running-the-solver)
11. [Running Tests](#running-tests)

---

## The Core Idea

The **Inverse Mapping Problem**: it is easy to embed Python code into a continuous vector space, but mathematically impossible to decode an arbitrary floating-point target vector back into valid code.

**Solution — Generate-and-Project:**

```
PSO target vector  ──────────────────────────────────────────────────────┐
                                                                          │
LLM generates K candidate code strings                                    │
      │                                                                   │
      ▼                                                                   ▼
embed each candidate ──► pick the candidate whose embedding is closest to target
```

This decouples **search direction** (done mathematically by PSO) from **solution generation** (done semantically by the LLM). The swarm systematically explores the latent space of programs rather than randomly sampling.

---

## Repository Structure

```
arc-agi-search/
│
├── arc/                        ← Core library (backend-agnostic)
│   ├── grid.py                 ← Grid type, I/O, colour utilities
│   ├── dsl.py                  ← Pure transformation primitives
│   ├── sandbox.py              ← Hardened subprocess code execution
│   └── evaluate.py             ← Binary eval + continuous fitness
│
├── agents/                     ← Agent layer
│   ├── llm_client.py           ← Unified LLM + embedding API
│   ├── roles.py                ← Hypothesizer, Coder, Critic, Decomposer, Verifier, PSOCoder
│   ├── dsl_reference.py        ← DSL primitives reference injected into LLM context
│   ├── multi_agent.py          ← Hypothesizer → Coder → Critic loop (with Decomposer + Verifier)
│   ├── orchestrator.py         ← Candidate-pooling wrapper (exposes fitness/perfect fields)
│   ├── ensemble.py             ← Pixel-weighted majority-vote ensemble with self-correction
│   ├── single_agent.py         ← Single-agent baseline
│   └── pso_orchestrator.py     ← ★ PSO swarm solver (main contribution)
│
├── tests/                      ← 496 tests, 94% coverage
│
├── data/
│   ├── training/               ← 400 ARC training tasks (JSON)
│   └── evaluation/             ← 400 ARC evaluation tasks (JSON)
│
├── run_pso.py                  ← CLI entry point for PSO solver
├── run_multi_agent.py          ← Unified CLI (--strategy flag)
├── start_ollama.sh             ← Launch Ollama with tiered model selection
└── requirements.txt
```

---

## Architecture Overview

The system has two layers that communicate through a single embedding bridge:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONTINUOUS SPACE                            │
│                                                                     │
│   x₁ ●──────────────────────────────────────────────────────►      │
│        velocity v₁                                   gbest ★        │
│   x₂ ●──────────────────────────────────────────────────────►      │
│   x₃ ●──────────────────────────────────────────────────────►      │
│   x₄ ●──────────────────────────────────────────────────────►      │
│                   PSO update equations                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  embed() / cosine distance
┌──────────────────────────▼──────────────────────────────────────────┐
│                         DISCRETE SPACE                              │
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
| `background_color(grid)` | Always 0 (black) if present; else most-frequent |

---

#### `arc/dsl.py` — Transformation Primitives

All functions are **pure** — they return a new array and never mutate the input.

```
Geometric:   crop · rotate · flip · translate · scale · tile
Colour:      recolor · mask · overlay
Fill:        flood_fill
Detection:   find_objects · bounding_box · crop_to_content
```

These are the only allowed operations inside generated `transform()` functions. They are injected into the sandbox execution namespace automatically.

---

#### `arc/sandbox.py` — Hardened Execution

Generated code is untrusted and could contain infinite loops, import statements, or crashes. Every execution runs in a **separate child process** with a hard wall-clock timeout.

```
┌─ Parent process ────────────────────────────────────────────┐
│                                                             │
│  execute(code, input_grid, timeout=10s)                     │
│       │                                                     │
│       ├── spawn child process                               │
│       ├── inject DSL_NAMESPACE (np, crop, rotate, ...)      │
│       ├── exec(code) in child                               │
│       ├── call transform(input_grid)                        │
│       ├── put result on mp.Queue                            │
│       └── join(timeout=10s) ──► kill if alive               │
│                                                             │
│  evaluate_code(code, task) ──► {pairs, n_correct, ...}      │
└─────────────────────────────────────────────────────────────┘
```

Security guards:
- `input()` / `sys.stdin` usage → rejected before spawning
- Hard timeout → process killed if exceeded
- No network, no file-write access (subprocess isolation)

**Parameterized CPU Search** (`param_search`): if generated code defines a `PARAM_GRID = dict(...)` mapping parameter names to lists of values, `param_search` sweeps all combinations (up to 1,000) in a single sandboxed subprocess, returning the best `(params, fitness)` pair — offloading constant-guessing from the LLM to the CPU:

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

**Continuous** (`calculate_continuous_fitness`): used by PSO to get a gradient signal beyond pass/fail.

```
fitness = 0.20 × dim_score + 0.30 × color_score + 0.50 × pixel_score

dim_score   = min(H_pred,H_target)/max(...) × min(W_pred,W_target)/max(...)
color_score = |colors_pred ∩ colors_target| / |colors_pred ∪ colors_target|
pixel_score = correct_cells / total_target_cells
```

| Case | Fitness |
|---|---|
| Prediction is `None` (crash) | `0.0` |
| Perfect pixel-perfect match | `1.0` |
| Right colours, wrong positions | `0.2–0.5` |
| Wrong shape | `< 0.5` |

**AST Complexity Penalty** (`calculate_complexity_penalty`): deducted from fitness to penalise memorisation code. Penalises `If` nodes (+0.005 each), large literals with >5 elements (+0.02 each), and `Compare` nodes (+0.002 each), capped at 0.15. A `SyntaxError` returns a flat 0.10 penalty.

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
Hypothesizer   generates 3 competing natural-language hypotheses
               about the transformation rule

Coder          translates one hypothesis into a Python transform()
               function using only the DSL primitives

Critic         reads the error diff and decides:
               ROUTE: hypothesizer  ← hypothesis is fundamentally wrong
               ROUTE: coder         ← implementation bug, same hypothesis

Decomposer     fires on stagnation (≥2 consecutive non-improving cycles);
               decomposes the task into sub-goals to break the agent out
               of a local-minimum hypothesis

Verifier       gates success — re-reads the code and training pairs and
               confirms all_correct before the loop exits; fail-safe
               (returns passes=True) on any malformed or missing response

PSOCoder       generates K distinct Python functions that blend the
               logic of pbest and gbest to move toward the PSO target
```

---

#### `agents/multi_agent.py` — Multi-Agent Orchestrator

A **state machine** that runs up to `max_cycles` total LLM calls. Phases 3–4 added Decomposer, Verifier, spatial diffs, and near-miss candidate collection:

```
┌────────────────┐
│  Hypothesizer  │◄──── feedback (if critic says: try new hypothesis)
│  generates 3   │
│  hypotheses    │
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
│     Critic     │     (uses spatial diff: direction, region, color names)
│  diagnoses     │     ROUTE=coder ──────────► Coder retries with feedback
└───────┬────────┘
        │ stagnation (≥2 non-improving cycles)
        ▼
┌────────────────┐
│  Decomposer    │──── breaks task into sub-goals ──► Hypothesizer restart
└────────────────┘
```

`compute_spatial_diff()` produces natural-language feedback (e.g. *"object shifted 2 rows down"*, *"bottom-right region: 3 wrong cells (expected blue, got red)"*) that the Critic uses to generate targeted fix instructions.

---

#### `agents/pso_orchestrator.py` — PSO Swarm Solver

The core contribution. See [The PSO Algorithm in Detail](#the-pso-algorithm-in-detail) below.

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
    pos:           np.ndarray   # embedding of current code  (768-dim, unit norm)
    velocity:      np.ndarray   # direction of travel in embedding space
    fitness:       float        # continuous fitness in [0, 1]
    pbest_code:    str          # best code this particle ever found
    pbest_pos:     np.ndarray   # embedding of pbest
    pbest_fitness: float        # fitness of pbest
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

### Update Equations

```
# Inertia keeps the particle moving in its current direction
# Cognitive term pulls toward personal best (memory)
# Social term pulls toward global best (collaboration)

v_i ← w · v_i
    + c1 · r1 · (pbest_i − x_i)
    + c2 · r2 · (gbest   − x_i)

target_i ← (x_i + v_i) / ‖x_i + v_i‖₂    # project back to unit sphere
```

Default hyperparameters: `w=0.5`, `c1=1.5`, `c2=1.5`

---

## The Generate-and-Project Bridge

This is the key innovation that solves the Inverse Mapping Problem:

```
Step 1 — PSO computes target_pos in ℝ⁷⁶⁸
         (unit sphere, L2-normalised)

Step 2 — LLM generates K=5 candidate functions
         Each is a blend of pbest_code and gbest_code
         guided by a prompt that shares both solutions
         and their fitness scores

Step 3 — Embed all K candidates
         embed(candidate_k) → ê_k ∈ ℝ⁷⁶⁸

Step 4 — Select the candidate closest to target_pos
         best = argmin_k  cosine_distance(ê_k, target_pos)

Step 5 — Update particle
         x_i     ← ê_best   (actual new position)
         v_i     ← x_i_new − x_i_old  (implied velocity)
         fitness ← sandbox.evaluate(best_candidate)
```

**Why cosine distance?** All embeddings are L2-normalised onto the unit sphere, so cosine distance and Euclidean distance are equivalent. The unit sphere constraint also prevents velocity vectors from diverging to infinity.

---

## Continuous Fitness Function

Standard ARC evaluation is binary (pass/fail). PSO needs a gradient to know if a particle is moving in the right direction. `calculate_continuous_fitness` decomposes correctness into three orthogonal signals:

```
                    ┌── 20% ──┐  ┌───── 30% ──────┐  ┌──── 50% ────┐
fitness =  weight × dim_score + weight × color_score + weight × pixel_score

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
              │           INITIALISATION          │
              │                                   │
              │  for each particle i in [1..N]:   │
              │    code_i  ← LLM.generate(        │
              │               role_i prompt,      │
              │               task_description)   │
              │    pos_i   ← embed(code_i)         │
              │    vel_i   ← zeros(768)            │
              │    fit_i   ← evaluate(code_i)     │
              │    pbest_i ← (code_i, pos_i, fit_i)│
              │                                   │
              │  gbest ← argmax_i(fit_i)           │
              └───────────────┬──────────────────┘
                              │
                    fit==1.0? ├──YES──► DONE (return gbest)
                              │
                         NO   ▼
              ┌───────────────────────────────────┐
              │  for iteration in [1..MAX_ITERS]:  │
              │                                   │
              │    for each particle i:           │
              │      ┌─────────────────────────┐  │
              │      │  PSO VELOCITY UPDATE    │  │
              │      │  r1,r2 ~ Uniform(0,1)   │  │
              │      │  v ← w·v                │  │
              │      │    + c1·r1·(pbest−x)    │  │
              │      │    + c2·r2·(gbest−x)    │  │
              │      │  target ← norm(x + v)   │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  LLM MUTATION (K=5)     │  │
              │      │  candidates ← LLM(      │  │
              │      │    pbest_code,           │  │
              │      │    gbest_code,           │  │
              │      │    current_fitness)      │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  GENERATE-AND-PROJECT   │  │
              │      │  for c in candidates:   │  │
              │      │    ê_c ← embed(c)       │  │
              │      │  best ← argmin cosine(  │  │
              │      │           ê_c, target)  │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │      ┌────────────▼────────────┐  │
              │      │  EVALUATE & UPDATE      │  │
              │      │  x_i   ← embed(best)    │  │
              │      │  fit_i ← sandbox(best)  │  │
              │      │  if fit > pbest: update │  │
              │      │  if fit > gbest: update │  │
              │      └────────────┬────────────┘  │
              │                   │               │
              │         fit==1.0? ├─YES─► DONE    │
              │                   │               │
              └───────────────────┘               │
                              │                   │
                    (budget   ▼   exhausted)       │
                    return gbest_code              │
                              │                   │
                    ┌─────────▼──────────┐         │
                    │  sandbox.execute(  │         │
                    │   gbest_code,      │         │
                    │   test_input)      │         │
                    └─────────┬──────────┘         │
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
# ── INITIALISATION ──────────────────────────────────────────────────────────
for i in range(N_PARTICLES):
    particle[i].code     = LLM.generate(role_prompt[i], task)
    particle[i].pos      = normalize(embed(particle[i].code))
    particle[i].velocity = zeros(768)
    particle[i].fitness  = continuous_fitness(sandbox(particle[i].code))
    particle[i].pbest    = copy(particle[i])

gbest = particle with highest fitness

# ── ITERATION LOOP ───────────────────────────────────────────────────────────
for iteration in range(MAX_ITERATIONS):
    for i in range(N_PARTICLES):

        # Step 1 — PSO velocity update (continuous embedding space)
        r1, r2 = random(0,1), random(0,1)
        velocity[i] = (W  * velocity[i]
                    + C1 * r1 * (pbest[i].pos - pos[i])
                    + C2 * r2 * (gbest.pos    - pos[i]))
        target_pos = normalize(pos[i] + velocity[i])

        # Step 2 — LLM generates K mutations in discrete code space
        candidates = LLM.generate_k_mutations(
            k              = K_CANDIDATES,
            pbest_code     = pbest[i].code,
            gbest_code     = gbest.code,
            current_fitness = fitness[i],
        )

        # Step 3 — Generate-and-Project: find closest candidate to target_pos
        embeddings    = [normalize(embed(c)) for c in candidates]
        distances     = [cosine(emb, target_pos) for emb in embeddings]
        best_idx      = argmin(distances)
        selected_code = candidates[best_idx]
        selected_emb  = embeddings[best_idx]

        # Step 4 — Update particle state
        velocity[i] = selected_emb - pos[i]   # implied displacement
        pos[i]      = selected_emb
        fitness[i]  = continuous_fitness(sandbox(selected_code))

        # Step 5 — Update personal best
        if fitness[i] > pbest[i].fitness:
            pbest[i] = (selected_code, selected_emb, fitness[i])

        # Step 6 — Update global best
        if fitness[i] > gbest.fitness:
            gbest = (selected_code, selected_emb, fitness[i])

    if gbest.fitness == 1.0:
        break   # solved all training pairs

# ── FINAL PREDICTION ─────────────────────────────────────────────────────────
return sandbox.execute(gbest.code, test_input)
```

### Continuous Fitness Calculation

```python
def continuous_fitness(pred, target):
    if pred is None:
        return 0.0

    # Dimension score (20%)
    if pred.shape == target.shape:
        dim_score = 1.0
    else:
        dim_score = (min(pred.rows, target.rows) / max(pred.rows, target.rows)
                   * min(pred.cols, target.cols) / max(pred.cols, target.cols))

    # Colour palette score (30%) — Jaccard index
    pred_colors   = set(unique(pred))
    target_colors = set(unique(target))
    color_score   = |pred_colors ∩ target_colors| / |pred_colors ∪ target_colors|

    # Pixel accuracy (50%)
    if pred.shape == target.shape:
        pixel_score = count(pred == target) / target.size
    else:
        overlap     = pred[:min_rows, :min_cols]
        pixel_score = count(overlap == target[:min_rows, :min_cols]) / target.size

    return 0.20 * dim_score + 0.30 * color_score + 0.50 * pixel_score
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

  Generate {K} variations. Each in its own ```python``` block.
  Named transform_1 ... transform_{K}.
  Blend the logic of both — do not simply copy one.

→ LLM returns K code blocks
→ normalise all function names to "transform"
→ embed each → pick closest to PSO target_pos
```

---

## Running the Solver

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama — tiered model selection based on available VRAM
TIER=small  ./start_ollama.sh      # ~8 GB  — deepseek-r1:8b  + qwen2.5-coder:7b
TIER=medium ./start_ollama.sh      # ~16 GB — deepseek-r1:14b + qwen2.5-coder:14b (default)
TIER=large  ./start_ollama.sh      # ~32 GB — deepseek-r1:32b + qwen2.5-coder:32b
TIER=ultra  ./start_ollama.sh      # ~25 GB — deepseek-r1:32b (reasoner) + qwen2.5-coder:7b (fast coder)
                                   #          optimal for 64 GB machines — 39 GB KV cache headroom
# Also pulls: nomic-embed-text (embeddings, 274 MB)
# Prints ready-to-paste CLI invocations for each strategy after startup

# Performance flags (set automatically by start_ollama.sh):
#   OLLAMA_FLASH_ATTENTION=1     — fused attention kernel for Apple Silicon
#   OLLAMA_KV_CACHE_TYPE=f16    — optimal KV cache precision for M-series bandwidth
#   OLLAMA_NUM_PARALLEL=4       — concurrent request slots (set to batch_generate workers)
```

### PSO Solver

```bash
# Solve a single task with defaults (6 particles, 10 iterations)
python run_pso.py --task data/training/007bbfb7.json

# Verbose mode shows per-particle fitness at each iteration
python run_pso.py --task data/training/007bbfb7.json --debug

# Custom swarm hyperparameters
python run_pso.py --task data/training/007bbfb7.json \
    --n-particles 6      \   # number of particles (max 6)
    --max-iterations 15  \   # PSO iteration budget
    --k-candidates 5     \   # LLM mutations per particle per iteration
    --w 0.5              \   # inertia weight
    --c1 1.5             \   # cognitive coefficient
    --c2 1.5             \   # social coefficient
    --temperature 0.7    \   # LLM sampling temperature
    --debug

# Batch evaluation — solve up to 50 tasks, save results
python run_pso.py --task-dir data/training/ \
    --max-tasks 50 \
    --output results/pso_run.json

# Use Anthropic backend for generation (embeddings still via Ollama)
python run_pso.py --task data/training/007bbfb7.json \
    --backend anthropic \
    --model claude-sonnet-4-6
```

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
# Run all 496 tests
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

Tests are fully offline — all LLM calls are mocked via `unittest.mock`. No Ollama server required to run the test suite.

### Test Coverage

| Module | Coverage |
|---|---|
| `arc/evaluate.py` | 100% |
| `arc/grid.py` | 100% |
| `arc/dsl.py` | 99% |
| `agents/roles.py` | 97% |
| `agents/ensemble.py` | 99% |
| `agents/single_agent.py` | 95% |
| `agents/orchestrator.py` | 100% |
| `agents/multi_agent.py` | 97% |
| `agents/pso_orchestrator.py` | 82% |
| `agents/llm_client.py` | 91% |
| `arc/sandbox.py` | 96% |
| **Total** | **94%** |

> `arc/sandbox.py` subprocess worker bodies are tested by calling them directly in-process with a threading `Queue`, giving full line coverage without requiring a real child process.

---

## Key Design Decisions

**Why nomic-embed-text?** It is a 768-dimensional open-weight model that runs locally via Ollama, producing high-quality code embeddings without API costs or rate limits.

**Why L2-normalise onto the unit sphere?** Cosine distance and Euclidean distance are equivalent on the unit sphere. Normalisation also prevents PSO velocities from diverging to infinity across iterations.

**Why re-compute velocity as displacement?** The standard PSO velocity update can accumulate floating-point drift when embeddings are regenerated each step. Computing `v_new = x_new − x_old` (the actual displacement) resets the velocity to a geometrically meaningful value each iteration.

**Why 6 fixed roles?** Diversity in the initial population is critical for PSO to avoid premature convergence. Six specialist roles (geometric, colour, pattern, object, rule, hybrid) cover the main reasoning strategies seen across ARC tasks. More roles would increase LLM cost without proportional benefit.
