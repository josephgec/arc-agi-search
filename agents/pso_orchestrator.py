"""Particle Swarm Optimization Orchestrator for ARC-AGI.

Architecture overview
---------------------
Standard LLM agent loops often get stuck in repetitive generation cycles
(local minima).  This module escapes that trap by coupling a *continuous*
mathematical optimizer (PSO) with a *discrete* LLM code generator via a
"Generate-and-Project" bridge:

  1. PSO computes a *target vector* in continuous embedding space.
  2. The LLM generates K candidate code strings by intelligently blending
     the particle's personal-best and the swarm's global-best solutions.
  3. We embed all K candidates and select the one whose embedding is
     *closest* (cosine distance) to the PSO target vector.

This decouples search direction (math) from solution generation (LLM), so
the swarm systematically explores the semantic space of programs rather than
just randomly sampling.

Particle roles
--------------
Each particle is seeded with a different "cognitive specialisation" prompt
to maximise diversity in the initial population and reduce the chance that
all particles converge to the same local optimum.

PSO update equations
--------------------
  v_i ← w·v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
  target_i ← x_i + v_i   (then L2-normalised to lie on unit hypersphere)

After selecting the best candidate via the bridge:
  x_i ← embed(best_candidate)   (actual position, not the target)
  v_i ← x_i_new − x_i_old      (implied velocity for next iteration)

Fitness
-------
Uses arc.evaluate.calculate_continuous_fitness for a smooth [0, 1] signal.
A fitness of 1.0 means all training pairs are pixel-perfect.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

from arc import sandbox
from arc.evaluate import calculate_continuous_fitness
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import PSOCoder, Hypothesizer
from agents.multi_agent import (
    _format_task_description,
    _format_training_examples,
    _format_eval_diff,
    _strip_thinking,
    _extract_code,
    _parse_hypotheses,
    _structural_note,
    _grid_to_str,
)

# ---------------------------------------------------------------------------
# Particle role catalogue
# ---------------------------------------------------------------------------

PARTICLE_ROLES: list[tuple[str, str]] = [
    (
        "geometric_specialist",
        "You excel at spatial transforms: rotation, flipping, translation, scaling, "
        "and tiling.  Think in terms of grid geometry and structural symmetry.",
    ),
    (
        "color_specialist",
        "You excel at colour-based operations: recoloring, masking, overlaying, and "
        "colour palette analysis.  Think in terms of colour relationships and mappings.",
    ),
    (
        "pattern_analyst",
        "You excel at finding repeating patterns, symmetries, and tile-level structure. "
        "Think in terms of periodic grids, reflections, and sub-grid repetition.",
    ),
    (
        "object_tracker",
        "You excel at finding discrete objects using find_objects(), analysing their "
        "properties (colour, size, bounding box), and applying object-level transforms.",
    ),
    (
        "rule_abstractor",
        "You excel at identifying the minimal abstract rule governing a transformation "
        "and expressing it as clean, general Python code with few hard-coded constants.",
    ),
    (
        "hybrid_solver",
        "You take a holistic approach, combining geometric, colour, and object-level "
        "reasoning.  You prefer elegant solutions that generalise across all pairs.",
    ),
]

# Role-specific fallback transforms used when LLM generation fails during init.
# Each is semantically different to preserve swarm diversity.
_ROLE_FALLBACK: dict[str, str] = {
    "geometric_specialist": "def transform(input_grid):\n    return rotate(input_grid, 1)",
    "color_specialist":     "def transform(input_grid):\n    return flip(input_grid, axis=1)",
    "pattern_analyst":      "def transform(input_grid):\n    return flip(input_grid, axis=0)",
    "object_tracker":       "def transform(input_grid):\n    return crop_to_content(input_grid)",
    "rule_abstractor":      "def transform(input_grid):\n    return tile(input_grid, 1, 1)",
    "hybrid_solver":        "def transform(input_grid):\n    return rotate(input_grid, 2)",
}
_IDENTITY_FALLBACK = "def transform(input_grid):\n    return input_grid.copy()"

_EMBED_DIM = 768  # nomic-embed-text output dimension


def _random_unit_vec(rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random L2-normalised float32 vector of length _EMBED_DIM."""
    if rng is None:
        v = np.random.randn(_EMBED_DIM).astype(np.float32)
    else:
        v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


_INIT_CODER_SYSTEM = """\
You are an ARC-AGI puzzle solver.  Follow these two steps:

STEP 1 — REASON (write a short analysis before the code block):
- Compare each input/output pair.  What changes and what stays the same?
- Is this a geometric transform (rotate/flip/crop/tile), a colour operation, or object manipulation?
- Does the output size depend on the input content?  If so, how is it computed?
- What is the single abstract rule that generates ALL outputs from ALL inputs?
Write 2-4 sentences summarising your conclusion.

STEP 2 — CODE:
Implement the rule as a Python function `transform(input_grid: np.ndarray) -> np.ndarray`.

Available DSL (already imported — do NOT re-import):
  crop, rotate, flip, translate, scale, tile,
  recolor, mask, overlay, flood_fill,
  find_objects, bounding_box, crop_to_content,
  pad, symmetrize, np

Rules:
- After your analysis, return EXACTLY one ```python … ``` code block.
- The function must be named `transform`.
- No print, no I/O, no side-effects.
- The solution must generalise to ALL training pairs, not just the first one.
- Output shape MUST match the training outputs (check every pair's dimensions).
- Do NOT hardcode specific cell values or coordinates from training examples.
  Discover the abstract rule; derive everything dynamically from the input.
"""


# ---------------------------------------------------------------------------
# Particle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    particle_id:   int
    role_name:     str
    role_desc:     str
    code:          str              = ""
    pos:           np.ndarray       = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    velocity:      np.ndarray       = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    fitness:       float            = 0.0
    last_eval:     dict             = field(default_factory=dict)
    pbest_code:    str              = ""
    pbest_pos:     np.ndarray       = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pbest_fitness: float            = -1.0

    def update_pbest(self, code: str, pos: np.ndarray, fitness: float) -> None:
        self.pbest_code    = code
        self.pbest_pos     = pos.copy()
        self.pbest_fitness = fitness


# ---------------------------------------------------------------------------
# PSO Orchestrator
# ---------------------------------------------------------------------------

class PSOOrchestrator:
    """Swarm of LLM-powered particles searching the code embedding space.

    Args:
        backend:        LLM backend — "ollama" or "anthropic".
        model:          Chat model name (default: deepseek-r1:32b for ollama).
        embed_model:    Ollama embedding model (default: nomic-embed-text).
        n_particles:    Number of particles (≤ len(PARTICLE_ROLES) = 6).
        max_iterations: PSO iteration budget.
        k_candidates:   LLM mutation candidates per particle per iteration.
        w:              Inertia weight (dampens velocity).
        c1:             Cognitive coefficient (pull toward personal best).
        c2:             Social coefficient (pull toward global best).
        temperature:    LLM sampling temperature for code generation.
        max_tokens:     Max tokens per LLM call.
        timeout:        Per-LLM-call timeout in seconds.
        debug:          Verbose progress logging.
    """

    def __init__(
        self,
        backend:        str        = "ollama",
        model:          str | None = None,
        embed_model:    str        = "nomic-embed-text",
        n_particles:    int        = 6,
        max_iterations: int        = 10,
        k_candidates:   int        = 5,
        w:              float      = 0.5,
        c1:             float      = 1.5,
        c2:             float      = 1.5,
        temperature:    float      = 0.7,
        max_tokens:     int        = 4096,
        timeout:        float      = 180.0,
        embed_timeout:  float      = 60.0,
        fitness_alpha:  float      = 0.4,
        debug:          bool       = False,
    ) -> None:
        self.n_particles    = min(n_particles, len(PARTICLE_ROLES))
        self.max_iterations = max_iterations
        self.k_candidates   = k_candidates
        self.w              = w
        self.c1             = c1
        self.c2             = c2
        self.fitness_alpha  = fitness_alpha
        self.debug          = debug
        self.embed_model    = embed_model

        self._llm = LLMClient(
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            embed_timeout=embed_timeout,
            debug=debug,
        )
        self._pso_coder    = PSOCoder(self._llm, k=k_candidates)
        self._hypothesizer = Hypothesizer(self._llm)

        # Expose for CLI / logging
        self.backend  = backend
        self.model    = self._llm.model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, code: str, fallback_random: bool = True) -> np.ndarray | None:
        """Return L2-normalised embedding vector for the given code string.

        Args:
            code:            Python source code to embed.
            fallback_random: If True (default), return a random unit vector on
                             failure so the particle stays in a valid position.
                             If False, return None so the caller can skip this
                             candidate entirely (used during candidate selection).
        """
        try:
            vec = self._llm.embed_code(code, model=self.embed_model)
        except Exception as exc:
            if self.debug:
                print(f"[PSO] Embedding error: {exc}")
            return _random_unit_vec() if fallback_random else None
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else (_random_unit_vec() if fallback_random else None)

    def _eval_fitness(self, code: str, task: dict) -> tuple[float, dict]:
        """Evaluate code; return (continuous_fitness, sandbox_result)."""
        if not code or not code.strip():
            empty = {"pairs": [], "n_correct": 0, "n_total": 0, "all_correct": False}
            return 0.0, empty

        result = sandbox.evaluate_code(code, task)
        if result["all_correct"]:
            return 1.0, result

        total = 0.0
        for pair_res in result["pairs"]:
            if pair_res["error"]:
                total += 0.0
            else:
                total += calculate_continuous_fitness(
                    pair_res["predicted"], pair_res["expected"]
                )
        fitness = total / result["n_total"] if result["n_total"] > 0 else 0.0
        return fitness, result

    def _init_particle_code(
        self,
        particle:          Particle,
        task_description:  str,
        training_examples: str,
        hypothesis:        str | None = None,
    ) -> tuple[str, bool]:
        """Ask the LLM to write an initial solution for this particle's role.

        Args:
            hypothesis: Optional natural-language rule hypothesis from Hypothesizer.
                        When provided, STEP 1 is replaced by the given hypothesis so
                        the LLM can focus entirely on implementing it.

        Returns:
            (code, generated) — generated=False when the LLM call failed so
            the caller can skip the embedding step and apply backoff.
        """
        system = (
            _INIT_CODER_SYSTEM
            + f"\n\nYour cognitive specialisation: {particle.role_name} — {particle.role_desc}"
        )
        if hypothesis:
            content = (
                f"{task_description}\n\n{training_examples}\n\n"
                f"Hypothesis (your STEP 1 reasoning is already done):\n{hypothesis}\n\n"
                "Now implement STEP 2 — write the `transform` function that realises this hypothesis."
            )
        else:
            content = f"{task_description}\n\n{training_examples}\n\nWrite the `transform` function."
        messages = [{"role": "user", "content": content}]

        try:
            response = self._llm.generate(system, messages)
        except Exception as exc:
            if self.debug:
                print(f"[PSO] Init generation failed for particle {particle.particle_id}: {exc}")
            fallback = _ROLE_FALLBACK.get(particle.role_name, _IDENTITY_FALLBACK)
            return fallback, False

        clean = _strip_thinking(response)
        code  = _extract_code(clean) or _extract_code(response)
        if code:
            return code, True
        fallback = _ROLE_FALLBACK.get(particle.role_name, _IDENTITY_FALLBACK)
        return fallback, False

    # ------------------------------------------------------------------
    # Core PSO loop
    # ------------------------------------------------------------------

    def solve(self, task: dict) -> dict:
        """Run the PSO loop and return the best solution found.

        Returns:
            {
              'success':       bool — True if gbest achieves fitness 1.0,
              'code':          str  — best transform function found,
              'gbest_fitness': float,
              'prediction':    np.ndarray | None,
              'test_correct':  bool | None  (only if test output is known),
              'log':           list of per-iteration dicts,
            }
        """
        task_description  = _format_task_description(task)
        training_examples = _format_training_examples(task)
        log: list[dict]   = []

        # ----------------------------------------------------------------
        # Phase 1 — Initialisation
        # ----------------------------------------------------------------
        if self.debug:
            print(f"\n[PSO] Initialising {self.n_particles} particles …")

        swarm: list[Particle] = [
            Particle(
                particle_id=i,
                role_name=PARTICLE_ROLES[i % len(PARTICLE_ROLES)][0],
                role_desc=PARTICLE_ROLES[i % len(PARTICLE_ROLES)][1],
            )
            for i in range(self.n_particles)
        ]

        # Pre-generate diverse hypotheses to seed particles with different starting rules.
        # This replaces STEP 1 (reasoning) in the init prompt so each particle implements
        # a different hypothesis, maximising initial diversity.
        hypotheses: list[str] = []
        try:
            n_hyp = max(3, self.n_particles)
            hyp_raw  = self._hypothesizer.generate(
                task_description + "\n\n" + training_examples, n=n_hyp
            )
            hypotheses = _parse_hypotheses(hyp_raw, max_n=n_hyp)
            if self.debug:
                print(f"  [PSO] Generated {len(hypotheses)} seed hypotheses")
        except Exception as exc:
            if self.debug:
                print(f"  [PSO] Hypothesis generation failed: {exc}")

        for p in swarm:
            seed_hyp = hypotheses[p.particle_id % len(hypotheses)] if hypotheses else None
            p.code, generated = self._init_particle_code(
                p, task_description, training_examples, hypothesis=seed_hyp
            )
            if generated:
                p.pos = self._embed(p.code)
            else:
                # LLM call failed (likely timeout): use a random unit vector to
                # preserve swarm diversity and avoid a cascaded embedding timeout.
                p.pos = _random_unit_vec()
                time.sleep(10)  # backoff so Ollama can drain the timed-out request
            p.velocity = np.zeros_like(p.pos)
            p.fitness, p.last_eval = self._eval_fitness(p.code, task)
            p.update_pbest(p.code, p.pos, p.fitness)
            if self.debug:
                print(f"  Particle {p.particle_id} ({p.role_name}): fitness={p.fitness:.4f}")

        # Global best
        gbest_particle = max(swarm, key=lambda p: p.pbest_fitness)
        gbest_code     = gbest_particle.pbest_code
        gbest_pos      = gbest_particle.pbest_pos.copy()
        gbest_fitness  = gbest_particle.pbest_fitness

        log.append({
            "phase":           "init",
            "gbest_fitness":   gbest_fitness,
            "particle_states": [
                {"id": p.particle_id, "role": p.role_name, "fitness": p.fitness}
                for p in swarm
            ],
        })

        if gbest_fitness >= 1.0:
            if self.debug:
                print("[PSO] Solved during initialisation!")
            return self._make_result(True, gbest_code, gbest_fitness, log, task)

        # ----------------------------------------------------------------
        # Phase 2 — PSO iteration loop
        # ----------------------------------------------------------------
        stagnation_count   = 0
        prev_gbest         = gbest_fitness
        stagnation_limit   = max(2, self.max_iterations // 3)

        for iteration in range(self.max_iterations):
            # Adaptive inertia: linearly anneal w from self.w → self.w*0.4
            # Early iterations explore; later ones exploit local optima.
            progress = iteration / max(1, self.max_iterations - 1)
            w_eff = self.w * (1.0 - 0.6 * progress)  # e.g. 0.50 → 0.20

            if self.debug:
                print(
                    f"\n[PSO] Iteration {iteration + 1}/{self.max_iterations}  "
                    f"gbest={gbest_fitness:.4f}  w={w_eff:.3f}"
                )

            iter_log: dict[str, Any] = {
                "iteration":     iteration + 1,
                "gbest_fitness": gbest_fitness,
                "particles":     [],
            }

            for p in swarm:
                dim = len(p.pos)

                # ------------------------------------------------------------
                # Step 1 — PSO velocity update (continuous embedding space)
                # ------------------------------------------------------------
                r1 = np.random.rand(dim).astype(np.float32)
                r2 = np.random.rand(dim).astype(np.float32)

                if len(p.velocity) != dim:
                    p.velocity = np.zeros(dim, dtype=np.float32)
                if len(gbest_pos) != dim:
                    gbest_pos_d = np.zeros(dim, dtype=np.float32)
                else:
                    gbest_pos_d = gbest_pos

                p.velocity = (
                    w_eff * p.velocity
                    + self.c1 * r1 * (p.pbest_pos - p.pos)
                    + self.c2 * r2 * (gbest_pos_d - p.pos)
                )

                target_pos = p.pos + p.velocity
                # Keep target on the unit hypersphere (embeddings are normalised)
                t_norm = np.linalg.norm(target_pos)
                if t_norm > 0:
                    target_pos = target_pos / t_norm

                # ------------------------------------------------------------
                # Step 2 — LLM generates K candidate mutations
                # ------------------------------------------------------------
                diff_str = _format_eval_diff(p.last_eval, max_pairs=len(task["train"]))

                try:
                    candidates = self._pso_coder.generate_mutations(
                        task_description=task_description,
                        training_context=training_examples,
                        current_code=p.code,
                        current_fitness=p.fitness,
                        pbest_code=p.pbest_code,
                        pbest_fitness=p.pbest_fitness,
                        gbest_code=gbest_code,
                        gbest_fitness=gbest_fitness,
                        role_name=p.role_name,
                        role_description=p.role_desc,
                        eval_diff=diff_str,
                    )
                except Exception as exc:
                    if self.debug:
                        print(f"  [PSO] Mutation failed for particle {p.particle_id}: {exc}")
                    candidates = [p.pbest_code]

                # ------------------------------------------------------------
                # Step 3 — Generate-and-Project: fitness-first hybrid selection
                # Evaluate all candidates in sandbox first (fast).
                # If any improve, pick by fitness (skip most embeddings).
                # Otherwise embed all and pick by PSO proximity.
                # ------------------------------------------------------------
                best_candidate = p.pbest_code  # safe fallback
                best_emb       = p.pbest_pos.copy()

                # (code, emb_or_None, dist_or_inf, fitness)
                cand_results: list[tuple[str, np.ndarray | None, float, float]] = []
                for cand in candidates:
                    if not cand or not cand.strip():
                        continue
                    fit, _ = self._eval_fitness(cand, task)
                    cand_results.append((cand, None, float("inf"), fit))

                if cand_results:
                    # Priority 1: any candidate strictly improves → pick best fitness.
                    # Only embed the winner (saves k-1 embedding calls).
                    improving = [t for t in cand_results if t[3] > p.fitness + 1e-6]
                    if improving:
                        winner_code, _, _, _ = max(improving, key=lambda x: x[3])
                        winner_emb = self._embed(winner_code)  # fallback_random=True
                        best_candidate = winner_code
                        best_emb       = winner_emb
                    else:
                        # Priority 2: no improvement — embed all for PSO exploration.
                        enriched = []
                        for c, _, _, f in cand_results:
                            e = self._embed(c, fallback_random=False)
                            if e is None or len(e) != len(target_pos):
                                continue
                            d = cosine(e, target_pos)
                            enriched.append((c, e, d, f))
                        if enriched:
                            best_c, best_e, _, _ = min(enriched, key=lambda x: x[2])
                            best_candidate = best_c
                            best_emb       = best_e

                # ------------------------------------------------------------
                # Step 4 — Update particle state
                # ------------------------------------------------------------
                prev_pos   = p.pos.copy()
                p.code     = best_candidate
                p.pos      = best_emb
                # Recompute actual velocity as displacement (more stable)
                if len(p.pos) == len(prev_pos):
                    p.velocity = p.pos - prev_pos

                # ------------------------------------------------------------
                # Step 5 — Evaluate in sandbox
                # ------------------------------------------------------------
                p.fitness, p.last_eval = self._eval_fitness(p.code, task)

                # ------------------------------------------------------------
                # Step 6 — Update personal best
                # ------------------------------------------------------------
                if p.fitness > p.pbest_fitness:
                    p.update_pbest(p.code, p.pos, p.fitness)
                    if self.debug:
                        print(
                            f"  Particle {p.particle_id} pbest update: "
                            f"{p.pbest_fitness:.4f}"
                        )

                # ------------------------------------------------------------
                # Step 7 — Update global best
                # ------------------------------------------------------------
                if p.fitness > gbest_fitness:
                    gbest_code    = p.code
                    gbest_pos     = p.pos.copy()
                    gbest_fitness = p.fitness
                    if self.debug:
                        print(
                            f"  *** New gbest! Particle {p.particle_id}: "
                            f"{gbest_fitness:.4f} ***"
                        )

                iter_log["particles"].append({
                    "particle_id":   p.particle_id,
                    "role":          p.role_name,
                    "fitness":       p.fitness,
                    "pbest_fitness": p.pbest_fitness,
                    "n_candidates":  len(cand_results),
                })

            iter_log["gbest_fitness"] = gbest_fitness
            log.append(iter_log)

            if gbest_fitness >= 1.0:
                if self.debug:
                    print(f"\n[PSO] Solved at iteration {iteration + 1}!")
                return self._make_result(True, gbest_code, gbest_fitness, log, task)

            # ----------------------------------------------------------------
            # Stagnation check — reinitialise worst particle if stuck
            # ----------------------------------------------------------------
            if gbest_fitness <= prev_gbest + 1e-6:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_gbest = gbest_fitness

            if stagnation_count >= stagnation_limit and len(swarm) > 1:
                worst = min(swarm, key=lambda pp: pp.pbest_fitness)
                if self.debug:
                    print(
                        f"  [PSO] Stagnation ({stagnation_count} iters) — "
                        f"reinitialising particle {worst.particle_id} ({worst.role_name})"
                    )
                # Use a divergence hypothesis: ask the Hypothesizer for fresh ideas
                # explicitly different from the current best solution.
                diverge_hyp: str | None = None
                try:
                    diverge_raw = self._hypothesizer.generate(
                        task_description + "\n\n" + training_examples,
                        feedback=(
                            f"The current best approach (fitness={gbest_fitness:.3f}) "
                            f"still fails.  Generate 3 COMPLETELY DIFFERENT hypotheses "
                            f"that try different strategies — avoid the same approach."
                        ),
                    )
                    diverge_hyps = _parse_hypotheses(diverge_raw, max_n=3)
                    if diverge_hyps:
                        # Pick one that's different from what was used at init
                        diverge_hyp = diverge_hyps[worst.particle_id % len(diverge_hyps)]
                except Exception:
                    pass
                worst.code, generated = self._init_particle_code(
                    worst, task_description, training_examples, hypothesis=diverge_hyp
                )
                if generated:
                    worst.pos = self._embed(worst.code)
                else:
                    worst.pos = _random_unit_vec()
                worst.velocity     = np.zeros_like(worst.pos)
                worst.fitness, worst.last_eval = self._eval_fitness(worst.code, task)
                worst.pbest_fitness = -1.0   # reset personal best to explore fresh
                worst.update_pbest(worst.code, worst.pos, worst.fitness)
                stagnation_count   = 0

        if self.debug:
            print(f"\n[PSO] Budget exhausted.  Best fitness: {gbest_fitness:.4f}")

        # ----------------------------------------------------------------
        # Phase 3 — Targeted refinement when fitness is near-perfect (≥0.85)
        # ----------------------------------------------------------------
        if 0.85 <= gbest_fitness < 1.0:
            gbest_code, gbest_fitness = self._refinement_phase(
                gbest_code, gbest_fitness, task,
                task_description, training_examples, log
            )

        return self._make_result(gbest_fitness >= 1.0, gbest_code, gbest_fitness, log, task)

    # ------------------------------------------------------------------
    # Refinement helpers
    # ------------------------------------------------------------------

    _REFINEMENT_SYSTEM = """\
You are an expert ARC-AGI code debugger.  The provided `transform` function
is ALMOST correct but has a systematic bug.  You will be shown which cells
are wrong — your job is to find and fix the ROOT CAUSE in the logic.

Available DSL (already imported): crop, rotate, flip, translate, scale, tile,
recolor, mask, overlay, flood_fill, find_objects, bounding_box, crop_to_content,
pad, symmetrize, np

Rules:
- Return ONLY one ```python … ``` code block named `transform`.
- Identify the SYSTEMATIC error pattern (e.g. wrong index, off-by-one, wrong axis).
- Fix the underlying logic — do NOT hardcode specific coordinates or add shape-based if-else.
- The fixed function must generalise to ALL training pairs, not just patch specific cells.
- The output shape MUST remain the same as in the training examples.
"""

    def _refinement_phase(
        self,
        code:          str,
        fitness:       float,
        task:          dict,
        task_desc:     str,
        training_ctx:  str,
        log:           list,
        max_attempts:  int = 5,
    ) -> tuple[str, float]:
        """Run targeted fix attempts when fitness is near-perfect."""
        if self.debug:
            print(f"\n[PSO] Refinement phase  (gbest={fitness:.4f})")

        best_code       = code
        best_fitness    = fitness
        no_improve_streak = 0

        for attempt in range(max_attempts):
            _, eval_res = self._eval_fitness(best_code, task)
            n_train = len(task["train"])
            diff = _format_eval_diff(eval_res, max_pairs=n_train, max_mismatches=24)

            # For near-perfect solutions show full predicted vs expected for each failing pair
            full_comparison = ""
            if best_fitness >= 0.97:
                fc_lines = []
                for i, pair_res in enumerate(eval_res.get("pairs", [])):
                    pred = pair_res.get("predicted")
                    exp  = pair_res.get("expected")
                    if pred is None or exp is None or pair_res.get("error"):
                        continue
                    if (pred == exp).all():
                        continue
                    fc_lines.append(f"Pair {i+1} — predicted:\n  {_grid_to_str(pred)}")
                    fc_lines.append(f"Pair {i+1} — expected:\n  {_grid_to_str(exp)}")
                if fc_lines:
                    full_comparison = "\n\nFull output comparison for failing pairs:\n" + "\n".join(fc_lines)

            # Pass/fail contrast: when some pairs pass and others fail, highlight what differs
            contrast_hint = ""
            pairs_res = eval_res.get("pairs", [])
            passing = [i for i, p in enumerate(pairs_res) if p["correct"]]
            failing = [i for i, p in enumerate(pairs_res) if not p["correct"]]
            if passing and failing and len(passing) < n_train:
                contrast_hint = (
                    f"\n\nNOTE: Pair(s) {[p+1 for p in passing]} pass but "
                    f"pair(s) {[f+1 for f in failing]} fail — compare those inputs "
                    "carefully to find what edge case the code mishandles."
                )

            # Structural hints from training data
            struct_notes = []
            for pair in task["train"]:
                sn = _structural_note(
                    pair["input"] if isinstance(pair["input"], np.ndarray) else np.array(pair["input"]),
                    pair["output"] if isinstance(pair["output"], np.ndarray) else np.array(pair["output"]),
                )
                if sn and sn not in struct_notes:
                    struct_notes.append(sn)
            struct_section = ""
            if struct_notes:
                struct_section = "\n\nPattern hints:\n" + "\n".join(struct_notes)

            content = (
                f"{task_desc}\n\n"
                f"{training_ctx}\n\n"
                f"Current best code (fitness={best_fitness:.4f}):\n"
                f"```python\n{best_code}\n```\n\n"
                f"Error analysis:\n{diff}"
                + full_comparison
                + contrast_hint
                + struct_section
                + "\n\nStudy the error pattern across all pairs — what SYSTEMATIC mistake "
                "causes these wrong cells?\n"
                "Fix the root cause in the algorithm.  Do NOT add if-else branches per "
                "input shape or hardcode specific cell coordinates.\n"
                "Return the corrected `transform` function."
            )
            messages = [{"role": "user", "content": content}]

            try:
                response = self._llm.generate(self._REFINEMENT_SYSTEM, messages)
            except Exception as exc:
                if self.debug:
                    print(f"  [Refine] LLM error: {exc}")
                break

            clean = _strip_thinking(response)
            fixed_code = _extract_code(clean) or _extract_code(response)
            if not fixed_code:
                continue

            fixed_fitness, _ = self._eval_fitness(fixed_code, task)
            if self.debug:
                print(f"  [Refine] attempt {attempt + 1}: fitness={fixed_fitness:.4f}")
            log.append({
                "phase":          "refine",
                "attempt":        attempt + 1,
                "fitness_before": best_fitness,
                "fitness_after":  fixed_fitness,
            })

            if fixed_fitness >= 1.0:
                return fixed_code, 1.0

            if fixed_fitness > best_fitness:
                best_code         = fixed_code
                best_fitness      = fixed_fitness
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                # Early stop: 3 consecutive non-improving attempts
                if no_improve_streak >= 3:
                    if self.debug:
                        print(f"  [Refine] Early stop after {no_improve_streak} non-improving attempts")
                    break

        return best_code, best_fitness

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _make_result(
        self,
        success:      bool,
        code:         str,
        gbest_fitness: float,
        log:          list,
        task:         dict,
    ) -> dict:
        test_pair    = (task.get("test") or [{}])[0]
        test_input   = test_pair.get("input")
        prediction   = None
        test_correct = None

        if code and test_input is not None:
            out, err = sandbox.execute(code, test_input)
            if out is not None:
                prediction = out
                if "output" in test_pair:
                    test_correct = grids_equal(prediction, test_pair["output"])

        return {
            "success":       success,
            "code":          code,
            "gbest_fitness": gbest_fitness,
            "prediction":    prediction,
            "test_correct":  test_correct,
            "log":           log,
        }

    def predict(self, task: dict) -> Grid | None:
        return self.solve(task)["prediction"]
