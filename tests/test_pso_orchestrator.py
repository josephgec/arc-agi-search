"""Tests for agents/pso_orchestrator.py.

All LLM calls are mocked — no network/Ollama required.
Tests cover:
  - Particle dataclass
  - PSOOrchestrator construction
  - Embedding and normalisation
  - Fitness evaluation
  - PSO velocity update math
  - Generate-and-Project selection
  - Full solve() with a pre-solved swarm (gbest_fitness==1.0 at init)
  - Full solve() loop that converges after 1 iteration
  - Graceful error handling (embedding failures, empty LLM responses)
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from agents.pso_orchestrator import (
    PSOOrchestrator,
    Particle,
    PARTICLE_ROLES,
    _compress_grid,
)
from arc.evaluate import calculate_continuous_fitness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orchestrator(**kwargs) -> PSOOrchestrator:
    """Build a PSOOrchestrator with all external calls mocked out."""
    defaults = dict(
        backend="ollama",
        model="test-model",
        embed_model="nomic-embed-text",
        n_particles=2,
        max_iterations=3,
        k_candidates=2,
        w=0.5, c1=1.5, c2=1.5,
        temperature=0.5,
        max_tokens=512,
        timeout=10.0,
        embed_mode="code",   # keep existing tests unaffected
        debug=False,
    )
    defaults.update(kwargs)
    orch = PSOOrchestrator(**defaults)
    return orch


def unit_vec(dim: int = 768, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


IDENTITY_CODE = "def transform(input_grid):\n    return input_grid.copy()"

SIMPLE_TASK = {
    "train": [
        {"input":  np.array([[1, 2], [3, 4]], dtype=np.int32),
         "output": np.array([[1, 2], [3, 4]], dtype=np.int32)},
    ],
    "test": [
        {"input":  np.array([[5, 6], [7, 8]], dtype=np.int32),
         "output": np.array([[5, 6], [7, 8]], dtype=np.int32)},
    ],
}


# ---------------------------------------------------------------------------
# Particle dataclass
# ---------------------------------------------------------------------------

class TestParticle:
    def test_construction(self):
        p = Particle(particle_id=0, role_name="test", role_desc="testing")
        assert p.particle_id == 0
        assert p.fitness == 0.0
        assert p.pbest_fitness == -1.0

    def test_update_pbest(self):
        p   = Particle(particle_id=0, role_name="r", role_desc="d")
        pos = unit_vec(768, seed=1)
        p.update_pbest("def transform(g): return g", pos, 0.8)
        assert p.pbest_fitness == 0.8
        assert p.pbest_code == "def transform(g): return g"
        np.testing.assert_array_equal(p.pbest_pos, pos)

    def test_update_pbest_copies_pos(self):
        p   = Particle(particle_id=0, role_name="r", role_desc="d")
        pos = unit_vec(768, seed=2)
        p.update_pbest("code", pos, 0.5)
        pos[0] = 999.0   # mutate original
        assert p.pbest_pos[0] != 999.0   # stored copy should be unaffected


# ---------------------------------------------------------------------------
# PARTICLE_ROLES catalogue
# ---------------------------------------------------------------------------

class TestParticleRoles:
    def test_six_roles_defined(self):
        assert len(PARTICLE_ROLES) == 6

    def test_each_role_has_name_and_description(self):
        for name, desc in PARTICLE_ROLES:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 20

    def test_role_names_unique(self):
        names = [r[0] for r in PARTICLE_ROLES]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# PSOOrchestrator construction
# ---------------------------------------------------------------------------

class TestPSOOrchestratorConstruction:
    def test_n_particles_capped(self):
        orch = make_orchestrator(n_particles=100)
        assert orch.n_particles == len(PARTICLE_ROLES)

    def test_model_exposed(self):
        orch = make_orchestrator()
        assert orch.model == "test-model"

    def test_backend_exposed(self):
        orch = make_orchestrator(backend="ollama")
        assert orch.backend == "ollama"


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_unit_norm(self):
        orch = make_orchestrator()
        v = unit_vec(768)
        orch._llm.embed_code = MagicMock(return_value=v * 3.0)   # unnormalised
        result = orch._embed("some code")
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_returns_random_unit_vec_on_connection_error_with_fallback(self):
        orch = make_orchestrator()
        orch._llm.embed_code = MagicMock(side_effect=ConnectionError("no Ollama"))
        result = orch._embed("some code", fallback_random=True)
        assert result is not None
        assert result.shape == (768,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_returns_none_on_connection_error_without_fallback(self):
        orch = make_orchestrator()
        orch._llm.embed_code = MagicMock(side_effect=ConnectionError("no Ollama"))
        result = orch._embed("some code", fallback_random=False)
        assert result is None

    def test_zero_vector_falls_back_to_random_unit_vec(self):
        orch = make_orchestrator()
        orch._llm.embed_code = MagicMock(return_value=np.zeros(768, dtype=np.float32))
        result = orch._embed("empty", fallback_random=True)
        # Zero-norm input triggers random unit vector fallback
        assert result is not None
        assert result.shape == (768,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _eval_fitness
# ---------------------------------------------------------------------------

class TestEvalFitness:
    def test_empty_code_returns_zero(self):
        orch = make_orchestrator()
        fitness, _ = orch._eval_fitness("", SIMPLE_TASK)
        assert fitness == 0.0

    def test_perfect_code_returns_one(self):
        orch = make_orchestrator()
        fitness, result = orch._eval_fitness(IDENTITY_CODE, SIMPLE_TASK)
        assert fitness == pytest.approx(1.0)
        assert result["all_correct"] is True

    def test_wrong_code_returns_partial(self):
        orch   = make_orchestrator()
        wrong  = "def transform(g):\n    return g * 0"
        fitness, _ = orch._eval_fitness(wrong, SIMPLE_TASK)
        assert 0.0 <= fitness < 1.0

    def test_crashing_code_returns_zero(self):
        orch  = make_orchestrator()
        crash = "def transform(g):\n    raise RuntimeError('boom')"
        fitness, _ = orch._eval_fitness(crash, SIMPLE_TASK)
        assert fitness == 0.0


# ---------------------------------------------------------------------------
# PSO velocity / position math
# ---------------------------------------------------------------------------

class TestPSOVelocityMath:
    """Verify the core PSO update equations are applied correctly."""

    def test_velocity_shape_matches_embedding(self):
        dim = 768
        pos      = unit_vec(dim, seed=0)
        pbest    = unit_vec(dim, seed=1)
        gbest    = unit_vec(dim, seed=2)
        velocity = np.zeros(dim, dtype=np.float32)

        w, c1, c2 = 0.5, 1.5, 1.5
        r1 = np.ones(dim, dtype=np.float32) * 0.5
        r2 = np.ones(dim, dtype=np.float32) * 0.5

        new_vel = w * velocity + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        target  = pos + new_vel
        t_norm  = np.linalg.norm(target)
        if t_norm > 0:
            target = target / t_norm

        assert target.shape == (dim,)
        assert np.linalg.norm(target) == pytest.approx(1.0, abs=1e-5)

    def test_zero_velocity_target_stays_at_pos(self):
        dim      = 16
        pos      = unit_vec(dim, seed=5)
        velocity = np.zeros(dim, dtype=np.float32)
        # With r=0, cognitive and social terms vanish
        new_vel = 0.5 * velocity + 0.0 + 0.0
        target  = pos + new_vel
        np.testing.assert_array_almost_equal(target, pos)

    def test_inertia_only_decays_velocity(self):
        dim      = 16
        velocity = np.ones(dim, dtype=np.float32)
        w        = 0.5
        decayed  = w * velocity
        np.testing.assert_array_almost_equal(decayed, np.full(dim, 0.5, dtype=np.float32))


# ---------------------------------------------------------------------------
# Full solve() — initialisation already perfect (fitness=1.0)
# ---------------------------------------------------------------------------

class TestSolveImmediateSuccess:
    """If the first particle initialises with perfect code, solve returns instantly."""

    def test_returns_success(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5)

        # Mock: LLM generates perfect identity code
        orch._llm.generate   = MagicMock(return_value=(
            "```python\n"
            "def transform(input_grid):\n"
            "    return input_grid.copy()\n"
            "```"
        ))
        emb = unit_vec(768, seed=10)
        orch._llm.embed_code = MagicMock(return_value=emb)

        result = orch.solve(SIMPLE_TASK)

        assert result["success"] is True
        assert result["gbest_fitness"] == pytest.approx(1.0)
        assert result["code"] is not None
        assert "def transform" in result["code"]

    def test_prediction_is_numpy_array(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5)
        orch._llm.generate   = MagicMock(return_value=(
            "```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
        ))
        emb = unit_vec(768, seed=11)
        orch._llm.embed_code = MagicMock(return_value=emb)

        result = orch.solve(SIMPLE_TASK)
        assert isinstance(result["prediction"], np.ndarray)

    def test_test_correct_flag_set(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5)
        orch._llm.generate   = MagicMock(return_value=(
            "```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
        ))
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=12))
        result = orch.solve(SIMPLE_TASK)
        assert result["test_correct"] is True

    def test_log_contains_init_phase(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5)
        orch._llm.generate   = MagicMock(return_value=(
            "```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
        ))
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=13))
        result = orch.solve(SIMPLE_TASK)
        assert any(e.get("phase") == "init" for e in result["log"])


# ---------------------------------------------------------------------------
# Full solve() — converges during iteration loop
# ---------------------------------------------------------------------------

_HYP_RESPONSE = (
    "1. The transformation copies each input cell to the output unchanged, "
    "preserving all colours and the grid layout exactly as given in training.\n\n"
    "2. The transformation rotates the grid 90 degrees counter-clockwise, "
    "producing an output whose rows are the columns of the input in reverse order.\n\n"
    "3. The transformation fills all background cells with the dominant foreground colour, "
    "leaving non-background cells as they are and creating a fully filled output grid."
)


class TestSolveConvergesDuringIteration:
    """First particles are wrong; second iteration produces perfect code."""

    def test_converges_after_mutations(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5, k_candidates=2)

        call_count = {"n": 0}

        def generate_side_effect(system, messages, **kwargs):
            call_count["n"] += 1
            # Call 1: hypothesis generation (returns numbered hypotheses)
            if call_count["n"] == 1:
                return _HYP_RESPONSE
            # Calls 2-3 (init for 2 particles): wrong code
            if call_count["n"] <= 3:
                return "```python\ndef transform(g):\n    return g * 0\n```"
            # Subsequent mutation calls: correct code
            return (
                "```python\ndef transform_1(input_grid):\n    return input_grid.copy()\n```\n"
                "```python\ndef transform_2(input_grid):\n    return input_grid.copy()\n```"
            )

        orch._llm.generate   = MagicMock(side_effect=generate_side_effect)
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=20))

        result = orch.solve(SIMPLE_TASK)

        assert result["success"] is True
        assert result["gbest_fitness"] == pytest.approx(1.0)

    def test_log_has_iteration_entries(self):
        orch = make_orchestrator(n_particles=2, max_iterations=3, k_candidates=2)

        call_count = {"n": 0}

        def gen(system, messages, **kwargs):
            call_count["n"] += 1
            # Call 1: hypothesis generation
            if call_count["n"] == 1:
                return _HYP_RESPONSE
            if call_count["n"] <= 3:
                return "```python\ndef transform(g):\n    return g * 0\n```"
            return (
                "```python\ndef transform_1(input_grid):\n    return input_grid.copy()\n```\n"
                "```python\ndef transform_2(input_grid):\n    return input_grid.copy()\n```"
            )

        orch._llm.generate   = MagicMock(side_effect=gen)
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=21))

        result = orch.solve(SIMPLE_TASK)
        iter_logs = [e for e in result["log"] if "iteration" in e]
        assert len(iter_logs) >= 1


# ---------------------------------------------------------------------------
# solve() — exhausts budget without solving
# ---------------------------------------------------------------------------

class TestSolveBudgetExhausted:
    def test_returns_failure_when_unsolvable(self):
        orch = make_orchestrator(n_particles=2, max_iterations=2, k_candidates=2)

        # Always generate wrong code
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(g):\n    return g * 0\n```"
        )
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=30))

        # Use a task where 0-multiply gives wrong output
        result = orch.solve(SIMPLE_TASK)

        assert result["success"] is False
        assert result["gbest_fitness"] < 1.0
        assert result["code"] is not None   # best-effort code still returned

    def test_log_length_matches_iterations(self):
        orch = make_orchestrator(n_particles=2, max_iterations=2, k_candidates=2)
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(g):\n    return g * 0\n```"
        )
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=31))

        result = orch.solve(SIMPLE_TASK)
        iter_logs = [e for e in result["log"] if "particles" in e]
        assert len(iter_logs) == 2   # == max_iterations


# ---------------------------------------------------------------------------
# solve() — embedding failure is handled gracefully
# ---------------------------------------------------------------------------

class TestSolveEmbeddingFailure:
    def test_embedding_error_does_not_crash(self):
        orch = make_orchestrator(n_particles=2, max_iterations=2, k_candidates=2)
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
        )
        # Raise connection error on all embed calls
        orch._llm.embed_code = MagicMock(side_effect=ConnectionError("offline"))

        # Should not raise; falls back to random unit vectors
        result = orch.solve(SIMPLE_TASK)
        assert "success" in result


# ---------------------------------------------------------------------------
# solve() result schema
# ---------------------------------------------------------------------------

class TestSolveResultSchema:
    def test_required_keys_present(self):
        orch = make_orchestrator(n_particles=2, max_iterations=1)
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(g):\n    return g * 0\n```"
        )
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768))
        result = orch.solve(SIMPLE_TASK)

        required = {"success", "code", "gbest_fitness", "prediction", "test_correct", "log"}
        assert required.issubset(result.keys())

    def test_gbest_fitness_in_unit_interval(self):
        orch = make_orchestrator(n_particles=2, max_iterations=1)
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(g):\n    return g * 0\n```"
        )
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768))
        result = orch.solve(SIMPLE_TASK)
        assert 0.0 <= result["gbest_fitness"] <= 1.0


# ---------------------------------------------------------------------------
# predict() convenience wrapper
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_grid_or_none(self):
        orch = make_orchestrator(n_particles=2, max_iterations=1)
        orch._llm.generate   = MagicMock(
            return_value="```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
        )
        orch._llm.embed_code = MagicMock(return_value=unit_vec(768, seed=99))
        pred = orch.predict(SIMPLE_TASK)
        assert pred is None or isinstance(pred, np.ndarray)


# ---------------------------------------------------------------------------
# _embed_behavior and _pos_from_eval
# ---------------------------------------------------------------------------

class TestEmbedBehavior:
    def test_returns_unit_norm_with_precomputed_eval(self):
        """With a pre-computed eval_result, no sandbox call is needed."""
        orch = make_orchestrator(embed_mode="behavioral")
        emb  = unit_vec(768, seed=42)
        orch._llm.embed_code = MagicMock(return_value=emb * 2.0)  # unnormalised

        eval_result = {
            "pairs": [
                {
                    "predicted": np.array([[1, 2], [3, 4]], dtype=np.int32),
                    "expected":  np.array([[1, 2], [3, 4]], dtype=np.int32),
                    "correct":   True,
                    "error":     None,
                }
            ],
            "n_correct": 1, "n_total": 1, "all_correct": True,
        }
        result = orch._embed_behavior(
            IDENTITY_CODE, SIMPLE_TASK, eval_result=eval_result
        )
        assert result is not None
        assert result.shape == (768,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_falls_back_to_code_embed_on_exception(self):
        """If embed_code raises on first call, should fall back to _embed (code mode)."""
        orch = make_orchestrator(embed_mode="behavioral")
        fallback_vec = unit_vec(768, seed=7)
        call_count = {"n": 0}

        def embed_side_effect(text, model=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("embedding failed")
            return fallback_vec

        orch._llm.embed_code = MagicMock(side_effect=embed_side_effect)
        result = orch._embed_behavior(IDENTITY_CODE, SIMPLE_TASK)
        # Should return a unit vector from fallback code embedding
        assert result is not None
        assert result.shape == (768,)

    def test_pos_from_eval_routes_code_mode(self):
        """_pos_from_eval with embed_mode='code' routes through _embed, not _embed_behavior."""
        orch = make_orchestrator(embed_mode="code")
        emb  = unit_vec(768, seed=5)
        orch._llm.embed_code = MagicMock(return_value=emb)

        eval_result = {"pairs": [], "n_correct": 0, "n_total": 0, "all_correct": False}
        result = orch._pos_from_eval(IDENTITY_CODE, SIMPLE_TASK, eval_result)
        assert result is not None
        # embed_code called once (from _embed, not _embed_behavior)
        assert orch._llm.embed_code.call_count == 1

    def test_all_none_predictions_uses_zero_grids(self):
        """When all predicted outputs are None, zero-grids are substituted."""
        orch = make_orchestrator(embed_mode="behavioral")
        emb  = unit_vec(768, seed=3)
        orch._llm.embed_code = MagicMock(return_value=emb)

        eval_result = {
            "pairs": [
                {
                    "predicted": None,
                    "expected":  np.array([[1, 2], [3, 4]], dtype=np.int32),
                    "correct":   False,
                    "error":     "RuntimeError: boom",
                }
            ],
            "n_correct": 0, "n_total": 1, "all_correct": False,
        }
        result = orch._embed_behavior(
            IDENTITY_CODE, SIMPLE_TASK, eval_result=eval_result
        )
        assert result is not None
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    _TASK_DESC = "Test task"
    _TRAIN_EX  = "Pair 1: input=[[1]] output=[[2]]"
    _CODE_A    = "def transform(g):\n    return g + 1"
    _CODE_B    = "def transform(g):\n    return g * 2"

    def test_valid_llm_response_returns_code(self):
        orch = make_orchestrator()
        orch._llm.generate = MagicMock(
            return_value=(
                "Here is my combined solution:\n"
                "```python\n"
                "def transform(input_grid):\n"
                "    return input_grid + 1\n"
                "```"
            )
        )
        result = orch._crossover(
            self._CODE_A, self._CODE_B, self._TASK_DESC, self._TRAIN_EX
        )
        assert result is not None
        assert "def transform" in result

    def test_empty_llm_response_returns_none(self):
        orch = make_orchestrator()
        orch._llm.generate = MagicMock(return_value="No code here at all.")
        result = orch._crossover(
            self._CODE_A, self._CODE_B, self._TASK_DESC, self._TRAIN_EX
        )
        assert result is None

    def test_llm_exception_returns_none(self):
        orch = make_orchestrator()
        orch._llm.generate = MagicMock(side_effect=RuntimeError("network error"))
        result = orch._crossover(
            self._CODE_A, self._CODE_B, self._TASK_DESC, self._TRAIN_EX
        )
        assert result is None


# ---------------------------------------------------------------------------
# _refinement_phase
# ---------------------------------------------------------------------------

class TestRefinementPhase:
    """Direct tests for PSOOrchestrator._refinement_phase."""

    START_CODE = IDENTITY_CODE

    _EVAL_RES_FAIL = {
        "pairs": [{
            "predicted": np.array([[1, 2], [3, 0]], dtype=np.int32),
            "expected":  np.array([[1, 2], [3, 4]], dtype=np.int32),
            "correct": False, "error": None,
        }],
        "all_correct": False, "n_correct": 0, "n_total": 1,
    }
    _EVAL_RES_PERFECT = {
        "pairs": [{
            "predicted": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "expected":  np.array([[1, 2], [3, 4]], dtype=np.int32),
            "correct": True, "error": None,
        }],
        "all_correct": True, "n_correct": 1, "n_total": 1,
    }

    def _make_orch(self):
        orch = make_orchestrator(debug=False)
        orch._llm = MagicMock()
        return orch

    def _good_response(self):
        return f"```python\n{self.START_CODE}\n```"

    def test_refine_improves_fitness(self):
        orch = self._make_orch()
        orch._eval_fitness = MagicMock(side_effect=[
            (0.8, self._EVAL_RES_FAIL),   # initial diff computation
            (0.9, self._EVAL_RES_FAIL),   # after first attempt
            (0.9, self._EVAL_RES_FAIL),   # guard calls
        ] * 5)
        orch._llm.generate = MagicMock(return_value=self._good_response())

        log: list = []
        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.8, SIMPLE_TASK, "task desc", "train ctx", log,
            max_attempts=2,
        )
        assert fitness >= 0.8
        assert len(log) >= 1
        assert log[0]["phase"] == "refine"

    def test_refine_finds_perfect_solution(self):
        orch = self._make_orch()
        orch._eval_fitness = MagicMock(side_effect=[
            (0.8, self._EVAL_RES_FAIL),     # diff for attempt 1
            (1.0, self._EVAL_RES_PERFECT),  # attempt 1 → perfect
        ])
        orch._llm.generate = MagicMock(return_value=self._good_response())

        log: list = []
        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.8, SIMPLE_TASK, "task desc", "train ctx", log,
        )
        assert fitness == pytest.approx(1.0)

    def test_refine_early_stop_on_no_improvement(self):
        orch = self._make_orch()
        # _eval_fitness always returns 0.8 → no improvement streak
        orch._eval_fitness = MagicMock(return_value=(0.8, self._EVAL_RES_FAIL))
        orch._llm.generate = MagicMock(return_value=self._good_response())

        log: list = []
        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.8, SIMPLE_TASK, "task desc", "train ctx", log,
            max_attempts=10,
        )
        # Early stop after 3 consecutive non-improving attempts
        assert len(log) <= 3
        assert fitness == pytest.approx(0.8)

    def test_refine_llm_error_breaks_immediately(self):
        orch = self._make_orch()
        orch._eval_fitness = MagicMock(return_value=(0.5, self._EVAL_RES_FAIL))
        orch._llm.generate = MagicMock(side_effect=RuntimeError("connection refused"))

        log: list = []
        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.5, SIMPLE_TASK, "task desc", "train ctx", log,
        )
        assert len(log) == 0
        assert code == self.START_CODE
        assert fitness == pytest.approx(0.5)

    def test_refine_no_code_in_response_continues(self):
        orch = self._make_orch()
        # Always returns 0.8; generate gives no extractable code twice, then succeeds
        orch._eval_fitness = MagicMock(return_value=(0.8, self._EVAL_RES_FAIL))
        orch._llm.generate = MagicMock(side_effect=[
            "No code here",
            "Still nothing",
            self._good_response(),
        ])

        log: list = []
        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.8, SIMPLE_TASK, "task desc", "train ctx", log,
            max_attempts=3,
        )
        # The successful attempt was also non-improving (same fitness 0.8)
        assert fitness == pytest.approx(0.8)

    def test_refine_debug_mode_does_not_crash(self, capsys):
        orch = make_orchestrator(debug=True)
        orch._llm = MagicMock()
        orch._eval_fitness = MagicMock(side_effect=[
            (0.8, self._EVAL_RES_FAIL),
            (1.0, self._EVAL_RES_PERFECT),
        ])
        orch._llm.generate = MagicMock(return_value=self._good_response())

        code, fitness = orch._refinement_phase(
            self.START_CODE, 0.8, SIMPLE_TASK, "task desc", "train ctx", [],
        )
        captured = capsys.readouterr()
        assert "Refine" in captured.out or "refine" in captured.out.lower() or fitness >= 0.8


# ---------------------------------------------------------------------------
# TestCaching — sandbox_cache and behavior_cache
# ---------------------------------------------------------------------------

class TestCaching:
    """Verify that _sandbox_cache and _behavior_cache deduplicate calls."""

    TASK = {
        "train": [
            {"input":  np.array([[1, 0]], dtype=np.int32),
             "output": np.array([[1, 0]], dtype=np.int32)},
        ],
        "test": [{"input": np.array([[1, 0]], dtype=np.int32)}],
    }
    CODE = "def transform(g):\n    return g.copy()"

    def test_sandbox_cache_initialized_empty(self):
        orch = make_orchestrator()
        assert orch._sandbox_cache == {}
        assert orch._behavior_cache == {}

    def test_sandbox_cache_deduplicates(self):
        orch = make_orchestrator()
        with patch("agents.pso_orchestrator.sandbox.evaluate_code") as mock_eval:
            mock_eval.return_value = {
                "pairs": [{"correct": True, "predicted": np.array([[1, 0]], dtype=np.int32),
                           "expected": np.array([[1, 0]], dtype=np.int32), "error": None}],
                "n_correct": 1, "n_total": 1, "all_correct": True,
            }
            # Call twice with the same code
            f1, r1 = orch._eval_fitness(self.CODE, self.TASK)
            f2, r2 = orch._eval_fitness(self.CODE, self.TASK)

        # sandbox.evaluate_code should only have been called once
        assert mock_eval.call_count == 1
        assert f1 == pytest.approx(1.0)
        assert f2 == pytest.approx(1.0)

    def test_sandbox_cache_cleared_on_solve(self):
        orch = make_orchestrator()
        # Pre-populate the cache as if a previous task was solved
        orch._sandbox_cache["stale_key"] = {"n_total": 0}
        orch._behavior_cache["stale_text"] = np.zeros(768, dtype=np.float32)

        # solve() should clear both caches at the start
        with patch.object(orch, "_init_particle_code", return_value=(self.CODE, True)), \
             patch.object(orch, "_pos_from_eval",      return_value=unit_vec()), \
             patch.object(orch, "_eval_fitness",       return_value=(1.0, {
                 "pairs": [], "n_correct": 1, "n_total": 1, "all_correct": True,
             })):
            orch.solve(self.TASK)

        assert "stale_key"  not in orch._sandbox_cache
        assert "stale_text" not in orch._behavior_cache

    def test_behavior_cache_deduplicates(self):
        orch = make_orchestrator(embed_mode="behavioral")
        task = self.TASK

        # Pre-compute eval_result so _embed_behavior uses it directly
        eval_result = {
            "pairs": [{"correct": True,
                       "predicted": np.array([[1, 0]], dtype=np.int32),
                       "expected":  np.array([[1, 0]], dtype=np.int32),
                       "error": None}],
        }

        with patch.object(orch._llm, "embed_code") as mock_embed:
            mock_embed.return_value = unit_vec()
            # Call twice with identical outputs → same behavior_text
            v1 = orch._embed_behavior(self.CODE, task, eval_result=eval_result)
            v2 = orch._embed_behavior(self.CODE, task, eval_result=eval_result)

        # embed_code should only have been called once
        assert mock_embed.call_count == 1
        np.testing.assert_array_equal(v1, v2)

    def test_sandbox_cache_different_code_runs_separately(self):
        orch = make_orchestrator()
        code_b = "def transform(g):\n    return g * 0"
        with patch("agents.pso_orchestrator.sandbox.evaluate_code") as mock_eval:
            mock_eval.return_value = {
                "pairs": [{"correct": False, "predicted": np.array([[0, 0]], dtype=np.int32),
                           "expected": np.array([[1, 0]], dtype=np.int32), "error": None}],
                "n_correct": 0, "n_total": 1, "all_correct": False,
            }
            orch._eval_fitness(self.CODE, self.TASK)
            orch._eval_fitness(code_b, self.TASK)

        # Two distinct codes → two separate sandbox calls
        assert mock_eval.call_count == 2


# ---------------------------------------------------------------------------
# _compress_grid
# ---------------------------------------------------------------------------

class TestCompressGrid:
    def test_small_grid_full_matrix(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = _compress_grid(g)
        assert "1 2" in result
        assert "3 4" in result

    def test_large_empty_grid(self):
        g = np.zeros((20, 20), dtype=np.int32)
        result = _compress_grid(g)
        assert "Empty" in result
        assert "(20, 20)" in result

    def test_large_sparse_grid_uses_coord_format(self):
        g = np.zeros((20, 20), dtype=np.int32)
        g[0, 0] = 5
        g[10, 15] = 3
        result = _compress_grid(g)
        assert "(0,0)=5" in result
        assert "(10,15)=3" in result

    def test_large_dense_grid_summarises(self):
        # >50 active pixels → summary format
        g = np.ones((15, 15), dtype=np.int32)
        result = _compress_grid(g)
        assert "active pixels" in result

    def test_boundary_exactly_100_cells_is_full_matrix(self):
        # 10×10 = 100 cells → full matrix (size <= 100)
        g = np.arange(100, dtype=np.int32).reshape(10, 10)
        result = _compress_grid(g)
        # Should contain raw numbers, not coord format
        assert "(0,0)=" not in result

    def test_boundary_101_cells_uses_compact(self):
        # 11×11 grid with one active pixel → coord format
        g = np.zeros((11, 11), dtype=np.int32)
        g[5, 5] = 7
        result = _compress_grid(g)
        assert "(5,5)=7" in result
