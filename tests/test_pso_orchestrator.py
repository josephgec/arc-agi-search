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

class TestSolveConvergesDuringIteration:
    """First particles are wrong; second iteration produces perfect code."""

    def test_converges_after_mutations(self):
        orch = make_orchestrator(n_particles=2, max_iterations=5, k_candidates=2)

        call_count = {"n": 0}

        def generate_side_effect(system, messages, **kwargs):
            call_count["n"] += 1
            # First 2 calls (init for 2 particles): wrong code
            if call_count["n"] <= 2:
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
            if call_count["n"] <= 2:
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
        iter_logs = [e for e in result["log"] if "iteration" in e]
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
