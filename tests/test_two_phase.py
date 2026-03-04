"""Tests for the TwoPhaseOrchestrator defined in run_multi_agent.py.

All LLM / PSO calls are mocked — no network or Ollama required.
Tests cover:
  - Phase 1 success short-circuits phase 2
  - Phase 1 failure triggers PSO phase 2
  - Best code from phase 1 is passed to PSOOrchestrator.seed_particles()
  - PSO phase is skipped when phase 1 has no code
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from run_multi_agent import TwoPhaseOrchestrator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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

IDENTITY_CODE = "def transform(input_grid):\n    return input_grid.copy()"

_MULTI_SUCCESS = {
    "success": True,
    "code": IDENTITY_CODE,
    "n_cycles": 2,
    "log": [],
}

_MULTI_FAILURE = {
    "success": False,
    "code": "def transform(g):\n    return g * 0",
    "n_cycles": 5,
    "log": [],
}

_MULTI_FAILURE_NO_CODE = {
    "success": False,
    "code": "",
    "n_cycles": 5,
    "log": [],
}

_PSO_SUCCESS = {
    "success":       True,
    "code":          IDENTITY_CODE,
    "gbest_fitness": 1.0,
    "prediction":    np.array([[5, 6], [7, 8]], dtype=np.int32),
    "test_correct":  True,
    "log":           [],
}

_PSO_FAILURE = {
    "success":       False,
    "code":          "def transform(g):\n    return g",
    "gbest_fitness": 0.4,
    "prediction":    None,
    "test_correct":  False,
    "log":           [],
}


def _make_two_phase(**kwargs) -> TwoPhaseOrchestrator:
    """Build a TwoPhaseOrchestrator with LLMClient patched out."""
    defaults = dict(
        backend="ollama",
        model="test-reasoner",
        phase1_model="test-coder-7b",
        timeout=10.0,
        debug=False,
        pso_n_particles=2,
        pso_max_iterations=2,
    )
    defaults.update(kwargs)
    with patch("agents.llm_client.LLMClient"):
        tp = TwoPhaseOrchestrator(**defaults)
    return tp


# ---------------------------------------------------------------------------
# Phase 1 success — PSO never called
# ---------------------------------------------------------------------------

class TestTwoPhasePhase1Success:
    """When MultiAgent solves the task, PSO is skipped."""

    def test_returns_phase1_result_on_success(self):
        tp = _make_two_phase()
        tp._multi.solve = MagicMock(return_value=_MULTI_SUCCESS)
        tp._pso.solve   = MagicMock(return_value=_PSO_SUCCESS)

        result = tp.solve(SIMPLE_TASK)

        assert result["success"] is True
        assert result is _MULTI_SUCCESS  # same object returned

    def test_pso_solve_not_called_when_phase1_succeeds(self):
        tp = _make_two_phase()
        tp._multi.solve = MagicMock(return_value=_MULTI_SUCCESS)
        tp._pso.solve   = MagicMock(return_value=_PSO_SUCCESS)

        tp.solve(SIMPLE_TASK)

        tp._pso.solve.assert_not_called()

    def test_pso_seed_not_called_when_phase1_succeeds(self):
        tp = _make_two_phase()
        tp._multi.solve       = MagicMock(return_value=_MULTI_SUCCESS)
        tp._pso.solve         = MagicMock(return_value=_PSO_SUCCESS)
        tp._pso.seed_particles = MagicMock()

        tp.solve(SIMPLE_TASK)

        tp._pso.seed_particles.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 1 failure — PSO is called
# ---------------------------------------------------------------------------

class TestTwoPhasePhase1Failure:
    """When MultiAgent fails, PSO phase 2 runs."""

    def test_pso_solve_called_when_phase1_fails(self):
        tp = _make_two_phase()
        tp._multi.solve = MagicMock(return_value=_MULTI_FAILURE)
        tp._pso.solve   = MagicMock(return_value=_PSO_SUCCESS)

        tp.solve(SIMPLE_TASK)

        tp._pso.solve.assert_called_once_with(SIMPLE_TASK)

    def test_returns_pso_result_when_phase1_fails(self):
        tp = _make_two_phase()
        tp._multi.solve = MagicMock(return_value=_MULTI_FAILURE)
        tp._pso.solve   = MagicMock(return_value=_PSO_SUCCESS)

        result = tp.solve(SIMPLE_TASK)

        assert result is _PSO_SUCCESS
        assert result["success"] is True
        assert result["gbest_fitness"] == pytest.approx(1.0)

    def test_phase1_is_always_called_first(self):
        """MultiAgent is called before PSO regardless of outcome."""
        call_order: list[str] = []

        tp = _make_two_phase()
        tp._multi.solve = MagicMock(side_effect=lambda t: (
            call_order.append("multi") or _MULTI_FAILURE
        ))
        tp._pso.solve = MagicMock(side_effect=lambda t: (
            call_order.append("pso") or _PSO_SUCCESS
        ))

        tp.solve(SIMPLE_TASK)

        assert call_order == ["multi", "pso"]


# ---------------------------------------------------------------------------
# Seeding PSO with phase 1 best code
# ---------------------------------------------------------------------------

class TestTwoPhaseSeedCode:
    """Best code from phase 1 is passed to PSO via seed_particles()."""

    def test_seed_particles_called_with_phase1_code(self):
        tp = _make_two_phase()
        tp._multi.solve        = MagicMock(return_value=_MULTI_FAILURE)
        tp._pso.solve          = MagicMock(return_value=_PSO_FAILURE)
        tp._pso.seed_particles = MagicMock()

        tp.solve(SIMPLE_TASK)

        tp._pso.seed_particles.assert_called_once_with([_MULTI_FAILURE["code"]])

    def test_seed_particles_receives_correct_code_string(self):
        EXPECTED_CODE = "def transform(g):\n    return rotate(g, 1)"
        phase1_result = dict(_MULTI_FAILURE, code=EXPECTED_CODE)

        tp = _make_two_phase()
        tp._multi.solve        = MagicMock(return_value=phase1_result)
        tp._pso.solve          = MagicMock(return_value=_PSO_FAILURE)
        tp._pso.seed_particles = MagicMock()

        tp.solve(SIMPLE_TASK)

        seeded_codes = tp._pso.seed_particles.call_args[0][0]
        assert seeded_codes == [EXPECTED_CODE]

    def test_seed_particles_not_called_when_phase1_code_is_empty(self):
        """If phase 1 produces no code, seed_particles is not called."""
        tp = _make_two_phase()
        tp._multi.solve        = MagicMock(return_value=_MULTI_FAILURE_NO_CODE)
        tp._pso.solve          = MagicMock(return_value=_PSO_FAILURE)
        tp._pso.seed_particles = MagicMock()

        tp.solve(SIMPLE_TASK)

        # PSO still runs, but seed_particles is skipped
        tp._pso.solve.assert_called_once()
        tp._pso.seed_particles.assert_not_called()

    def test_pso_runs_even_without_seed(self):
        """PSO phase runs even when phase 1 returned no code."""
        tp = _make_two_phase()
        tp._multi.solve        = MagicMock(return_value=_MULTI_FAILURE_NO_CODE)
        tp._pso.solve          = MagicMock(return_value=_PSO_FAILURE)
        tp._pso.seed_particles = MagicMock()

        result = tp.solve(SIMPLE_TASK)

        assert result is _PSO_FAILURE


# ---------------------------------------------------------------------------
# Attribute exposure
# ---------------------------------------------------------------------------

class TestTwoPhaseAttributes:
    def test_model_attribute_from_pso(self):
        tp = _make_two_phase(model="deepseek-r1:32b")
        # model is exposed on the orchestrator (from PSO sub-orchestrator)
        assert hasattr(tp, "model")

    def test_backend_attribute_exposed(self):
        tp = _make_two_phase(backend="ollama")
        assert tp.backend == "ollama"
