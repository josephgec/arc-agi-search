"""Tests for agents/ensemble.py — voting helpers and Ensemble.solve()."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agents.ensemble import Ensemble, _majority_vote, _vote_summary, _grids_equal


# ---------------------------------------------------------------------------
# _grids_equal
# ---------------------------------------------------------------------------

class TestGridsEqual:
    def test_equal_arrays(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert _grids_equal(a, a.copy())

    def test_different_values(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        assert not _grids_equal(a, b)

    def test_different_shapes(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1], [2]], dtype=np.int32)
        assert not _grids_equal(a, b)


# ---------------------------------------------------------------------------
# _majority_vote
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_empty_returns_none(self):
        assert _majority_vote([]) is None

    def test_single_grid_returned(self):
        g = np.array([[1, 2]], dtype=np.int32)
        result = _majority_vote([g])
        np.testing.assert_array_equal(result, g)

    def test_majority_wins(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        result = _majority_vote([a, b, a])   # a appears twice
        np.testing.assert_array_equal(result, a)

    def test_all_same(self):
        g = np.array([[9, 8], [7, 6]], dtype=np.int32)
        result = _majority_vote([g, g.copy(), g.copy()])
        np.testing.assert_array_equal(result, g)

    def test_tie_returns_first(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        result = _majority_vote([a, b])   # tie → first representative
        np.testing.assert_array_equal(result, a)


# ---------------------------------------------------------------------------
# _vote_summary
# ---------------------------------------------------------------------------

class TestVoteSummary:
    def test_empty_returns_empty(self):
        assert _vote_summary([]) == []

    def test_groups_correctly(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        summary = _vote_summary([a, b, a, a])
        assert summary[0]["count"] == 3   # a wins
        assert summary[1]["count"] == 1   # b second

    def test_sorted_descending(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        c = np.array([[3]], dtype=np.int32)
        summary = _vote_summary([a, b, b, c, c, c])
        counts = [s["count"] for s in summary]
        assert counts == sorted(counts, reverse=True)

    def test_candidate_indices_recorded(self):
        a = np.array([[1]], dtype=np.int32)
        b = np.array([[2]], dtype=np.int32)
        summary = _vote_summary([a, b, a])
        # a is at indices 0, 2
        group_a = next(s for s in summary if s["count"] == 2)
        assert set(group_a["candidate_indices"]) == {0, 2}


# ---------------------------------------------------------------------------
# Ensemble.solve()
# ---------------------------------------------------------------------------

class TestEnsembleSolve:
    def _make_ensemble(self, orchestrator_results: list[dict]) -> Ensemble:
        """Build Ensemble with a mocked Orchestrator that yields preset results."""
        ens = Ensemble.__new__(Ensemble)
        ens.target_candidates = 3
        ens.max_runs          = 5
        ens.debug             = False
        ens.backend           = "ollama"
        ens.hypothesizer_model = "mock"
        ens.coder_model        = "mock"
        ens.critic_model       = "mock"
        ens.model              = "mock"
        ens.hypothesizer_temperature = 0.6
        ens.coder_temperature        = 0.1
        ens.critic_temperature       = 0.2
        ens.hypothesizer_max_tokens  = 1024
        ens.coder_max_tokens         = 1024
        ens.critic_max_tokens        = 1024

        mock_orch = MagicMock()
        mock_orch.solve.side_effect = orchestrator_results
        ens._orchestrator = mock_orch
        return ens

    IDENTITY_CODE = "def transform(input_grid):\n    return input_grid.copy()"
    TASK = {
        "train": [
            {"input":  np.array([[1, 2], [3, 4]], dtype=np.int32),
             "output": np.array([[1, 2], [3, 4]], dtype=np.int32)},
        ],
        "test": [
            {"input":  np.array([[5, 6], [7, 8]], dtype=np.int32),
             "output": np.array([[5, 6], [7, 8]], dtype=np.int32)},
        ],
    }

    def _orch_result(self, code: str | None) -> dict:
        if code is None:
            return {"candidates": []}
        return {"candidates": [{"code": code}]}

    def _make_unique_codes(self, n: int) -> list[str]:
        """Return n syntactically valid but distinct transform functions."""
        return [
            f"def transform(input_grid):\n    return input_grid.copy()  # v{i}"
            for i in range(n)
        ]

    def test_returns_success_when_candidates_found(self):
        # Provide distinct codes so dedup doesn't block candidate accumulation
        codes = self._make_unique_codes(3)
        results = [self._orch_result(c) for c in codes]
        ens = self._make_ensemble(results)
        ens.target_candidates = 3
        result = ens.solve(self.TASK)
        assert result["success"] is True

    def test_returns_failure_when_no_candidates(self):
        results = [{"candidates": []}] * 5
        ens = self._make_ensemble(results)
        result = ens.solve(self.TASK)
        assert result["success"] is False

    def test_result_schema(self):
        codes   = self._make_unique_codes(3)
        results = [self._orch_result(c) for c in codes]
        ens = self._make_ensemble(results)
        ens.target_candidates = 3
        result = ens.solve(self.TASK)
        required = {"success", "prediction", "test_correct", "candidates", "vote_summary", "n_runs"}
        assert required.issubset(result.keys())

    def test_stops_early_when_target_reached(self):
        # 5 distinct results available; target=1 so should stop after run 1
        codes   = self._make_unique_codes(5)
        results = [self._orch_result(c) for c in codes]
        ens = self._make_ensemble(results)
        ens.target_candidates = 1
        result = ens.solve(self.TASK)
        assert result["n_runs"] == 1

    def test_deduplicates_identical_code(self):
        # Two runs returning the same code → only 1 unique candidate
        results = [
            {"candidates": [{"code": self.IDENTITY_CODE}]},
            {"candidates": [{"code": self.IDENTITY_CODE}]},
        ]
        ens = self._make_ensemble(results)
        ens.target_candidates = 5
        ens.max_runs = 2
        result = ens.solve(self.TASK)
        assert len(result["candidates"]) == 1

    def test_prediction_is_ndarray_on_success(self):
        codes   = self._make_unique_codes(3)
        results = [self._orch_result(c) for c in codes]
        ens = self._make_ensemble(results)
        ens.target_candidates = 3
        result = ens.solve(self.TASK)
        if result["prediction"] is not None:
            assert isinstance(result["prediction"], np.ndarray)

    def test_predict_convenience(self):
        ens = self._make_ensemble([{"candidates": []}] * 5)
        pred = ens.predict(self.TASK)
        assert pred is None   # no candidates → None
