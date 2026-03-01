"""Tests for agents/ensemble.py — voting helpers and Ensemble.solve()."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agents.ensemble import (
    Ensemble,
    _majority_vote,
    _pixel_majority_vote,
    _avg_fitness,
    _vote_summary,
    _grids_equal,
    _check_prediction,
)


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
        ens.near_miss_threshold      = 0.85
        ens.near_miss_weight         = 0.5
        ens.max_corrections          = 2
        ens.use_correction           = False   # disabled by default so existing tests pass
        ens._correction_client       = MagicMock()

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

    def _orch_result(self, code: str | None, fitness: float = 1.0, perfect: bool = True) -> dict:
        if code is None:
            return {"candidates": []}
        return {"candidates": [{"code": code, "fitness": fitness, "perfect": perfect}]}

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


# ---------------------------------------------------------------------------
# _pixel_majority_vote
# ---------------------------------------------------------------------------

class TestPixelMajorityVote:
    def test_empty_returns_none(self):
        assert _pixel_majority_vote([]) is None

    def test_single_grid(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = _pixel_majority_vote([g])
        np.testing.assert_array_equal(result, g)

    def test_pixel_majority(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[1, 0], [3, 4]], dtype=np.int32)
        # a and b agree on all pixels except [0,1]; a wins 2-to-1 at [0,1]
        result = _pixel_majority_vote([a, a, b])
        np.testing.assert_array_equal(result, a)

    def test_weighted_vote(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[1, 0], [3, 4]], dtype=np.int32)
        # b has higher weight so wins at [0,1] even though a has more grids
        result = _pixel_majority_vote([a, b], weights=[0.3, 1.0])
        np.testing.assert_array_equal(result, b)

    def test_different_shapes_uses_most_weighted_shape(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)     # 2×2, weight=2.0
        b = np.array([[9, 8, 7]], dtype=np.int32)          # 1×3, weight=0.5
        # 2×2 shape has higher aggregate weight → b is ignored
        result = _pixel_majority_vote([a, b, a], weights=[1.0, 0.5, 1.0])
        np.testing.assert_array_equal(result, a)


# ---------------------------------------------------------------------------
# _check_prediction
# ---------------------------------------------------------------------------

class TestCheckPrediction:
    TASK = {
        "train": [
            {"input":  np.array([[1, 2]], dtype=np.int32),
             "output": np.array([[1, 2]], dtype=np.int32)},
        ],
        "test": [
            {"input":  np.array([[3, 4]], dtype=np.int32)},
        ],
    }
    PREDICTION = np.array([[3, 4]], dtype=np.int32)

    def _make_client(self, response: str):
        client = MagicMock()
        client.generate.return_value = response
        return client

    def test_accept_response(self):
        client = self._make_client("VERDICT: ACCEPT\nREASON: looks good")
        result = _check_prediction(self.PREDICTION, self.TASK, client)
        assert result["accept"] is True
        assert result["reason"] == "looks good"

    def test_reject_response(self):
        client = self._make_client("VERDICT: REJECT\nREASON: wrong color")
        result = _check_prediction(self.PREDICTION, self.TASK, client)
        assert result["accept"] is False
        assert result["reason"] == "wrong color"

    def test_exception_is_fail_safe(self):
        client = MagicMock()
        client.generate.side_effect = RuntimeError("connection error")
        result = _check_prediction(self.PREDICTION, self.TASK, client)
        assert result["accept"] is True
        assert result["reason"] == ""


# ---------------------------------------------------------------------------
# Candidate filtering (perfect_pool vs near_miss_pool)
# ---------------------------------------------------------------------------

class TestCandidateFiltering(TestEnsembleSolve):
    """Tests for near-miss threshold filtering in Ensemble.solve()."""

    TASK = TestEnsembleSolve.TASK
    IDENTITY_CODE = TestEnsembleSolve.IDENTITY_CODE

    def test_perfect_candidate_in_perfect_pool(self):
        results = [self._orch_result(self.IDENTITY_CODE, fitness=1.0, perfect=True)]
        ens = self._make_ensemble(results)
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["success"] is True
        assert result["candidates"][0]["weight"] == 1.0

    def test_near_miss_included(self):
        nm_code = "def transform(input_grid):\n    return input_grid.copy()  # near"
        results = [self._orch_result(nm_code, fitness=0.9, perfect=False)]
        ens = self._make_ensemble(results)
        ens.near_miss_threshold = 0.85
        ens.near_miss_weight    = 0.5
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["success"] is True
        assert result["candidates"][0]["weight"] == 0.5

    def test_low_fitness_excluded(self):
        low_code = "def transform(input_grid):\n    return input_grid.copy()  # low"
        results = [self._orch_result(low_code, fitness=0.5, perfect=False)]
        ens = self._make_ensemble(results)
        ens.near_miss_threshold = 0.85
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["success"] is False
        assert result["candidates"] == []


# ---------------------------------------------------------------------------
# Self-correction loop
# ---------------------------------------------------------------------------

class TestSelfCorrectionLoop(TestEnsembleSolve):
    """Tests for the critic-driven self-correction loop in Ensemble.solve()."""

    TASK = TestEnsembleSolve.TASK

    def _make_correction_ensemble(
        self,
        orchestrator_results: list[dict],
        critic_responses: list[str],
        max_corrections: int = 2,
    ) -> Ensemble:
        ens = self._make_ensemble(orchestrator_results)
        ens.use_correction  = True
        ens.max_corrections = max_corrections
        mock_client = MagicMock()
        mock_client.generate.side_effect = critic_responses
        ens._correction_client = mock_client
        return ens

    def test_use_correction_false_skips_critic(self):
        codes = self._make_unique_codes(1)
        results = [self._orch_result(codes[0])]
        ens = self._make_ensemble(results)
        ens.use_correction = False
        ens.max_runs = 1
        ens.solve(self.TASK)
        ens._correction_client.generate.assert_not_called()

    def test_critic_accepts_first_prediction(self):
        codes   = self._make_unique_codes(1)
        results = [self._orch_result(codes[0])]
        ens = self._make_correction_ensemble(
            results,
            critic_responses=["VERDICT: ACCEPT\nREASON: fine"],
        )
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["success"] is True
        assert result["corrections_done"] == 0

    def test_critic_rejects_leading_group_revotes(self):
        # Build two distinct codes that produce different outputs
        code_a = "def transform(input_grid):\n    return input_grid.copy()  # A"
        code_b = "def transform(input_grid):\n    import numpy as np; return np.zeros_like(input_grid)"

        # code_a appears twice, code_b once
        results = [
            {"candidates": [
                {"code": code_a, "fitness": 1.0, "perfect": True},
                {"code": code_a + " ", "fitness": 1.0, "perfect": True},  # distinct code, same output
                {"code": code_b, "fitness": 1.0, "perfect": True},
            ]}
        ]
        ens = self._make_correction_ensemble(
            results,
            # first REJECT → exclude leading group, second ACCEPT
            critic_responses=[
                "VERDICT: REJECT\nREASON: inconsistent",
                "VERDICT: ACCEPT\nREASON: ok",
            ],
            max_corrections=2,
        )
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["corrections_done"] >= 1

    def test_max_corrections_cap(self):
        codes = self._make_unique_codes(3)
        results = [{"candidates": [
            {"code": c, "fitness": 1.0, "perfect": True} for c in codes
        ]}]
        # Critic always rejects
        ens = self._make_correction_ensemble(
            results,
            critic_responses=["VERDICT: REJECT\nREASON: bad"] * 10,
            max_corrections=2,
        )
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        assert result["corrections_done"] <= 2

    @patch("agents.ensemble.sandbox.execute")
    def test_fallback_leader_exclusion_triggers(self, mock_execute):
        """When pixel vote produces a grid matching no candidate, the leader group
        is excluded instead (lines 371-375).  With max_corrections=1 the loop
        also hits the corrections_done >= max_corrections break (line 358)."""
        # 4 outputs, each with a 1 in a different corner.
        # Pixel vote at each position: 0 wins with 3 votes → [[0,0],[0,0]]
        # which matches NONE of the 4 candidates → fallback path fires.
        A = np.array([[1, 0], [0, 0]], dtype=np.int32)
        B = np.array([[0, 1], [0, 0]], dtype=np.int32)
        C = np.array([[0, 0], [1, 0]], dtype=np.int32)
        D = np.array([[0, 0], [0, 1]], dtype=np.int32)
        mock_execute.side_effect = [(A, None), (B, None), (C, None), (D, None)]

        codes = [f"def transform(g):\n    return g  # v{i}" for i in range(4)]
        result_dict = {"candidates": [
            {"code": codes[i], "fitness": 1.0, "perfect": True} for i in range(4)
        ]}
        ens = self._make_correction_ensemble(
            [result_dict],
            critic_responses=["VERDICT: REJECT\nREASON: wrong"] * 10,
            max_corrections=1,
        )
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        # Fallback exclusion ran (corrections_done==1) then max_corrections break fired
        assert result["corrections_done"] == 1


# ---------------------------------------------------------------------------
# _avg_fitness
# ---------------------------------------------------------------------------

class TestCheckPredictionNoVerdict:
    """Cover the `if verdict_match is None` fail-safe branch (line 195)."""
    TASK = {
        "train": [{"input": np.array([[1]], dtype=np.int32),
                   "output": np.array([[1]], dtype=np.int32)}],
        "test":  [{"input": np.array([[2]], dtype=np.int32)}],
    }

    def test_no_verdict_in_response_returns_accept(self):
        client = MagicMock()
        client.generate.return_value = "I cannot determine the answer."
        pred = np.array([[2]], dtype=np.int32)
        result = _check_prediction(pred, self.TASK, client)
        assert result["accept"] is True
        assert result["reason"] == ""


class TestMaxCorrectionsBroken:
    """Cover corrections_done >= max_corrections break (line 358)."""

    TASK = TestSelfCorrectionLoop.TASK

    def test_second_iteration_hits_max_corrections_break(self):
        """Two distinct outputs; critic rejects first prediction (identity wins),
        zeros remains, second loop iteration hits corrections_done >= max_corrections."""
        ens = TestSelfCorrectionLoop()._make_correction_ensemble(
            [{"candidates": [
                {"code": "def transform(g):\n    return g.copy()  # id",
                 "fitness": 1.0, "perfect": True},
                {"code": "def transform(g):\n    import numpy as np; return np.zeros_like(g)",
                 "fitness": 1.0, "perfect": True},
            ]}],
            critic_responses=["VERDICT: REJECT\nREASON: bad"] * 5,
            max_corrections=1,
        )
        ens.max_runs = 1
        result = ens.solve(self.TASK)
        # First iteration: identity wins, critic rejects, zeros remain.
        # Second iteration: corrections_done(1) >= max_corrections(1) → break.
        assert result["corrections_done"] == 1


class TestAvgFitness:
    def test_empty_pairs_returns_zero(self):
        assert _avg_fitness([]) == 0.0

    def test_pairs_with_no_expected_returns_zero(self):
        pairs = [{"predicted": np.array([[1]], dtype=np.int32)}]
        assert _avg_fitness(pairs) == 0.0

    def test_perfect_match_returns_one(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        pairs = [{"predicted": g, "expected": g}]
        assert _avg_fitness(pairs) == pytest.approx(1.0)

    def test_averages_across_pairs(self):
        g = np.array([[1]], dtype=np.int32)
        h = np.array([[2]], dtype=np.int32)
        pairs = [
            {"predicted": g, "expected": g},   # fitness=1.0
            {"predicted": g, "expected": h},   # fitness<1.0
        ]
        result = _avg_fitness(pairs)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# Ensemble.__init__ constructor
# ---------------------------------------------------------------------------

class TestEnsembleConstructor:
    def test_real_init_sets_new_params(self):
        with patch("agents.ensemble.Orchestrator") as MockOrch, \
             patch("agents.ensemble.LLMClient"):
            mock_orch_inst = MagicMock(
                backend="ollama", model="m",
                hypothesizer_model="m", coder_model="m", critic_model="m",
                hypothesizer_temperature=0.6, coder_temperature=0.1,
                critic_temperature=0.2,
                hypothesizer_max_tokens=32768, coder_max_tokens=8192,
                critic_max_tokens=16384,
            )
            MockOrch.return_value = mock_orch_inst
            ens = Ensemble(backend="ollama", model="test-model")
            assert ens.target_candidates == 3
            assert ens.near_miss_threshold == 0.85
            assert ens.near_miss_weight == 0.5
            assert ens.max_corrections == 2
            assert ens.use_correction is True

    def test_custom_params_stored(self):
        with patch("agents.ensemble.Orchestrator") as MockOrch, \
             patch("agents.ensemble.LLMClient"):
            mock_orch_inst = MagicMock(
                backend="ollama", model="m",
                hypothesizer_model="m", coder_model="m", critic_model="m",
                hypothesizer_temperature=0.6, coder_temperature=0.1,
                critic_temperature=0.2,
                hypothesizer_max_tokens=32768, coder_max_tokens=8192,
                critic_max_tokens=16384,
            )
            MockOrch.return_value = mock_orch_inst
            ens = Ensemble(
                near_miss_threshold=0.7,
                near_miss_weight=0.3,
                max_corrections=5,
                use_correction=False,
                target_candidates=7,
                max_runs=10,
            )
            assert ens.near_miss_threshold == 0.7
            assert ens.near_miss_weight == 0.3
            assert ens.max_corrections == 5
            assert ens.use_correction is False
            assert ens.target_candidates == 7
            assert ens.max_runs == 10


# ---------------------------------------------------------------------------
# Ensemble.solve() debug output
# ---------------------------------------------------------------------------

class TestEnsembleSolveDebug(TestEnsembleSolve):
    def test_debug_prints_run_info(self, capsys):
        ens = self._make_ensemble([{"candidates": []}] * 5)
        ens.debug = True
        ens.solve(self.TASK)
        captured = capsys.readouterr()
        assert "[ensemble]" in captured.out

    def test_debug_prints_after_vote(self, capsys):
        codes = self._make_unique_codes(1)
        results = [self._orch_result(codes[0])]
        ens = self._make_ensemble(results)
        ens.debug = True
        ens.max_runs = 1
        ens.solve(self.TASK)
        captured = capsys.readouterr()
        assert "[ensemble]" in captured.out
