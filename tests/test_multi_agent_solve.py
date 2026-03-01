"""Tests for MultiAgent.solve() and Orchestrator.solve().

Uses mocked Hypothesizer / Coder / Critic to test the orchestration logic
(routing, cycle counting, best-code tracking) without LLM calls.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agents.multi_agent import MultiAgent
from agents.orchestrator import Orchestrator
from agents.roles import ROUTE_HYPOTHESIZER, ROUTE_CODER


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SIMPLE_TASK = {
    "train": [
        {"input":  np.array([[1, 2], [3, 4]], dtype=np.int32),
         "output": np.array([[1, 2], [3, 4]], dtype=np.int32)},
        {"input":  np.array([[5, 6], [7, 8]], dtype=np.int32),
         "output": np.array([[5, 6], [7, 8]], dtype=np.int32)},
    ],
    "test": [
        {"input":  np.array([[9, 0], [0, 9]], dtype=np.int32),
         "output": np.array([[9, 0], [0, 9]], dtype=np.int32)},
    ],
}

IDENTITY_CODE = "def transform(input_grid):\n    return input_grid.copy()"
WRONG_CODE    = "def transform(input_grid):\n    return input_grid * 0"

THREE_HYPS = (
    "1. The output grid is identical to the input — no transformation is applied at all.\n\n"
    "2. Each row is reversed horizontally to produce a mirror-image of the input grid.\n\n"
    "3. The grid is rotated 90 degrees counter-clockwise to produce the final output."
)


def make_agent(
    hyp_responses=None,
    code_responses=None,
    critic_responses=None,
    decomposer_responses=None,
    verifier_responses=None,
    max_cycles: int = 9,
    use_decomposer: bool = True,
    use_verifier: bool = True,
) -> MultiAgent:
    agent = MultiAgent.__new__(MultiAgent)
    agent.max_cycles               = max_cycles
    agent.debug                    = False
    agent.backend                  = "ollama"
    agent.hypothesizer_model       = "mock"
    agent.coder_model              = "mock"
    agent.critic_model             = "mock"
    agent.model                    = "mock"
    agent.hypothesizer_temperature = 0.6
    agent.coder_temperature        = 0.1
    agent.critic_temperature       = 0.2
    agent.hypothesizer_max_tokens  = 1024
    agent.coder_max_tokens         = 1024
    agent.critic_max_tokens        = 1024
    agent.use_decomposer           = use_decomposer
    agent.use_verifier             = use_verifier

    hyp = MagicMock()
    if hyp_responses:
        hyp.generate.side_effect = hyp_responses
    else:
        hyp.generate.return_value = THREE_HYPS

    coder = MagicMock()
    if code_responses:
        coder.generate.side_effect = code_responses
    else:
        coder.generate.return_value = (
            "```python\n" + IDENTITY_CODE + "\n```"
        )

    critic = MagicMock()
    if critic_responses:
        critic.analyze.side_effect = critic_responses
    else:
        critic.analyze.return_value = {
            "route":    ROUTE_CODER,
            "feedback": "fix the code",
        }

    decomposer = MagicMock()
    if decomposer_responses:
        decomposer.decompose.side_effect = decomposer_responses
    else:
        decomposer.decompose.return_value = (
            "1. Identify the pattern.\n2. Apply the transformation.\n3. Return result."
        )

    verifier = MagicMock()
    if verifier_responses:
        verifier.verify.side_effect = verifier_responses
    else:
        verifier.verify.return_value = {"passes": True, "issues": "", "suggestion": ""}

    agent._hypothesizer = hyp
    agent._coder        = coder
    agent._critic       = critic
    agent._decomposer   = decomposer
    agent._verifier     = verifier
    return agent


# ---------------------------------------------------------------------------
# MultiAgent.solve()
# ---------------------------------------------------------------------------

class TestMultiAgentSolveSuccess:
    def test_returns_success_on_correct_code(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is True

    def test_correct_code_in_result(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert result["code"] is not None
        assert "def transform" in result["code"]

    def test_test_correct_flag(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert result["test_correct"] is True

    def test_n_cycles_tracked(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert result["n_cycles"] >= 2   # at least hypothesizer + coder

    def test_log_has_entries(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert len(result["log"]) > 0

    def test_hypothesizer_called_once_on_first_pass(self):
        agent  = make_agent()
        agent.solve(SIMPLE_TASK)
        agent._hypothesizer.generate.assert_called_once()


class TestMultiAgentSolveFailure:
    def test_returns_failure_on_wrong_code(self):
        code_resp = "```python\n" + WRONG_CODE + "\n```"
        agent = make_agent(
            code_responses=[code_resp] * 10,
            critic_responses=[{"route": ROUTE_CODER, "feedback": "still wrong"}] * 10,
            max_cycles=6,
        )
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False

    def test_best_code_still_returned_on_failure(self):
        code_resp = "```python\n" + WRONG_CODE + "\n```"
        agent = make_agent(
            code_responses=[code_resp] * 10,
            critic_responses=[{"route": ROUTE_CODER, "feedback": "fix"}] * 10,
            max_cycles=5,
        )
        result = agent.solve(SIMPLE_TASK)
        # best_code is set even when unsuccessful
        assert result["code"] is not None

    def test_hypothesizer_error_breaks_loop(self):
        agent = make_agent(max_cycles=9)
        agent._hypothesizer.generate.side_effect = ConnectionError("offline")
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False
        error_entries = [e for e in result["log"] if "error" in e]
        assert len(error_entries) >= 1


class TestMultiAgentCriticRouting:
    def test_critic_routes_to_hypothesizer_tries_next(self):
        """When Critic says hypothesizer, the agent should advance hyp_index."""
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "```python\n" + WRONG_CODE + "\n```"
            return "```python\n" + IDENTITY_CODE + "\n```"

        agent = make_agent(
            code_responses=code_side_effect,
            critic_responses=[
                {"route": ROUTE_HYPOTHESIZER, "feedback": "wrong hypothesis"},
            ],
            max_cycles=9,
        )
        result = agent.solve(SIMPLE_TASK)
        # After critic routes to hypothesizer, next hypothesis is tried
        # Eventually identity code is generated → success
        assert result["success"] is True

    def test_critic_routes_to_coder_retries_same_hyp(self):
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return "```python\n" + WRONG_CODE + "\n```"
            return "```python\n" + IDENTITY_CODE + "\n```"

        agent = make_agent(
            code_responses=code_side_effect,
            critic_responses=[{"route": ROUTE_CODER, "feedback": "fix crop bounds"}],
            max_cycles=9,
        )
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is True


class TestMultiAgentNoCodeBlock:
    def test_no_code_block_advances_hypothesis(self):
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "I cannot write code."   # no code block
            return "```python\n" + IDENTITY_CODE + "\n```"

        agent = make_agent(code_responses=code_side_effect, max_cycles=9)
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Orchestrator.solve()
# ---------------------------------------------------------------------------

class TestOrchestratorSolve:
    def _make_orchestrator(self, **kwargs) -> Orchestrator:
        orch = Orchestrator.__new__(Orchestrator)
        orch.max_cycles               = kwargs.get("max_cycles", 9)
        orch.debug                    = False
        orch.backend                  = "ollama"
        orch.hypothesizer_model       = "mock"
        orch.coder_model              = "mock"
        orch.critic_model             = "mock"
        orch.model                    = "mock"
        orch.n_hypotheses             = 3
        orch.max_retries              = 2
        orch.hypothesizer_temperature = 0.6
        orch.coder_temperature        = 0.1
        orch.critic_temperature       = 0.2
        orch.hypothesizer_max_tokens  = 1024
        orch.coder_max_tokens         = 1024
        orch.critic_max_tokens        = 1024
        orch.use_decomposer           = False
        orch.use_verifier             = False

        hyp = MagicMock()
        hyp.generate.return_value = THREE_HYPS

        coder = MagicMock()
        coder.generate.return_value = "```python\n" + IDENTITY_CODE + "\n```"

        critic = MagicMock()
        critic.analyze.return_value = {"route": ROUTE_CODER, "feedback": "fix"}

        decomposer = MagicMock()
        decomposer.decompose.return_value = "1. Step one.\n2. Step two."

        verifier = MagicMock()
        verifier.verify.return_value = {"passes": True, "issues": "", "suggestion": ""}

        orch._hypothesizer = hyp
        orch._coder        = coder
        orch._critic       = critic
        orch._decomposer   = decomposer
        orch._verifier     = verifier
        return orch

    def test_candidates_key_present(self):
        orch   = self._make_orchestrator()
        result = orch.solve(SIMPLE_TASK)
        assert "candidates" in result

    def test_correct_code_added_to_candidates(self):
        orch   = self._make_orchestrator()
        result = orch.solve(SIMPLE_TASK)
        # At least one correct candidate should be pooled
        if result["success"]:
            assert len(result["candidates"]) >= 1
            for cand in result["candidates"]:
                assert "code" in cand

    def test_success_flag(self):
        orch   = self._make_orchestrator()
        result = orch.solve(SIMPLE_TASK)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# result schema checks
# ---------------------------------------------------------------------------

class TestResultSchema:
    def test_multi_agent_result_keys(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        assert {"success", "code", "test_correct", "n_cycles", "log"} <= set(result.keys())

    def test_log_entries_have_cycle_and_agent(self):
        agent  = make_agent()
        result = agent.solve(SIMPLE_TASK)
        for entry in result["log"]:
            assert "cycle" in entry or "phase" in entry


# ---------------------------------------------------------------------------
# TestDecomposerIntegration
# ---------------------------------------------------------------------------

class TestDecomposerIntegration:
    def test_decomposer_called_when_stuck_at_zero(self):
        """Decomposer invoked after stagnation (n_correct=0, no_improve_count>=2).

        The stagnation state can only accumulate at hyp_index > 0, because the
        loop resets prev_n_correct / no_improve_count when hyp_index == 0.
        We therefore route the first Critic call to HYPOTHESIZER (advancing
        hyp_index to 1), then route subsequent calls to CODER so stagnation
        can accumulate.
        """
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            # First 5 calls produce wrong code; from 6th onward produce correct code
            if call_count["n"] <= 5:
                return "```python\n" + WRONG_CODE + "\n```"
            return "```python\n" + IDENTITY_CODE + "\n```"

        critic_seq = (
            [{"route": ROUTE_HYPOTHESIZER, "feedback": "wrong hypothesis"}]  # advance hyp_index→1
            + [{"route": ROUTE_CODER, "feedback": "fix"}] * 10
        )

        agent = make_agent(
            code_responses=code_side_effect,
            critic_responses=critic_seq,
            max_cycles=20,
            use_decomposer=True,
            use_verifier=False,
        )
        agent.solve(SIMPLE_TASK)
        # Decomposer must have been called at least once
        assert agent._decomposer.decompose.called

    def test_decomposer_skipped_when_use_decomposer_false(self):
        """When use_decomposer=False, Decomposer is never called."""
        agent = make_agent(
            code_responses=["```python\n" + WRONG_CODE + "\n```"] * 20,
            critic_responses=[{"route": ROUTE_CODER, "feedback": "fix"}] * 20,
            max_cycles=8,
            use_decomposer=False,
            use_verifier=False,
        )
        agent.solve(SIMPLE_TASK)
        agent._decomposer.decompose.assert_not_called()

    def test_decomposer_replaces_hypothesis_and_loop_continues(self):
        """After Decomposer runs, the loop should continue with the new hypothesis."""
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return "```python\n" + WRONG_CODE + "\n```"
            return "```python\n" + IDENTITY_CODE + "\n```"

        agent = make_agent(
            code_responses=code_side_effect,
            max_cycles=15,
            use_decomposer=True,
            use_verifier=False,
        )
        result = agent.solve(SIMPLE_TASK)
        # Loop must continue after decomposer (eventually succeeds)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# TestVerifierIntegration
# ---------------------------------------------------------------------------

class TestVerifierIntegration:
    def test_verifier_called_when_all_correct(self):
        """Verifier should be invoked when all training pairs pass."""
        agent = make_agent(use_verifier=True)
        agent.solve(SIMPLE_TASK)
        assert agent._verifier.verify.called

    def test_verifier_pass_accepts_solution(self):
        """When Verifier passes, solve() returns success."""
        agent = make_agent(
            verifier_responses=[{"passes": True, "issues": "", "suggestion": ""}],
            use_verifier=True,
        )
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is True

    def test_verifier_fail_sends_feedback_to_coder(self):
        """When Verifier fails, the loop continues and Coder gets the verifier feedback."""
        call_count = {"n": 0}

        def verifier_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"passes": False, "issues": "Hardcoded shape", "suggestion": "Use dynamic shape"}
            return {"passes": True, "issues": "", "suggestion": ""}

        agent = make_agent(
            verifier_responses=verifier_side_effect,
            max_cycles=15,
            use_verifier=True,
            use_decomposer=False,
        )
        result = agent.solve(SIMPLE_TASK)
        # Verifier called at least twice (fail then pass)
        assert agent._verifier.verify.call_count >= 2
        assert result["success"] is True

    def test_verifier_skipped_when_use_verifier_false(self):
        """When use_verifier=False, Verifier is never called."""
        agent = make_agent(use_verifier=False, use_decomposer=False)
        agent.solve(SIMPLE_TASK)
        agent._verifier.verify.assert_not_called()


# ---------------------------------------------------------------------------
# MultiAgent.predict() and _evaluate_test() (lines 949-960)
# ---------------------------------------------------------------------------

class TestMultiAgentPredictAndEvaluate:
    def test_predict_returns_grid_on_success(self):
        agent  = make_agent()
        result = agent.predict(SIMPLE_TASK)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_predict_returns_none_when_no_code(self):
        agent = make_agent(code_responses=["no code here"] * 20,
                           max_cycles=9, use_decomposer=False)
        result = agent.predict(SIMPLE_TASK)
        assert result is None

    def test_evaluate_test_correct(self):
        agent = make_agent()
        test_pair = SIMPLE_TASK["test"][0]
        assert agent._evaluate_test(IDENTITY_CODE, test_pair) is True

    def test_evaluate_test_wrong_code(self):
        agent = make_agent()
        test_pair = SIMPLE_TASK["test"][0]
        assert agent._evaluate_test(WRONG_CODE, test_pair) is False

    def test_evaluate_test_sandbox_error(self):
        agent     = make_agent()
        test_pair = SIMPLE_TASK["test"][0]
        assert agent._evaluate_test("def f(: pass", test_pair) is False


# ---------------------------------------------------------------------------
# Exception paths in solve() (lines 791-800, 915-919)
# ---------------------------------------------------------------------------

class TestSolveExceptionPaths:
    def test_coder_exception_advances_hypothesis(self):
        # Coder raises on first call, then returns valid code
        agent = make_agent(
            code_responses=[
                Exception("LLM timeout"),
                "```python\n" + IDENTITY_CODE + "\n```",
            ],
            max_cycles=9,
        )
        result = agent.solve(SIMPLE_TASK)
        # Should still succeed on the second attempt
        assert result.get("code") is not None

    def test_critic_exception_advances_hypothesis(self):
        # Critic raises → should advance hypothesis index, not crash
        critic_resp = [
            Exception("critic down"),
            {"route": ROUTE_CODER, "feedback": "ok"},
        ]
        agent = make_agent(
            code_responses=[
                "```python\n" + WRONG_CODE + "\n```",
                "```python\n" + IDENTITY_CODE + "\n```",
            ],
            critic_responses=critic_resp,
            max_cycles=9,
        )
        result = agent.solve(SIMPLE_TASK)
        # Loop continues after critic exception
        assert "log" in result
