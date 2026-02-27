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
    max_cycles: int = 9,
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

    agent._hypothesizer = hyp
    agent._coder        = coder
    agent._critic       = critic
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

        hyp = MagicMock()
        hyp.generate.return_value = THREE_HYPS

        coder = MagicMock()
        coder.generate.return_value = "```python\n" + IDENTITY_CODE + "\n```"

        critic = MagicMock()
        critic.analyze.return_value = {"route": ROUTE_CODER, "feedback": "fix"}

        orch._hypothesizer = hyp
        orch._coder        = coder
        orch._critic       = critic
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
