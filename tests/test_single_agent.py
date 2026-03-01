"""Tests for agents/single_agent.py."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agents.single_agent import SingleAgent


IDENTITY_CODE_RESPONSE = (
    "```python\n"
    "def transform(input_grid):\n"
    "    return input_grid.copy()\n"
    "```"
)

THREE_HYPOTHESES = (
    "1. The output is identical to the input — no transformation is applied whatsoever.\n\n"
    "2. Each row is individually reversed to produce a horizontally mirrored output.\n\n"
    "3. The grid is rotated 90 degrees counter-clockwise to produce the final output."
)

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


class TestSingleAgentConstruction:
    def test_model_is_set(self):
        with patch("agents.single_agent.LLMClient") as mock_cls:
            mock_cls.return_value.model = "test-model"
            agent = SingleAgent(backend="ollama", model="test-model")
        assert agent.model is not None

    def test_backend_stored(self):
        with patch("agents.single_agent.LLMClient") as mock_cls:
            mock_cls.return_value.model = "m"
            agent = SingleAgent(backend="anthropic")
        assert agent.backend == "anthropic"


class TestSingleAgentSolve:
    def _make_agent(self, hyp_response: str, code_response: str) -> SingleAgent:
        agent = SingleAgent.__new__(SingleAgent)
        agent.debug   = False
        agent.backend = "ollama"
        agent.model   = "mock"

        agent._hypothesizer = MagicMock()
        agent._hypothesizer.generate.return_value = hyp_response

        agent._coder = MagicMock()
        agent._coder.generate.return_value = code_response

        return agent

    def test_success_when_code_correct(self):
        agent = self._make_agent(THREE_HYPOTHESES, IDENTITY_CODE_RESPONSE)
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is True
        assert "def transform" in result["code"]

    def test_result_has_required_keys(self):
        agent = self._make_agent(THREE_HYPOTHESES, IDENTITY_CODE_RESPONSE)
        result = agent.solve(SIMPLE_TASK)
        assert "success" in result
        assert "code" in result
        assert "test_correct" in result

    def test_test_correct_set_when_ground_truth_available(self):
        agent = self._make_agent(THREE_HYPOTHESES, IDENTITY_CODE_RESPONSE)
        result = agent.solve(SIMPLE_TASK)
        assert result["test_correct"] is True

    def test_failure_when_code_wrong(self):
        wrong = "```python\ndef transform(g):\n    return g * 0\n```"
        agent = self._make_agent(THREE_HYPOTHESES, wrong)
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False

    def test_hypothesizer_error_returns_failure(self):
        agent = SingleAgent.__new__(SingleAgent)
        agent.debug = False
        agent._hypothesizer = MagicMock()
        agent._hypothesizer.generate.side_effect = ConnectionError("offline")
        agent._coder = MagicMock()
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False

    def test_no_code_in_response_skipped(self):
        agent = self._make_agent(THREE_HYPOTHESES, "I cannot write code right now.")
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False

    def test_predict_returns_grid_on_success(self):
        agent = self._make_agent(THREE_HYPOTHESES, IDENTITY_CODE_RESPONSE)
        pred = agent.predict(SIMPLE_TASK)
        assert isinstance(pred, np.ndarray)

    def test_predict_returns_none_on_failure(self):
        agent = self._make_agent(THREE_HYPOTHESES, "no code here")
        pred = agent.predict(SIMPLE_TASK)
        assert pred is None


class TestSingleAgentCoderException:
    """Cover coder exception path (lines 72-73 in single_agent.py)."""

    def _make_agent_with_coder_error(self) -> "SingleAgent":
        from agents.single_agent import SingleAgent
        from unittest.mock import MagicMock
        agent = SingleAgent.__new__(SingleAgent)
        agent.debug = False

        hyp = MagicMock()
        hyp.generate.return_value = (
            "1. Identity transform — output equals input.\n\n"
            "2. Rotate 90 degrees CCW.\n\n"
            "3. Flip horizontally."
        )
        coder = MagicMock()
        coder.generate.side_effect = Exception("timeout")
        agent._hypothesizer = hyp
        agent._coder = coder
        return agent

    def test_coder_exception_returns_failure(self):
        from tests.test_single_agent import SIMPLE_TASK
        agent = self._make_agent_with_coder_error()
        result = agent.solve(SIMPLE_TASK)
        assert result["success"] is False
