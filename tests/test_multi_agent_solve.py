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
    def test_decomposer_skipped_when_stuck_at_zero(self):
        """Decomposer is NOT invoked when completely stuck at 0/N correct.

        When best_n_correct never exceeds 0, the hypothesis is fundamentally
        wrong — the loop should escalate directly to the next hypothesis instead
        of wasting ~100s on a Decomposer call that cannot help.
        """
        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            n = call_count["n"]
            # All calls produce wrong code — each unique (avoids dedup block)
            return (
                f"```python\n"
                f"def transform(input_grid):\n"
                f"    # attempt {n}\n"
                f"    return input_grid * 0\n"
                f"```"
            )

        critic_seq = (
            [{"route": ROUTE_HYPOTHESIZER, "feedback": "wrong hypothesis"}]
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
        # Decomposer must NOT have been called — 0/N means wrong hypothesis
        agent._decomposer.decompose.assert_not_called()

    def test_decomposer_called_with_partial_progress(self):
        """Decomposer fires when there is partial progress (n_correct > 0) but stagnation."""
        from unittest.mock import patch

        call_count = {"n": 0}

        def code_side_effect(*a, **kw):
            call_count["n"] += 1
            n = call_count["n"]
            return (
                f"```python\n"
                f"def transform(input_grid):\n"
                f"    # attempt {n}\n"
                f"    return input_grid * 0\n"
                f"```"
            )

        critic_seq = (
            [{"route": ROUTE_HYPOTHESIZER, "feedback": "wrong hypothesis"}]
            + [{"route": ROUTE_CODER, "feedback": "fix"}] * 10
        )

        agent = make_agent(
            code_responses=code_side_effect,
            critic_responses=critic_seq,
            max_cycles=20,
            use_decomposer=True,
            use_verifier=False,
        )

        # Decomposer fires when n_correct drops to 0 AFTER best_n_correct > 0
        # (regression). Sequence: hyp0 call → ROUTE_HYPOTHESIZER; then at hyp1:
        #   eval 1: n_correct=1 (partial) → best_n_correct becomes 1
        #   eval 2: n_correct=0 (regression) → no_improve_count=1
        #   eval 3: n_correct=0 → no_improve_count=2 → stuck=True, best>0 → Decomposer
        eval_seq = [
            # hyp 0, attempt 1: n_correct=1 (Critic → hypothesizer from critic_seq)
            {"all_correct": False, "n_correct": 1, "n_total": 2,
             "pairs": [{"correct": True,  "predicted": np.array([[1,2],[3,4]]),
                        "expected": np.array([[1,2],[3,4]]), "error": None},
                       {"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[5,6],[7,8]]), "error": None}]},
            # hyp 1, attempt 1: n_correct=1 (partial — best_n_correct now 1)
            {"all_correct": False, "n_correct": 1, "n_total": 2,
             "pairs": [{"correct": True,  "predicted": np.array([[1,2],[3,4]]),
                        "expected": np.array([[1,2],[3,4]]), "error": None},
                       {"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[5,6],[7,8]]), "error": None}]},
            # hyp 1, attempt 2: n_correct=0 (regression → no_improve=1)
            {"all_correct": False, "n_correct": 0, "n_total": 2,
             "pairs": [{"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[1,2],[3,4]]), "error": None},
                       {"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[5,6],[7,8]]), "error": None}]},
            # hyp 1, attempt 3: n_correct=0 (no_improve=2 → stuck=True, best>0 → Decomposer)
            {"all_correct": False, "n_correct": 0, "n_total": 2,
             "pairs": [{"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[1,2],[3,4]]), "error": None},
                       {"correct": False, "predicted": np.array([[0,0],[0,0]]),
                        "expected": np.array([[5,6],[7,8]]), "error": None}]},
        ]
        with patch("agents.multi_agent.sandbox.evaluate_code", side_effect=eval_seq * 5):
            agent.solve(SIMPLE_TASK)

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
        """When Verifier fails, the loop continues and Coder gets the verifier feedback.

        The Coder must produce a unique code on each call so the deduplication
        check does not block the second all-correct pass from reaching the Verifier.
        """
        ver_calls = {"n": 0}
        cod_calls = {"n": 0}

        def verifier_side_effect(*a, **kw):
            ver_calls["n"] += 1
            if ver_calls["n"] == 1:
                return {"passes": False, "issues": "Hardcoded shape", "suggestion": "Use dynamic shape"}
            return {"passes": True, "issues": "", "suggestion": ""}

        def coder_side_effect(*a, **kw):
            # Each call produces functionally identical but textually unique code
            # so the deduplication check doesn't prevent the second verifier call.
            cod_calls["n"] += 1
            return (
                f"```python\n"
                f"def transform(input_grid):\n"
                f"    # v{cod_calls['n']}\n"
                f"    return input_grid.copy()\n"
                f"```"
            )

        agent = make_agent(
            code_responses=coder_side_effect,
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


# ---------------------------------------------------------------------------
# prior_coder_failures passed to Coder after Critic→coder route
# ---------------------------------------------------------------------------

class TestPriorCoderFailures:
    """Verify prior_coder_failures accumulates and is passed to Coder.generate()."""

    def _make_failing_task(self):
        """Task where WRONG_CODE produces wrong output."""
        return {
            "train": [
                {"input":  np.array([[1, 2], [3, 4]], dtype=np.int32),
                 "output": np.array([[1, 2], [3, 4]], dtype=np.int32)},
            ],
            "test": [
                {"input": np.array([[1, 2], [3, 4]], dtype=np.int32)},
            ],
        }

    def test_prior_failures_passed_after_critic_coder_route(self):
        """After Critic routes to coder, next Coder call receives prior_failures."""
        agent = make_agent(
            code_responses=[
                "```python\n" + WRONG_CODE + "\n```",   # first attempt → fails
                "```python\n" + IDENTITY_CODE + "\n```", # second attempt → succeeds
            ],
            critic_responses=[
                {"route": ROUTE_CODER, "feedback": "fix the dimensions"},
            ],
            max_cycles=9,
        )
        agent.solve(self._make_failing_task())

        # The second Coder call should have received prior_failures
        calls = agent._coder.generate.call_args_list
        assert len(calls) >= 2
        _, second_kwargs = calls[1]
        prior = second_kwargs.get("prior_failures")
        assert prior is not None and len(prior) >= 1
        # The snippet from the first attempt should be in the list
        first_snippet, first_feedback = prior[0]
        assert "transform" in first_snippet
        assert "fix the dimensions" in first_feedback

    def test_prior_failures_reset_on_hypothesis_change(self):
        """When Critic routes to hypothesizer, prior_failures resets for new hyp."""
        agent = make_agent(
            code_responses=[
                "```python\n" + WRONG_CODE + "\n```",    # hyp 0, attempt 1
                "```python\n" + WRONG_CODE + "\n```",    # hyp 0, attempt 2 (stuck)
                "```python\n" + WRONG_CODE + "\n```",    # hyp 0, attempt 3 (stuck)
                "```python\n" + IDENTITY_CODE + "\n```", # hyp 1 → success
            ],
            critic_responses=[
                {"route": ROUTE_HYPOTHESIZER, "feedback": "wrong rule"},
            ],
            max_cycles=12,
        )
        agent.solve(self._make_failing_task())

        calls = agent._coder.generate.call_args_list
        # Find the call that corresponds to hyp 1 (the one that gets IDENTITY_CODE)
        # It should have prior_failures=None or [] (fresh slate)
        last_call = calls[-1]
        _, last_kwargs = last_call
        prior = last_kwargs.get("prior_failures")
        # After hypothesis reset, prior_failures must be empty/None
        assert prior is None or prior == []

    def test_first_coder_call_has_no_prior_failures(self):
        """The very first Coder call has no prior failure history."""
        agent = make_agent(
            code_responses=["```python\n" + IDENTITY_CODE + "\n```"],
            max_cycles=5,
        )
        agent.solve(self._make_failing_task())

        first_call = agent._coder.generate.call_args_list[0]
        _, first_kwargs = first_call
        prior = first_kwargs.get("prior_failures")
        assert prior is None or prior == []


# ---------------------------------------------------------------------------
# Budget-aware loop (task_timeout parameter) — lines 1131-1134, 1401
# ---------------------------------------------------------------------------

class TestBudgetAwareLoop:
    def test_budget_exhausted_skips_hypothesizer(self):
        """When task_timeout is nearly elapsed, Hypothesizer is not called."""
        agent = make_agent(max_cycles=9, use_decomposer=False, use_verifier=False)
        # A tiny timeout forces _budget_ok() to return False immediately
        result = agent.solve(SIMPLE_TASK, task_timeout=0.0001)
        # Hypothesizer never called
        agent._hypothesizer.generate.assert_not_called()
        assert result["success"] is False

    def test_budget_exhausted_skips_critic(self):
        """When budget runs out just before the Critic, loop exits cleanly."""
        import itertools

        # task_timeout=200; _budget_ok() returns False when elapsed >= 90
        # Provide unlimited time values: first call returns start (0s elapsed),
        # all subsequent calls return start+150 (budget exhausted).
        T0 = 1000.0
        time_vals = itertools.chain([T0], itertools.repeat(T0 + 150))

        with patch("agents.multi_agent.time") as mock_time:
            mock_time.time.side_effect = time_vals
            agent = make_agent(
                code_responses=["```python\n" + WRONG_CODE + "\n```"] * 20,
                max_cycles=9,
                use_decomposer=False,
                use_verifier=False,
            )
            result = agent.solve(SIMPLE_TASK, task_timeout=200)

        agent._critic.analyze.assert_not_called()
        assert result["success"] is False

    def test_partial_best_set_during_solve(self):
        """self._partial_best is updated whenever n_correct improves."""
        agent = make_agent(use_verifier=False, use_decomposer=False)
        agent.solve(SIMPLE_TASK)
        # IDENTITY_CODE passes all pairs → best code should be stored
        assert agent._partial_best.get("code") is not None
        assert agent._partial_best.get("n_correct", -1) >= 0


# ---------------------------------------------------------------------------
# Hypothesizer exception → break (lines 1133–1137)
# ---------------------------------------------------------------------------

class TestHypothesizerException:
    def test_hypothesizer_exception_breaks_loop(self):
        """If Hypothesizer raises, solve() breaks and returns failure."""
        agent = make_agent(max_cycles=9, use_decomposer=False, use_verifier=False)
        agent._hypothesizer.generate.side_effect = ConnectionError("model offline")

        result = agent.solve(SIMPLE_TASK)

        assert result["success"] is False
        assert result["code"] is None or result["code"] == ""
        # Error must be recorded in the log
        hyp_errors = [e for e in result["log"] if e.get("agent") == "hypothesizer"
                      and "error" in e]
        assert len(hyp_errors) >= 1

    def test_hypothesizer_empty_response_continues(self):
        """An un-parseable Hypothesizer response is skipped (continue, not break)."""
        call_n = {"n": 0}

        def hyp_side(*a, **kw):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return ""   # un-parseable → no hypotheses
            return THREE_HYPS

        agent = make_agent(max_cycles=9, use_decomposer=False, use_verifier=False)
        agent._hypothesizer.generate.side_effect = hyp_side

        result = agent.solve(SIMPLE_TASK)
        # Second call returns valid hypotheses → should still succeed
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Verifier exception → fallback pass (lines 1294–1299)
# ---------------------------------------------------------------------------

class TestVerifierException:
    def test_verifier_exception_treated_as_pass(self):
        """A Verifier exception is caught and treated as passes=True."""
        agent = make_agent(
            use_verifier=True,
            use_decomposer=False,
        )
        agent._verifier.verify.side_effect = RuntimeError("verifier crashed")

        result = agent.solve(SIMPLE_TASK)
        # Despite the exception, solution is accepted
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Debug mode print paths (line 1456)
# ---------------------------------------------------------------------------

class TestDebugMode:
    def test_debug_true_prints_critic_route(self, capsys):
        """With debug=True, solve() prints 'Critic → <route>'."""
        agent = make_agent(
            code_responses=[
                "```python\n" + WRONG_CODE + "\n```",
                "```python\n" + IDENTITY_CODE + "\n```",
            ],
            critic_responses=[
                {"route": ROUTE_CODER, "feedback": "fix it"},
            ],
            max_cycles=9,
            use_decomposer=False,
            use_verifier=False,
        )
        agent.debug = True
        agent.solve(SIMPLE_TASK)

        captured = capsys.readouterr()
        assert "Critic" in captured.out

    def test_debug_true_prints_hypothesizer_count(self, capsys):
        """With debug=True, Hypothesizer count is printed."""
        agent = make_agent(use_decomposer=False, use_verifier=False)
        agent.debug = True
        agent.solve(SIMPLE_TASK)
        captured = capsys.readouterr()
        assert "hypothesis" in captured.out.lower()


# ---------------------------------------------------------------------------
# Critic grid-comparison context (lines 1408–1409)
# ---------------------------------------------------------------------------

class TestCriticGridComparison:
    def test_critic_receives_grid_comparison_when_pair0_fails(self):
        """When pair 0 fails, Critic.analyze() receives a spatial diff with
        grid comparison text appended."""
        agent = make_agent(
            code_responses=[
                "```python\n" + WRONG_CODE + "\n```",
                "```python\n" + IDENTITY_CODE + "\n```",
            ],
            critic_responses=[{"route": ROUTE_CODER, "feedback": "fix"}],
            max_cycles=9,
            use_decomposer=False,
            use_verifier=False,
        )
        agent.solve(SIMPLE_TASK)

        # The spatial_diff argument passed to Critic should contain "Training pair 0"
        assert agent._critic.analyze.called
        call_args = agent._critic.analyze.call_args
        # 4th positional arg is spatial_diff
        spatial_diff_arg = call_args[0][3] if call_args[0] else call_args[1].get("spatial_diff", "")
        assert "Training pair 0" in spatial_diff_arg


# ---------------------------------------------------------------------------
# _best_fitness tracked correctly in failure return (line ~1472)
# ---------------------------------------------------------------------------

class TestBestFitnessReturn:
    def test_failure_result_has_correct_best_code(self):
        """When solve() fails, 'code' in result is the best code seen."""
        # Use a task where training and test outputs differ from WRONG_CODE
        partial_task = {
            "train": [
                {"input":  np.array([[1]], dtype=np.int32),
                 "output": np.array([[1]], dtype=np.int32)},
                {"input":  np.array([[2]], dtype=np.int32),
                 "output": np.array([[0]], dtype=np.int32)},  # WRONG_CODE gets 0 here
            ],
            "test": [{"input": np.array([[3]], dtype=np.int32)}],
        }

        call_n = {"n": 0}
        def coder_side(*a, **kw):
            call_n["n"] += 1
            # attempt 1: IDENTITY_CODE (gets 1/2 correct)
            # attempt 2+: WRONG_CODE (gets 0/2 — regression)
            if call_n["n"] == 1:
                return "```python\ndef transform(input_grid):\n    return input_grid.copy()\n```"
            return f"```python\ndef transform(input_grid):\n    # v{call_n['n']}\n    return input_grid * 0\n```"

        agent = make_agent(
            code_responses=coder_side,
            critic_responses=[{"route": ROUTE_CODER, "feedback": "not right"}] * 20,
            max_cycles=9,
            use_decomposer=False,
            use_verifier=False,
        )
        result = agent.solve(partial_task)

        assert result["success"] is False
        # Best code (1/2 correct) should be stored, not the later 0/2 code
        assert result["code"] is not None
        assert "input_grid.copy()" in result["code"]
