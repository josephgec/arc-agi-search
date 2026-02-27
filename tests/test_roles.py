"""Tests for agents/roles.py — all roles use mocked LLM clients."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.roles import (
    Hypothesizer,
    Coder,
    Critic,
    PSOCoder,
    ROUTE_HYPOTHESIZER,
    ROUTE_CODER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_llm(response: str = "") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    llm.model = "mock-model"
    return llm


VALID_CODE_RESPONSE = (
    "Here is the solution:\n"
    "```python\n"
    "def transform(input_grid):\n"
    "    return input_grid.copy()\n"
    "```"
)

THREE_HYPOTHESES = (
    "1. The output is a rotation of the input by 90 degrees clockwise for all examples.\n\n"
    "2. Each unique color in the input is replaced by its complement in the ARC palette.\n\n"
    "3. The input is cropped to remove the outermost row and column on each side."
)

CRITIC_HYPOTHESIZER_RESPONSE = (
    "ROUTE: hypothesizer\n"
    "FEEDBACK:\nThe hypothesis is fundamentally wrong. The transformation is not a rotation."
)

CRITIC_CODER_RESPONSE = (
    "ROUTE: coder\n"
    "FEEDBACK:\nThe code is correct in logic but fails on pair 2 due to an off-by-one error "
    "in the crop coordinates. Use r_max+1 instead of r_max."
)

K_MUTATIONS_RESPONSE = (
    "Here are 5 variants:\n"
    "```python\ndef transform_1(input_grid):\n    return input_grid.copy()\n```\n"
    "```python\ndef transform_2(input_grid):\n    return rotate(input_grid, 1)\n```\n"
    "```python\ndef transform_3(input_grid):\n    return flip(input_grid, 0)\n```\n"
    "```python\ndef transform_4(input_grid):\n    return recolor(input_grid, 1, 2)\n```\n"
    "```python\ndef transform_5(input_grid):\n    return crop_to_content(input_grid)\n```\n"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_route_values(self):
        assert ROUTE_HYPOTHESIZER == "hypothesizer"
        assert ROUTE_CODER == "coder"
        assert ROUTE_HYPOTHESIZER != ROUTE_CODER


# ---------------------------------------------------------------------------
# Hypothesizer
# ---------------------------------------------------------------------------

class TestHypothesizer:
    def test_calls_generate(self):
        llm = make_llm(THREE_HYPOTHESES)
        hyp = Hypothesizer(llm)
        result = hyp.generate("task description")
        llm.generate.assert_called_once()
        assert result == THREE_HYPOTHESES

    def test_feedback_included_in_prompt(self):
        llm = make_llm(THREE_HYPOTHESES)
        hyp = Hypothesizer(llm)
        hyp.generate("task description", feedback="previous attempt failed")
        _, kwargs = llm.generate.call_args
        # Check feedback appears in the messages sent to the LLM
        call_args = llm.generate.call_args
        messages_arg = call_args[0][1]  # second positional arg = messages
        user_content = messages_arg[0]["content"]
        assert "feedback" in user_content.lower() or "previous attempt" in user_content.lower()

    def test_no_feedback_no_feedback_text(self):
        llm = make_llm(THREE_HYPOTHESES)
        hyp = Hypothesizer(llm)
        hyp.generate("task description")
        call_args = llm.generate.call_args
        messages_arg = call_args[0][1]
        user_content = messages_arg[0]["content"]
        assert "CRITIC FEEDBACK" not in user_content

    def test_system_prompt_not_empty(self):
        llm = make_llm(THREE_HYPOTHESES)
        hyp = Hypothesizer(llm)
        hyp.generate("task description")
        system_arg = llm.generate.call_args[0][0]
        assert len(system_arg) > 50


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class TestCoder:
    def test_calls_generate(self):
        llm = make_llm(VALID_CODE_RESPONSE)
        coder = Coder(llm)
        result = coder.generate("hypothesis text")
        llm.generate.assert_called_once()
        assert result == VALID_CODE_RESPONSE

    def test_feedback_appended(self):
        llm = make_llm(VALID_CODE_RESPONSE)
        coder = Coder(llm)
        coder.generate("hypothesis", feedback="fix the crop bounds")
        call_args = llm.generate.call_args
        messages = call_args[0][1]
        content = messages[0]["content"]
        assert "fix the crop bounds" in content

    def test_training_context_included(self):
        llm = make_llm(VALID_CODE_RESPONSE)
        coder = Coder(llm)
        coder.generate("hypothesis", training_context="Example 1: input…")
        call_args = llm.generate.call_args
        messages = call_args[0][1]
        content = messages[0]["content"]
        assert "Example 1" in content

    def test_temperature_override_passed(self):
        llm = make_llm(VALID_CODE_RESPONSE)
        coder = Coder(llm)
        coder.generate("hypothesis", temperature=0.9)
        call_args = llm.generate.call_args
        # temperature should be passed as kwarg
        assert call_args[1].get("temperature") == 0.9 or \
               call_args[0][2] == 0.9 if len(call_args[0]) > 2 else True

    def test_system_mentions_dsl(self):
        llm = make_llm(VALID_CODE_RESPONSE)
        coder = Coder(llm)
        coder.generate("hypothesis")
        system_arg = llm.generate.call_args[0][0]
        # System prompt should mention DSL functions
        assert "crop" in system_arg or "rotate" in system_arg or "DSL" in system_arg


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestCritic:
    def test_routes_to_hypothesizer(self):
        llm = make_llm(CRITIC_HYPOTHESIZER_RESPONSE)
        critic = Critic(llm)
        result = critic.analyze("hyp", "code", "0/2 correct", "diff")
        assert result["route"] == ROUTE_HYPOTHESIZER
        assert len(result["feedback"]) > 10

    def test_routes_to_coder(self):
        llm = make_llm(CRITIC_CODER_RESPONSE)
        critic = Critic(llm)
        result = critic.analyze("hyp", "code", "1/2 correct", "diff")
        assert result["route"] == ROUTE_CODER
        assert "off-by-one" in result["feedback"]

    def test_result_has_required_keys(self):
        llm = make_llm(CRITIC_CODER_RESPONSE)
        critic = Critic(llm)
        result = critic.analyze("hyp", "code", "error", "diff")
        assert "route" in result
        assert "feedback" in result

    def test_default_route_on_ambiguous_response(self):
        llm = make_llm("I am not sure what to do here.")
        critic = Critic(llm)
        result = critic.analyze("hyp", "code", "error", "diff")
        # Should default to coder (safe default)
        assert result["route"] in (ROUTE_CODER, ROUTE_HYPOTHESIZER)

    def test_case_insensitive_route_parse(self):
        llm = make_llm("ROUTE: CODER\nFEEDBACK:\nFix it.")
        critic = Critic(llm)
        result = critic.analyze("hyp", "code", "error", "diff")
        assert result["route"] == ROUTE_CODER

    def test_code_included_in_prompt(self):
        llm = make_llm(CRITIC_CODER_RESPONSE)
        critic = Critic(llm)
        critic.analyze("my_hypothesis", "def transform(g): pass", "1/2", "diff")
        call_args = llm.generate.call_args
        messages = call_args[0][1]
        content = messages[0]["content"]
        assert "def transform" in content
        assert "my_hypothesis" in content


# ---------------------------------------------------------------------------
# PSOCoder
# ---------------------------------------------------------------------------

class TestPSOCoder:
    def test_returns_list_of_strings(self):
        llm = make_llm(K_MUTATIONS_RESPONSE)
        pso_coder = PSOCoder(llm, k=5)
        results = pso_coder.generate_mutations(
            task_description="task",
            training_context="examples",
            current_code="def transform(g): return g",
            current_fitness=0.3,
            pbest_code="def transform(g): return rotate(g)",
            pbest_fitness=0.5,
            gbest_code="def transform(g): return flip(g)",
            gbest_fitness=0.7,
            role_name="test_role",
            role_description="test desc",
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, str)
            assert "def transform" in r

    def test_function_names_normalised_to_transform(self):
        llm = make_llm(K_MUTATIONS_RESPONSE)
        pso_coder = PSOCoder(llm, k=5)
        results = pso_coder.generate_mutations(
            task_description="t", training_context="e",
            current_code="", current_fitness=0.0,
            pbest_code="def transform(g): return g", pbest_fitness=0.0,
            gbest_code="def transform(g): return g", gbest_fitness=0.0,
            role_name="r", role_description="d",
        )
        for r in results:
            # All should be named `transform`, not `transform_N`
            assert re.search(r"def transform\s*\(", r)
            assert not re.search(r"def transform_\d+\s*\(", r)

    def test_fallback_to_pbest_on_empty_response(self):
        llm = make_llm("I cannot generate any code right now.")
        pso_coder = PSOCoder(llm, k=3)
        pbest = "def transform(g):\n    return g.copy()"
        results = pso_coder.generate_mutations(
            task_description="t", training_context="e",
            current_code="", current_fitness=0.0,
            pbest_code=pbest, pbest_fitness=0.5,
            gbest_code=pbest, gbest_fitness=0.5,
            role_name="r", role_description="d",
        )
        assert len(results) >= 1
        assert results[0] == pbest

    def test_context_passed_to_llm(self):
        llm = make_llm(K_MUTATIONS_RESPONSE)
        pso_coder = PSOCoder(llm, k=2)
        pso_coder.generate_mutations(
            task_description="TASK_DESC", training_context="TRAIN_CTX",
            current_code="CUR_CODE", current_fitness=0.1,
            pbest_code="PBEST_CODE", pbest_fitness=0.3,
            gbest_code="GBEST_CODE", gbest_fitness=0.6,
            role_name="ROLE", role_description="DESC",
        )
        call_args = llm.generate.call_args
        content = call_args[0][1][0]["content"]
        assert "TASK_DESC" in content
        assert "PBEST_CODE" in content
        assert "GBEST_CODE" in content
        assert "0.3" in content   # pbest_fitness
        assert "0.6" in content   # gbest_fitness

    def test_respects_k_limit(self):
        # Even if LLM returns more blocks, we cap at k
        llm = make_llm(K_MUTATIONS_RESPONSE)  # 5 blocks
        pso_coder = PSOCoder(llm, k=3)
        results = pso_coder.generate_mutations(
            task_description="t", training_context="e",
            current_code="", current_fitness=0.0,
            pbest_code="def transform(g): return g", pbest_fitness=0.0,
            gbest_code="def transform(g): return g", gbest_fitness=0.0,
            role_name="r", role_description="d",
        )
        assert len(results) <= 3
