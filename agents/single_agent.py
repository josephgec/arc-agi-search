"""Single-agent baseline for ARC-AGI.

Simpler than MultiAgent: one Hypothesizer call followed by one Coder call
per hypothesis, with no Critic feedback loop.  Useful as a performance
baseline and for quick sanity checks.
"""
from __future__ import annotations

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder
from agents.multi_agent import (
    _format_task_description,
    _format_training_examples,
    _strip_thinking,
    _extract_code,
    _parse_hypotheses,
)


class SingleAgent:
    """One-shot hypothesize → code solver (no Critic loop)."""

    def __init__(
        self,
        backend:     str        = "ollama",
        model:       str | None = None,
        temperature: float      = 0.4,
        max_tokens:  int        = 8192,
        timeout:     float      = 120.0,
        debug:       bool       = False,
    ) -> None:
        client = LLMClient(
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            debug=debug,
        )
        self._hypothesizer = Hypothesizer(client)
        self._coder        = Coder(client)
        self.model         = client.model
        self.backend       = backend
        self.debug         = debug

    def solve(self, task: dict) -> dict:
        task_description  = _format_task_description(task)
        training_examples = _format_training_examples(task)
        test_pair         = task.get("test", [{}])[0]
        has_truth         = "output" in test_pair

        # Hypothesizer
        try:
            hyp_response = self._hypothesizer.generate(task_description)
        except Exception as e:
            return {"success": False, "code": None, "error": str(e)}

        hypotheses = _parse_hypotheses(hyp_response, max_n=3)
        if not hypotheses:
            hypotheses = [hyp_response.strip()]

        best_code      = None
        best_n_correct = -1

        for hypothesis in hypotheses:
            try:
                code_response = self._coder.generate(
                    hypothesis, training_context=training_examples
                )
            except Exception:
                continue

            clean = _strip_thinking(code_response)
            code  = _extract_code(clean) or _extract_code(code_response)
            if code is None:
                continue

            eval_result = sandbox.evaluate_code(code, task)
            n_correct   = eval_result["n_correct"]
            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code      = code

            if eval_result["all_correct"]:
                break

        success = best_code is not None and sandbox.evaluate_code(best_code, task)["all_correct"]

        test_correct = None
        if has_truth and best_code:
            out, _ = sandbox.execute(best_code, test_pair["input"])
            if out is not None:
                test_correct = grids_equal(out, test_pair["output"])

        return {
            "success":      success,
            "code":         best_code,
            "test_correct": test_correct,
        }

    def predict(self, task: dict) -> Grid | None:
        result = self.solve(task)
        if not result.get("code"):
            return None
        out, _ = sandbox.execute(result["code"], task["test"][0]["input"])
        return out
