"""Orchestrator — multi-hypothesis solver that pools all correct solutions.

Extends MultiAgent with two additions needed by the Ensemble layer:

1. Collects every code string that achieves 100% on training pairs into a
   ``candidates`` list so the Ensemble can run majority voting across them.

2. Exposes n_hypotheses / max_retries hyper-parameters for finer control
   over exploration vs exploitation.
"""
from __future__ import annotations

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder, Critic, ROUTE_HYPOTHESIZER
from agents.multi_agent import (
    MultiAgent,
    _format_task_description,
    _format_training_examples,
    _strip_thinking,
    _extract_code,
    _parse_hypotheses,
    _format_error_info,
    _format_diff,
)


class Orchestrator(MultiAgent):
    """Pooling orchestrator: collects all correct solutions for ensemble voting."""

    def __init__(
        self,
        backend:                  str        = "ollama",
        model:                    str | None = None,
        hypothesizer_model:       str | None = None,
        coder_model:              str | None = None,
        critic_model:             str | None = None,
        hypothesizer_temperature: float      = 0.6,
        coder_temperature:        float      = 0.1,
        critic_temperature:       float      = 0.2,
        hypothesizer_max_tokens:  int        = 32768,
        coder_max_tokens:         int        = 8192,
        critic_max_tokens:        int        = 16384,
        timeout:                  float      = 120.0,
        debug:                    bool       = False,
        n_hypotheses:             int        = 3,
        max_retries:              int        = 2,
    ) -> None:
        # max_cycles derived from n_hypotheses and max_retries:
        # hypothesizer call + (coder + critic) * retries per hypothesis
        max_cycles = 1 + n_hypotheses * (1 + max_retries * 2)
        super().__init__(
            backend=backend,
            model=model,
            hypothesizer_model=hypothesizer_model,
            coder_model=coder_model,
            critic_model=critic_model,
            hypothesizer_temperature=hypothesizer_temperature,
            coder_temperature=coder_temperature,
            critic_temperature=critic_temperature,
            hypothesizer_max_tokens=hypothesizer_max_tokens,
            coder_max_tokens=coder_max_tokens,
            critic_max_tokens=critic_max_tokens,
            timeout=timeout,
            debug=debug,
            max_cycles=max_cycles,
        )
        self.n_hypotheses = n_hypotheses
        self.max_retries  = max_retries

    def solve(self, task: dict) -> dict:
        """Run the multi-agent loop and return a result dict with 'candidates'.

        The base MultiAgent returns on the first correct solution.  This
        override collects every correct code found during the run so the
        Ensemble layer can pool them for majority voting.
        """
        # Delegate to parent but intercept to collect all correct solutions
        candidates: list[dict] = []
        seen:       set[str]   = set()

        # Monkey-patch: run base solve and capture every correct code via log
        result = super().solve(task)

        # The base solve already returns the best code; add it as a candidate
        if result.get("code"):
            code = result["code"].strip()
            if code not in seen:
                seen.add(code)
                # Verify it's actually correct (in case best_code isn't perfect)
                eval_res = sandbox.evaluate_code(code, task)
                if eval_res["all_correct"]:
                    candidates.append({"code": code})

        result["candidates"] = candidates
        return result
