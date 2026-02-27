"""Test-time compute scaling via ensembling and majority voting.

Runs the Orchestrator repeatedly, pools programs that achieve 100% on all
training pairs, then submits the output that a majority of them agree on.

Flow
----
1. Run the Orchestrator repeatedly, collecting programs that achieve 100%
   on all training pairs.  Stop when target_candidates unique programs are
   pooled or max_runs is exhausted.
2. Execute every pooled program on the test input.
3. Group identical output grids (pixel-perfect equality).
4. Submit the output grid with the most supporting programs (votes).
   Ties are broken in favour of the first group formed.
"""
from __future__ import annotations

import numpy as np

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Voting primitives
# ---------------------------------------------------------------------------

def _grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _majority_vote(grids: list[np.ndarray]) -> np.ndarray | None:
    if not grids:
        return None

    representatives: list[np.ndarray] = []
    counts:          list[int]         = []

    for g in grids:
        for i, rep in enumerate(representatives):
            if _grids_equal(g, rep):
                counts[i] += 1
                break
        else:
            representatives.append(g)
            counts.append(1)

    return representatives[counts.index(max(counts))]


def _vote_summary(grids: list[np.ndarray]) -> list[dict]:
    if not grids:
        return []

    representatives:   list[np.ndarray] = []
    indices_per_group: list[list[int]]  = []

    for i, g in enumerate(grids):
        for j, rep in enumerate(representatives):
            if _grids_equal(g, rep):
                indices_per_group[j].append(i)
                break
        else:
            representatives.append(g)
            indices_per_group.append([i])

    groups = [
        {
            "output":            rep,
            "count":             len(idx),
            "candidate_indices": idx,
        }
        for rep, idx in zip(representatives, indices_per_group)
    ]
    return sorted(groups, key=lambda g: -g["count"])


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class Ensemble:
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
        target_candidates:        int        = 3,
        max_runs:                 int        = 5,
        n_hypotheses:             int        = 3,
        max_retries:              int        = 2,
    ) -> None:
        self._orchestrator = Orchestrator(
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
            n_hypotheses=n_hypotheses,
            max_retries=max_retries,
        )
        self.target_candidates        = target_candidates
        self.max_runs                 = max_runs
        self.debug                    = debug
        self.backend                  = self._orchestrator.backend
        self.hypothesizer_model       = self._orchestrator.hypothesizer_model
        self.coder_model              = self._orchestrator.coder_model
        self.critic_model             = self._orchestrator.critic_model
        self.hypothesizer_temperature = self._orchestrator.hypothesizer_temperature
        self.coder_temperature        = self._orchestrator.coder_temperature
        self.critic_temperature       = self._orchestrator.critic_temperature
        self.hypothesizer_max_tokens  = self._orchestrator.hypothesizer_max_tokens
        self.coder_max_tokens         = self._orchestrator.coder_max_tokens
        self.critic_max_tokens        = self._orchestrator.critic_max_tokens
        self.model                    = self._orchestrator.model

    def solve(self, task: dict) -> dict:
        seen_codes: set[str]   = set()
        candidates: list[dict] = []
        n_runs:     int        = 0

        test_pair             = (task.get("test") or [{}])[0]
        has_test_ground_truth = "output" in test_pair

        for _ in range(self.max_runs):
            n_runs += 1
            if self.debug:
                print(
                    f"[ensemble] run {n_runs}/{self.max_runs} "
                    f"(pool: {len(candidates)}/{self.target_candidates})"
                )

            result = self._orchestrator.solve(task)

            for cand in result.get("candidates", []):
                code = cand["code"].strip()
                if code not in seen_codes:
                    seen_codes.add(code)
                    candidates.append(cand)

            if len(candidates) >= self.target_candidates:
                break

        if not candidates:
            return {
                "success":      False,
                "prediction":   None,
                "test_correct": None,
                "candidates":   [],
                "vote_summary": [],
                "n_runs":       n_runs,
            }

        test_input = test_pair["input"]
        outputs: list[np.ndarray | None] = []

        for cand in candidates:
            out, _ = sandbox.execute(cand["code"], test_input)
            outputs.append(out)

        valid_outputs = [o for o in outputs if o is not None]
        prediction    = _majority_vote(valid_outputs)
        vote_summary  = _vote_summary(valid_outputs)

        if self.debug:
            top = vote_summary[0]["count"] if vote_summary else 0
            print(
                f"[ensemble] {len(valid_outputs)} valid outputs, "
                f"{len(vote_summary)} distinct group(s), "
                f"top vote: {top}/{len(valid_outputs)}"
            )

        test_correct: bool | None = None
        if has_test_ground_truth and prediction is not None:
            test_correct = grids_equal(prediction, test_pair["output"])

        return {
            "success":      prediction is not None,
            "prediction":   prediction,
            "test_correct": test_correct,
            "candidates":   candidates,
            "vote_summary": vote_summary,
            "n_runs":       n_runs,
        }

    def predict(self, task: dict) -> Grid | None:
        return self.solve(task)["prediction"]
