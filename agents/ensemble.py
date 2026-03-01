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

import re

import numpy as np

from arc import sandbox
from arc.evaluate import calculate_continuous_fitness
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.multi_agent import _format_task_description, _grid_to_str
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
# Fitness helpers
# ---------------------------------------------------------------------------

def _avg_fitness(pairs: list[dict]) -> float:
    """Average continuous fitness across evaluation pairs."""
    if not pairs:
        return 0.0
    total = sum(
        calculate_continuous_fitness(p["predicted"], p["expected"])
        if p.get("predicted") is not None else 0.0
        for p in pairs if p.get("expected") is not None
    )
    valid = [p for p in pairs if p.get("expected") is not None]
    return total / len(valid) if valid else 0.0


# ---------------------------------------------------------------------------
# Pixel-level weighted majority vote
# ---------------------------------------------------------------------------

def _pixel_majority_vote(
    grids: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray | None:
    """Per-pixel weighted plurality vote across a list of grids.

    Grids that share the most-weighted shape are included; others are ignored.
    At each pixel position, the value with the highest total weight wins.
    """
    if not grids:
        return None
    if weights is None:
        weights = [1.0] * len(grids)

    # Determine the shape with the highest aggregate weight
    shape_weight: dict = {}
    for g, w in zip(grids, weights):
        shape_weight[g.shape] = shape_weight.get(g.shape, 0.0) + w
    target_shape = max(shape_weight, key=shape_weight.get)

    valid = [(g, w) for g, w in zip(grids, weights) if g.shape == target_shape]
    rows, cols = target_shape
    result = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            tally: dict[int, float] = {}
            for g, w in valid:
                v = int(g[r, c])
                tally[v] = tally.get(v, 0.0) + w
            result[r, c] = max(tally, key=tally.get)
    return result


# ---------------------------------------------------------------------------
# Self-correction critic
# ---------------------------------------------------------------------------

_CORRECTION_CRITIC_SYSTEM = """\
You are an expert ARC-AGI evaluator.  Given training input/output pairs and a
candidate prediction for the test input, determine whether the prediction is
logically consistent with the transformation rule inferred from the training pairs.

Respond in EXACTLY this format:

VERDICT: ACCEPT
REASON: <brief explanation>

  or

VERDICT: REJECT
REASON: <specific inconsistency with the training-pair rule>

Rules:
- VERDICT must be exactly "ACCEPT" or "REJECT" (uppercase).
- REASON must follow on the next line.
- REJECT only for a clear, specific violation of the inferred rule.
  When uncertain, default to ACCEPT.
"""


def _check_prediction(
    prediction: np.ndarray,
    task: dict,
    client: LLMClient,
) -> dict:
    """Ask the correction critic whether a prediction is logically consistent.

    Returns {"accept": bool, "reason": str}.
    Fail-safe: returns accept=True on any exception or malformed response.
    """
    try:
        task_desc = _format_task_description(task)
        pred_str  = _grid_to_str(prediction)
        prompt = (
            f"{task_desc}\n\n"
            f"### Candidate prediction for the test input:\n{pred_str}\n\n"
            "Is this prediction logically consistent with the transformation rule?\n"
            "Reply with VERDICT and REASON as instructed."
        )
        response = client.generate(
            system=_CORRECTION_CRITIC_SYSTEM,
            user=prompt,
        )
        verdict_match = re.search(
            r"VERDICT\s*:\s*(ACCEPT|REJECT)", response, re.IGNORECASE
        )
        reason_match = re.search(
            r"REASON\s*:\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE
        )
        if verdict_match is None:
            return {"accept": True, "reason": ""}
        accept = verdict_match.group(1).upper() == "ACCEPT"
        reason = reason_match.group(1).strip() if reason_match else ""
        return {"accept": accept, "reason": reason}
    except Exception:
        return {"accept": True, "reason": ""}


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
        near_miss_threshold:      float      = 0.85,
        near_miss_weight:         float      = 0.5,
        max_corrections:          int        = 2,
        use_correction:           bool       = True,
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
        self.near_miss_threshold      = near_miss_threshold
        self.near_miss_weight         = near_miss_weight
        self.max_corrections          = max_corrections
        self.use_correction           = use_correction
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
        self._correction_client = LLMClient(
            backend=backend,
            model=critic_model or model,
            temperature=critic_temperature,
            max_tokens=critic_max_tokens,
            timeout=timeout,
            debug=debug,
        )

    def solve(self, task: dict) -> dict:
        seen_codes:     set[str]   = set()
        perfect_pool:   list[dict] = []   # weight=1.0
        near_miss_pool: list[dict] = []   # weight=near_miss_weight
        n_runs:         int        = 0

        test_pair             = (task.get("test") or [{}])[0]
        has_test_ground_truth = "output" in test_pair

        for _ in range(self.max_runs):
            n_runs += 1
            if self.debug:
                print(
                    f"[ensemble] run {n_runs}/{self.max_runs} "
                    f"(perfect: {len(perfect_pool)}/{self.target_candidates}, "
                    f"near-miss: {len(near_miss_pool)})"
                )

            result = self._orchestrator.solve(task)

            for cand in result.get("candidates", []):
                code = cand["code"].strip()
                if code in seen_codes:
                    continue
                seen_codes.add(code)

                # Use pre-computed fields if available; else compute from scratch
                perfect = cand.get("perfect")
                fitness = cand.get("fitness")
                if perfect is None or fitness is None:
                    eval_res = sandbox.evaluate_code(code, task)
                    perfect  = eval_res["all_correct"]
                    fitness  = _avg_fitness(eval_res.get("pairs", []))

                if perfect:
                    perfect_pool.append({"code": code, "fitness": 1.0, "weight": 1.0})
                elif fitness >= self.near_miss_threshold:
                    near_miss_pool.append({
                        "code":    code,
                        "fitness": fitness,
                        "weight":  self.near_miss_weight,
                    })

            if len(perfect_pool) >= self.target_candidates:
                break

        all_candidates = perfect_pool + near_miss_pool

        if not all_candidates:
            return {
                "success":          False,
                "prediction":       None,
                "test_correct":     None,
                "candidates":       [],
                "vote_summary":     [],
                "n_runs":           n_runs,
                "corrections_done": 0,
            }

        test_input = test_pair["input"]
        outputs: list[np.ndarray] = []
        weights: list[float]      = []

        for cand in all_candidates:
            out, _ = sandbox.execute(cand["code"], test_input)
            if out is not None:
                outputs.append(out)
                weights.append(cand["weight"])

        # Pixel-level weighted vote with self-correction loop
        remaining_out    = list(outputs)
        remaining_wt     = list(weights)
        prediction       = None
        corrections_done = 0

        while remaining_out:
            prediction = _pixel_majority_vote(remaining_out, remaining_wt)
            if (
                not self.use_correction
                or corrections_done >= self.max_corrections
                or prediction is None
            ):
                break
            check = _check_prediction(prediction, task, self._correction_client)
            if check["accept"]:
                break

            # Exclude outputs that match the rejected pixel-voted prediction
            filtered = [
                (o, w) for o, w in zip(remaining_out, remaining_wt)
                if not _grids_equal(o, prediction)
            ]
            if len(filtered) == len(remaining_out):
                # Pixel-voted result didn't exactly match any candidate output —
                # exclude the leading whole-grid group instead
                groups = _vote_summary(remaining_out)
                if not groups:
                    break
                leader = groups[0]["output"]
                filtered = [
                    (o, w) for o, w in zip(remaining_out, remaining_wt)
                    if not _grids_equal(o, leader)
                ]
            corrections_done += 1
            if not filtered:
                break
            remaining_out, remaining_wt = (list(col) for col in zip(*filtered))

        vote_summary = _vote_summary(remaining_out)

        if self.debug:
            top = vote_summary[0]["count"] if vote_summary else 0
            print(
                f"[ensemble] {len(remaining_out)} valid outputs, "
                f"{len(vote_summary)} distinct group(s), "
                f"top vote: {top}/{len(remaining_out)}, "
                f"corrections: {corrections_done}"
            )

        test_correct: bool | None = None
        if has_test_ground_truth and prediction is not None:
            test_correct = grids_equal(prediction, test_pair["output"])

        return {
            "success":          prediction is not None,
            "prediction":       prediction,
            "test_correct":     test_correct,
            "candidates":       all_candidates,
            "vote_summary":     vote_summary,
            "n_runs":           n_runs,
            "corrections_done": corrections_done,
        }

    def predict(self, task: dict) -> Grid | None:
        return self.solve(task)["prediction"]
