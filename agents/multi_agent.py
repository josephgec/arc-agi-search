"""Multi-agent orchestrator for ARC-AGI puzzles.

Coordinates three specialized agents in a feedback loop:

  Hypothesizer → generates 3 competing natural-language transformation rules
  Coder        → translates one rule into executable Python DSL code
  Critic       → diagnoses failures and routes feedback to the right agent

The loop runs up to ``max_cycles`` total agent calls.  On each cycle either:
  - A correct solution is found (success), or
  - The Critic routes to the Hypothesizer (try next/new hypothesis), or
  - The Critic routes to the Coder (fix the current implementation).
"""
from __future__ import annotations

import re

import numpy as np

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder, Critic, ROUTE_HYPOTHESIZER


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_LARGE_GRID_CELL_THRESHOLD = 300
_LARGE_GRID_MAX_PAIRS      = 2

_COLOR_NAMES = {
    0: "black", 1: "blue",   2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "fuschia", 7: "orange", 8: "azure",  9: "maroon",
}


# ---------------------------------------------------------------------------
# Grid / task formatting helpers (also used by pso_orchestrator)
# ---------------------------------------------------------------------------

def _grid_to_str(grid) -> str:
    return (
        "["
        + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in grid.tolist())
        + "]"
    )


def _block_analysis(inp, out) -> str | None:
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    br, bc = oh // ih, ow // iw
    lines  = [f"  (Output divided into {br}×{bc} blocks, each {ih}×{iw}:)"]
    for r in range(br):
        for c in range(bc):
            block    = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
            cell_val = inp[r, c] if r < ih and c < iw else "?"
            if (block == 0).all():
                content = "all zeros"
            else:
                content = _grid_to_str(block)
                if block.shape == inp.shape and (block == inp).all():
                    content += " (= input)"
            lines.append(f"  block({r},{c}): input[{r}][{c}]={cell_val} → {content}")
    return "\n".join(lines)


def _diff_annotation(inp: np.ndarray, out: np.ndarray) -> str | None:
    """For same-shape pairs, briefly describe which cells changed."""
    if inp.shape != out.shape:
        return None
    diff = np.argwhere(inp != out)
    n = len(diff)
    if n == 0:
        return "  (no cells changed)"
    if n > 20:
        return f"  ({n} cells changed)"
    parts = [f"[{r},{c}] {inp[r,c]}→{out[r,c]}" for r, c in diff[:8]]
    suffix = "…" if n > 8 else ""
    return f"  ({n} cells changed: {', '.join(parts)}{suffix})"


def _format_training_examples(task: dict) -> str:
    lines = ["Training examples (use these to verify your implementation):"]
    for i, pair in enumerate(task["train"]):
        inp, out = pair["input"], pair["output"]
        ih, iw   = inp.shape
        oh, ow   = out.shape
        lines.append(f"Example {i + 1}: input ({ih}×{iw}) → output ({oh}×{ow})")
        lines.append(f"  Input:  {_grid_to_str(inp)}")
        lines.append(f"  Output: {_grid_to_str(out)}")
        ann = _diff_annotation(inp, out)
        if ann:
            lines.append(ann)
    return "\n".join(lines)


def _output_shape_constraint(task: dict) -> str:
    """Return a shape hint for LLM: hard constraint when fixed, guidance when variable."""
    pairs      = task["train"]
    in_shapes  = [tuple(p["input"].shape)  for p in pairs]
    out_shapes = [tuple(p["output"].shape) for p in pairs]

    if len(set(out_shapes)) == 1:
        oh, ow = out_shapes[0]
        return (
            f"\n⚠ HARD CONSTRAINT: transform() MUST return an array of shape "
            f"({oh}, {ow}) — i.e. exactly {oh} rows × {ow} columns.  "
            "Any other output size is WRONG regardless of content."
        )

    # Variable output sizes — give rich guidance
    lines = ["\n⚠ Note: output size VARIES across training pairs — it must be computed "
             "dynamically from the input content, not hardcoded."]
    for i, (ish, osh) in enumerate(zip(in_shapes, out_shapes)):
        lines.append(f"  Pair {i+1}: input {ish[0]}×{ish[1]} → output {osh[0]}×{osh[1]}")
        if ish == osh:
            lines.append("    (same shape as input)")
        elif osh[0] < ish[0] or osh[1] < ish[1]:
            lines.append("    (output is SMALLER than input — likely a crop or selection)")
        elif osh[0] > ish[0] or osh[1] > ish[1]:
            lines.append("    (output is LARGER than input — likely expanded/tiled)")
    lines.append("Study how each output size relates to what's in the corresponding input.")
    return "\n".join(lines)


def _format_task_description(task: dict) -> str:
    pairs    = task["train"]
    max_cells = max(max(p["input"].size, p["output"].size) for p in pairs)
    if max_cells > _LARGE_GRID_CELL_THRESHOLD:
        pairs = pairs[:_LARGE_GRID_MAX_PAIRS]
        note  = f"(Note: grids are large; showing {len(pairs)} of {len(task['train'])} training pairs.)\n"
    else:
        note = ""

    lines = [f"Here is an ARC-AGI puzzle.\n{note}"]
    for i, pair in enumerate(pairs):
        inp, out = pair["input"], pair["output"]
        ih, iw   = inp.shape
        oh, ow   = out.shape
        lines.append(f"### Training pair {i + 1}")
        lines.append(f"Input  ({ih}×{iw}):\n{_grid_to_str(inp)}")
        lines.append(f"Output ({oh}×{ow}):\n{_grid_to_str(out)}")
        ba = _block_analysis(inp, out)
        if ba:
            lines.append(ba)
        lines.append("")

    test_inp = task["test"][0]["input"]
    th, tw   = test_inp.shape
    lines.append(f"### Test input ({th}×{tw}):\n{_grid_to_str(test_inp)}")
    lines.append(_output_shape_constraint(task))
    lines.append("\nStudy the training pairs and identify the transformation rule.")
    return "\n".join(lines)


def _format_eval_diff(eval_result: dict, max_pairs: int = 2, max_mismatches: int = 8) -> str:
    """Format a concise diff of sandbox evaluation failures for LLM feedback."""
    lines = []
    for i, pair in enumerate(eval_result.get("pairs", [])[:max_pairs]):
        if pair.get("error"):
            lines.append(f"Pair {i + 1}: runtime error — {pair['error'][:120]}")
            continue
        pred = pair.get("predicted")
        exp  = pair.get("expected")
        if pred is None or exp is None:
            lines.append(f"Pair {i + 1}: no prediction")
            continue
        if pred.shape != exp.shape:
            lines.append(
                f"Pair {i + 1}: wrong shape — got {pred.shape[0]}×{pred.shape[1]}, "
                f"expected {exp.shape[0]}×{exp.shape[1]}"
            )
        else:
            wrong = int(np.sum(pred != exp))
            total = int(exp.size)
            lines.append(f"Pair {i + 1}: {wrong}/{total} cells wrong")
            mismatches = np.argwhere(pred != exp)[:max_mismatches]
            for r, c in mismatches:
                got_name = _COLOR_NAMES.get(int(pred[r, c]), str(pred[r, c]))
                exp_name = _COLOR_NAMES.get(int(exp[r, c]),  str(exp[r, c]))
                lines.append(f"  [{r},{c}]: got {pred[r,c]}({got_name}), expected {exp[r,c]}({exp_name})")
            if wrong > max_mismatches:
                lines.append(f"  … and {wrong - max_mismatches} more wrong cells")
    return "\n".join(lines) if lines else "No diff available"


# ---------------------------------------------------------------------------
# Code extraction / response cleaning
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _truncate_to_valid_function(text: str) -> str:
    lines   = text.splitlines()
    in_func = False
    result  = []
    for line in lines:
        if line.startswith("def "):
            in_func = True
        if in_func:
            stripped = line.rstrip()
            if stripped and not stripped[0].isspace() and not stripped.startswith("def "):
                break
            result.append(line)
    return "\n".join(result).rstrip() if result else text


def _extract_code(text: str) -> str | None:
    text = _strip_thinking(text)
    if not text.strip():
        return None

    # 1. ```python … ``` — last block preferred
    matches = list(re.finditer(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip()
        return _truncate_to_valid_function(candidate) if "def " in candidate else candidate

    # 2. ``` … ``` with a def — last qualifying block
    for m in reversed(list(re.finditer(r"```\s*(.*?)\s*```", text, re.DOTALL))):
        candidate = m.group(1).strip()
        if "def " in candidate:
            return _truncate_to_valid_function(candidate)

    # 3. Bare def transform( — last occurrence
    all_transforms = list(re.finditer(r"def transform\(", text))
    if all_transforms:
        return _truncate_to_valid_function(text[all_transforms[-1].start():])

    # 4. Any bare def — last occurrence
    all_defs = list(re.finditer(r"def \w+\(", text))
    if all_defs:
        return _truncate_to_valid_function(text[all_defs[-1].start():])

    # 5. import numpy … def (no fence)
    if "import numpy" in text and "def " in text:
        start = text.rfind("import numpy")
        return _truncate_to_valid_function(text[start:])

    return None


# ---------------------------------------------------------------------------
# Hypothesis parsing
# ---------------------------------------------------------------------------

_MIN_HYPOTHESIS_CHARS = 80

def _parse_hypotheses(text: str, max_n: int | None = None) -> list[str]:
    text     = _strip_thinking(text)
    stripped = text.strip()

    paragraphs = re.split(r"\n\s*\n", stripped)
    hyp_paras  = [p.strip() for p in paragraphs
                  if re.match(r"^[1-9]\.\s", p.strip())]
    if len(hyp_paras) >= 2:
        hypotheses = hyp_paras
    else:
        parts      = re.split(r"(?m)^(?=[1-9]\.\s)", stripped)
        hypotheses = [p.strip() for p in parts if p.strip()]
        if len(hypotheses) < 2:
            return [stripped] if stripped else []

    hypotheses = [h for h in hypotheses if len(h) >= _MIN_HYPOTHESIS_CHARS]

    if max_n is not None:
        hypotheses = hypotheses[:max_n]

    return hypotheses if hypotheses else [stripped]


# ---------------------------------------------------------------------------
# Error / diff formatting for the Critic
# ---------------------------------------------------------------------------

def _format_error_info(eval_result: dict) -> str:
    lines = [f"{eval_result['n_correct']}/{eval_result['n_total']} training pairs correct."]
    for i, pair in enumerate(eval_result["pairs"]):
        if pair["error"]:
            lines.append(f"Pair {i + 1} error: {pair['error']}")
        elif not pair["correct"]:
            lines.append(f"Pair {i + 1}: produced wrong output (no exception).")
    return "\n".join(lines)


def _diff_summary(expected, predicted, max_show: int = 12) -> str:
    if predicted is None:
        return "(no output produced)"
    if expected.shape != predicted.shape:
        return f"Shape mismatch: expected {expected.shape}, got {predicted.shape}."
    diffs = list(zip(*np.where(expected != predicted)))
    if not diffs:
        return "(no differences)"
    lines = []
    for r, c in diffs[:max_show]:
        ev = int(expected[r, c])
        pv = int(predicted[r, c])
        lines.append(
            f"  [{r},{c}] expected {ev} ({_COLOR_NAMES.get(ev, ev)}), "
            f"got {pv} ({_COLOR_NAMES.get(pv, pv)})"
        )
    if len(diffs) > max_show:
        lines.append(f"  … and {len(diffs) - max_show} more differences")
    return "\n".join(lines)


def _format_diff(eval_result: dict) -> str:
    _FULL_GRID_CELL_LIMIT = 200
    parts = []
    first_fail = True
    for i, pair in enumerate(eval_result["pairs"]):
        if not pair["correct"]:
            diff = _diff_summary(pair["expected"], pair["predicted"])
            section = f"Pair {i + 1}:\n{diff}"
            if first_fail and pair["predicted"] is not None:
                exp = pair["expected"]
                pred = pair["predicted"]
                if exp.size <= _FULL_GRID_CELL_LIMIT:
                    section += (
                        f"\n  Full expected:  {_grid_to_str(exp)}"
                        f"\n  Full predicted: {_grid_to_str(pred)}"
                    )
                first_fail = False
            parts.append(section)
    return "\n\n".join(parts) if parts else "(all pairs correct)"


# ---------------------------------------------------------------------------
# MultiAgent orchestrator
# ---------------------------------------------------------------------------

class MultiAgent:
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
        max_cycles:               int        = 9,
    ) -> None:
        def _make_client(role_model: str | None, temperature: float, max_tokens: int) -> LLMClient:
            return LLMClient(
                backend=backend,
                model=role_model or model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                debug=debug,
            )

        hyp_client = _make_client(hypothesizer_model, hypothesizer_temperature, hypothesizer_max_tokens)
        cod_client = _make_client(coder_model,        coder_temperature,        coder_max_tokens)
        cri_client = _make_client(critic_model,       critic_temperature,       critic_max_tokens)

        self._hypothesizer            = Hypothesizer(hyp_client)
        self._coder                   = Coder(cod_client)
        self._critic                  = Critic(cri_client)
        self.max_cycles               = max_cycles
        self.debug                    = debug
        self.backend                  = backend
        self.hypothesizer_model       = hyp_client.model
        self.coder_model              = cod_client.model
        self.critic_model             = cri_client.model
        self.hypothesizer_temperature = hypothesizer_temperature
        self.coder_temperature        = coder_temperature
        self.critic_temperature       = critic_temperature
        self.hypothesizer_max_tokens  = hypothesizer_max_tokens
        self.coder_max_tokens         = coder_max_tokens
        self.critic_max_tokens        = critic_max_tokens
        self.model                    = self.hypothesizer_model

    def solve(self, task: dict) -> dict:
        log:            list[dict] = []
        best_code:      str | None = None
        best_n_correct: int        = -1
        cycle:          int        = 0

        test_pair             = task.get("test", [{}])[0]
        has_test_ground_truth = "output" in test_pair

        task_description: str        = _format_task_description(task)
        training_examples: str       = _format_training_examples(task)
        hypotheses:        list[str] = []
        hyp_index:         int       = 0
        hyp_feedback:      str | None = None
        coder_feedback:    str | None = None
        prev_n_correct:    int        = -1
        no_improve_count:  int        = 0
        coder_attempt:     int        = 0

        while cycle < self.max_cycles:

            if not hypotheses or hyp_index >= len(hypotheses):
                cycle += 1
                if cycle > self.max_cycles:
                    break
                try:
                    hyp_response = self._hypothesizer.generate(task_description, hyp_feedback)
                except Exception as e:
                    log.append({"cycle": cycle, "agent": "hypothesizer", "error": str(e)})
                    break

                hyp_feedback = None
                hypotheses   = _parse_hypotheses(hyp_response, max_n=3)
                hyp_index    = 0

                log.append({
                    "cycle": cycle, "agent": "hypothesizer",
                    "n_hypotheses": len(hypotheses),
                })
                if self.debug:
                    print(f"[debug] Hypothesizer: {len(hypotheses)} hypothesis(es)")

            if not hypotheses or hyp_index == 0 or (
                log and log[-1].get("agent") == "hypothesizer"
            ):
                coder_attempt    = 0
                prev_n_correct   = -1
                no_improve_count = 0

            current_hypothesis = hypotheses[hyp_index]
            coder_attempt += 1

            temperature = min(self.coder_temperature + (coder_attempt - 1) * 0.3, 0.9)

            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                code_response = self._coder.generate(
                    current_hypothesis, coder_feedback,
                    training_context=training_examples,
                    temperature=temperature,
                )
            except Exception as e:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": str(e),
                })
                hyp_index     += 1
                coder_feedback = None
                coder_attempt  = 0
                continue

            coder_feedback = None

            clean  = _strip_thinking(code_response)
            code   = _extract_code(clean) or _extract_code(code_response)

            if self.debug:
                print(
                    f"[debug] Coder (hyp {hyp_index}): "
                    f"response={len(code_response)} chars, code={code is not None}"
                )

            if code is None:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": "no_code_block",
                })
                hyp_index += 1
                continue

            eval_result = sandbox.evaluate_code(code, task)
            n_correct   = eval_result["n_correct"]

            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code      = code

            log.append({
                "cycle":            cycle,
                "agent":            "coder",
                "hypothesis_index": hyp_index,
                "n_correct":        n_correct,
                "n_total":          eval_result["n_total"],
                "all_correct":      eval_result["all_correct"],
            })

            if eval_result["all_correct"]:
                return {
                    "success":      True,
                    "test_correct": self._evaluate_test(best_code, test_pair) if has_test_ground_truth else None,
                    "code":         best_code,
                    "n_cycles":     cycle,
                    "log":          log,
                }

            if n_correct <= prev_n_correct:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_n_correct = n_correct

            _hypothesis_stuck = (n_correct == 0 and no_improve_count >= 2)
            if _hypothesis_stuck:
                if self.debug:
                    print(f"[debug] Stuck at 0/{eval_result['n_total']} — skipping Critic, next hyp")
                hyp_index    += 1
                coder_attempt = 0
                continue

            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                critic_result = self._critic.analyze(
                    current_hypothesis, code,
                    _format_error_info(eval_result),
                    _format_diff(eval_result),
                )
            except Exception as e:
                log.append({"cycle": cycle, "agent": "critic", "error": str(e)})
                hyp_index    += 1
                coder_attempt = 0
                continue

            log.append({
                "cycle":    cycle,
                "agent":    "critic",
                "route":    critic_result["route"],
                "feedback": critic_result["feedback"],
            })
            if self.debug:
                print(f"[debug] Critic → {critic_result['route']}")

            if critic_result["route"] == ROUTE_HYPOTHESIZER:
                hyp_feedback  = critic_result["feedback"]
                hyp_index    += 1
                coder_attempt = 0
            else:
                coder_feedback = critic_result["feedback"]

        return {
            "success":      False,
            "test_correct": (
                self._evaluate_test(best_code, test_pair)
                if has_test_ground_truth and best_code else None
            ),
            "code":         best_code,
            "n_cycles":     cycle,
            "log":          log,
        }

    def predict(self, task: dict) -> Grid | None:
        result = self.solve(task)
        if not result["code"]:
            return None
        out, _ = sandbox.execute(result["code"], task["test"][0]["input"])
        return out

    def _evaluate_test(self, code: str, test_pair: dict) -> bool:
        out, err = sandbox.execute(code, test_pair["input"])
        if err or out is None:
            return False
        return grids_equal(out, test_pair["output"])
