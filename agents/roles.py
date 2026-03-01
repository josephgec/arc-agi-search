"""Agent role definitions for the ARC-AGI multi-agent framework.

Each class wraps an LLMClient with a specific system prompt and calling
convention.  Roles:

  Hypothesizer  — generates competing natural-language transformation rules
  Coder         — translates a rule hypothesis into executable Python DSL code
  Critic        — diagnoses failures and routes feedback to the right agent

PSO extension:
  PSOCoder      — specialised Coder variant for the swarm's mutation step;
                  generates K distinct candidate functions blending pbest/gbest.

Routing constants used by Critic:
  ROUTE_HYPOTHESIZER  — send feedback to the Hypothesizer (new hypothesis needed)
  ROUTE_CODER         — send feedback to the Coder (fix the implementation)
"""
from __future__ import annotations

import re

from agents.llm_client import LLMClient
from agents.dsl_reference import _DSL_REFERENCE

# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------

ROUTE_HYPOTHESIZER = "hypothesizer"
ROUTE_CODER        = "coder"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_HYPOTHESIZER_SYSTEM = """\
You are an expert ARC-AGI pattern analyst.  Your job is to study input/output
grid pairs and generate exactly 3 distinct, plausible hypotheses about the
underlying transformation rule.

Rules:
- Each hypothesis must be a clear, concise English description (2-5 sentences).
- Cover meaningfully different approaches (geometric, colour-based, object-level…).
- Number them exactly: "1.", "2.", "3." on their own lines.
- Do NOT write any Python code.
- Do NOT repeat the same idea with minor wording changes.
"""

_CODER_SYSTEM = (
    "You are an expert Python programmer solving ARC-AGI grid transformations.\n"
    "Translate the given hypothesis into a Python function called `transform` that\n"
    "takes `input_grid: np.ndarray` and returns `np.ndarray`.\n\n"
    + _DSL_REFERENCE
    + "\nRules:\n"
    "- Return ONLY a single ```python \u2026 ``` code block.\n"
    "- The function must be called `transform`.\n"
    "- Use only the DSL primitives listed above plus plain Python/numpy.\n"
    "- The function must handle every training example shown.\n"
    "- No print statements, no side effects, no I/O.\n"
)

_CRITIC_SYSTEM = """\
You are an expert ARC-AGI code debugger.  Analyse why the current Python
solution fails and decide on the best remediation.

You must respond in EXACTLY this format (no extra text):

ROUTE: hypothesizer   ← use this when the hypothesis is fundamentally wrong
  or
ROUTE: coder          ← use this when the hypothesis is correct but the code is buggy

FEEDBACK:
<one concise paragraph of actionable feedback for the chosen agent>

Rules:
- ROUTE must be exactly "hypothesizer" or "coder" (lowercase).
- FEEDBACK must immediately follow on the next line.
- Be specific: quote actual cell values, mention which pairs fail, suggest fixes.

You will receive a spatial diff describing WHERE errors occur geometrically.
Use spatially-grounded language in your feedback: reference directions
(left/right/up/down), color names, and object positions rather than raw cell coordinates.
"""

_PSO_CODER_SYSTEM = (
    "You are a specialised code-mutation agent in a Particle Swarm Optimization\n"
    "loop solving ARC-AGI puzzles.  You will be given two reference solutions\n"
    "(personal best and global best) together with their fitness scores, and you\n"
    "must generate {k} DISTINCT Python `transform` functions that creatively\n"
    "recombine and improve upon both.\n\n"
    "Before generating each function, briefly reason about what the current\n"
    "solutions get right, what they get wrong, and how to fix the root cause.\n\n"
    + _DSL_REFERENCE
    + "\nRules:\n"
    "- Generate exactly {k} code blocks, each in its own ```python \u2026 ``` fence.\n"
    "- Name the functions `transform_1`, `transform_2`, \u2026 `transform_{k}`.\n"
    "- Each must be complete and executable (takes np.ndarray, returns np.ndarray).\n"
    "- Each variation should try a DIFFERENT strategy or fix \u2014 not minor wording tweaks.\n"
    "- Do NOT hardcode specific cell coordinates or shape-based if-else branches.\n"
    "- Prefer solutions that fix the systematic root cause of the listed failures.\n"
    "- No print, no I/O, no side effects.\n"
)

# ---------------------------------------------------------------------------
# Hypothesizer
# ---------------------------------------------------------------------------

class Hypothesizer:
    """Generates competing natural-language transformation hypotheses."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(
        self,
        task_description: str,
        feedback: str | None = None,
        n: int = 3,
    ) -> str:
        """Return raw LLM response containing n numbered hypotheses.

        Args:
            task_description: Formatted ARC task (training pairs + test input).
            feedback:         Optional Critic feedback from a previous attempt.
            n:                Number of hypotheses to generate (default 3).
        """
        # Build a system prompt that requests exactly n hypotheses
        sys = _HYPOTHESIZER_SYSTEM.replace("exactly 3 distinct", f"exactly {n} distinct")
        sys = sys.replace(
            'Number them exactly: "1.", "2.", "3." on their own lines.',
            f'Number them exactly: "1.", "2.", … "{n}." on their own lines.',
        )

        content = task_description
        if feedback:
            content += (
                "\n\n--- CRITIC FEEDBACK FROM PREVIOUS ATTEMPT ---\n"
                + feedback
                + f"\n\nGenerate {n} NEW hypotheses that address the above feedback."
            )
        else:
            content += f"\n\nGenerate {n} hypotheses for the transformation rule."

        messages = [{"role": "user", "content": content}]
        return self._client.generate(sys, messages)


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class Coder:
    """Translates a hypothesis into executable Python DSL code."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(
        self,
        hypothesis:       str,
        feedback:         str | None = None,
        training_context: str | None = None,
        temperature:      float | None = None,
    ) -> str:
        """Return raw LLM response containing a ```python``` code block.

        Args:
            hypothesis:        The natural-language rule to implement.
            feedback:          Optional Critic feedback on a previous code attempt.
            training_context:  Formatted string of all training pairs.
            temperature:       Override the client's default temperature.
        """
        content = f"Hypothesis:\n{hypothesis}"
        if training_context:
            content += f"\n\n{training_context}"
        if feedback:
            content += (
                "\n\n--- CRITIC FEEDBACK ---\n"
                + feedback
                + "\n\nFix the code based on the above feedback."
            )
        else:
            content += "\n\nImplement the `transform` function."

        messages = [{"role": "user", "content": content}]
        return self._client.generate(_CODER_SYSTEM, messages, temperature=temperature)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic:
    """Analyses failures and routes feedback to Hypothesizer or Coder."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def analyze(
        self,
        hypothesis:  str,
        code:        str,
        error_info:  str,
        diff_info:   str,
    ) -> dict[str, str]:
        """Return a dict with keys 'route' and 'feedback'.

        Args:
            hypothesis: The natural-language rule that was being implemented.
            code:       The Python code that was evaluated.
            error_info: Summary of pass/fail counts and error messages.
            diff_info:  Cell-by-cell diff for failing pairs.
        """
        content = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Evaluation results:\n{error_info}\n\n"
            f"Diff of failures:\n{diff_info}"
        )
        messages = [{"role": "user", "content": content}]
        response = self._client.generate(_CRITIC_SYSTEM, messages)

        # Parse ROUTE and FEEDBACK from response
        route    = ROUTE_CODER  # safe default
        feedback = response.strip()

        route_match = re.search(
            r"ROUTE\s*:\s*(hypothesizer|coder)", response, re.IGNORECASE
        )
        if route_match:
            route = route_match.group(1).lower()

        feedback_match = re.search(
            r"FEEDBACK\s*:\s*(.+)", response, re.DOTALL | re.IGNORECASE
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()

        return {"route": route, "feedback": feedback}


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

_DECOMPOSER_SYSTEM = """\
You are an expert ARC-AGI task analyst specialising in breaking down complex
transformations that have resisted previous solution attempts.

Given the task and failed approaches, decompose the transformation into 2–4
numbered, concrete sub-steps in plain English.  Do NOT write any Python code.

Rules:
- Each sub-step must be actionable and specific (not vague).
- Number them 1., 2., 3., (4.) and keep each to 1–3 sentences.
- Refer to spatial positions (top-left, bottom-right), colors by name, and
  object counts rather than raw coordinates.
- If stuck approaches are given, avoid those directions entirely.
"""


class Decomposer:
    """Breaks a stuck ARC task into explicit sub-steps for the swarm to solve."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def decompose(
        self,
        task_description: str,
        training_examples: str,
        stuck_approaches: str | None = None,
    ) -> str:
        """Return a numbered decomposition of the transformation.

        Args:
            task_description:  Formatted ARC task (all training pairs + test input).
            training_examples: Additional formatted training examples.
            stuck_approaches:  Description of approaches already tried (to avoid).
        """
        content = f"{task_description}\n\n{training_examples}"
        if stuck_approaches:
            content += (
                "\n\n--- APPROACHES THAT HAVE ALREADY FAILED ---\n"
                + stuck_approaches
                + "\n\nAvoid these directions. Decompose the problem differently."
            )
        else:
            content += "\n\nDecompose the transformation into 2–4 concrete sub-steps."

        messages = [{"role": "user", "content": content}]
        return self._client.generate(_DECOMPOSER_SYSTEM, messages)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM = """\
You are a QA agent for ARC-AGI solutions.  Your job is to probe a solution for
edge-case fragility BEFORE it is accepted as correct.

Common failure modes to check:
- Hardcoded grid shapes or specific coordinate values
- Assumptions about input size that may not hold for unseen examples
- Off-by-one errors at grid boundaries
- Color assumptions (e.g. background = 0) that may not generalise
- Logic that only works because all training examples happen to share a property

Respond in EXACTLY this format:

VERDICT: PASS
ISSUES: none
SUGGESTION: none

  or

VERDICT: FAIL
ISSUES: <concise description of the fragility found>
SUGGESTION: <concrete fix to make the code more robust>
"""


class Verifier:
    """QA agent that gates acceptance of solutions by probing for fragility."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def verify(
        self,
        code: str,
        task_description: str,
        training_examples: str,
        eval_result: dict,
    ) -> dict:
        """Return {'passes': bool, 'issues': str, 'suggestion': str}.

        Defaults to passes=True on malformed response or exception (fail-safe).

        Args:
            code:              The Python solution to verify.
            task_description:  Formatted ARC task.
            training_examples: Formatted training examples.
            eval_result:       Result dict from sandbox.evaluate_code.
        """
        n_correct = eval_result.get("n_correct", 0)
        n_total   = eval_result.get("n_total",   0)
        content = (
            f"{task_description}\n\n{training_examples}\n\n"
            f"Solution code:\n```python\n{code}\n```\n\n"
            f"Evaluation: {n_correct}/{n_total} training pairs pass.\n\n"
            "Check for hardcoded values, size assumptions, off-by-ones, and other "
            "edge-case fragilities.  Respond in the required format."
        )
        messages = [{"role": "user", "content": content}]
        try:
            response = self._client.generate(_VERIFIER_SYSTEM, messages)
        except Exception:
            return {"passes": True, "issues": "", "suggestion": ""}

        try:
            verdict_match = re.search(
                r"VERDICT\s*:\s*(PASS|FAIL)", response, re.IGNORECASE
            )
            issues_match = re.search(
                r"ISSUES\s*:\s*(.+?)(?=SUGGESTION\s*:|$)", response,
                re.DOTALL | re.IGNORECASE,
            )
            suggest_match = re.search(
                r"SUGGESTION\s*:\s*(.+)", response, re.DOTALL | re.IGNORECASE
            )

            if verdict_match is None:
                return {"passes": True, "issues": "", "suggestion": ""}

            passes     = verdict_match.group(1).upper() == "PASS"
            issues     = issues_match.group(1).strip()  if issues_match  else ""
            suggestion = suggest_match.group(1).strip() if suggest_match else ""
            return {"passes": passes, "issues": issues, "suggestion": suggestion}
        except Exception:
            return {"passes": True, "issues": "", "suggestion": ""}


# ---------------------------------------------------------------------------
# PSO Coder (mutation agent)
# ---------------------------------------------------------------------------

class PSOCoder:
    """Generates K distinct code mutations for the PSO swarm's mutation step.

    Each call returns a list of up to K Python function strings, each
    attempting to blend the logic of the personal-best and global-best
    solutions to move towards a better region of the solution space.
    """

    def __init__(self, client: LLMClient, k: int = 5) -> None:
        self._client = client
        self.k       = k

    def generate_mutations(
        self,
        task_description:  str,
        training_context:  str,
        current_code:      str,
        current_fitness:   float,
        pbest_code:        str,
        pbest_fitness:     float,
        gbest_code:        str,
        gbest_fitness:     float,
        role_name:         str,
        role_description:  str,
        eval_diff:         str | None = None,
        temperature:       float | None = None,
    ) -> list[str]:
        """Return up to self.k candidate code strings.

        The caller embeds them and selects the one closest to the PSO
        target position vector.
        """
        k   = self.k
        sys = (
            _PSO_CODER_SYSTEM.format(k=k)
            + f"\n\nYour particle role: {role_name} — {role_description}"
        )

        diff_section = ""
        if eval_diff:
            diff_section = (
                "--- CURRENT FAILURES (address these in your mutations) ---\n"
                + eval_diff
                + "\n\n"
            )

        # Deduplicate when pbest == gbest (avoid confusing the LLM with duplicates)
        if pbest_code.strip() == gbest_code.strip():
            ref_section = (
                f"Best code so far (fitness={gbest_fitness:.4f}):\n"
                f"```python\n{gbest_code}\n```\n\n"
            )
        else:
            ref_section = (
                "Personal best code:\n"
                f"```python\n{pbest_code}\n```\n\n"
                "Global best code:\n"
                f"```python\n{gbest_code}\n```\n\n"
            )

        # Generate diverse candidate descriptions based on k
        strategies = [
            "Fix the known failures in the best code above",
            "Try a different algorithmic approach entirely (different DSL primitives or logic)",
            "Combine the strongest elements of both reference codes",
            "Simplify to the minimal correct solution",
        ]
        strategy_lines = "\n".join(
            f"  transform_{i+1}: {strategies[i % len(strategies)]}"
            for i in range(k)
        )

        content = (
            f"{task_description}\n\n"
            f"{training_context}\n\n"
            "--- OPTIMIZATION CONTEXT ---\n"
            f"Current code fitness : {current_fitness:.4f} / 1.0000\n"
            f"Personal best fitness: {pbest_fitness:.4f} / 1.0000\n"
            f"Global best fitness  : {gbest_fitness:.4f} / 1.0000\n\n"
            + diff_section
            + ref_section
            + f"Generate {k} distinct `transform` functions.  "
            f"Use these {k} different strategies:\n"
            + strategy_lines
            + f"\n\nProvide exactly {k} ```python``` blocks (named transform_1 … transform_{k})."
        )

        messages = [{"role": "user", "content": content}]
        response = self._client.generate(sys, messages, temperature=temperature)

        # Extract all python code blocks
        blocks = re.findall(r"```python\s*(.*?)\s*```", response, re.DOTALL)

        candidates: list[str] = []
        for block in blocks:
            # Normalise function name to `transform` for sandbox compatibility
            normalised = re.sub(r"def\s+transform_\d+\s*\(", "def transform(", block.strip())
            if "def " in normalised:
                candidates.append(normalised)

        # Fallback: try extracting any single code block
        if not candidates:
            all_blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
            for b in all_blocks:
                if "def " in b:
                    candidates.append(re.sub(r"def\s+transform_\d+\s*\(", "def transform(", b.strip()))

        # Last resort: return personal best so particle doesn't go silent
        if not candidates:
            candidates.append(pbest_code or "def transform(input_grid):\n    return input_grid.copy()")

        return candidates[:k]
