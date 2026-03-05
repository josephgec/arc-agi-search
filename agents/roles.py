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

Important categories to consider (pay attention to [Structural] hints in the examples):
- MOVEMENT / ATTRACTION: Do non-zero cells change position between input and output?
  If so, consider: do they compress toward a fixed "anchor" cell?  Do they encode
  a direction (displacement from anchor → which adjacent cell to occupy)?
- SATELLITE patterns: Some colors may act as "anchors" (stay fixed) while others
  are "satellites" (move, collapse, or reflect relative to the anchor).
- DIRECTION ENCODING: If scattered cells surround an anchor, their direction
  (up/down/left/right from anchor) may determine where they land in the output.
- COLOR PERMUTATION / SWAP: Check if output colors are always a permutation of
  input colors. If the [Cross-pair analysis] shows consistent swap pairs
  (e.g. 1↔5, 2↔6), the rule may be a FIXED GLOBAL SWAP TABLE applied to every
  cell. Implement with a dict mapping each color to its partner — never swap
  in-place, always read from the original and write to a copy.
- OUTPUT SYMMETRY: If [Cross-pair analysis] reports all outputs have 4-way,
  horizontal, or vertical symmetry, the rule likely CONSTRUCTS a symmetric
  output (e.g. tile the input then mirror, or concatenate original + flipped
  halves). The output shape will typically be 2× the input in one or both dims.
- BLOCK SELECTION: If the input is evenly divided into stacked equal-height
  blocks and [Cross-pair analysis] reports the output IS one of those blocks,
  the rule SELECTS a block based on some discriminating property (e.g. highest
  non-zero cell count, most unique colors, specific color present, etc.).
"""

_CODER_SYSTEM = (
    "You are an expert Python programmer solving ARC-AGI grid transformations.\n"
    "Translate the given hypothesis into a Python function called `transform` that\n"
    "takes `input_grid: np.ndarray` and returns `np.ndarray`.\n\n"
    + _DSL_REFERENCE
    + """
OUTPUT SHAPE — determine this BEFORE writing any logic:
  Check every training pair. Common cases:
    Same shape: output_grid = np.zeros(input_grid.shape, dtype=np.int32)
    Scaled N×:  output_grid = np.zeros((rows*N, cols*N), dtype=np.int32)
    Cropped:    output_grid = np.zeros(out_shape, dtype=np.int32)
  Add a comment at the top of your function stating the shape rule.

CRITICAL BUG TO AVOID — sequential color replacement corrupts itself:
  WRONG:  grid[grid==1]=2; grid[grid==2]=1  # second line overwrites the first!
  RIGHT:  out = input_grid.copy()
          out[input_grid==1] = 2
          out[input_grid==2] = 1           # read from original, write to copy

DEFENSIVE CODING — always guard against empty collections:
  objects = find_objects(input_grid)
  if not objects:
      return input_grid.copy()   # bail out rather than crash
  # Never call max()/min() on an empty list — check first.

  Similarly guard filtered results:
  large = [o for o in objects if get_size(o) > 10]
  if not large:
      large = objects  # fallback: use all objects

COMMON PATTERNS:
  Anchor+satellite: For each scattered satellite cell, compute (dr, dc) from nearest
    anchor cell. Dominant axis: if abs(dr)>=abs(dc) → vertical dir (up/down), else
    horizontal (left/right). Place satellite color at anchor ± 1 step in that direction.
    out = np.zeros_like(input_grid)
    for anchor in anchor_positions:
        out[anchor] = anchor_color
        for sat_r, sat_c in satellite_positions:
            dr = sat_r - anchor[0]; dc = sat_c - anchor[1]
            if abs(dr) >= abs(dc): nr, nc = anchor[0] + np.sign(dr), anchor[1]
            else:                  nr, nc = anchor[0], anchor[1] + np.sign(dc)
            out[nr, nc] = satellite_color
  Region fill: flood_fill(grid, r, c, color) for connected area; use fill_enclosed_regions()
    for cells fully enclosed by a border color.
  Object rank-sort: sorted(find_objects(g), key=get_size) — process in ascending size order.
  Cross/plus around center (r,c): place color at (r-1,c),(r+1,c),(r,c-1),(r,c+1).

Rules:
- Return ONLY a single ```python ... ``` code block.
- The function must be called `transform`.
- Use only the DSL primitives listed above plus plain Python/numpy.
- The function must handle every training example shown.
- No print statements, no side effects, no I/O.
- CRITICAL: Always clamp array indices near grid edges:
    rows, cols = grid.shape
    r_min, r_max = max(0, r-1), min(rows, r+2)
    c_min, c_max = max(0, c-1), min(cols, c+2)
  Never use grid[r-1:r+2] without clamping. Use the safe_neighbors()
  helper when you need a cell's neighbourhood.

MANDATORY OUTPUT FORMAT: You MUST wrap your code in a fenced block:
```python
def transform(input_grid):
    ...
```
Do NOT output code outside a fenced block. This is required.
"""
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
    + "\nOUTPUT SHAPE — determine this BEFORE writing any logic:\n"
    "  Check every training pair. Common cases:\n"
    "    Same shape: output_grid = np.zeros(input_grid.shape, dtype=np.int32)\n"
    "    Scaled N times: output_grid = np.zeros((rows*N, cols*N), dtype=np.int32)\n"
    "    Cropped: output_grid = np.zeros(out_shape, dtype=np.int32)\n\n"
    "CRITICAL BUG TO AVOID — sequential color replacement corrupts itself:\n"
    "  WRONG:  grid[grid==1]=2; grid[grid==2]=1  # second line overwrites the first!\n"
    "  RIGHT:  out = input_grid.copy()\n"
    "          out[input_grid==1] = 2\n"
    "          out[input_grid==2] = 1  # read from original, write to copy\n\n"
    "DEFENSIVE CODING — always guard against empty collections:\n"
    "  objects = find_objects(input_grid)\n"
    "  if not objects:\n"
    "      return input_grid.copy()   # bail out rather than crash\n"
    "  # Never call max()/min() on an empty list — check first.\n\n"
    "  Similarly guard filtered results:\n"
    "  large = [o for o in objects if get_size(o) > 10]\n"
    "  if not large:\n"
    "      large = objects  # fallback: use all objects\n\n"
    "COMMON PATTERNS:\n"
    "  Anchor+satellite: compute (dr,dc) from satellite to anchor. Dominant axis:\n"
    "    abs(dr)>=abs(dc) -> vertical (up/down), else horizontal (left/right).\n"
    "    Place satellite color at anchor +- 1 step in that direction.\n"
    "  Region fill: flood_fill(grid,r,c,color); fill_enclosed_regions() for cavities.\n"
    "  Object rank-sort: sorted(find_objects(g), key=get_size) — ascending size.\n"
    "  Cross/plus at (r,c): place color at (r-1,c),(r+1,c),(r,c-1),(r,c+1).\n"
    "\nRules:\n"
    "- Generate exactly {k} code blocks, each in its own ```python ... ``` fence.\n"
    "- Name the functions `transform_1`, `transform_2`, ... `transform_{k}`.\n"
    "- Each must be complete and executable (takes np.ndarray, returns np.ndarray).\n"
    "- Each variation should try a DIFFERENT strategy or fix — not minor wording tweaks.\n"
    "- Do NOT hardcode specific cell coordinates or shape-based if-else branches.\n"
    "- Prefer solutions that fix the systematic root cause of the listed failures.\n"
    "- No print, no I/O, no side effects.\n"
    "- CRITICAL: Always clamp array indices near grid edges:\n"
    "    rows, cols = grid.shape\n"
    "    r_min, r_max = max(0, r-1), min(rows, r+2)\n"
    "    c_min, c_max = max(0, c-1), min(cols, c+2)\n"
    "  Never use grid[r-1:r+2] without clamping. Use the safe_neighbors()\n"
    "  helper when you need a cell's neighbourhood.\n"
    "\nMANDATORY OUTPUT FORMAT: You MUST wrap each function in a fenced block:\n"
    "```python\n"
    "def transform_1(input_grid):\n"
    "    ...\n"
    "```\n"
    "Do NOT output code outside fenced blocks. This is required.\n"
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
        prior_failures:   list[tuple[str, str]] | None = None,
    ) -> str:
        """Return raw LLM response containing a ```python``` code block.

        Args:
            hypothesis:        The natural-language rule to implement.
            feedback:          Optional Critic feedback on a previous code attempt.
            training_context:  Formatted string of all training pairs.
            temperature:       Override the client's default temperature.
            prior_failures:    List of (code_snippet, what_went_wrong) from
                               previous failed attempts.  Shown as explicit
                               negative examples so the Coder avoids them.
        """
        content = f"Hypothesis:\n{hypothesis}"
        if training_context:
            content += f"\n\n{training_context}"
        if prior_failures:
            lines = [
                "\n--- PREVIOUSLY FAILED IMPLEMENTATIONS — do NOT repeat these patterns ---"
            ]
            for i, (snippet, why) in enumerate(prior_failures, 1):
                short = "\n".join(snippet.splitlines()[:8])
                lines.append(f"\n[{i}] What went wrong: {why}")
                lines.append(f"Code snippet:\n```python\n{short}\n```")
            content += "\n".join(lines)
        if feedback:
            content += (
                "\n\n--- CRITIC FEEDBACK ---\n"
                + feedback
                + "\n\nFix the code based on the above feedback."
            )
        else:
            content += "\n\nImplement the `transform` function."

        messages = [{"role": "user", "content": content}]
        return self._client.generate(
            _CODER_SYSTEM, messages,
            temperature=temperature,
            model_override="qwen2.5-coder:7b",
        )


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
        task_description:   str,
        training_context:   str,
        current_code:       str,
        current_fitness:    float,
        pbest_code:         str,
        pbest_fitness:      float,
        gbest_code:         str,
        gbest_fitness:      float,
        role_name:          str,
        role_description:   str,
        eval_diff:          str | None = None,
        temperature:        float | None = None,
        failed_examples:    list[tuple[str, float, str]] | None = None,
        pair_fitness_scores: list[float] | None = None,
        hypothesis:         str | None = None,
    ) -> list[str]:
        """Return up to self.k candidate code strings.

        The caller evaluates them in the sandbox and selects the highest-fitness one.

        Args:
            failed_examples:     List of (code_snippet, fitness, error_description)
                                 from previously tried candidates that were below
                                 pbest.  Shown to the LLM as explicit negative
                                 examples to avoid.
            pair_fitness_scores: Per-training-pair fitness list, e.g. [0.8, 0.2, 0.9].
                                 Highlights which specific pairs are hardest to fix.
            hypothesis:          Optional fresh hypothesis from the Hypothesizer to
                                 guide mutations toward a new strategy.
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

        failed_section = ""
        if failed_examples:
            lines = [
                "--- PREVIOUSLY FAILED APPROACHES — do NOT repeat these patterns ---"
            ]
            for i, (snippet, fit, err_desc) in enumerate(failed_examples, 1):
                short = "\n".join(snippet.splitlines()[:8])
                lines.append(f"[{i}] Fitness: {fit:.4f} — {err_desc}")
                lines.append(f"```python\n{short}\n```")
            failed_section = "\n".join(lines) + "\n\n"

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

        # Per-pair fitness breakdown — highlights the weakest training pairs
        pair_section = ""
        if pair_fitness_scores:
            pair_lines = ["--- PER-PAIR FITNESS (focus mutations on failing pairs) ---"]
            for idx, pf in enumerate(pair_fitness_scores):
                marker = " ← WEAKEST" if pf == min(pair_fitness_scores) else ""
                pair_lines.append(f"  Pair {idx+1}: {pf:.4f}{marker}")
            pair_section = "\n".join(pair_lines) + "\n\n"

        # Fresh hypothesis to guide a new strategy direction
        hyp_section = ""
        if hypothesis:
            hyp_section = (
                "--- FRESH HYPOTHESIS (try implementing this new approach) ---\n"
                + hypothesis
                + "\n\n"
            )

        content = (
            f"{task_description}\n\n"
            f"{training_context}\n\n"
            "--- OPTIMIZATION CONTEXT ---\n"
            f"Current code fitness : {current_fitness:.4f} / 1.0000\n"
            f"Personal best fitness: {pbest_fitness:.4f} / 1.0000\n"
            f"Global best fitness  : {gbest_fitness:.4f} / 1.0000\n\n"
            + pair_section
            + diff_section
            + failed_section
            + hyp_section
            + ref_section
            + f"Generate {k} distinct `transform` functions.  "
            f"Use these {k} different strategies:\n"
            + strategy_lines
            + f"\n\nProvide exactly {k} ```python``` blocks (named transform_1 … transform_{k})."
        )

        messages = [{"role": "user", "content": content}]
        response = self._client.generate(
            sys, messages,
            temperature=temperature,
            model_override="qwen2.5-coder:7b",
        )

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
