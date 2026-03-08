"""Monte Carlo Tree Search over partial DSL programs for ARC-AGI.

Builds programs incrementally — one DSL operation per tree level.
Uses UCB1 for selection, DSL enumeration for expansion, random
rollouts for simulation, and continuous fitness for backpropagation.

This solver is LLM-free: it explores the space of DSL compositions
using only the CPU and the existing sandbox evaluation infrastructure.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from arc.grid import Grid, grids_equal
from arc.evaluate import calculate_continuous_fitness
from arc import sandbox
from arc.dsl_actions import (
    DslAction,
    enumerate_actions,
    get_target_shape,
    prune_actions,
)
from arc.sandbox import DSL_NAMESPACE

logger = logging.getLogger(__name__)

# Floating-point threshold for "solved" (1.0 can be ~0.9999999999999999)
_SOLVED_THRESHOLD = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """A node in the MCTS tree, representing a partial DSL pipeline."""

    action: DslAction | None = None        # None for root
    parent: "MCTSNode | None" = None       # None for root
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0               # max reward in subtree
    best_code: str = ""                    # code string that achieved best_reward
    untried_actions: list[DslAction] = field(default_factory=list)
    _intermediate_grid: Grid | None = None  # cached grid after this action

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.best_reward
        explore = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploit + explore

    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def action_sequence(self) -> list[DslAction]:
        """Return the list of actions from root to this node."""
        actions: list[DslAction] = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions


# ---------------------------------------------------------------------------
# MCTSSolver
# ---------------------------------------------------------------------------

# Progressive widening constants
_PW_C = 4.0
_PW_ALPHA = 0.5


class MCTSSolver:
    """MCTS solver that builds DSL pipelines incrementally."""

    def __init__(
        self,
        max_depth: int = 5,
        max_iterations: int = 2000,
        max_time: float = 120.0,
        exploration: float = 1.414,
        rollout_depth: int = 3,
        n_rollouts: int = 1,
        debug: bool = False,
    ) -> None:
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.n_rollouts = n_rollouts
        self.debug = debug

        # Evaluation cache: code string → fitness
        self._eval_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict, task_timeout: float = 0.0) -> dict:
        """Run MCTS and return the best solution found.

        Returns the same dict shape as other solvers:
            success, code, test_correct, gbest_fitness, prediction, log
        """
        self._eval_cache.clear()
        train = task.get("train", [])
        if not train:
            return self._make_result(False, "", 0.0, [], task)

        t0 = time.time()

        # Use first training pair for intermediate grid tracking
        first_input = np.asarray(train[0]["input"], dtype=np.int32)
        first_target = np.asarray(train[0]["output"], dtype=np.int32)

        # Check identity (output == input for all pairs)
        identity_code = "def transform(input_grid):\n    return input_grid.copy()"
        identity_fitness = self._evaluate_pipeline([], task)
        if identity_fitness >= _SOLVED_THRESHOLD:
            if self.debug:
                print("[MCTS] Identity transform solves all training pairs")
            return self._make_result(True, identity_code, 1.0, [], task)

        # Build root node
        rng = np.random.default_rng()
        root = MCTSNode(_intermediate_grid=first_input)
        actions = enumerate_actions(first_input, task)
        root.untried_actions = prune_actions(actions, first_input, first_target)
        rng.shuffle(root.untried_actions)
        root.best_reward = identity_fitness
        root.best_code = identity_code

        log: list[dict] = []
        best_fitness = identity_fitness
        best_code = identity_code

        for iteration in range(self.max_iterations):
            elapsed = time.time() - t0
            if elapsed >= self.max_time:
                if self.debug:
                    print(f"[MCTS] Time limit reached at iteration {iteration}")
                break

            # Selection
            node = self._select(root)

            # Expansion
            if (not node.is_fully_expanded
                    and node.depth() < self.max_depth
                    and node._intermediate_grid is not None):
                # Progressive widening check
                pw_limit = math.ceil(_PW_C * (node.visits ** _PW_ALPHA)) if node.visits > 0 else 1
                if len(node.children) < pw_limit:
                    node = self._expand(node, task, first_target)

            # Simulation
            reward, code = self._simulate(node, task)

            # Backpropagation
            self._backpropagate(node, reward, code)

            # Track global best
            if reward > best_fitness:
                best_fitness = reward
                best_code = code
                if self.debug:
                    print(
                        f"[MCTS] iter={iteration + 1} new best fitness={best_fitness:.4f} "
                        f"depth={node.depth()} actions={node.action_sequence()}"
                    )

            # Early termination
            if best_fitness >= _SOLVED_THRESHOLD:
                if self.debug:
                    print(f"[MCTS] Solved at iteration {iteration + 1}!")
                break

            # Progress logging
            if self.debug and (iteration + 1) % 200 == 0:
                print(
                    f"[MCTS] iter={iteration + 1}/{self.max_iterations} "
                    f"best={best_fitness:.4f} "
                    f"nodes={self._count_nodes(root)} "
                    f"cache={len(self._eval_cache)} "
                    f"time={elapsed:.1f}s"
                )

        elapsed = time.time() - t0
        logger.info(
            "[MCTS] solved=%s fitness=%.4f iterations=%d time=%.1fs cache=%d",
            best_fitness >= _SOLVED_THRESHOLD, best_fitness, iteration + 1,
            elapsed, len(self._eval_cache),
        )
        return self._make_result(
            best_fitness >= _SOLVED_THRESHOLD, best_code, best_fitness, log, task,
        )

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 tree descent until reaching an expandable or leaf node."""
        while (node.is_fully_expanded
               and node.children
               and node.depth() < self.max_depth):
            node = max(
                node.children, key=lambda c: c.ucb1(self.exploration)
            )
        return node

    def _expand(
        self, node: MCTSNode, task: dict, first_target: Grid,
    ) -> MCTSNode:
        """Create a child for one untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        child_grid = self._apply_action(node._intermediate_grid, action)

        child = MCTSNode(
            action=action,
            parent=node,
            _intermediate_grid=child_grid,
        )

        if child_grid is not None and child.depth() < self.max_depth:
            actions = enumerate_actions(child_grid, task)
            child.untried_actions = prune_actions(
                actions, child_grid, first_target,
            )
            np.random.default_rng().shuffle(child.untried_actions)

        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode, task: dict) -> tuple[float, str]:
        """Random rollout from this node, return (fitness, code)."""
        actions = node.action_sequence()
        grid = node._intermediate_grid

        if grid is None:
            code = self._actions_to_code(actions)
            return self._evaluate_pipeline(actions, task), code

        rng = np.random.default_rng()
        rollout_actions = list(actions)

        for _ in range(self.rollout_depth):
            if len(rollout_actions) >= self.max_depth:
                break
            available = enumerate_actions(grid, task)
            if not available:
                break
            action = available[int(rng.integers(len(available)))]
            new_grid = self._apply_action(grid, action)
            if new_grid is None:
                break
            grid = new_grid
            rollout_actions.append(action)

        code = self._actions_to_code(rollout_actions)
        return self._evaluate_pipeline(rollout_actions, task), code

    def _backpropagate(
        self, node: MCTSNode, reward: float, code: str,
    ) -> None:
        """Walk from node to root, updating stats."""
        n: MCTSNode | None = node
        while n is not None:
            n.visits += 1
            n.total_reward += reward
            if reward > n.best_reward:
                n.best_reward = reward
                n.best_code = code
            n = n.parent

    # ------------------------------------------------------------------
    # Action execution (in-process, no subprocess)
    # ------------------------------------------------------------------

    def _apply_action(self, grid: Grid, action: DslAction) -> Grid | None:
        """Apply a single DSL action to a grid. Returns None on failure."""
        if grid is None:
            return None
        fn = DSL_NAMESPACE.get(action.name)
        if fn is None:
            return None
        try:
            result = fn(grid, *action.args)
            if not isinstance(result, np.ndarray):
                return None
            if result.ndim != 2:
                return None
            if result.size == 0:
                return None
            return result.astype(np.int32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Pipeline evaluation (via sandbox for full safety)
    # ------------------------------------------------------------------

    def _actions_to_code(self, actions: list[DslAction]) -> str:
        """Convert an action sequence to a transform() function string."""
        if not actions:
            return "def transform(input_grid):\n    return input_grid.copy()"
        lines = ["def transform(input_grid):"]
        lines.append("    grid = input_grid.copy()")
        for action in actions:
            lines.append(f"    grid = {action.to_code_fragment()}")
        lines.append("    return grid")
        return "\n".join(lines)

    def _evaluate_pipeline(
        self, actions: list[DslAction], task: dict,
    ) -> float:
        """Evaluate a pipeline against ALL training pairs. Returns avg fitness."""
        code = self._actions_to_code(actions)

        cached = self._eval_cache.get(code)
        if cached is not None:
            return cached

        result = sandbox.evaluate_code(code, task)
        pairs = result.get("pairs", [])
        if not pairs:
            self._eval_cache[code] = 0.0
            return 0.0

        total = 0.0
        for pair_res in pairs:
            pred = pair_res.get("predicted")
            expected = pair_res.get("expected")
            if expected is None:
                continue
            expected_arr = np.asarray(expected, dtype=np.int32)
            total += calculate_continuous_fitness(pred, expected_arr)

        fitness = total / len(pairs)
        self._eval_cache[code] = fitness
        return fitness

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def _make_result(
        self,
        success: bool,
        code: str,
        fitness: float,
        log: list,
        task: dict,
    ) -> dict:
        """Build the standard result dict."""
        prediction = None
        test_correct = None

        test_pairs = task.get("test", [])
        if test_pairs and code:
            test_input = test_pairs[0].get("input")
            test_output = test_pairs[0].get("output")

            if test_input is not None:
                inp_arr = np.asarray(test_input, dtype=np.int32)
                out, _ = sandbox.execute(code, inp_arr)
                if out is not None:
                    prediction = out
                    if test_output is not None:
                        exp_arr = np.asarray(test_output, dtype=np.int32)
                        test_correct = grids_equal(out, exp_arr)

        return {
            "success": success,
            "code": code,
            "test_correct": test_correct,
            "gbest_fitness": fitness,
            "prediction": prediction,
            "log": log,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in the tree (for debug logging)."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
