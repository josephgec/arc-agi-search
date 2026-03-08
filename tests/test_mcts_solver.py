"""Tests for agents.mcts_solver — Monte Carlo Tree Search over DSL programs."""
from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from unittest.mock import MagicMock

from agents.mcts_solver import MCTSNode, MCTSSolver, _SOLVED_THRESHOLD
from arc.dsl_actions import DslAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(inp, out, n_train=1):
    """Build a minimal ARC task dict."""
    inp = np.asarray(inp, dtype=np.int32)
    out = np.asarray(out, dtype=np.int32)
    return {
        "train": [{"input": inp.copy(), "output": out.copy()} for _ in range(n_train)],
        "test":  [{"input": inp.copy(), "output": out.copy()}],
    }


def _rotate_task():
    """Task where output = rotate(input, 1) — 90° CCW."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = np.rot90(inp, 1).copy()  # [[2,4],[1,3]]
    return _make_task(inp, out, n_train=2)


def _flip_task():
    """Task where output = flip(input, 0) — vertical flip on non-square grid."""
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    out = np.flip(inp, axis=0).copy()  # [[4,5,6],[1,2,3]]
    return _make_task(inp, out, n_train=2)


def _scale_task():
    """Task where output = scale(input, 2) — 2× upscale."""
    inp = np.array([[1, 2], [3, 0]], dtype=np.int32)
    out = np.kron(inp, np.ones((2, 2), dtype=np.int32))
    return _make_task(inp, out, n_train=2)


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_root_depth_is_zero(self):
        root = MCTSNode()
        assert root.depth() == 0

    def test_child_depth(self):
        root = MCTSNode()
        child = MCTSNode(action=DslAction("rotate", (1,)), parent=root)
        assert child.depth() == 1

    def test_action_sequence_root(self):
        root = MCTSNode()
        assert root.action_sequence() == []

    def test_action_sequence_chain(self):
        root = MCTSNode()
        child = MCTSNode(action=DslAction("rotate", (1,)), parent=root)
        grandchild = MCTSNode(action=DslAction("flip", (0,)), parent=child)
        seq = grandchild.action_sequence()
        assert len(seq) == 2
        assert seq[0].name == "rotate"
        assert seq[1].name == "flip"

    def test_ucb1_unvisited_is_inf(self):
        root = MCTSNode(visits=10)
        child = MCTSNode(parent=root, visits=0)
        assert child.ucb1() == float("inf")

    def test_ucb1_visited(self):
        root = MCTSNode(visits=100)
        child = MCTSNode(parent=root, visits=10, best_reward=0.5)
        score = child.ucb1()
        assert score > 0.5  # exploit + explore > exploit alone

    def test_is_fully_expanded(self):
        node = MCTSNode(untried_actions=[])
        assert node.is_fully_expanded
        node2 = MCTSNode(untried_actions=[DslAction("rotate", (1,))])
        assert not node2.is_fully_expanded

    def test_avg_reward(self):
        node = MCTSNode(visits=4, total_reward=2.0)
        assert node.avg_reward == 0.5

    def test_avg_reward_zero_visits(self):
        node = MCTSNode(visits=0, total_reward=0.0)
        assert node.avg_reward == 0.0


# ---------------------------------------------------------------------------
# MCTSSolver — actions_to_code
# ---------------------------------------------------------------------------

class TestActionsToCode:
    def test_empty_actions_returns_identity(self):
        solver = MCTSSolver()
        code = solver._actions_to_code([])
        assert "return input_grid.copy()" in code

    def test_single_action(self):
        solver = MCTSSolver()
        code = solver._actions_to_code([DslAction("rotate", (1,))])
        assert "def transform(input_grid):" in code
        assert "grid = rotate(grid, 1)" in code
        assert "return grid" in code

    def test_multiple_actions(self):
        solver = MCTSSolver()
        actions = [
            DslAction("flip", (0,)),
            DslAction("recolor", (1, 2)),
        ]
        code = solver._actions_to_code(actions)
        lines = code.split("\n")
        # Should have def, grid=copy, flip, recolor, return
        assert len(lines) == 5
        assert "flip(grid, 0)" in code
        assert "recolor(grid, 1, 2)" in code


# ---------------------------------------------------------------------------
# MCTSSolver — pipeline evaluation (deterministic)
# ---------------------------------------------------------------------------

class TestEvaluatePipeline:
    """Deterministic tests that the evaluation pipeline scores correctly."""

    def test_rotate_pipeline_scores_high(self):
        task = _rotate_task()
        solver = MCTSSolver()
        fitness = solver._evaluate_pipeline([DslAction("rotate", (1,))], task)
        assert fitness >= _SOLVED_THRESHOLD

    def test_flip_pipeline_scores_high(self):
        task = _flip_task()
        solver = MCTSSolver()
        fitness = solver._evaluate_pipeline([DslAction("flip", (0,))], task)
        assert fitness >= _SOLVED_THRESHOLD

    def test_scale_pipeline_scores_high(self):
        task = _scale_task()
        solver = MCTSSolver()
        fitness = solver._evaluate_pipeline([DslAction("scale", (2,))], task)
        assert fitness >= _SOLVED_THRESHOLD

    def test_wrong_pipeline_scores_low(self):
        task = _rotate_task()
        solver = MCTSSolver()
        # flip(0) is wrong for a rotate task
        fitness = solver._evaluate_pipeline([DslAction("flip", (0,))], task)
        assert fitness < _SOLVED_THRESHOLD


# ---------------------------------------------------------------------------
# MCTSSolver — solve() integration tests
# ---------------------------------------------------------------------------

class TestSolveIdentity:
    def test_identity_task_returns_success(self, identity_task):
        solver = MCTSSolver(max_iterations=100, max_time=10.0, debug=False)
        result = solver.solve(identity_task)
        assert result["success"] is True
        assert result["gbest_fitness"] >= _SOLVED_THRESHOLD
        assert "return input_grid.copy()" in result["code"]

    def test_identity_task_result_format(self, identity_task):
        solver = MCTSSolver(max_iterations=10, max_time=5.0)
        result = solver.solve(identity_task)
        assert "success" in result
        assert "code" in result
        assert "test_correct" in result
        assert "gbest_fitness" in result
        assert "prediction" in result
        assert "log" in result


class TestSolveRotate:
    def test_improves_beyond_identity(self):
        task = _rotate_task()
        solver = MCTSSolver(
            max_depth=3, max_iterations=500, max_time=10.0, debug=False,
        )
        identity_fitness = solver._evaluate_pipeline([], task)
        result = solver.solve(task)
        assert result["gbest_fitness"] > identity_fitness
        assert result["code"]


class TestSolveFlip:
    def test_improves_beyond_identity(self):
        task = _flip_task()
        solver = MCTSSolver(
            max_depth=3, max_iterations=500, max_time=10.0, debug=False,
        )
        identity_fitness = solver._evaluate_pipeline([], task)
        result = solver.solve(task)
        assert result["gbest_fitness"] > identity_fitness
        assert result["code"]


class TestSolveScale:
    def test_improves_beyond_identity(self):
        task = _scale_task()
        solver = MCTSSolver(
            max_depth=3, max_iterations=500, max_time=10.0, debug=False,
        )
        identity_fitness = solver._evaluate_pipeline([], task)
        result = solver.solve(task)
        assert result["gbest_fitness"] > identity_fitness


# ---------------------------------------------------------------------------
# MCTSSolver — caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_populates(self, identity_task):
        solver = MCTSSolver(max_iterations=10, max_time=5.0)
        solver.solve(identity_task)
        assert len(solver._eval_cache) > 0

    def test_cache_cleared_on_new_solve(self, identity_task):
        solver = MCTSSolver(max_iterations=10, max_time=5.0)
        solver.solve(identity_task)
        first_cache_size = len(solver._eval_cache)
        assert first_cache_size > 0
        solver.solve(identity_task)
        # Cache was cleared and repopulated — could be same or different size
        # but should not accumulate
        assert len(solver._eval_cache) <= first_cache_size + 5


# ---------------------------------------------------------------------------
# MCTSSolver — dead ends and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_train(self):
        task = {"train": [], "test": []}
        solver = MCTSSolver(max_iterations=10)
        result = solver.solve(task)
        assert result["success"] is False
        assert result["gbest_fitness"] == 0.0

    def test_max_time_respected(self):
        """Solver should stop within a reasonable margin of max_time."""
        inp = np.zeros((5, 5), dtype=np.int32)
        # Unsolvable: output is all 9s but no recolor path from 0→9
        out = np.full((5, 5), 9, dtype=np.int32)
        task = _make_task(inp, out, n_train=2)
        solver = MCTSSolver(
            max_iterations=100000, max_time=2.0, debug=False,
        )
        t0 = time.time()
        result = solver.solve(task)
        elapsed = time.time() - t0
        # Should finish within max_time + generous margin for overhead
        assert elapsed < 5.0
        # Should not have found a perfect solution
        assert result["gbest_fitness"] < 1.0

    def test_apply_action_returns_none_for_bad_function(self):
        solver = MCTSSolver()
        grid = np.zeros((3, 3), dtype=np.int32)
        result = solver._apply_action(grid, DslAction("nonexistent_func", ()))
        assert result is None

    def test_apply_action_returns_none_for_none_grid(self):
        solver = MCTSSolver()
        result = solver._apply_action(None, DslAction("rotate", (1,)))
        assert result is None

    def test_task_timeout_param_accepted(self, identity_task):
        """solve() accepts task_timeout kwarg without error."""
        solver = MCTSSolver(max_iterations=10, max_time=5.0)
        result = solver.solve(identity_task, task_timeout=30.0)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# MCTSSolver — multi-pair consistency
# ---------------------------------------------------------------------------

class TestMultiPair:
    def test_multi_pair_improves(self):
        """Solver should improve beyond identity with multiple training pairs."""
        pairs = []
        for i in range(3):
            rng = np.random.default_rng(i)
            inp = rng.integers(0, 10, size=(2, 2), dtype=np.int32)
            out = np.rot90(inp, 1).copy()
            pairs.append({"input": inp, "output": out})
        task = {
            "train": pairs,
            "test": [{"input": pairs[0]["input"].copy(),
                       "output": pairs[0]["output"].copy()}],
        }
        solver = MCTSSolver(
            max_depth=3, max_iterations=500, max_time=30.0, debug=False,
        )
        identity_fitness = solver._evaluate_pipeline([], task)
        result = solver.solve(task)
        assert result["gbest_fitness"] > identity_fitness
        assert result["code"]  # non-empty code


# ---------------------------------------------------------------------------
# MCTSSolver — test_correct field
# ---------------------------------------------------------------------------

class TestTestCorrect:
    def test_test_correct_true_on_solve(self, identity_task):
        solver = MCTSSolver(max_iterations=100, max_time=10.0)
        result = solver.solve(identity_task)
        assert result["test_correct"] is True

    def test_prediction_not_none_on_solve(self, identity_task):
        solver = MCTSSolver(max_iterations=100, max_time=10.0)
        result = solver.solve(identity_task)
        assert result["prediction"] is not None


# ---------------------------------------------------------------------------
# MCTSSolver — debug mode coverage
# ---------------------------------------------------------------------------

class TestDebugMode:
    def test_identity_debug_prints(self, identity_task, capsys):
        solver = MCTSSolver(max_iterations=10, max_time=5.0, debug=True)
        result = solver.solve(identity_task)
        captured = capsys.readouterr()
        assert "Identity transform" in captured.out

    def test_time_limit_debug_prints(self, capsys):
        # Use a complex task that can't be solved quickly — random target
        rng = np.random.default_rng(99)
        inp = rng.integers(0, 10, size=(10, 10), dtype=np.int32)
        out = rng.integers(0, 10, size=(8, 12), dtype=np.int32)
        task = _make_task(inp, out, n_train=1)
        solver = MCTSSolver(
            max_iterations=100000, max_time=0.01, debug=True,
        )
        solver.solve(task)
        captured = capsys.readouterr()
        assert "Time limit" in captured.out

    def test_solved_debug_prints(self, capsys):
        task = _rotate_task()
        solver = MCTSSolver(
            max_depth=3, max_iterations=2000, max_time=30.0, debug=True,
        )
        # Pre-seed a perfect solution in the cache won't help — we need to
        # actually find one. Use a task that's quick to solve.
        inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
        out = np.rot90(inp, 1).copy()
        task = _make_task(inp, out, n_train=1)
        solver.solve(task)
        captured = capsys.readouterr()
        # Should print either "Solved" or "new best fitness"
        assert "new best fitness" in captured.out or "Solved" in captured.out

    def test_progress_logging_at_200_iterations(self, capsys):
        rng = np.random.default_rng(99)
        inp = rng.integers(0, 10, size=(10, 10), dtype=np.int32)
        out = rng.integers(0, 10, size=(8, 12), dtype=np.int32)
        task = _make_task(inp, out, n_train=1)
        solver = MCTSSolver(
            max_iterations=201, max_time=30.0, debug=True,
        )
        solver.solve(task)
        captured = capsys.readouterr()
        assert "iter=200/" in captured.out


# ---------------------------------------------------------------------------
# MCTSSolver — _count_nodes
# ---------------------------------------------------------------------------

class TestCountNodes:
    def test_single_root(self):
        solver = MCTSSolver()
        root = MCTSNode()
        assert solver._count_nodes(root) == 1

    def test_root_with_children(self):
        solver = MCTSSolver()
        root = MCTSNode()
        child1 = MCTSNode(parent=root)
        child2 = MCTSNode(parent=root)
        root.children = [child1, child2]
        grandchild = MCTSNode(parent=child1)
        child1.children = [grandchild]
        assert solver._count_nodes(root) == 4


# ---------------------------------------------------------------------------
# MCTSSolver — _apply_action edge cases
# ---------------------------------------------------------------------------

class TestApplyActionEdgeCases:
    def test_non_ndarray_result_returns_none(self):
        """DSL function returning non-ndarray → None."""
        solver = MCTSSolver()
        grid = np.zeros((3, 3), dtype=np.int32)
        # Patch DSL_NAMESPACE to return a non-array
        with patch("agents.mcts_solver.DSL_NAMESPACE", {"bad_fn": lambda g: "not_array"}):
            result = solver._apply_action(grid, DslAction("bad_fn", ()))
            assert result is None

    def test_1d_result_returns_none(self):
        """DSL function returning 1D array → None."""
        solver = MCTSSolver()
        grid = np.zeros((3, 3), dtype=np.int32)
        with patch("agents.mcts_solver.DSL_NAMESPACE", {"bad_fn": lambda g: np.array([1, 2, 3])}):
            result = solver._apply_action(grid, DslAction("bad_fn", ()))
            assert result is None

    def test_empty_result_returns_none(self):
        """DSL function returning empty 2D array → None."""
        solver = MCTSSolver()
        grid = np.zeros((3, 3), dtype=np.int32)
        with patch("agents.mcts_solver.DSL_NAMESPACE", {"bad_fn": lambda g: np.zeros((0, 0), dtype=np.int32)}):
            result = solver._apply_action(grid, DslAction("bad_fn", ()))
            assert result is None

    def test_exception_in_dsl_returns_none(self):
        """DSL function that raises → None."""
        solver = MCTSSolver()
        grid = np.zeros((3, 3), dtype=np.int32)
        def raise_fn(g):
            raise ValueError("boom")
        with patch("agents.mcts_solver.DSL_NAMESPACE", {"bad_fn": raise_fn}):
            result = solver._apply_action(grid, DslAction("bad_fn", ()))
            assert result is None


# ---------------------------------------------------------------------------
# MCTSSolver — _simulate with None grid
# ---------------------------------------------------------------------------

class TestSimulateNoneGrid:
    def test_simulate_node_with_none_grid(self):
        """When a node has _intermediate_grid=None, simulate uses action sequence only."""
        task = _rotate_task()
        solver = MCTSSolver()
        node = MCTSNode(
            action=DslAction("rotate", (1,)),
            _intermediate_grid=None,
        )
        # Create a fake parent so action_sequence works
        root = MCTSNode()
        node.parent = root
        reward, code = solver._simulate(node, task)
        assert isinstance(reward, float)
        assert "rotate" in code


# ---------------------------------------------------------------------------
# MCTSSolver — _expand with empty untried_actions
# ---------------------------------------------------------------------------

class TestExpandEmpty:
    def test_expand_returns_same_node_when_no_untried(self):
        solver = MCTSSolver()
        task = _rotate_task()
        target = np.rot90(np.array([[1, 2], [3, 4]], dtype=np.int32), 1)
        node = MCTSNode(untried_actions=[])
        result = solver._expand(node, task, target)
        assert result is node


# ---------------------------------------------------------------------------
# MCTSSolver — _evaluate_pipeline edge cases
# ---------------------------------------------------------------------------

class TestEvaluatePipelineEdgeCases:
    def test_cache_hit(self):
        """Second call with same pipeline returns cached value."""
        task = _rotate_task()
        solver = MCTSSolver()
        actions = [DslAction("rotate", (1,))]
        f1 = solver._evaluate_pipeline(actions, task)
        f2 = solver._evaluate_pipeline(actions, task)
        assert f1 == f2
        # Should have exactly one cache entry for this code
        code = solver._actions_to_code(actions)
        assert code in solver._eval_cache

    def test_no_pairs_returns_zero(self):
        """evaluate_code returning no pairs → 0.0."""
        solver = MCTSSolver()
        task = _rotate_task()
        with patch("agents.mcts_solver.sandbox") as mock_sb:
            mock_sb.evaluate_code.return_value = {"pairs": []}
            fitness = solver._evaluate_pipeline([DslAction("rotate", (1,))], task)
            assert fitness == 0.0

    def test_pair_with_none_expected_skipped(self):
        """Pairs where expected is None are skipped."""
        solver = MCTSSolver()
        task = _rotate_task()
        pred_arr = np.array([[1, 2]], dtype=np.int32)
        with patch("agents.mcts_solver.sandbox") as mock_sb:
            mock_sb.evaluate_code.return_value = {
                "pairs": [
                    {"predicted": pred_arr, "expected": None},
                    {"predicted": pred_arr, "expected": pred_arr},
                ]
            }
            fitness = solver._evaluate_pipeline([DslAction("flip", (0,))], task)
            # Only one valid pair, so fitness is based on that pair
            assert fitness > 0


# ---------------------------------------------------------------------------
# MCTSSolver — _make_result with no test pairs
# ---------------------------------------------------------------------------

class TestMakeResultEdgeCases:
    def test_no_test_pairs(self):
        task = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}
        solver = MCTSSolver()
        result = solver._make_result(False, "", 0.0, [], task)
        assert result["prediction"] is None
        assert result["test_correct"] is None

    def test_no_code(self):
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[1]], "output": [[1]]}],
        }
        solver = MCTSSolver()
        result = solver._make_result(False, "", 0.0, [], task)
        assert result["prediction"] is None
