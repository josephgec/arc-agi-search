"""Tests for arc/evaluate.py — binary evaluation + continuous fitness."""
from __future__ import annotations

import numpy as np
import pytest

from arc.evaluate import calculate_continuous_fitness, evaluate_task, evaluate_directory, _count_objects_total


# ---------------------------------------------------------------------------
# calculate_continuous_fitness
# ---------------------------------------------------------------------------

class TestContinuousFitness:
    def test_perfect_match_is_one(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert calculate_continuous_fitness(g, g) == pytest.approx(1.0)

    def test_none_pred_is_zero(self):
        target = np.array([[1, 2]], dtype=np.int32)
        assert calculate_continuous_fitness(None, target) == pytest.approx(0.0)

    def test_all_wrong_pixels_still_positive_if_colors_match(self):
        # Same shape, same colour palette, but every pixel wrong
        pred   = np.array([[1, 2], [2, 1]], dtype=np.int32)
        target = np.array([[2, 1], [1, 2]], dtype=np.int32)
        f = calculate_continuous_fitness(pred, target)
        assert 0.0 < f < 1.0   # colour score lifts it above zero

    def test_wrong_shape_penalised(self):
        pred   = np.array([[1, 2, 3]], dtype=np.int32)   # 1×3
        target = np.array([[1, 2], [3, 4]], dtype=np.int32)  # 2×2
        f = calculate_continuous_fitness(pred, target)
        assert 0.0 < f < 1.0

    def test_partial_pixel_accuracy(self):
        pred   = np.array([[1, 2], [3, 9]], dtype=np.int32)  # 3/4 correct
        target = np.array([[1, 2], [3, 4]], dtype=np.int32)
        f = calculate_continuous_fitness(pred, target)
        # pixel_score ≈ 3/4=0.75, dim=1.0, color < 1 (extra color 9)
        # 0.2*1.0 + 0.3*color + 0.5*0.75
        assert 0.5 < f < 1.0

    def test_color_jaccard_partial(self):
        # pred has colors {0,1}, target has {0,2}: intersection={0}, union={0,1,2}
        pred   = np.array([[0, 1]], dtype=np.int32)
        target = np.array([[0, 2]], dtype=np.int32)
        f = calculate_continuous_fitness(pred, target)
        # dim=1.0, color=1/3≈0.33, pixel=0.5 (only [0,0] matches)
        expected = 0.20 * 1.0 + 0.30 * (1/3) + 0.50 * 0.5
        assert f == pytest.approx(expected, abs=0.01)

    def test_completely_wrong_shape_and_colors(self):
        pred   = np.array([[5, 6, 7]], dtype=np.int32)   # 1×3
        target = np.array([[1, 2], [3, 4]], dtype=np.int32)  # 2×2
        f = calculate_continuous_fitness(pred, target)
        assert 0.0 <= f < 1.0

    def test_returns_float(self):
        g = np.array([[1]], dtype=np.int32)
        assert isinstance(calculate_continuous_fitness(g, g), float)

    def test_score_in_unit_interval(self):
        """Fitness must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            pred   = rng.integers(0, 9, size=(rng.integers(1, 5), rng.integers(1, 5))).astype(np.int32)
            target = rng.integers(0, 9, size=(rng.integers(1, 5), rng.integers(1, 5))).astype(np.int32)
            f = calculate_continuous_fitness(pred, target)
            assert 0.0 <= f <= 1.0, f"Fitness {f} out of [0,1]"


# ---------------------------------------------------------------------------
# evaluate_task
# ---------------------------------------------------------------------------

class TestEvaluateTask:
    def test_all_correct(self, identity_task):
        fn = lambda g: g.copy()
        result = evaluate_task(identity_task, fn)
        assert result["all_correct"] is True
        assert result["n_correct"] == result["n_total"]

    def test_all_wrong(self, identity_task):
        fn = lambda g: np.zeros_like(g)
        result = evaluate_task(identity_task, fn)
        # identity_task has non-zero values; zeroing makes them wrong
        assert result["n_correct"] < result["n_total"]

    def test_exception_caught(self, identity_task):
        def exploding(g):
            raise ValueError("boom")

        result = evaluate_task(identity_task, exploding)
        assert result["n_correct"] == 0
        for pair in result["pairs"]:
            assert pair["error"] is not None
            assert "ValueError" in pair["error"]

    def test_result_keys(self, identity_task):
        result = evaluate_task(identity_task, lambda g: g.copy())
        assert "pairs" in result
        assert "all_correct" in result
        assert "n_correct" in result
        assert "n_total" in result

    def test_n_total_equals_train_length(self, identity_task):
        result = evaluate_task(identity_task, lambda g: g.copy())
        assert result["n_total"] == len(identity_task["train"])

    def test_pair_result_shape(self, identity_task):
        result = evaluate_task(identity_task, lambda g: g.copy())
        for pair in result["pairs"]:
            assert "correct" in pair
            assert "predicted" in pair
            assert "expected" in pair
            assert "error" in pair

    def test_mixed_correct(self, recolor_task):
        # Only correct for pairs where input has no 1s
        fn = lambda g: g.copy()   # wrong for recolor task
        result = evaluate_task(recolor_task, fn)
        assert not result["all_correct"]


# ---------------------------------------------------------------------------
# evaluate_directory
# ---------------------------------------------------------------------------

class TestCountObjectsTotal:
    def test_empty_grid(self):
        g = np.zeros((0, 0), dtype=np.int32)
        assert _count_objects_total(g) == 0

    def test_single_blob(self):
        # One connected region of color 1 against background 0
        g = np.array([[0, 1, 1],
                      [0, 1, 0],
                      [0, 0, 0]], dtype=np.int32)
        assert _count_objects_total(g) == 1

    def test_two_disjoint_same_color_blobs(self):
        g = np.array([[1, 0, 1],
                      [0, 0, 0],
                      [0, 0, 0]], dtype=np.int32)
        assert _count_objects_total(g) == 2

    def test_two_adjacent_different_colors(self):
        # Colors 1 and 2 touching against zero background — each is its own component
        g = np.array([[0, 1, 2, 0]], dtype=np.int32)
        assert _count_objects_total(g) == 2


class TestDynamicFitnessWeighting:
    def test_perfect_match_progress_none(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert calculate_continuous_fitness(g, g) == pytest.approx(1.0)

    def test_color_jaccard_formula_unchanged_with_progress_none(self):
        # Reproduce existing test_color_jaccard_partial with progress=None
        pred   = np.array([[0, 1]], dtype=np.int32)
        target = np.array([[0, 2]], dtype=np.int32)
        f = calculate_continuous_fitness(pred, target, progress=None)
        expected = 0.20 * 1.0 + 0.30 * (1/3) + 0.50 * 0.5
        assert f == pytest.approx(expected, abs=0.01)

    def test_progress_values_in_unit_interval(self):
        pred   = np.array([[1, 0]], dtype=np.int32)
        target = np.array([[1, 2]], dtype=np.int32)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            f = calculate_continuous_fitness(pred, target, progress=p)
            assert 0.0 <= f <= 1.0, f"progress={p} gave f={f}"

    def test_early_vs_late_pixel_weight(self):
        # Pixel-perfect but object count mismatch: at progress=1.0 pixel dominates
        # so the score should be higher (pixel accuracy=1.0 gets 65% weight late)
        pred   = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target = np.array([[1, 2], [3, 4]], dtype=np.int32)
        early  = calculate_continuous_fitness(pred, target, progress=0.0)
        late   = calculate_continuous_fitness(pred, target, progress=1.0)
        # Both should be 1.0 for perfect match
        assert early == pytest.approx(1.0)
        assert late  == pytest.approx(1.0)

    def test_object_count_mismatch_reduces_score(self):
        # pred has 1 object, target has 3; otherwise same palette & shape
        pred   = np.array([[1, 0, 0],
                           [0, 0, 0]], dtype=np.int32)
        target = np.array([[1, 0, 1],
                           [0, 0, 1]], dtype=np.int32)
        f_mismatch = calculate_continuous_fitness(pred, target, progress=0.5)
        # Compare against perfect object match (same grids)
        f_perfect  = calculate_continuous_fitness(target, target, progress=0.5)
        assert f_mismatch < f_perfect

    def test_perfect_match_late_progress(self):
        g = np.array([[1, 2, 3]], dtype=np.int32)
        f = calculate_continuous_fitness(g, g, progress=1.0)
        assert f == pytest.approx(1.0)


class TestEvaluateDirectory:
    def test_single_file(self, tmp_task_file):
        import json
        from arc.evaluate import evaluate_directory

        result = evaluate_directory(tmp_task_file.parent, lambda g: g.copy())
        assert result["n_tasks"] >= 1
        assert "n_solved" in result
        assert "tasks" in result

    def test_empty_directory(self, tmp_path):
        result = evaluate_directory(tmp_path, lambda g: g.copy())
        assert result["n_tasks"] == 0
        assert result["n_solved"] == 0
