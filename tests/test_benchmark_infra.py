"""Tests for benchmark infrastructure — task tiers, runner, and analysis."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from benchmarks.task_tiers import (
    TaskSpec,
    TIER_1,
    TIER_2,
    TIER_3,
    TIER_4,
    TIER_5,
    TIER_6,
    ALL_TIERS,
    SMOKE_TEST,
    get_tier,
    get_all_tasks,
)
from benchmarks.run_benchmark import (
    build_task_list,
    evaluate_task,
    run_benchmark,
    print_summary,
)
from benchmarks.analyze_results import (
    fitness_histogram,
    speed_analysis,
    failure_analysis,
    head_to_head,
)


# ---------------------------------------------------------------------------
# TaskSpec and tier data
# ---------------------------------------------------------------------------

class TestTaskTiers:
    def test_task_spec_frozen(self):
        spec = TaskSpec("abc", 1, 1, "desc")
        with pytest.raises(AttributeError):
            spec.task_id = "xyz"

    def test_all_tiers_contains_6_tiers(self):
        assert set(ALL_TIERS.keys()) == {1, 2, 3, 4, 5, 6}

    def test_each_tier_has_tasks(self):
        for tier_num, tasks in ALL_TIERS.items():
            assert len(tasks) > 0, f"Tier {tier_num} is empty"

    def test_task_ids_unique(self):
        all_ids = [t.task_id for t in get_all_tasks()]
        assert len(all_ids) == len(set(all_ids)), "Duplicate task IDs"

    def test_tier_matches_task_tier_field(self):
        for tier_num, tasks in ALL_TIERS.items():
            for t in tasks:
                assert t.tier == tier_num

    def test_smoke_test_has_3_tasks(self):
        assert len(SMOKE_TEST) == 3

    def test_smoke_test_covers_3_tiers(self):
        tiers = {t.tier for t in SMOKE_TEST}
        assert len(tiers) == 3

    def test_get_tier_returns_correct_list(self):
        assert get_tier(1) is TIER_1
        assert get_tier(99) == []

    def test_get_all_tasks_returns_all(self):
        total = sum(len(t) for t in ALL_TIERS.values())
        assert len(get_all_tasks()) == total


# ---------------------------------------------------------------------------
# build_task_list
# ---------------------------------------------------------------------------

class TestBuildTaskList:
    def test_smoke_flag(self):
        tasks = build_task_list(smoke=True)
        assert len(tasks) == 3

    def test_specific_tiers(self):
        tasks = build_task_list(tiers=[1, 2])
        assert all(t.tier in (1, 2) for t in tasks)
        assert len(tasks) == len(TIER_1) + len(TIER_2)

    def test_max_tasks(self):
        tasks = build_task_list(tiers=[1], max_tasks=2)
        assert len(tasks) == 2

    def test_no_tiers_returns_all(self):
        tasks = build_task_list()
        assert len(tasks) == len(get_all_tasks())


# ---------------------------------------------------------------------------
# evaluate_task
# ---------------------------------------------------------------------------

class TestEvaluateTask:
    def test_missing_task_file(self, tmp_path):
        spec = TaskSpec("nonexistent", 1, 1, "missing")
        solver = MagicMock()
        result = evaluate_task(solver, spec, tmp_path)
        assert result["solved"] is False
        assert "not found" in result["error"]

    def test_successful_solve(self, tmp_path):
        # Create a trivial task file
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "test": [
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
            ],
        }
        task_file = tmp_path / "test123.json"
        task_file.write_text(json.dumps(task))

        solver = MagicMock()
        solver.solve.return_value = {
            "success": True,
            "code": "def transform(input_grid):\n    return input_grid.copy()",
            "gbest_fitness": 1.0,
            "test_correct": True,
        }

        spec = TaskSpec("test123", 1, 1, "identity")
        result = evaluate_task(solver, spec, tmp_path, task_timeout=5.0)
        assert result["solved"] is True
        assert result["fitness"] == 1.0

    def test_solver_exception(self, tmp_path):
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [],
        }
        task_file = tmp_path / "crash.json"
        task_file.write_text(json.dumps(task))

        solver = MagicMock()
        solver.solve.side_effect = RuntimeError("boom")

        spec = TaskSpec("crash", 1, 1, "crash test")
        result = evaluate_task(solver, spec, tmp_path, task_timeout=5.0)
        assert result["solved"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_results_for_all_tasks(self, tmp_path):
        # Create two task files
        for tid in ["t1", "t2"]:
            task = {
                "train": [{"input": [[0]], "output": [[0]]}],
                "test": [],
            }
            (tmp_path / f"{tid}.json").write_text(json.dumps(task))

        solver = MagicMock()
        solver.solve.return_value = {
            "success": False,
            "code": "",
            "gbest_fitness": 0.5,
        }

        specs = [
            TaskSpec("t1", 1, 1, "task 1"),
            TaskSpec("t2", 2, 1, "task 2"),
        ]

        results = run_benchmark(solver, specs, tmp_path, task_timeout=5.0)
        assert len(results) == 2
        assert results[0]["task_id"] == "t1"
        assert results[1]["task_id"] == "t2"


# ---------------------------------------------------------------------------
# print_summary (just check it doesn't crash)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_empty_results(self, capsys):
        print_summary([])
        assert "No results" in capsys.readouterr().out

    def test_with_results(self, capsys):
        results = [
            {"task_id": "a", "tier": 1, "solved": True, "test_correct": True,
             "fitness": 1.0, "elapsed_s": 2.0, "error": None},
            {"task_id": "b", "tier": 2, "solved": False, "test_correct": False,
             "fitness": 0.3, "elapsed_s": 10.0, "error": None},
        ]
        print_summary(results)
        out = capsys.readouterr().out
        assert "1/2" in out
        assert "BENCHMARK" in out


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

class TestFitnessHistogram:
    def test_all_zeros(self):
        results = [{"fitness": 0.0}, {"fitness": 0.0}]
        hist = fitness_histogram(results, bins=5)
        assert hist["0.00-0.20"] == 2

    def test_perfect_scores(self):
        results = [{"fitness": 1.0}]
        hist = fitness_histogram(results, bins=5)
        # 1.0 maps to last bin
        assert hist["0.80-1.00"] == 1

    def test_mixed(self):
        results = [{"fitness": 0.1}, {"fitness": 0.5}, {"fitness": 0.9}]
        hist = fitness_histogram(results, bins=5)
        assert sum(hist.values()) == 3


class TestSpeedAnalysis:
    def test_basic(self):
        results = [
            {"elapsed_s": 1.0},
            {"elapsed_s": 3.0},
            {"elapsed_s": 5.0},
        ]
        speed = speed_analysis(results)
        assert speed["min_s"] == 1.0
        assert speed["max_s"] == 5.0
        assert speed["avg_s"] == 3.0
        assert speed["total_s"] == 9.0

    def test_empty(self):
        assert speed_analysis([]) == {}


class TestFailureAnalysis:
    def test_returns_only_failures(self):
        results = [
            {"task_id": "a", "solved": True, "fitness": 1.0},
            {"task_id": "b", "solved": False, "fitness": 0.7},
            {"task_id": "c", "solved": False, "fitness": 0.3},
        ]
        failed = failure_analysis(results)
        assert len(failed) == 2
        # Sorted by fitness descending
        assert failed[0]["task_id"] == "b"
        assert failed[1]["task_id"] == "c"


class TestHeadToHead:
    def test_basic_comparison(self):
        results_a = [
            {"task_id": "t1", "fitness": 0.5},
            {"task_id": "t2", "fitness": 0.8},
            {"task_id": "t3", "fitness": 1.0},
        ]
        results_b = [
            {"task_id": "t1", "fitness": 0.9},
            {"task_id": "t2", "fitness": 0.8},
            {"task_id": "t3", "fitness": 0.6},
        ]
        h2h = head_to_head(results_a, results_b)
        assert h2h["common_tasks"] == 3
        assert h2h["wins_b"] == 1   # t1 improved
        assert h2h["wins_a"] == 1   # t3 regressed
        assert h2h["ties"] == 1     # t2 same

    def test_no_common_tasks(self):
        h2h = head_to_head(
            [{"task_id": "a", "fitness": 1.0}],
            [{"task_id": "b", "fitness": 1.0}],
        )
        assert h2h["common_tasks"] == 0
