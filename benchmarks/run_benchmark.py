#!/usr/bin/env python3
"""Run MCTS (or other solver) benchmarks on curated task tiers.

Usage examples
--------------
# Smoke test (3 tasks, fast):
python -m benchmarks.run_benchmark --tiers 1 --max-tasks 3 --solver mcts

# Full tier-1 benchmark:
python -m benchmarks.run_benchmark --tiers 1

# All tiers with custom MCTS params:
python -m benchmarks.run_benchmark --tiers 1 2 3 4 5 6 \
    --mcts-max-iterations 5000 --mcts-max-time 300

# Save results to JSON:
python -m benchmarks.run_benchmark --tiers 1 2 3 --output results.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np

from arc.grid import load_task, grids_equal
from arc import sandbox
from arc.evaluate import calculate_continuous_fitness
from agents.mcts_solver import MCTSSolver
from benchmarks.task_tiers import TaskSpec, get_tier, get_all_tasks, SMOKE_TEST


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    solver,
    spec: TaskSpec,
    data_dir: Path,
    task_timeout: float = 120.0,
) -> dict:
    """Run solver on a single task and return result dict."""
    task_path = data_dir / f"{spec.task_id}.json"
    if not task_path.exists():
        return {
            "task_id": spec.task_id,
            "tier": spec.tier,
            "description": spec.description,
            "solved": False,
            "train_correct": False,
            "test_correct": None,
            "fitness": 0.0,
            "elapsed_s": 0.0,
            "error": f"Task file not found: {task_path}",
        }

    task = load_task(task_path)
    t0 = time.time()

    try:
        # Run with wall-clock timeout via ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(solver.solve, task, task_timeout=task_timeout)
            try:
                result = future.result(timeout=task_timeout + 5)
            except FuturesTimeout:
                result = {
                    "success": False,
                    "code": "",
                    "gbest_fitness": 0.0,
                    "error": "timeout",
                }
    except Exception as exc:
        result = {
            "success": False,
            "code": "",
            "gbest_fitness": 0.0,
            "error": str(exc),
        }

    elapsed = time.time() - t0

    # Evaluate on test pair if available
    test_correct = result.get("test_correct")
    test_fitness = 0.0
    test_pairs = task.get("test", [])
    code = result.get("code", "")

    if test_pairs and code and test_correct is None:
        test_input = test_pairs[0].get("input")
        test_output = test_pairs[0].get("output")
        if test_input is not None:
            inp_arr = np.asarray(test_input, dtype=np.int32)
            out, _ = sandbox.execute(code, inp_arr)
            if out is not None:
                if test_output is not None:
                    exp_arr = np.asarray(test_output, dtype=np.int32)
                    test_correct = grids_equal(out, exp_arr)
                    test_fitness = calculate_continuous_fitness(out, exp_arr)

    return {
        "task_id": spec.task_id,
        "tier": spec.tier,
        "description": spec.description,
        "expected_ops": spec.expected_ops,
        "solved": result.get("success", False),
        "train_correct": result.get("success", False),
        "test_correct": test_correct,
        "fitness": result.get("gbest_fitness", 0.0),
        "test_fitness": test_fitness,
        "elapsed_s": round(elapsed, 2),
        "code": code,
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# Build task list
# ---------------------------------------------------------------------------

def build_task_list(
    tiers: list[int] | None = None,
    smoke: bool = False,
    max_tasks: int | None = None,
) -> list[TaskSpec]:
    """Build the list of tasks to benchmark."""
    if smoke:
        return list(SMOKE_TEST)

    if tiers:
        tasks: list[TaskSpec] = []
        for t in tiers:
            tasks.extend(get_tier(t))
    else:
        tasks = get_all_tasks()

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


# ---------------------------------------------------------------------------
# Build solver
# ---------------------------------------------------------------------------

def build_solver(args: argparse.Namespace) -> MCTSSolver:
    """Construct solver from CLI args."""
    return MCTSSolver(
        max_depth=args.mcts_max_depth,
        max_iterations=args.mcts_max_iterations,
        max_time=args.mcts_max_time,
        exploration=args.mcts_exploration,
        rollout_depth=args.mcts_rollout_depth,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    solver,
    tasks: list[TaskSpec],
    data_dir: Path,
    task_timeout: float = 120.0,
    debug: bool = False,
) -> list[dict]:
    """Run solver on all tasks and return list of result dicts."""
    results: list[dict] = []

    for i, spec in enumerate(tasks, 1):
        if debug:
            print(f"[{i}/{len(tasks)}] {spec.task_id} (tier {spec.tier}): "
                  f"{spec.description} ... ", end="", flush=True)

        result = evaluate_task(solver, spec, data_dir, task_timeout=task_timeout)
        results.append(result)

        if debug:
            status = "SOLVED" if result["solved"] else f"fitness={result['fitness']:.4f}"
            print(f"{status} ({result['elapsed_s']:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print a concise summary of benchmark results."""
    if not results:
        print("No results.")
        return

    # Overall
    total = len(results)
    solved = sum(1 for r in results if r["solved"])
    test_ok = sum(1 for r in results if r.get("test_correct"))
    avg_fitness = sum(r["fitness"] for r in results) / total
    avg_time = sum(r["elapsed_s"] for r in results) / total
    total_time = sum(r["elapsed_s"] for r in results)

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Tasks:         {total}")
    print(f"Train solved:  {solved}/{total} ({100*solved/total:.1f}%)")
    print(f"Test correct:  {test_ok}/{total} ({100*test_ok/total:.1f}%)")
    print(f"Avg fitness:   {avg_fitness:.4f}")
    print(f"Avg time:      {avg_time:.1f}s")
    print(f"Total time:    {total_time:.1f}s")

    # Per-tier breakdown
    tiers = sorted(set(r["tier"] for r in results))
    if len(tiers) > 1:
        print(f"\n{'Tier':<6} {'Solved':<10} {'Fitness':<10} {'Time':<10}")
        print("-" * 36)
        for tier in tiers:
            tier_results = [r for r in results if r["tier"] == tier]
            t_total = len(tier_results)
            t_solved = sum(1 for r in tier_results if r["solved"])
            t_fitness = sum(r["fitness"] for r in tier_results) / t_total
            t_time = sum(r["elapsed_s"] for r in tier_results) / t_total
            print(f"{tier:<6} {t_solved}/{t_total:<8} {t_fitness:<10.4f} {t_time:<10.1f}s")

    # Failed tasks
    failed = [r for r in results if not r["solved"]]
    if failed:
        print(f"\nFailed tasks:")
        for r in failed:
            err = f" ({r['error']})" if r.get("error") else ""
            print(f"  {r['task_id']} (tier {r['tier']}): "
                  f"fitness={r['fitness']:.4f}{err}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ARC-AGI solver benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--tiers", type=int, nargs="+", default=None,
                    help="Which tiers to run (e.g. 1 2 3)")
    p.add_argument("--smoke", action="store_true",
                    help="Run smoke test (3 tasks)")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data/training"))
    p.add_argument("--task-timeout", type=float, default=120.0)

    # Solver selection
    p.add_argument("--solver", default="mcts", choices=["mcts"],
                    help="Solver to benchmark")

    # MCTS params
    p.add_argument("--mcts-max-depth", type=int, default=5)
    p.add_argument("--mcts-max-iterations", type=int, default=2000)
    p.add_argument("--mcts-max-time", type=float, default=120.0)
    p.add_argument("--mcts-exploration", type=float, default=1.414)
    p.add_argument("--mcts-rollout-depth", type=int, default=3)

    # Output
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    solver = build_solver(args)
    tasks = build_task_list(tiers=args.tiers, smoke=args.smoke, max_tasks=args.max_tasks)

    if not tasks:
        print("No tasks selected. Use --tiers or --smoke.")
        return

    print(f"Benchmarking {len(tasks)} tasks (solver={args.solver})")
    print(f"MCTS params: depth={args.mcts_max_depth}, "
          f"iters={args.mcts_max_iterations}, "
          f"time={args.mcts_max_time}s")
    print()

    results = run_benchmark(
        solver, tasks, args.data_dir,
        task_timeout=args.task_timeout,
        debug=args.debug,
    )

    print_summary(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serializable fields
        serializable = []
        for r in results:
            sr = dict(r)
            sr.pop("code", None)  # code can be very long
            serializable.append(sr)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
