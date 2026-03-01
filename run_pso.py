#!/usr/bin/env python3
"""CLI entry point for the PSO swarm solver.

Usage examples
--------------
# Solve a single task (prints JSON result):
python run_pso.py --task data/training/007bbfb7.json

# Solve with custom swarm parameters:
python run_pso.py --task data/training/007bbfb7.json \\
    --n-particles 6 --max-iterations 12 --k-candidates 10 \\
    --w 0.5 --c1 1.5 --c2 1.5 \\
    --temperature 0.7 --debug

# Solve all tasks in a directory, print summary:
python run_pso.py --task-dir data/training/ --max-tasks 10

# Use Anthropic backend for generation (embeddings still use Ollama):
python run_pso.py --task data/training/007bbfb7.json --backend anthropic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np

from arc.grid import load_task, grid_to_list
from agents.pso_orchestrator import PSOOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ARC-AGI Particle Swarm Optimization Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task selection
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", type=Path, metavar="FILE",
                            help="Path to a single ARC task JSON file.")
    task_group.add_argument("--task-dir", type=Path, metavar="DIR",
                            help="Directory of ARC task JSON files to evaluate.")

    # LLM
    p.add_argument("--backend",     default="ollama",
                   choices=["ollama", "anthropic"],
                   help="LLM backend for code generation.")
    p.add_argument("--model",       default=None,
                   help="Chat model name override.")
    p.add_argument("--embed-model", default="nomic-embed-text",
                   help="Ollama embedding model name.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="LLM sampling temperature.")
    p.add_argument("--max-tokens",  type=int,   default=4096,
                   help="Max tokens per LLM call.")
    p.add_argument("--timeout",       type=float, default=180.0,
                   help="Per-LLM-generation timeout (seconds).")
    p.add_argument("--embed-timeout", type=float, default=60.0,
                   help="Per-embedding timeout (seconds).")
    p.add_argument("--fitness-alpha", type=float, default=0.4,
                   help="Weight of actual fitness vs PSO proximity in candidate selection (0=pure PSO, 1=pure greedy).")

    # PSO hyperparameters
    p.add_argument("--n-particles",    type=int,   default=6,
                   help="Number of swarm particles (max 6).")
    p.add_argument("--max-iterations", type=int,   default=10,
                   help="PSO iteration budget.")
    p.add_argument("--k-candidates",   type=int,   default=10,
                   help="LLM mutation candidates per particle per iteration.")
    p.add_argument("--w",  type=float, default=0.5,
                   help="PSO inertia weight.")
    p.add_argument("--c1", type=float, default=1.5,
                   help="PSO cognitive coefficient (pull toward personal best).")
    p.add_argument("--c2", type=float, default=1.5,
                   help="PSO social coefficient (pull toward global best).")

    # Evaluation
    p.add_argument("--max-tasks", type=int, default=None,
                   help="(--task-dir only) Maximum number of tasks to evaluate.")
    p.add_argument("--task-timeout", type=float, default=None,
                   help="(--task-dir only) Max seconds per task before skipping it.")

    # Output
    p.add_argument("--output", type=Path, default=None,
                   help="Write JSON results to this file.")
    p.add_argument("--debug",  action="store_true",
                   help="Verbose progress logging.")

    return p.parse_args()


def build_orchestrator(args: argparse.Namespace) -> PSOOrchestrator:
    return PSOOrchestrator(
        backend=args.backend,
        model=args.model,
        embed_model=args.embed_model,
        n_particles=args.n_particles,
        max_iterations=args.max_iterations,
        k_candidates=args.k_candidates,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        embed_timeout=args.embed_timeout,
        fitness_alpha=args.fitness_alpha,
        debug=args.debug,
    )


def solve_single(orchestrator: PSOOrchestrator, task_path: Path, debug: bool) -> dict:
    task   = load_task(task_path)
    t0     = time.time()
    result = orchestrator.solve(task)
    elapsed = time.time() - t0

    prediction_list = None
    if result.get("prediction") is not None:
        prediction_list = grid_to_list(result["prediction"])

    summary = {
        "task":          str(task_path),
        "success":       result["success"],
        "gbest_fitness": round(result["gbest_fitness"], 6),
        "test_correct":  result["test_correct"],
        "elapsed_s":     round(elapsed, 2),
        "prediction":    prediction_list,
        "code":          result.get("code", ""),
        "iterations":    len([e for e in result["log"] if "iteration" in e]),
    }
    return summary


def run_directory(
    orchestrator: PSOOrchestrator,
    task_dir: Path,
    max_tasks: int | None,
    debug: bool,
    task_timeout: float | None = None,
) -> list[dict]:
    paths = sorted(task_dir.glob("*.json"))
    if max_tasks is not None:
        paths = paths[:max_tasks]

    results = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name} … ", end="", flush=True)
        if task_timeout is not None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_single, orchestrator, path, debug)
                try:
                    summary = future.result(timeout=task_timeout)
                except FuturesTimeoutError:
                    summary = {
                        "task":          str(path),
                        "success":       False,
                        "gbest_fitness": 0.0,
                        "test_correct":  None,
                        "elapsed_s":     task_timeout,
                        "prediction":    None,
                        "code":          "",
                        "iterations":    0,
                        "timed_out":     True,
                    }
                    print(f"TIMEOUT (>{task_timeout:.0f}s)", flush=True)
                    results.append(summary)
                    continue
        else:
            summary = solve_single(orchestrator, path, debug)

        status  = "SOLVED" if summary["success"] else f"fitness={summary['gbest_fitness']:.4f}"
        print(status, flush=True)
        results.append(summary)

    n_solved = sum(r["success"] for r in results)
    print(f"\nSummary: {n_solved}/{len(results)} tasks solved "
          f"({100 * n_solved / len(results):.1f}%)")
    return results


def main() -> None:
    args         = parse_args()
    orchestrator = build_orchestrator(args)

    task_to_str = f"{args.task_timeout:.0f}s" if args.task_timeout else "none"
    print(
        f"PSO Swarm Solver\n"
        f"  Backend : {args.backend} / {orchestrator.model}\n"
        f"  Embed   : {args.embed_model}\n"
        f"  Swarm   : {args.n_particles} particles × {args.max_iterations} iterations "
        f"× {args.k_candidates} candidates\n"
        f"  PSO     : w={args.w}, c1={args.c1}, c2={args.c2}\n"
        f"  Timeout : {args.timeout:.0f}s/LLM, task-limit={task_to_str}\n"
    )

    if args.task:
        summary = solve_single(orchestrator, args.task, args.debug)
        print(json.dumps(summary, indent=2, default=str))
        all_results = [summary]
    else:
        all_results = run_directory(orchestrator, args.task_dir, args.max_tasks, args.debug,
                                    task_timeout=args.task_timeout)
        if not args.debug:
            # Print compact per-task table
            print("\nPer-task results:")
            for r in all_results:
                status = "✓" if r["success"] else "✗"
                print(f"  {status} {Path(r['task']).name:<40} "
                      f"fit={r['gbest_fitness']:.4f}  {r['elapsed_s']:.1f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
