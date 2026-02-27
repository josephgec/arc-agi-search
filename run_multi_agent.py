#!/usr/bin/env python3
"""CLI entry point for multi-agent ARC-AGI solvers.

Supports three strategies selectable via --strategy:
  multi    — Hypothesizer → Coder → Critic feedback loop (default)
  ensemble — Multiple Orchestrator runs with majority voting
  pso      — Particle Swarm Optimization swarm (see run_pso.py for full options)
  single   — Simple single-agent baseline

Usage examples
--------------
# Multi-agent on one task:
python run_multi_agent.py --task data/training/007bbfb7.json

# Ensemble with majority voting:
python run_multi_agent.py --task data/training/007bbfb7.json --strategy ensemble

# PSO swarm:
python run_multi_agent.py --task data/training/007bbfb7.json --strategy pso \\
    --pso-n-particles 6 --pso-max-iterations 10

# Run on a directory:
python run_multi_agent.py --task-dir data/training/ --max-tasks 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arc.grid import load_task, grid_to_list
from agents.multi_agent import MultiAgent
from agents.ensemble import Ensemble
from agents.single_agent import SingleAgent
from agents.pso_orchestrator import PSOOrchestrator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ARC-AGI multi-agent solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task",     type=Path, metavar="FILE")
    task_group.add_argument("--task-dir", type=Path, metavar="DIR")

    # Strategy
    p.add_argument("--strategy", default="multi",
                   choices=["multi", "ensemble", "pso", "single"],
                   help="Solving strategy.")

    # Shared LLM config
    p.add_argument("--backend",     default="ollama",
                   choices=["ollama", "anthropic"])
    p.add_argument("--model",       default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens",  type=int,   default=8192)
    p.add_argument("--timeout",     type=float, default=120.0)
    p.add_argument("--max-cycles",  type=int,   default=9)

    # Ensemble-specific
    p.add_argument("--ensemble-runs",       type=int, default=5)
    p.add_argument("--ensemble-candidates", type=int, default=3)

    # PSO-specific
    p.add_argument("--pso-n-particles",    type=int,   default=6)
    p.add_argument("--pso-max-iterations", type=int,   default=10)
    p.add_argument("--pso-k-candidates",   type=int,   default=5)
    p.add_argument("--pso-w",              type=float, default=0.5)
    p.add_argument("--pso-c1",             type=float, default=1.5)
    p.add_argument("--pso-c2",             type=float, default=1.5)
    p.add_argument("--pso-embed-model",    default="nomic-embed-text")

    # Output
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--output",    type=Path, default=None)
    p.add_argument("--debug",     action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def build_solver(args: argparse.Namespace):
    if args.strategy == "pso":
        return PSOOrchestrator(
            backend=args.backend,
            model=args.model,
            embed_model=args.pso_embed_model,
            n_particles=args.pso_n_particles,
            max_iterations=args.pso_max_iterations,
            k_candidates=args.pso_k_candidates,
            w=args.pso_w,
            c1=args.pso_c1,
            c2=args.pso_c2,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            debug=args.debug,
        )
    elif args.strategy == "ensemble":
        return Ensemble(
            backend=args.backend,
            model=args.model,
            timeout=args.timeout,
            debug=args.debug,
            target_candidates=args.ensemble_candidates,
            max_runs=args.ensemble_runs,
        )
    elif args.strategy == "single":
        return SingleAgent(
            backend=args.backend,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            debug=args.debug,
        )
    else:  # "multi"
        return MultiAgent(
            backend=args.backend,
            model=args.model,
            timeout=args.timeout,
            debug=args.debug,
            max_cycles=args.max_cycles,
        )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def solve_task(solver, task_path: Path) -> dict:
    task    = load_task(task_path)
    t0      = time.time()
    result  = solver.solve(task)
    elapsed = time.time() - t0

    pred = result.get("prediction") or result.get("code")
    pred_list = None
    if result.get("prediction") is not None:
        try:
            pred_list = grid_to_list(result["prediction"])
        except Exception:
            pass

    return {
        "task":         str(task_path),
        "success":      result.get("success", False),
        "test_correct": result.get("test_correct"),
        "elapsed_s":    round(elapsed, 2),
        "code":         result.get("code", ""),
        "prediction":   pred_list,
        # PSO-specific
        "gbest_fitness": result.get("gbest_fitness"),
    }


def main() -> None:
    args   = parse_args()
    solver = build_solver(args)

    print(f"Strategy : {args.strategy}")
    print(f"Backend  : {args.backend} / {getattr(solver, 'model', '?')}")
    print()

    if args.task:
        summary = solve_task(solver, args.task)
        print(json.dumps(summary, indent=2, default=str))
        all_results = [summary]
    else:
        paths = sorted(args.task_dir.glob("*.json"))
        if args.max_tasks:
            paths = paths[:args.max_tasks]
        all_results = []
        for i, path in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] {path.name} … ", end="", flush=True)
            summary = solve_task(solver, path)
            status  = "SOLVED" if summary["success"] else "failed"
            print(status)
            all_results.append(summary)

        n_solved = sum(r["success"] for r in all_results)
        print(f"\n{n_solved}/{len(all_results)} solved "
              f"({100 * n_solved / max(len(all_results), 1):.1f}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
