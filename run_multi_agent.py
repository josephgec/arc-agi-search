#!/usr/bin/env python3
"""CLI entry point for multi-agent ARC-AGI solvers.

Supports five strategies selectable via --strategy:
  multi      — Hypothesizer → Coder → Critic feedback loop (default)
  ensemble   — Multiple Orchestrator runs with majority voting
  pso        — Particle Swarm Optimization swarm (see run_pso.py for full options)
  single     — Simple single-agent baseline
  two_phase  — Fast MultiAgent (phase 1) then seed-PSO if unsolved (phase 2)

Usage examples
--------------
# Multi-agent on one task:
python run_multi_agent.py --task data/training/007bbfb7.json

# Ensemble with majority voting:
python run_multi_agent.py --task data/training/007bbfb7.json --strategy ensemble

# PSO swarm:
python run_multi_agent.py --task data/training/007bbfb7.json --strategy pso \\
    --pso-n-particles 6 --pso-max-iterations 10

# Two-phase (fast 7b MultiAgent → PSO seeded with best code):
python run_multi_agent.py --task data/training/007bbfb7.json --strategy two_phase \\
    --coder-model qwen2.5-coder:7b

# Run on a directory:
python run_multi_agent.py --task-dir data/training/ --max-tasks 20
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import sys
import time
from pathlib import Path

from arc import sandbox
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
                   choices=["multi", "ensemble", "pso", "single", "two_phase"],
                   help="Solving strategy.")

    # Shared LLM config
    p.add_argument("--backend",     default="ollama",
                   choices=["ollama", "anthropic"])
    p.add_argument("--model",        default=None)
    p.add_argument("--coder-model",  default=None,
                   help="Override model for Coder role (multi strategy only)")
    p.add_argument("--critic-model", default=None,
                   help="Override model for Critic/Decomposer/Verifier roles")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens",  type=int,   default=8192)
    p.add_argument("--timeout",     type=float, default=600.0)
    p.add_argument("--max-cycles",  type=int,   default=9)

    # Ensemble-specific
    p.add_argument("--ensemble-runs",       type=int, default=5)
    p.add_argument("--ensemble-candidates", type=int, default=3)

    # PSO-specific
    p.add_argument("--pso-n-particles",    type=int,   default=6)
    p.add_argument("--pso-max-iterations", type=int,   default=10)
    p.add_argument("--pso-k-candidates",   type=int,   default=10)
    p.add_argument("--pso-w",              type=float, default=0.5)
    p.add_argument("--pso-c1",             type=float, default=1.5)
    p.add_argument("--pso-c2",             type=float, default=1.5)
    p.add_argument("--pso-embed-model",    default="nomic-embed-text")

    # Output
    p.add_argument("--max-tasks",    type=int, default=None)
    p.add_argument("--task-timeout", type=int, default=0,
                   help="Hard wall-clock limit per task in seconds (0 = unlimited)")
    p.add_argument("--output",    type=Path, default=None)
    p.add_argument("--debug",     action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Two-phase orchestrator
# ---------------------------------------------------------------------------

class TwoPhaseOrchestrator:
    """Run a fast MultiAgent pass (phase 1), then seed PSO if unsolved (phase 2).

    Phase 1 uses ``phase1_model`` (typically the 7b coder) with only
    ``phase1_cycles`` cycles so it completes quickly.  If it succeeds, the
    PSO phase is skipped entirely.  If it fails, the best code found is
    passed to ``PSOOrchestrator.seed_particles()`` so PSO starts from an
    already-reasonable solution rather than from random initialisation.
    """

    def __init__(
        self,
        backend:            str        = "ollama",
        model:              str | None = None,
        phase1_model:       str | None = None,
        coder_model:        str | None = None,
        critic_model:       str | None = None,
        phase1_cycles:      int        = 5,
        timeout:            float      = 120.0,
        debug:              bool       = False,
        pso_n_particles:    int        = 6,
        pso_max_iterations: int        = 10,
        pso_k_candidates:   int        = 10,
        pso_w:              float      = 0.5,
        pso_c1:             float      = 1.5,
        pso_c2:             float      = 1.5,
        embed_model:        str        = "nomic-embed-text",
        temperature:        float      = 0.6,
        max_tokens:         int        = 8192,
    ) -> None:
        # Phase 1 uses the fast model (e.g. qwen2.5-coder:7b)
        _p1_model = phase1_model or coder_model or model
        self._multi = MultiAgent(
            backend=backend,
            model=_p1_model,
            coder_model=coder_model or _p1_model,
            critic_model=critic_model,
            decomposer_model=critic_model,
            verifier_model=critic_model,
            timeout=timeout,
            debug=debug,
            max_cycles=phase1_cycles,
        )
        # Phase 2 uses the full reasoner model
        self._pso = PSOOrchestrator(
            backend=backend,
            model=model,
            embed_model=embed_model,
            n_particles=pso_n_particles,
            max_iterations=pso_max_iterations,
            k_candidates=pso_k_candidates,
            w=pso_w,
            c1=pso_c1,
            c2=pso_c2,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            debug=debug,
        )
        self.model   = self._pso.model
        self.backend = backend

    def solve(self, task: dict) -> dict:
        """Run phase 1 (MultiAgent), then phase 2 (PSO) if unsolved."""
        phase1 = self._multi.solve(task)
        if phase1.get("success"):
            return phase1

        best_code = phase1.get("code") or ""
        if best_code:
            self._pso.seed_particles([best_code])

        return self._pso.solve(task)


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
    elif args.strategy == "two_phase":
        return TwoPhaseOrchestrator(
            backend=args.backend,
            model=args.model,
            phase1_model=args.coder_model,
            coder_model=args.coder_model,
            critic_model=args.critic_model,
            phase1_cycles=5,
            timeout=args.timeout,
            debug=args.debug,
            pso_n_particles=args.pso_n_particles,
            pso_max_iterations=args.pso_max_iterations,
            pso_k_candidates=args.pso_k_candidates,
            pso_w=args.pso_w,
            pso_c1=args.pso_c1,
            pso_c2=args.pso_c2,
            embed_model=args.pso_embed_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:  # "multi"
        return MultiAgent(
            backend=args.backend,
            model=args.model,
            coder_model=args.coder_model,
            critic_model=args.critic_model,
            decomposer_model=args.critic_model,
            verifier_model=args.critic_model,
            timeout=args.timeout,
            debug=args.debug,
            max_cycles=args.max_cycles,
        )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def solve_task(solver, task_path: Path, task_timeout: int = 0) -> dict:
    task = load_task(task_path)
    t0   = time.time()

    # Use a daemon thread + SIGINT to enforce the wall-clock limit.
    # signal.alarm() + SIGALRM is unreliable when the main thread is blocked
    # inside a C-level socket.recv() (Ollama streaming); SIGINT reliably
    # interrupts blocking sockets in CPython by raising KeyboardInterrupt.
    _timer: threading.Timer | None = None
    if task_timeout > 0:
        def _kill():
            import os
            os.kill(os.getpid(), signal.SIGINT)
        _timer = threading.Timer(task_timeout, _kill)
        _timer.daemon = True
        _timer.start()

    try:
        result = solver.solve(task)
    except (KeyboardInterrupt, TimeoutError) as exc:
        return {
            "task":         str(task_path),
            "success":      False,
            "test_correct": None,
            "elapsed_s":    round(time.time() - t0, 2),
            "code":         "",
            "prediction":   None,
            "gbest_fitness": None,
            "error":        f"Task exceeded {task_timeout}s wall-clock limit",
        }
    finally:
        if _timer is not None:
            _timer.cancel()

    elapsed = time.time() - t0

    pred_list = None
    if result.get("prediction") is not None:
        # PSO / Ensemble already executed the code and return a numpy array
        try:
            pred_list = grid_to_list(result["prediction"])
        except Exception:
            pass
    elif result.get("code"):
        # multi / single return code but no prediction — execute it now
        test_input = task.get("test", [{}])[0].get("input")
        if test_input is not None:
            import numpy as np
            grid = np.array(test_input, dtype=np.int32)
            out, _ = sandbox.execute(result["code"], grid)
            if out is not None:
                try:
                    pred_list = grid_to_list(out)
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
        summary = solve_task(solver, args.task, task_timeout=args.task_timeout)
        print(json.dumps(summary, indent=2, default=str))
        all_results = [summary]
    else:
        import random as _random
        paths = sorted(args.task_dir.glob("*.json"))
        _random.shuffle(paths)
        if args.max_tasks:
            paths = paths[:args.max_tasks]
        all_results = []
        for i, path in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] {path.name} … ", end="", flush=True)
            summary = solve_task(solver, path, task_timeout=args.task_timeout)
            elapsed = summary.get("elapsed_s", 0)
            if summary.get("error"):
                status = f"TIMEOUT ({elapsed:.0f}s)"
            else:
                status = f"{'SOLVED' if summary['success'] else 'failed'} ({elapsed:.0f}s)"
            print(status)
            # Running tally every 10 tasks
            all_results.append(summary)
            if i % 10 == 0:
                n_so_far = sum(r["success"] for r in all_results)
                avg_t = sum(r.get("elapsed_s", 0) for r in all_results) / len(all_results)
                print(f"  → {n_so_far}/{i} solved so far  |  avg {avg_t:.0f}s/task", flush=True)

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
