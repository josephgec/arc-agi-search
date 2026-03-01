"""Debug single-task run with verbose cycle tracking."""
from __future__ import annotations
import time

from arc.grid import load_task
from agents.multi_agent import MultiAgent

TASK_PATH = "data/training/28e73c20.json"


def main() -> None:
    print(f"[main] Starting task: {TASK_PATH}", flush=True)

    solver = MultiAgent(
        backend="ollama",
        model="qwen2.5-coder:7b",
        timeout=60.0,
        debug=True,
        max_cycles=4,
    )

    task = load_task(TASK_PATH)
    print(f"[main] Task loaded: {len(task['train'])} train pairs", flush=True)

    t0 = time.time()
    result = solver.solve(task)
    elapsed = round(time.time() - t0, 1)

    print(f"\n[main] Done in {elapsed}s — success={result.get('success')}", flush=True)
    print(f"[main] n_cycles={result.get('n_cycles')}", flush=True)
    print(f"[main] code:\n{result.get('code','(none)')}", flush=True)


if __name__ == "__main__":
    main()
