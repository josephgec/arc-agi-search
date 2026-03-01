"""Quick batch tester: run 5 tasks and report results."""
from __future__ import annotations
import json, time, traceback

from arc.grid import load_task
from agents.multi_agent import MultiAgent

TASKS = [
    "data/training/d22278a0.json",
    "data/training/28e73c20.json",
    "data/training/0a938d79.json",
    "data/training/ed36ccf7.json",
    "data/training/623ea044.json",
]


def main() -> None:
    solver = MultiAgent(
        backend="ollama",
        model="qwen2.5-coder:7b",
        timeout=60.0,
        debug=True,
        max_cycles=4,
    )

    results = []
    for task_path in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_path}")
        print('='*60, flush=True)
        task = load_task(task_path)
        t0 = time.time()
        try:
            result = solver.solve(task)
        except Exception as e:
            traceback.print_exc()
            result = {"success": False, "error": str(e)}
        elapsed = round(time.time() - t0, 1)

        status = "SOLVED" if result.get("success") else "failed"
        print(f"\n[{status}] {task_path} ({elapsed}s)", flush=True)
        results.append({
            "task":     task_path,
            "success":  result.get("success", False),
            "elapsed_s": elapsed,
            "code":     result.get("code", ""),
            "error":    result.get("error"),
        })

    print("\n\n=== SUMMARY ===")
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(f"  {mark}  {r['task']} ({r['elapsed_s']}s)")

    n = sum(r["success"] for r in results)
    print(f"\n{n}/{len(results)} solved")

    with open("batch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to batch_results.json")


if __name__ == "__main__":
    main()
