"""Quick batch tester: run 5 tasks and report results."""
from __future__ import annotations
import json, time, traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from arc.grid import load_task
from agents.multi_agent import MultiAgent

TASKS = [
    "data/training/46442a0e.json",
    "data/training/49d1d64f.json",
    "data/training/8d5021e8.json",
    "data/training/4522001f.json",
    "data/training/6150a2bd.json",
]

TASK_TIMEOUT = 240  # seconds per task


def main() -> None:
    # Use qwen2.5-coder:14b for ALL roles — ~10-20s/call vs 60-70s for deepseek-r1
    # This is for iterative testing/bug-finding, not peak accuracy.
    solver = MultiAgent(
        backend="ollama",
        model="qwen2.5-coder:14b",
        hypothesizer_model="qwen2.5-coder:14b",
        coder_model="qwen2.5-coder:14b",
        critic_model="qwen2.5-coder:14b",
        hypothesizer_max_tokens=4096,
        coder_max_tokens=2048,
        critic_max_tokens=2048,
        timeout=60.0,
        debug=True,
        max_cycles=8,
    )

    results = []
    for task_path in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_path}")
        print('='*60, flush=True)
        task = load_task(task_path)
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(solver.solve, task)
                try:
                    result = fut.result(timeout=TASK_TIMEOUT)
                except FuturesTimeoutError:
                    result = {"success": False, "error": f"task_timeout>{TASK_TIMEOUT}s"}
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
