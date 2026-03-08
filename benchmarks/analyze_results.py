#!/usr/bin/env python3
"""Post-hoc analysis of benchmark results.

Usage:
    python -m benchmarks.analyze_results results.json
    python -m benchmarks.analyze_results results.json --compare baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    """Load benchmark results from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def fitness_histogram(results: list[dict], bins: int = 10) -> dict[str, int]:
    """Bucket fitness scores into bins and return counts."""
    step = 1.0 / bins
    buckets: dict[str, int] = {}
    for i in range(bins):
        lo = round(i * step, 2)
        hi = round((i + 1) * step, 2)
        label = f"{lo:.2f}-{hi:.2f}"
        buckets[label] = 0

    for r in results:
        f = r.get("fitness", 0.0)
        idx = min(int(f * bins), bins - 1)
        lo = round(idx * step, 2)
        hi = round((idx + 1) * step, 2)
        label = f"{lo:.2f}-{hi:.2f}"
        buckets[label] = buckets.get(label, 0) + 1

    return buckets


def speed_analysis(results: list[dict]) -> dict:
    """Compute speed statistics."""
    times = [r["elapsed_s"] for r in results]
    if not times:
        return {}
    return {
        "min_s": min(times),
        "max_s": max(times),
        "avg_s": sum(times) / len(times),
        "median_s": sorted(times)[len(times) // 2],
        "total_s": sum(times),
    }


def failure_analysis(results: list[dict]) -> list[dict]:
    """Return details of failed tasks sorted by fitness (highest first)."""
    failed = [r for r in results if not r.get("solved")]
    return sorted(failed, key=lambda r: r.get("fitness", 0.0), reverse=True)


def head_to_head(results_a: list[dict], results_b: list[dict]) -> dict:
    """Compare two result sets on the same tasks.

    Returns dict with wins_a, wins_b, ties, regressions, improvements.
    """
    map_a = {r["task_id"]: r for r in results_a}
    map_b = {r["task_id"]: r for r in results_b}

    common = set(map_a) & set(map_b)
    wins_a = 0
    wins_b = 0
    ties = 0
    improvements: list[dict] = []
    regressions: list[dict] = []

    for tid in sorted(common):
        ra, rb = map_a[tid], map_b[tid]
        fa = ra.get("fitness", 0.0)
        fb = rb.get("fitness", 0.0)

        if abs(fa - fb) < 1e-6:
            ties += 1
        elif fb > fa:
            wins_b += 1
            improvements.append({
                "task_id": tid,
                "fitness_a": fa,
                "fitness_b": fb,
                "delta": round(fb - fa, 4),
            })
        else:
            wins_a += 1
            regressions.append({
                "task_id": tid,
                "fitness_a": fa,
                "fitness_b": fb,
                "delta": round(fb - fa, 4),
            })

    return {
        "common_tasks": len(common),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "improvements": sorted(improvements, key=lambda x: x["delta"], reverse=True),
        "regressions": sorted(regressions, key=lambda x: x["delta"]),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_analysis(results: list[dict]) -> None:
    """Print comprehensive analysis to stdout."""
    print(f"\n{'='*60}")
    print("BENCHMARK ANALYSIS")
    print(f"{'='*60}")

    # Fitness histogram
    hist = fitness_histogram(results)
    print("\nFitness Distribution:")
    max_count = max(hist.values()) if hist else 1
    for label, count in hist.items():
        bar = "#" * int(40 * count / max(max_count, 1))
        print(f"  {label}: {count:3d} {bar}")

    # Speed
    speed = speed_analysis(results)
    if speed:
        print(f"\nSpeed:")
        print(f"  Min:    {speed['min_s']:.1f}s")
        print(f"  Max:    {speed['max_s']:.1f}s")
        print(f"  Avg:    {speed['avg_s']:.1f}s")
        print(f"  Median: {speed['median_s']:.1f}s")
        print(f"  Total:  {speed['total_s']:.1f}s")

    # Near-misses (fitness > 0.5 but not solved)
    failed = failure_analysis(results)
    near = [r for r in failed if r.get("fitness", 0) > 0.3]
    if near:
        print(f"\nNear-misses (fitness > 0.3, not solved):")
        for r in near[:10]:
            print(f"  {r['task_id']} (tier {r.get('tier', '?')}): "
                  f"fitness={r.get('fitness', 0):.4f} — {r.get('description', '')}")


def print_comparison(h2h: dict, label_a: str, label_b: str) -> None:
    """Print head-to-head comparison."""
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"Common tasks: {h2h['common_tasks']}")
    print(f"  {label_a} wins: {h2h['wins_a']}")
    print(f"  {label_b} wins: {h2h['wins_b']}")
    print(f"  Ties:          {h2h['ties']}")

    if h2h["improvements"]:
        print(f"\nImprovements ({label_b} better):")
        for r in h2h["improvements"][:5]:
            print(f"  {r['task_id']}: {r['fitness_a']:.4f} → {r['fitness_b']:.4f} "
                  f"(+{r['delta']:.4f})")

    if h2h["regressions"]:
        print(f"\nRegressions ({label_a} better):")
        for r in h2h["regressions"][:5]:
            print(f"  {r['task_id']}: {r['fitness_a']:.4f} → {r['fitness_b']:.4f} "
                  f"({r['delta']:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze benchmark results")
    p.add_argument("results", type=Path, help="Results JSON file")
    p.add_argument("--compare", type=Path, default=None,
                    help="Second results file for head-to-head comparison")
    args = p.parse_args()

    results = load_results(args.results)
    print_analysis(results)

    if args.compare:
        results_b = load_results(args.compare)
        h2h = head_to_head(results, results_b)
        print_comparison(h2h, str(args.results.stem), str(args.compare.stem))


if __name__ == "__main__":
    main()
