# Implementation Plan

## Baseline (overnight run)
1/40 solved (2.5%), 746s avg/task, Critic loop never ran.

## Status
Phase: 0 (critical blockers)
Last completed: none
Next: 0a — Eliminate Ollama model swapping

## PHASE 0 — CRITICAL BLOCKERS
- [ ] 0a: OLLAMA_KEEP_ALIVE=-1 + pre-warm models → start_ollama.sh
- [ ] 0b: Socket timeout 300→600s + retry logic → agents/llm_client.py
- [ ] 0c: Edge-guard prompts + safe_neighbors DSL helper → agents/roles.py, arc/sandbox.py, agents/dsl_reference.py
- [ ] 0d: Structured stage logging → agents/multi_agent.py, agents/pso_orchestrator.py
→ GATE: run 10 tasks, verify Critic loop runs, <4min/task, >10% solve

## PHASE 1 — ACCURACY
- [ ] 1a: Visual grid format in prompts → agents/roles.py
- [ ] 1b: Negative examples in PSOCoder prompts → agents/roles.py, agents/pso_orchestrator.py
- [ ] 1c: DSL primitives (fill_enclosed_regions, gravity) → arc/dsl.py, arc/sandbox.py
- [ ] 1d: Stronger memorization penalty → arc/evaluate.py
→ GATE: run 40 tasks, verify >15% solve rate

## PHASE 2 — SPEED
- [ ] 2a: Model routing (7b coder, 32b reasoner) → agents/llm_client.py, agents/roles.py
- [ ] 2b: NUM_PARALLEL=6 + q8_0 KV → start_ollama.sh
- [ ] 2c: Sandbox ProcessPoolExecutor (8 workers) → arc/sandbox.py
- [ ] 2d: Parallel particle updates → agents/pso_orchestrator.py
- [ ] 2e: Two-phase multi-agent → PSO → run_multi_agent.py, agents/pso_orchestrator.py
- [ ] 2f: Early termination + staleness → agents/pso_orchestrator.py
→ GATE: run 40 tasks, verify <120s/task, >25% solve

## PHASE 3 — ADVANCED
- [ ] 3a: Ring topology PSO
- [ ] 3b: Structural fitness scoring
- [ ] 3c: Per-color F1 pixel score
- [ ] 3d: Code-specific embeddings (mxbai-embed-large)
- [ ] 3e: Test-time data augmentation
- [ ] 3f: Adaptive K candidates
