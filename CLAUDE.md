# ARC-AGI PSO Swarm Solver

## What this is
Hybrid PSO + LLM for ARC-AGI. PSO explores continuous embedding space; LLM
generates discrete Python code; Generate-and-Project bridge connects them.
Multi-agent orchestrator has Hypothesizer → Coder → Critic → Decomposer loop.

## Architecture
- `arc/` — grid ops, DSL primitives, sandbox (subprocess), fitness functions
- `agents/` — llm_client, roles (Hypothesizer/Coder/Critic/PSOCoder), multi_agent, pso_orchestrator, ensemble
- Key flow: Hypothesizer infers rule → Coder writes transform() → Sandbox evals → Critic routes fix

## Hardware
M1 Ultra Mac Studio, 64 GB unified memory, macOS.
Ollama ultra tier: deepseek-r1:32b + qwen2.5-coder:7b + nomic-embed-text.
Both chat models must stay loaded simultaneously (~25 GB total).

## Known issues being fixed
- Ollama swaps models out after 5min idle → causes timeouts. Fix: OLLAMA_KEEP_ALIVE=-1
- Socket timeout at 300s kills Coder before it finishes → needs 600s + retry
- Generated code crashes on grid edges → needs bounds-checking prompts + safe_neighbors DSL helper
- Silent failures → need structured stage logging
See PLAN.md for current task and progress.

## Commands
- All tests (offline, mocked): `python -m pytest`
- Single module: `python -m pytest tests/test_<module>.py -v -x`
- Coverage: `python -m pytest --cov=arc --cov=agents --cov-report=term-missing`
- Single task: `python run_multi_agent.py --task data/training/007bbfb7.json --strategy multi --debug`
- Batch test: `python run_multi_agent.py --task-dir data/training/ --strategy multi --max-tasks 10 --debug`

## Code style
- Python 3.11+, type hints on public functions
- DSL functions are pure — return new array, never mutate
- Sandbox code is untrusted — subprocess isolation with timeout
- numpy int32, values 0-9
- Always mock LLM calls in tests — never require Ollama running
- Coverage: 94%. Don't drop below 90%.
