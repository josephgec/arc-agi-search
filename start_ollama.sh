#!/usr/bin/env bash
# Start Ollama with optimised parallel-request settings for the PSO swarm.
#
# The PSO swarm makes concurrent embedding + generation calls, so we raise
# the parallel request limit.  Adjust OLLAMA_NUM_PARALLEL to match your GPU
# VRAM (lower if you hit OOM errors).

set -euo pipefail

export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-6}
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}

echo "Starting Ollama (parallel=$OLLAMA_NUM_PARALLEL, max_models=$OLLAMA_MAX_LOADED_MODELS)…"
ollama serve &
OLLAMA_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama to be ready…"
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    sleep 1
done

# Pull required models if not present
echo "Checking required models…"
ollama pull nomic-embed-text || true   # Embedding model for PSO
ollama pull deepseek-r1:32b  || true   # Reasoning / code generation
# ollama pull qwen2.5-coder:32b || true  # Alternative coder model (uncomment if preferred)

echo ""
echo "Ollama PID: $OLLAMA_PID"
echo "To stop: kill $OLLAMA_PID"
echo ""
echo "Quick test:"
echo "  python run_pso.py --task data/training/007bbfb7.json --debug"
echo "  python run_multi_agent.py --task data/training/007bbfb7.json --strategy pso"

wait $OLLAMA_PID
