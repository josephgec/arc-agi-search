#!/usr/bin/env bash
# Start Ollama and pull the models best suited for ARC-AGI.
#
# Model strategy
# --------------
# ARC tasks require two distinct capabilities:
#   • Spatial / pattern reasoning  → deepseek-r1  (native chain-of-thought)
#   • Python code generation       → qwen2.5-coder (highest Pass@1 on code benchmarks)
#
# Choose a tier based on available RAM/VRAM:
#   TIER=small   ~8 GB    deepseek-r1:8b   + qwen2.5-coder:7b
#   TIER=medium  ~16 GB   deepseek-r1:14b  + qwen2.5-coder:14b  (default)
#   TIER=large   ~32 GB   deepseek-r1:32b  + qwen2.5-coder:32b
#   TIER=ultra   ~64 GB   deepseek-r1:32b  + qwen2.5-coder:7b   (asymmetric: big reasoner + fast coder)
#
# M1/M2/M3 Ultra with 64 GB: use TIER=ultra.  The 32B reasoner (~20 GB at Q4)
# leaves ~44 GB for the KV cache — massive context headroom for large ARC grids.
# The 7B coder runs at 70+ t/s so the Coder step costs almost nothing.
#
# Override by exporting TIER, REASONER_MODEL, or CODER_MODEL before running:
#   TIER=ultra ./start_ollama.sh
#   REASONER_MODEL=deepseek-r1:32b CODER_MODEL=qwen2.5-coder:7b ./start_ollama.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Tier selection
# ---------------------------------------------------------------------------

TIER=${TIER:-medium}

case "$TIER" in
  small)
    DEFAULT_REASONER="deepseek-r1:8b"
    DEFAULT_CODER="qwen2.5-coder:7b"
    ;;
  large)
    DEFAULT_REASONER="deepseek-r1:32b"
    DEFAULT_CODER="qwen2.5-coder:32b"
    ;;
  ultra)
    # Asymmetric: 32B for reasoning (Hypothesizer + Critic), 7B for fast coding
    # Optimal for 64 GB Apple Silicon — 32B @ Q4 ~20 GB, 7B ~5 GB, 39 GB left for KV cache
    DEFAULT_REASONER="deepseek-r1:32b"
    DEFAULT_CODER="qwen2.5-coder:7b"
    ;;
  medium|*)
    DEFAULT_REASONER="deepseek-r1:14b"
    DEFAULT_CODER="qwen2.5-coder:14b"
    ;;
esac

REASONER_MODEL=${REASONER_MODEL:-$DEFAULT_REASONER}
CODER_MODEL=${CODER_MODEL:-$DEFAULT_CODER}

echo "Tier        : $TIER"
echo "Reasoner    : $REASONER_MODEL  (Hypothesizer + Critic roles)"
echo "Coder       : $CODER_MODEL     (Coder role)"
echo "Embed       : nomic-embed-text (PSO vector space)"
echo ""

# ---------------------------------------------------------------------------
# Ollama server settings
# ---------------------------------------------------------------------------
# OLLAMA_FLASH_ATTENTION  — fused attention kernel; critical speedup on M-series
# OLLAMA_KV_CACHE_TYPE    — f16 KV cache is ideal for Apple Silicon (800 GB/s bandwidth)
# OLLAMA_NUM_PARALLEL     — simultaneous token streams; match to GPU core count
# OLLAMA_MAX_LOADED_MODELS — keep reasoner + coder both hot in VRAM
# Lower OLLAMA_NUM_PARALLEL or set OLLAMA_MAX_LOADED_MODELS=1 if you hit OOM.

export OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-1}
export OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-"f16"}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}

echo "Starting Ollama (parallel=$OLLAMA_NUM_PARALLEL, max_models=$OLLAMA_MAX_LOADED_MODELS)…"
ollama serve &
OLLAMA_PID=$!

# Wait for server readiness
echo "Waiting for Ollama to be ready…"
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Pull models
# ---------------------------------------------------------------------------

echo ""
echo "Pulling models (skipped if already cached)…"
ollama pull nomic-embed-text  || true
ollama pull "$REASONER_MODEL" || true
# Only pull coder model if it differs from the reasoner
if [ "$CODER_MODEL" != "$REASONER_MODEL" ]; then
    ollama pull "$CODER_MODEL" || true
fi

# ---------------------------------------------------------------------------
# Usage guide
# ---------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Recommended CLI invocations for tier: $TIER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Multi-agent (Hypothesizer + Coder + Critic):"
echo "  python run_multi_agent.py \\"
echo "    --task data/training/007bbfb7.json \\"
echo "    --strategy multi \\"
echo "    --model $REASONER_MODEL"
echo ""
echo "# Ensemble with pixel-vote + self-correction (Phase 4):"
echo "  python run_multi_agent.py \\"
echo "    --task data/training/007bbfb7.json \\"
echo "    --strategy ensemble \\"
echo "    --model $REASONER_MODEL \\"
echo "    --ensemble-runs 5 --ensemble-candidates 3"
echo ""
echo "# PSO swarm:"
echo "  python run_pso.py \\"
echo "    --task data/training/007bbfb7.json \\"
echo "    --model $REASONER_MODEL \\"
echo "    --n-particles 6 --max-iterations 10 --debug"
echo ""
echo "# Batch evaluation:"
echo "  python run_multi_agent.py \\"
echo "    --task-dir data/training/ --max-tasks 50 \\"
echo "    --strategy ensemble --model $REASONER_MODEL \\"
echo "    --output results_${TIER}.json"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Ollama PID: $OLLAMA_PID   |   To stop: kill $OLLAMA_PID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait $OLLAMA_PID
