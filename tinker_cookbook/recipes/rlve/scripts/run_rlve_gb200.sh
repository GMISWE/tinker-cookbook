#!/bin/bash
# RLVE Training via Tinker API - GB200 aligned parameters
# Based on: kgateway/.../RLVE/scripts/training/Qwen2.5-7B/rlve-4xGB200.sh
#
# Usage:
#   ./run_rlve_gb200.sh                    # Default: 7B model, 1000 batches
#   ./run_rlve_gb200.sh 7B 1000            # 7B model, 1000 batches
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./run_rlve_gb200.sh

set -ex

# Configuration
MODEL_SIZE=${1:-"7B"}
N_BATCHES=${2:-1000}

#######################################
# Pre-run setup steps
#######################################

echo "=== 1. Kill GPU processes ==="
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 2
nvidia-smi | grep -E "MiB|%"

echo "=== 2. Restart Ray ==="
ray stop --force 2>/dev/null || true
sleep 3
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== 3. Restart tinkercloud ==="
pkill -9 -f "uvicorn.*training" 2>/dev/null || true
sleep 2
ORIG_DIR="$(pwd)"
cd /root/gavin/tinkercloud
ALLOW_PARTIAL_BATCHES=true \
PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/miles:$PYTHONPATH \
nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 >> /data/logs/tinkercloud.log 2>&1 &
cd "$ORIG_DIR"
echo "Waiting for server to start..."
sleep 10

# Wait for server to be healthy
for i in {1..30}; do
  if curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | grep -q "healthy"; then
    echo "Server is healthy"
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 2
done

curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python3 -m json.tool

echo "=== 4. Run cleanup ==="
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
python3 /root/gavin/tinker_gmi/tests_integration/cleanup_test_env.py 2>/dev/null || true

#######################################
# RLVE Training Configuration
#######################################

# Model selection
case "$MODEL_SIZE" in
    "0.5B"|"0.5b")
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
        WANDB_GROUP="tinker-0.5B-4xGB200"
        ;;
    "1.5B"|"1.5b"|"deepseek"|"r1")
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        WANDB_GROUP="tinker-deepseek-1.5B-4xGB200"
        ;;
    "7B"|"7b")
        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
        WANDB_GROUP="tinker-7B-4xGB200"
        ;;
    *)
        MODEL_NAME="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
        WANDB_GROUP="tinker-custom-4xGB200"
        ;;
esac

# Environments (can be overridden by ENVIRONMENTS env var)
# Default: 4 core environments for faster iteration
ENVIRONMENTS="${ENVIRONMENTS:-Division,EuclidGame,Multiplication,Sorting}"

# API configuration
export TINKER_API_KEY="${TINKER_API_KEY:-slime-dev-key}"
export WANDB_API_KEY="${WANDB_API_KEY:-0ed1fa8a77196635759510132f81ea55ced801bd}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

# Paths - use readlink -f to get absolute path regardless of how script is invoked
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# Go up 4 levels: scripts -> rlve -> recipes -> tinker_cookbook -> tinker-cookbook
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="/data/logs/rlve-tinker"
LOG_FILE="/data/logs/tinker-rlve.log"

echo "=========================================="
echo "RLVE Training via Tinker API"
echo "=========================================="
echo "Model:        $MODEL_NAME"
echo "Environments: $ENVIRONMENTS"
echo "Batches:      $N_BATCHES"
echo "Base URL:     $BASE_URL"
echo "Wandb Group:  $WANDB_GROUP"
echo "=========================================="

# Batch size configuration (memory-constrained for 4xGB200):
# - 7B model: groups_per_batch=4, group_size=16 → 64 samples/batch
# - 0.5B model: groups_per_batch=8, group_size=16 → 128 samples/batch
# - max_tokens=1024 (reduced from 4096 for memory)
# - GPU memory: ~50-80% during sampling, ~7-10% during training
# - Higher batch sizes cause OOM on SGLang KV cache

cd "$COOKBOOK_DIR"

PYTHONPATH="${COOKBOOK_DIR}:${PYTHONPATH}" \
python -m tinker_cookbook.recipes.rlve.train \
    model_name="$MODEL_NAME" \
    base_url="$BASE_URL" \
    environment_list="$ENVIRONMENTS" \
    groups_per_batch=64 \
    group_size=16 \
    max_tokens=4096 \
    learning_rate=1e-6 \
    loss_fn=ppo \
    advantage_estimator=grpo \
    grpo_reward_center=True \
    grpo_std_normalization=True \
    kl_penalty_coef=0 \
    log_path="$LOG_DIR" \
    wandb_project=rlve-qwen \
    wandb_name="$WANDB_GROUP" \
    behavior_if_log_dir_exists=delete \
    eval_every=0 \
    save_every=100 \
    n_batches="$N_BATCHES" 2>&1 | tee "$LOG_FILE"
