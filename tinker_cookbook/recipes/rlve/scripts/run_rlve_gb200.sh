#!/bin/bash
# RLVE Training via Tinker API - GB200 aligned parameters
# Based on: kgateway/.../RLVE/scripts/training/Qwen2.5-7B/rlve-4xGB200.sh
#
# Usage:
#   ./run_rlve_gb200.sh                    # Default: 0.5B model, 100 batches
#   ./run_rlve_gb200.sh 7B 1000            # 7B model, 1000 batches
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./run_rlve_gb200.sh

set -e

# Configuration
MODEL_SIZE=${1:-"7B"}
N_BATCHES=${2:-1000}

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

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(dirname "$SCRIPT_DIR")"
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
