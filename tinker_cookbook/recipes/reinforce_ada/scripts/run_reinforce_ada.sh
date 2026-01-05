#!/bin/bash
# Reinforce-Ada Training via Tinker API
# Based on: tinker_cookbook/recipes/rlve/scripts/run_rlve_gb200.sh
#
# Usage:
#   ./run_reinforce_ada.sh                    # Default: 4B model, 10 batches
#   ./run_reinforce_ada.sh 7B 100             # 7B model, 100 batches

set -e

# Configuration
MODEL_SIZE=${1:-"4B"}
N_BATCHES=${2:-10}

# Model selection
case "$MODEL_SIZE" in
    "4B"|"4b"|"qwen3")
        MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
        WANDB_GROUP="reinforce-ada-qwen3-4B"
        ;;
    "7B"|"7b")
        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
        WANDB_GROUP="reinforce-ada-qwen2.5-7B"
        ;;
    "0.5B"|"0.5b")
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
        WANDB_GROUP="reinforce-ada-qwen2.5-0.5B"
        ;;
    *)
        MODEL_NAME="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
        WANDB_GROUP="reinforce-ada-custom"
        ;;
esac

# API configuration
export TINKER_API_KEY="${TINKER_API_KEY:-slime-dev-key}"
export WANDB_API_KEY="${WANDB_API_KEY:-0ed1fa8a77196635759510132f81ea55ced801bd}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="/data/logs/reinforce-ada"
LOG_FILE="/data/logs/reinforce-ada.log"

echo "=========================================="
echo "Reinforce-Ada Training via Tinker API"
echo "=========================================="
echo "Model:        $MODEL_NAME"
echo "Batches:      $N_BATCHES"
echo "Base URL:     $BASE_URL"
echo "Wandb Group:  $WANDB_GROUP"
echo "=========================================="

# Create log directory
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Run training
cd /root/gavin/tinker-cookbook

PYTHONPATH="/root/gavin/tinker-cookbook:${PYTHONPATH}" \
python -m tinker_cookbook.recipes.reinforce_ada.train \
    model_name="$MODEL_NAME" \
    base_url="$BASE_URL" \
    groups_per_batch=32 \
    group_size=4 \
    max_tokens=2048 \
    learning_rate=5e-6 \
    loss_fn=importance_sampling \
    multiround_adaptive_downsampling=True \
    global_stat_est=True \
    reinforce_ada_choice=balanced \
    max_rounds=4 \
    round_repeat=8 \
    n_batches="$N_BATCHES" \
    wandb_project=reinforce-ada \
    wandb_name="$WANDB_GROUP" \
    log_path="$LOG_DIR" \
    behavior_if_log_dir_exists=delete \
    eval_every=0 \
    save_every=50 2>&1 | tee "$LOG_FILE"
