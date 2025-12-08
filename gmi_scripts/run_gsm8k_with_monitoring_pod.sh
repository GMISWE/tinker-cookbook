#!/bin/bash
# GSM8K test with comprehensive OOM monitoring (for execution inside pod)
# This script monitors GPU memory and Ray actors at each stage to diagnose OOM issues

set -e  # Exit on error

LOG_FILE="/tmp/gsm8k_monitoring_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "GSM8K OOM Monitoring Test"
echo "Log file: $LOG_FILE"
echo "================================================================================"

# Function to log GPU memory
log_gpu_memory() {
    local stage="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "=== GPU Memory @ $stage ===" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Function to log Ray actors
log_ray_actors() {
    local stage="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "=== Ray Actors @ $stage ===" | tee -a "$LOG_FILE"
    ray list actors --filter "state=ALIVE" 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Function to log system memory
log_system_memory() {
    local stage="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "=== System Memory @ $stage ===" | tee -a "$LOG_FILE"
    free -h | grep -E 'Mem:|Swap:' | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Initial state
echo "$(date): Starting GSM8K test with monitoring" | tee -a "$LOG_FILE"
log_gpu_memory "INITIAL"
log_ray_actors "INITIAL"
log_system_memory "INITIAL"

# Set Tinker API key
export TINKER_API_KEY=slime-dev-key

# Set HuggingFace to use local cache
export HF_HOME=/data
export HF_DATASETS_CACHE=/data/datasets
export HF_DATASETS_OFFLINE=0

# Create symlink for HuggingFace dataset cache structure if needed
CACHE_DIR="$HOME/.cache/huggingface/datasets"
mkdir -p "$CACHE_DIR"

# Symlink the local gsm8k to expected HuggingFace cache location
if [ ! -e "$CACHE_DIR/openai___gsm8k" ]; then
    ln -s /data/datasets/gsm8k "$CACHE_DIR/openai___gsm8k"
    echo "Created symlink: $CACHE_DIR/openai___gsm8k -> /data/datasets/gsm8k" | tee -a "$LOG_FILE"
fi

# Start background monitoring (check every 10 seconds)
(
    while true; do
        sleep 10
        echo "$(date): Periodic GPU check" >> "$LOG_FILE"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$LOG_FILE"
    done
) &
MONITOR_PID=$!

echo "" | tee -a "$LOG_FILE"
echo "=== Starting Training ===" | tee -a "$LOG_FILE"
echo "$(date): Background monitoring started (PID: $MONITOR_PID)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the training with output tee'd to log
# MODIFIED: Reduce max_tokens to 128 (was 256) to reduce OOM risk
python -u -m tinker_cookbook.recipes.math_rl.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    env=gsm8k \
    base_url=http://kgateway-training.slime-gmi:8000 \
    group_size=4 \
    groups_per_batch=2 \
    learning_rate=2e-4 \
    max_tokens=128 \
    eval_every=5 \
    save_every=10 \
    num_substeps=1 \
    behavior_if_log_dir_exists=delete "$@" 2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!

# Monitor the training process
echo "Training PID: $TRAIN_PID" | tee -a "$LOG_FILE"

# Wait for model creation (check every 5 seconds)
sleep 20
log_gpu_memory "AFTER_MODEL_CREATION"
log_ray_actors "AFTER_MODEL_CREATION"

# Wait for training to complete or fail
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Stop background monitoring
kill $MONITOR_PID 2>/dev/null || true

# Final state
echo "" | tee -a "$LOG_FILE"
echo "=== Training Completed ===" | tee -a "$LOG_FILE"
echo "$(date): Training exit code: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
log_gpu_memory "FINAL"
log_ray_actors "FINAL"
log_system_memory "FINAL"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================"
echo "Monitoring log saved to: $LOG_FILE"
echo "================================================================================"

exit $TRAIN_EXIT_CODE
