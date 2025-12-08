#!/bin/bash
# Wrapper script to run GSM8K with local dataset transparently
#
# This script sets up HuggingFace environment variables to use local datasets
# without requiring code changes in tinker-cookbook

# Set HuggingFace to use local cache
export HF_HOME=/data
export HF_DATASETS_CACHE=/data/datasets
export HF_DATASETS_OFFLINE=0

# Create symlink for HuggingFace dataset cache structure if needed
CACHE_DIR="$HOME/.cache/huggingface/datasets"
mkdir -p "$CACHE_DIR"

# Symlink the local gsm8k to expected HuggingFace cache location
# HuggingFace expects: datasets/openai___gsm8k/main/...
if [ ! -e "$CACHE_DIR/openai___gsm8k" ]; then
    ln -s /data/datasets/gsm8k "$CACHE_DIR/openai___gsm8k"
fi

# Run the math RL training with GSM8K
python -u -m tinker_cookbook.recipes.math_rl.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    env=gsm8k \
    base_url=http://kgateway-training.slime-gmi:8000 \
    group_size=4 \
    groups_per_batch=2 \
    learning_rate=2e-4 \
    max_tokens=256 \
    eval_every=5 \
    save_every=10 \
    num_substeps=1 \
    behavior_if_log_dir_exists=delete "$@"
